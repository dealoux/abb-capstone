"""Detection system page with calibration-based coordinate system."""

import datetime
import glob
import json
from typing import Optional
import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime  # Add this import
from abbvisionsystem.camera.camera import BaslerCamera, WebcamCamera
from abbvisionsystem.models.defect_detection_model import DefectDetectionModel
from abbvisionsystem.models.yolo_model import YOLODefectModel
from abbvisionsystem.preprocessing.preprocessing import (
    prepare_for_detection,
    apply_image_enhancement,
)
from abbvisionsystem.utils.visualization import draw_detection_summary
from abbvisionsystem.camera.calibration import CameraCalibrator
import logging

logger = logging.getLogger(__name__)


@st.cache_resource
def _get_model(model_type):
    """Local model factory function to avoid circular imports"""
    MODEL_BASE_PATH = "trained_models"

    # Map of model types to their respective model paths and classes
    model_configs = {
        "defect_classification": {
            "path": "resnet_defect_classifier.keras",
            "class": DefectDetectionModel,
            "extra_args": {
                "class_mapping_path": os.path.join(
                    MODEL_BASE_PATH, "class_mapping.json"
                )
            },
        },
        "defect_yolo": {
            "path": "enhanced_yolo_defect_detector/weights/best.pt",
            "class": YOLODefectModel,
            "extra_args": {},
        },
    }

    # Check if model type is supported
    if model_type not in model_configs:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(model_configs.keys())}"
        )

    config = model_configs[model_type]

    model_path = os.path.join(MODEL_BASE_PATH, config["path"])

    if not os.path.exists(model_path):
        if model_type == "defect_yolo":
            alt_paths = [
                os.path.join(
                    MODEL_BASE_PATH,
                    "enhanced_yolo_defect_detector",
                    "weights",
                    "best.pt",
                ),
                os.path.join(
                    MODEL_BASE_PATH,
                    "enhanced_yolo_defect_detector",
                    "weights",
                    "last.pt",
                ),
                os.path.join(MODEL_BASE_PATH, "best.pt"),
                os.path.join(MODEL_BASE_PATH, "yolo_best.pt"),
                "best.pt",
                "enhanced_yolo_defect_detector/weights/best.pt",
                "yolo11s.pt",
            ]

            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    st.sidebar.info(f"📍 Using model: {os.path.basename(alt_path)}")
                    break
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    # Initialize the appropriate model class
    model_class = config["class"]
    extra_args = config["extra_args"]

    model = model_class(model_path=model_path, **extra_args)

    # Load the model
    if not model.load():
        raise RuntimeError(f"Failed to load {model_type} model from {model_path}")

    return model


def detection_system_page():
    """Main detection system interface with calibration-based coordinates."""
    st.title("🏠 Image Detection System")
    st.markdown(
        "Upload images or connect camera for real-time defect detection with calibrated measurements"
    )

    if "main_calibrator" not in st.session_state:
        st.session_state.main_calibrator = CameraCalibrator()
        # Auto-load latest calibration if available
        try:
            latest_calibration = find_latest_calibration_file()
            if latest_calibration:
                st.session_state.main_calibrator.load_calibration(latest_calibration)
                st.session_state.loaded_calibration_file = latest_calibration
        except Exception as e:
            logger.warning(f"Could not auto-load calibration: {e}")

    calibrator = st.session_state.main_calibrator

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Detection Settings")

        # Model type selection
        model_type = st.selectbox(
            "Detection Framework", ["defect_yolo", "defect_classification"], index=0
        )

        # Model selection section for YOLO
        if model_type == "defect_yolo":
            model_selection_section()

        # Detection parameters
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

        # Calibration management section
        st.subheader("📏 Calibration Management")

        # Current calibration status
        if calibrator.calibration_result:
            st.success("✅ System Calibrated")

            # Show loaded calibration file info
            if hasattr(st.session_state, "loaded_calibration_file"):
                filename = os.path.basename(st.session_state.loaded_calibration_file)
                st.info(f"📁 Loaded: {filename}")

            st.info(f"Scale: {calibrator.calibration_result.pixels_per_mm:.2f} px/mm")
            st.info(f"Error: {calibrator.calibration_result.reprojection_error:.3f}")

            # Show calibration date if available
            if calibrator.calibration_result.calibration_date:
                st.info(f"Date: {calibrator.calibration_result.calibration_date}")

        else:
            st.warning("⚠️ System Not Calibrated")
            st.info("Upload a calibration file or go to Camera Calibration page")

        # Calibration file upload
        calibration_upload_section()

        # Coordinate system settings
        st.subheader("🎯 Coordinate Display")
        show_coordinate_system = st.checkbox("Show Coordinate System", value=True)
        show_measurements = st.checkbox("Show Object Measurements", value=True)
        show_origin_lines = st.checkbox("Show Origin Connection Lines", value=True)

    # Main content area with dynamic model selection
    selected_model_path = get_selected_model_path(model_type)

    tab1, tab2 = st.tabs(["📤 Image Upload", "📷 Camera Detection"])

    with tab1:
        image_detection_interface(
            calibrator,
            model_type,
            selected_model_path,
            conf_threshold,
            iou_threshold,
            show_coordinate_system,
            show_measurements,
            show_origin_lines,
        )

    with tab2:
        camera_detection_interface(
            calibrator,
            model_type,
            selected_model_path,
            conf_threshold,
            iou_threshold,
            show_coordinate_system,
            show_measurements,
            show_origin_lines,
        )


def model_selection_section():
    """Handle YOLO model selection and management."""
    st.subheader("🤖 YOLO Model Selection")

    # Get available models
    available_models = get_available_yolo_models()

    # Current model status
    if hasattr(st.session_state, "selected_model_path"):
        current_model = st.session_state.selected_model_path
        if os.path.exists(current_model):
            st.success(f"✅ Model Loaded")
            model_name = os.path.basename(current_model)
            st.info(f"📁 Current: {model_name}")

            # Show model size
            model_size_mb = os.path.getsize(current_model) / (1024 * 1024)
            st.info(f"📦 Size: {model_size_mb:.1f} MB")
        else:
            st.warning(f"⚠️ Model file not found: {os.path.basename(current_model)}")
    else:
        st.info("No model selected")

    # Model categories
    trained_models = available_models.get("trained", [])
    pretrained_models = available_models.get("pretrained", [])

    # Trained models section
    if trained_models:
        st.write("**🎯 Trained Models (Recommended)**")

        # Create display names and paths
        trained_options = ["None"] + [
            f"{m['name']} ({m['size_mb']:.1f}MB)" for m in trained_models
        ]
        trained_paths = [None] + [m["path"] for m in trained_models]

        selected_trained_idx = 0
        if hasattr(st.session_state, "selected_model_path"):
            current_path = st.session_state.selected_model_path
            if current_path in trained_paths:
                selected_trained_idx = trained_paths.index(current_path)

        selected_trained = st.selectbox(
            "Choose trained model:",
            options=trained_options,
            index=selected_trained_idx,
            key="trained_model_selector",
        )

        if selected_trained != "None":
            selected_path = trained_paths[trained_options.index(selected_trained)]

            # Show model details
            model_info = next(m for m in trained_models if m["path"] == selected_path)

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Modified:** {model_info['date_str']}")
                st.write(f"**Type:** {model_info['model_type']}")
            with col2:
                st.write(f"**Accuracy:** {model_info.get('accuracy', 'Unknown')}")
                st.write(f"**Epochs:** {model_info.get('epochs', 'Unknown')}")

            if st.button("🔄 Load Trained Model"):
                st.session_state.selected_model_path = selected_path
                st.success(f"✅ Loaded: {model_info['name']}")
                st.rerun()
    else:
        st.info("No trained models found")

    st.write("---")

    # Pretrained models section
    st.write("**⚡ Pretrained Models (General Purpose)**")

    pretrained_options = ["None"] + [
        f"{m['name']} - {m['description']}" for m in pretrained_models
    ]
    pretrained_paths = [None] + [m["path"] for m in pretrained_models]

    selected_pretrained_idx = 0
    if hasattr(st.session_state, "selected_model_path"):
        current_path = st.session_state.selected_model_path
        if current_path in pretrained_paths:
            selected_pretrained_idx = pretrained_paths.index(current_path)

    selected_pretrained = st.selectbox(
        "Choose pretrained model:",
        options=pretrained_options,
        index=selected_pretrained_idx,
        key="pretrained_model_selector",
    )

    if selected_pretrained != "None":
        selected_path = pretrained_paths[pretrained_options.index(selected_pretrained)]

        # Show model details
        model_info = next(m for m in pretrained_models if m["path"] == selected_path)
        st.info(f"ℹ️ {model_info['description']}")
        st.warning("⚠️ Pretrained models may not be optimized for defect detection")

        if st.button("🔄 Load Pretrained Model"):
            st.session_state.selected_model_path = selected_path
            st.success(f"✅ Loaded: {model_info['name']}")
            st.rerun()

    # Model upload section
    st.write("---")
    st.write("**📤 Upload Custom Model**")

    uploaded_model = st.file_uploader(
        "Upload YOLO model (.pt file)",
        type=["pt"],
        key="model_upload",
        help="Upload a custom trained YOLO model file",
    )

    if uploaded_model is not None:
        try:
            # Save uploaded model
            model_dir = "uploaded_models"
            os.makedirs(model_dir, exist_ok=True)

            model_path = os.path.join(model_dir, uploaded_model.name)

            with open(model_path, "wb") as f:
                f.write(uploaded_model.read())

            # Validate model
            if validate_yolo_model(model_path):
                st.session_state.selected_model_path = model_path
                st.success(f"✅ Uploaded and loaded: {uploaded_model.name}")

                # Show model info
                model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                st.info(f"📦 Size: {model_size_mb:.1f} MB")

                st.rerun()
            else:
                st.error("Invalid YOLO model file")
                os.remove(model_path)

        except Exception as e:
            st.error(f"Error uploading model: {str(e)}")

    # Clear model option
    if st.button("🗑️ Clear Selected Model"):
        if hasattr(st.session_state, "selected_model_path"):
            del st.session_state.selected_model_path
        st.success("Model selection cleared")
        st.rerun()


def get_available_yolo_models():
    """Get all available YOLO models categorized by type."""
    models = {"trained": [], "pretrained": []}

    # Search for trained models
    trained_search_patterns = [
        "trained_models/*/weights/best.pt",
        "trained_models/*/weights/last.pt",
        "trained_models/*.pt",
        "models/*/best.pt",
        "models/*/last.pt",
        "weights/*.pt",
        "best.pt",
        "last.pt",
    ]

    found_trained = set()  # Avoid duplicates

    for pattern in trained_search_patterns:
        for model_path in glob.glob(pattern):
            if os.path.isfile(model_path) and model_path not in found_trained:
                try:
                    # Extract model information
                    model_name = extract_model_name(model_path)
                    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    modified_time = os.path.getmtime(model_path)

                    # Fix the datetime usage here
                    date_str = datetime.fromtimestamp(modified_time).strftime(
                        "%Y-%m-%d %H:%M"
                    )

                    # Determine model type from path/name
                    model_type = "Custom"
                    if "enhanced" in model_path.lower():
                        model_type = "Enhanced YOLO"
                    elif "defect" in model_path.lower():
                        model_type = "Defect Detection"
                    elif "best" in model_path.lower():
                        model_type = "Best Checkpoint"
                    elif "last" in model_path.lower():
                        model_type = "Latest Checkpoint"

                    # Try to extract training info if available
                    accuracy, epochs = extract_training_info(model_path)

                    models["trained"].append(
                        {
                            "name": model_name,
                            "path": model_path,
                            "size_mb": model_size_mb,
                            "modified": modified_time,
                            "date_str": date_str,
                            "model_type": model_type,
                            "accuracy": accuracy,
                            "epochs": epochs,
                        }
                    )

                    found_trained.add(model_path)

                except Exception as e:
                    logger.warning(f"Error processing model {model_path}: {e}")
                    continue

    # Sort trained models by modification time (newest first)
    models["trained"].sort(key=lambda x: x["modified"], reverse=True)

    # Define pretrained models
    pretrained_models_config = [
        {
            "name": "YOLOv8n",
            "path": "yolov8n.pt",
            "description": "Nano - Fastest, lowest accuracy",
        },
        {
            "name": "YOLOv8s",
            "path": "yolov8s.pt",
            "description": "Small - Good balance of speed and accuracy",
        },
        {
            "name": "YOLOv8m",
            "path": "yolov8m.pt",
            "description": "Medium - Higher accuracy, slower",
        },
        {
            "name": "YOLOv8l",
            "path": "yolov8l.pt",
            "description": "Large - High accuracy, slower",
        },
        {
            "name": "YOLOv8x",
            "path": "yolov8x.pt",
            "description": "Extra Large - Highest accuracy, slowest",
        },
        {
            "name": "YOLO11s",
            "path": "yolo11s.pt",
            "description": "YOLO11 Small - Latest version",
        },
        {
            "name": "YOLO11m",
            "path": "yolo11m.pt",
            "description": "YOLO11 Medium - Latest version",
        },
    ]

    models["pretrained"] = pretrained_models_config

    return models


def extract_model_name(model_path):
    """Extract a readable model name from the file path."""
    # Get directory and filename info
    dir_name = os.path.basename(os.path.dirname(model_path))
    file_name = os.path.basename(model_path)

    # Create meaningful name
    if dir_name and dir_name not in ["weights", "models", "."]:
        # Use directory name if meaningful
        model_name = dir_name.replace("_", " ").title()
        if file_name != "best.pt":
            model_name += f" ({file_name})"
    else:
        # Use filename
        model_name = file_name.replace(".pt", "").replace("_", " ").title()

    return model_name


def extract_training_info(model_path):
    """Try to extract training information from model file or associated files."""
    accuracy = "Unknown"
    epochs = "Unknown"

    try:
        # Look for results.csv or results.txt in parent directory
        model_dir = os.path.dirname(model_path)
        parent_dir = os.path.dirname(model_dir)

        # Check for results files
        for results_file in ["results.csv", "results.txt"]:
            results_path = os.path.join(parent_dir, results_file)
            if os.path.exists(results_path):
                try:
                    # Try to read training results
                    with open(results_path, "r") as f:
                        lines = f.readlines()
                        if lines:
                            # Parse last line for final metrics
                            last_line = lines[-1].strip()
                            if "," in last_line:
                                parts = last_line.split(",")
                                if len(parts) > 1:
                                    epochs = parts[0].strip()
                                if len(parts) > 6:  # Typical CSV format
                                    accuracy = f"{float(parts[6].strip()):.3f}"
                    break
                except Exception as parse_error:
                    logger.debug(
                        f"Could not parse results file {results_path}: {parse_error}"
                    )
                    continue

        # Try to load model metadata if available
        try:
            import torch

            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location="cpu")
                if isinstance(checkpoint, dict):
                    if "epoch" in checkpoint:
                        epochs = str(checkpoint["epoch"])
                    if "best_fitness" in checkpoint:
                        accuracy = f"{checkpoint['best_fitness']:.3f}"
        except Exception as torch_error:
            logger.debug(
                f"Could not load model metadata for {model_path}: {torch_error}"
            )
            pass

    except Exception as e:
        logger.debug(f"Could not extract training info for {model_path}: {e}")

    return accuracy, epochs


def validate_yolo_model(model_path):
    """Validate that the uploaded file is a valid YOLO model."""
    try:
        # Check file extension
        if not model_path.endswith(".pt"):
            return False

        # Check file size (should be reasonable for a model)
        file_size = os.path.getsize(model_path)
        if file_size < 1024 * 1024:  # Less than 1MB is suspicious
            return False

        if file_size > 500 * 1024 * 1024:  # More than 500MB is too large
            return False

        # Try to load with torch (basic validation)
        import torch

        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            # Basic checks for YOLO model structure
            if isinstance(checkpoint, dict):
                return True
            return False
        except:
            return False

    except Exception as e:
        logger.error(f"Model validation error: {e}")
        return False


def get_selected_model_path(model_type):
    """Get the currently selected model path."""
    if model_type == "defect_yolo":
        if hasattr(st.session_state, "selected_model_path"):
            return st.session_state.selected_model_path
        else:
            # Return default model path
            return None
    else:
        # For non-YOLO models, return None to use default behavior
        return None


@st.cache_resource
def _get_model_with_path(model_type, model_path=None):
    """Enhanced model factory function with custom model path support."""
    MODEL_BASE_PATH = "trained_models"

    # Map of model types to their respective model paths and classes
    model_configs = {
        "defect_classification": {
            "path": "resnet_defect_classifier.keras",
            "class": DefectDetectionModel,
            "extra_args": {
                "class_mapping_path": os.path.join(
                    MODEL_BASE_PATH, "class_mapping.json"
                )
            },
        },
        "defect_yolo": {
            "path": "enhanced_yolo_defect_detector/weights/best.pt",
            "class": YOLODefectModel,
            "extra_args": {},
        },
    }

    # Check if model type is supported
    if model_type not in model_configs:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(model_configs.keys())}"
        )

    config = model_configs[model_type]

    # Use custom model path if provided, otherwise use default
    if model_path and os.path.exists(model_path):
        final_model_path = model_path
        st.sidebar.success(f"✅ Using selected model: {os.path.basename(model_path)}")
    else:
        # Use default path resolution
        final_model_path = os.path.join(MODEL_BASE_PATH, config["path"])

        if not os.path.exists(final_model_path):
            if model_type == "defect_yolo":
                alt_paths = [
                    os.path.join(
                        MODEL_BASE_PATH,
                        "enhanced_yolo_defect_detector",
                        "weights",
                        "best.pt",
                    ),
                    os.path.join(
                        MODEL_BASE_PATH,
                        "enhanced_yolo_defect_detector",
                        "weights",
                        "last.pt",
                    ),
                    os.path.join(MODEL_BASE_PATH, "best.pt"),
                    os.path.join(MODEL_BASE_PATH, "yolo_best.pt"),
                    "best.pt",
                    "enhanced_yolo_defect_detector/weights/best.pt",
                    "yolov8n.pt",  # Fallback to pretrained
                ]

                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        final_model_path = alt_path
                        st.sidebar.info(
                            f"📍 Using fallback: {os.path.basename(alt_path)}"
                        )
                        break
            else:
                raise FileNotFoundError(f"Model file not found: {final_model_path}")

    # Initialize the appropriate model class
    model_class = config["class"]
    extra_args = config["extra_args"]

    model = model_class(model_path=final_model_path, **extra_args)

    # Load the model
    if not model.load():
        raise RuntimeError(f"Failed to load {model_type} model from {final_model_path}")

    return model


def image_detection_interface(
    calibrator,
    model_type,
    model_path,
    conf_threshold,
    iou_threshold,
    show_coordinate_system,
    show_measurements,
    show_origin_lines,
):
    """Image upload detection interface."""
    st.subheader("Upload Image for Detection")

    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        key="detection_upload",
    )

    if uploaded_file is not None:
        try:
            # Convert uploaded file to image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Failed to load image. Please check the file format.")
                return

            # Display original image
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Detection Results")

                # Load and run model with custom path
                try:
                    model = _get_model_with_path(model_type, model_path)

                    # Run detection
                    with st.spinner("Running detection..."):
                        detections = model.predict(
                            image,
                            conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold,
                        )

                    # Process results with calibration-based coordinates
                    result_image = process_detection_results(
                        image,
                        detections,
                        calibrator,
                        show_coordinate_system,
                        show_measurements,
                        show_origin_lines,
                    )

                    # Display result
                    st.image(
                        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                        use_column_width=True,
                    )

                except Exception as e:
                    st.error(f"Detection failed: {str(e)}")
                    st.image(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                        caption="Original Image",
                        use_column_width=True,
                    )

            with col2:
                st.subheader("📊 Detection Summary")

                if (
                    "detections" in locals()
                    and detections
                    and detections.get("num_detections", 0) > 0
                ):
                    display_detection_measurements(
                        detections, calibrator, image.shape[:2]
                    )
                else:
                    st.info("No objects detected")

                # Show calibration info
                st.subheader("📏 Measurement Info")
                if calibrator.calibration_result:
                    st.success(f"✅ Calibrated System")
                    st.write(
                        f"**Scale:** {calibrator.calibration_result.pixels_per_mm:.2f} pixels/mm"
                    )
                    st.write(
                        f"**Resolution:** {1/calibrator.calibration_result.pixels_per_mm:.4f} mm/pixel"
                    )
                else:
                    st.warning("⚠️ Uncalibrated - showing pixel coordinates")
                    st.write(
                        "Go to Camera Calibration to enable real-world measurements"
                    )

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")


def camera_detection_interface(
    calibrator,
    model_type,
    model_path,
    conf_threshold,
    iou_threshold,
    show_coordinate_system,
    show_measurements,
    show_origin_lines,
):
    """Camera-based detection interface."""
    st.subheader("Live Camera Detection")

    # Camera connection controls
    col1, col2 = st.columns(2)

    with col1:
        camera_type = st.selectbox(
            "Camera Type", ["Basler Camera", "Webcam"], key="cam_type"
        )
        device_index = st.number_input(
            "Device Index", min_value=0, max_value=5, value=0, key="cam_index"
        )

    with col2:
        if st.button("📷 Connect Camera"):
            try:
                if camera_type == "Basler Camera":
                    camera = BaslerCamera(device_index=device_index)
                else:
                    camera = WebcamCamera(camera_id=device_index)

                if camera.connect():
                    st.session_state.camera = camera
                    st.success("Camera connected successfully!")
                else:
                    st.error("Failed to connect to camera")
            except Exception as e:
                st.error(f"Camera connection error: {str(e)}")

        if st.button("🔌 Disconnect Camera"):
            if hasattr(st.session_state, "camera") and st.session_state.camera:
                st.session_state.camera.disconnect()
                del st.session_state.camera
                st.success("Camera disconnected")

    # Live detection
    if (
        hasattr(st.session_state, "camera")
        and st.session_state.camera
        and st.session_state.camera.connected
    ):
        st.success("📷 Camera Ready")

        col_capture, col_live = st.columns(2)

        with col_capture:
            if st.button("📸 Capture & Detect"):
                try:
                    # Capture image
                    image = (
                        st.session_state.camera.capture_corrected_image()
                    )  # Uses auto-undistort if calibrated

                    if image is not None:
                        # Store captured image
                        st.session_state.captured_image = image

                        # Run detection with custom model path
                        model = _get_model_with_path(model_type, model_path)
                        detections = model.predict(
                            image,
                            conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold,
                        )

                        # Process with calibration
                        result_image = process_detection_results(
                            image,
                            detections,
                            calibrator,
                            show_coordinate_system,
                            show_measurements,
                            show_origin_lines,
                        )

                        st.session_state.detection_result = result_image
                        st.session_state.detections = detections

                    else:
                        st.error("Failed to capture image")

                except Exception as e:
                    st.error(f"Capture error: {str(e)}")

        # Display results
        if hasattr(st.session_state, "detection_result"):
            st.subheader("Latest Detection Result")

            col_result, col_measurements = st.columns([2, 1])

            with col_result:
                st.image(
                    cv2.cvtColor(st.session_state.detection_result, cv2.COLOR_BGR2RGB),
                    use_column_width=True,
                )

            with col_measurements:
                st.subheader("📊 Measurements")
                if (
                    hasattr(st.session_state, "detections")
                    and st.session_state.detections
                ):
                    display_detection_measurements(
                        st.session_state.detections,
                        calibrator,
                        st.session_state.captured_image.shape[:2],
                    )
    else:
        st.info("Connect a camera to start live detection")


def find_latest_calibration_file() -> Optional[str]:
    """Find the latest calibration file in the calibrations directory."""
    calibrations_dir = "calibrations"

    if not os.path.exists(calibrations_dir):
        return None

    calibration_files = []
    for filename in os.listdir(calibrations_dir):
        if filename.startswith("calibration_") and filename.endswith(".json"):
            filepath = os.path.join(calibrations_dir, filename)
            if os.path.isfile(filepath):
                # Extract timestamp from filename
                try:
                    timestamp_part = filename.replace("calibration_", "").replace(
                        ".json", ""
                    )
                    calibration_files.append((filepath, timestamp_part))
                except:
                    continue

    if not calibration_files:
        return None

    # Sort by timestamp (most recent first)
    calibration_files.sort(key=lambda x: x[1], reverse=True)
    return calibration_files[0][0]


def calibration_upload_section():
    """Handle calibration file upload and management."""
    st.subheader("📤 Upload Calibration")

    # File upload for calibration
    uploaded_calibration = st.file_uploader(
        "Choose a calibration file",
        type=["json"],
        key="calibration_upload",
        help="Upload a calibration JSON file (calibration_YYYYMMDD_HHMMSS.json)",
    )

    if uploaded_calibration is not None:
        try:
            # Read and parse the uploaded file
            calibration_content = uploaded_calibration.read()
            calibration_data = json.loads(calibration_content.decode("utf-8"))

            # Validate calibration file format
            if validate_calibration_format(calibration_data):
                # Create temporary file to load calibration
                temp_calibration_path = f"temp_calibration_{uploaded_calibration.name}"

                # Save temporary file
                with open(temp_calibration_path, "w") as f:
                    json.dump(calibration_data, f, indent=2)

                # Load calibration
                if st.session_state.main_calibrator.load_calibration(
                    temp_calibration_path
                ):
                    st.success(f"✅ Calibration loaded: {uploaded_calibration.name}")
                    st.session_state.loaded_calibration_file = uploaded_calibration.name

                    # Display calibration info
                    display_calibration_info(calibration_data)

                    # Clean up temporary file
                    if os.path.exists(temp_calibration_path):
                        os.remove(temp_calibration_path)

                    # Force refresh
                    st.rerun()
                else:
                    st.error("Failed to load calibration file")
                    if os.path.exists(temp_calibration_path):
                        os.remove(temp_calibration_path)
            else:
                st.error("Invalid calibration file format")

        except json.JSONDecodeError:
            st.error("Invalid JSON format")
        except Exception as e:
            st.error(f"Error loading calibration: {str(e)}")

    # Option to browse available calibration files
    st.subheader("📂 Available Calibrations")

    calibrations_dir = "calibrations"
    if os.path.exists(calibrations_dir):
        available_calibrations = []
        for filename in os.listdir(calibrations_dir):
            if filename.startswith("calibration_") and filename.endswith(".json"):
                filepath = os.path.join(calibrations_dir, filename)
                if os.path.isfile(filepath):
                    # Get file modification time
                    mtime = os.path.getmtime(filepath)
                    available_calibrations.append((filename, filepath, mtime))

        if available_calibrations:
            # Sort by modification time (most recent first)
            available_calibrations.sort(key=lambda x: x[2], reverse=True)

            selected_calibration = st.selectbox(
                "Select calibration file:",
                options=["None"] + [f[0] for f in available_calibrations],
                key="calibration_selector",
            )

            if selected_calibration != "None":
                selected_path = next(
                    f[1] for f in available_calibrations if f[0] == selected_calibration
                )

                if st.button("🔄 Load Selected Calibration"):
                    if st.session_state.main_calibrator.load_calibration(selected_path):
                        st.success(f"✅ Loaded: {selected_calibration}")
                        st.session_state.loaded_calibration_file = selected_path
                        st.rerun()
                    else:
                        st.error("Failed to load calibration")
        else:
            st.info("No calibration files found in calibrations directory")
    else:
        st.info("Calibrations directory not found")

    # Clear calibration option
    if st.button("🗑️ Clear Current Calibration"):
        st.session_state.main_calibrator.calibration_result = None
        if hasattr(st.session_state, "loaded_calibration_file"):
            del st.session_state.loaded_calibration_file
        st.success("Calibration cleared")
        st.rerun()


def validate_calibration_format(calibration_data: dict) -> bool:
    """Validate that the uploaded file has the correct calibration format."""
    required_fields = [
        "camera_matrix",
        "distortion_coefficients",
        "reprojection_error",
        "image_size",
    ]

    try:
        # Check required fields exist
        for field in required_fields:
            if field not in calibration_data:
                logger.error(f"Missing required field: {field}")
                return False

        # Validate camera matrix format
        camera_matrix = calibration_data["camera_matrix"]
        if not isinstance(camera_matrix, list) or len(camera_matrix) != 3:
            return False

        for row in camera_matrix:
            if not isinstance(row, list) or len(row) != 3:
                return False

        # Validate distortion coefficients
        dist_coeffs = calibration_data["distortion_coefficients"]
        if not isinstance(dist_coeffs, list) or len(dist_coeffs) == 0:
            return False

        # Validate numeric fields
        if not isinstance(calibration_data["reprojection_error"], (int, float)):
            return False

        # Validate image size
        image_size = calibration_data["image_size"]
        if not isinstance(image_size, list) or len(image_size) != 2:
            return False

        return True

    except Exception as e:
        logger.error(f"Calibration validation error: {e}")
        return False


def display_calibration_info(calibration_data: dict):
    """Display information about the loaded calibration."""
    st.subheader("📊 Calibration Details")

    # Basic info
    col1, col2 = st.columns(2)

    with col1:
        st.write(
            f"**Reprojection Error:** {calibration_data['reprojection_error']:.4f}"
        )
        st.write(
            f"**Image Size:** {calibration_data['image_size'][0]} × {calibration_data['image_size'][1]}"
        )

        if "pixels_per_mm" in calibration_data and calibration_data["pixels_per_mm"]:
            st.write(f"**Scale:** {calibration_data['pixels_per_mm']:.2f} px/mm")
            st.write(f"**Resolution:** {1/calibration_data['pixels_per_mm']:.4f} mm/px")

    with col2:
        if "calibration_date" in calibration_data:
            st.write(f"**Date:** {calibration_data['calibration_date']}")

        if "chessboard_size" in calibration_data:
            size = calibration_data["chessboard_size"]
            st.write(f"**Pattern:** {size[0]} × {size[1]}")

        if "square_size_mm" in calibration_data:
            st.write(f"**Square Size:** {calibration_data['square_size_mm']} mm")

    # Camera matrix (collapsible)
    with st.expander("🔍 Camera Matrix"):
        camera_matrix = np.array(calibration_data["camera_matrix"])
        st.text(f"fx: {camera_matrix[0,0]:.2f}")
        st.text(f"fy: {camera_matrix[1,1]:.2f}")
        st.text(f"cx: {camera_matrix[0,2]:.2f}")
        st.text(f"cy: {camera_matrix[1,2]:.2f}")

    # Distortion coefficients (collapsible)
    with st.expander("📐 Distortion Coefficients"):
        dist_coeffs = calibration_data["distortion_coefficients"][0]
        st.text(f"k1: {dist_coeffs[0]:.6f}")
        st.text(f"k2: {dist_coeffs[1]:.6f}")
        if len(dist_coeffs) > 2:
            st.text(f"p1: {dist_coeffs[2]:.6f}")
        if len(dist_coeffs) > 3:
            st.text(f"p2: {dist_coeffs[3]:.6f}")
        if len(dist_coeffs) > 4:
            st.text(f"k3: {dist_coeffs[4]:.6f}")


def process_detection_results(
    image,
    detections,
    calibrator,
    show_coordinate_system=True,
    show_measurements=True,
    show_origin_lines=True,
):
    """Process detection results with calibration-based coordinate system."""
    result_image = image.copy()

    # Get calibrated origin point
    origin_point = get_calibrated_origin(calibrator, image.shape[:2])
    ox, oy = origin_point

    # Draw coordinate system if requested
    if show_coordinate_system:
        draw_coordinate_system(result_image, origin_point, calibrator)

    # Process detections
    if detections and detections.get("num_detections", 0) > 0:
        for i in range(detections["num_detections"]):
            try:
                # Get bounding box
                if (
                    "absolute_boxes" in detections
                    and len(detections["absolute_boxes"]) > 0
                ):
                    box = detections["absolute_boxes"][i]
                else:
                    # Convert from relative coordinates
                    h, w = image.shape[:2]
                    box = detections["boxes"][i] * [w, h, w, h]

                x1, y1, x2, y2 = map(int, box)

                # Calculate object center and dimensions
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1

                # Get object region for contour analysis
                object_region = image[y1:y2, x1:x2]
                if object_region.size > 0:
                    # Find the most accurate object center using contours
                    center_x, center_y, angle = find_object_center_and_orientation(
                        object_region, (x1, y1)
                    )

                # Draw bounding box
                class_id = detections["classes"][i] if "classes" in detections else 0
                color = (
                    (0, 0, 255) if class_id == 1 else (0, 255, 0)
                )  # Red for defect, Green for normal
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

                # Draw object center
                cv2.circle(result_image, (center_x, center_y), 4, (0, 0, 255), -1)

                # Draw connection line to origin if requested
                if show_origin_lines:
                    cv2.line(
                        result_image,
                        (ox, oy),
                        (center_x, center_y),
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                # Calculate and display measurements
                if show_measurements:
                    display_object_measurements(
                        result_image,
                        (center_x, center_y),
                        (width, height),
                        origin_point,
                        calibrator,
                        i + 1,
                        angle if "angle" in locals() else 0,
                    )

            except Exception as e:
                logger.error(f"Error processing detection {i}: {e}")
                continue

    return result_image


def get_calibrated_origin(calibrator, image_shape):
    """Get the calibrated origin point or fallback to default."""
    if calibrator.calibration_result and calibrator.calibration_images:
        try:
            # Get origin from first calibration image (first detected corner)
            first_calib = calibrator.calibration_images[0]
            if first_calib["corners"] is not None and len(first_calib["corners"]) > 0:
                origin_point = tuple(first_calib["corners"][0].ravel().astype(int))
                return origin_point
        except Exception as e:
            logger.warning(f"Could not get calibrated origin: {e}")

    # Fallback to image center or default position
    height, width = image_shape
    return (width // 4, height // 4)  # Top-left quadrant as default


def draw_coordinate_system(image, origin_point, calibrator):
    """Draw coordinate system at the calibrated origin."""
    ox, oy = origin_point

    # Draw origin point (larger yellow circle like in calibration)
    cv2.circle(image, (ox, oy), 8, (255, 255, 0), -1)

    # Draw coordinate axes (matching calibration style)
    axis_length = 100

    # X-axis (Red) - horizontal right
    cv2.arrowedLine(
        image, (ox, oy), (ox + axis_length, oy), (0, 0, 255), 3, tipLength=0.1
    )
    # Y-axis (Green) - vertical down
    cv2.arrowedLine(
        image, (ox, oy), (ox, oy + axis_length), (0, 255, 0), 3, tipLength=0.1
    )

    # Add labels (matching calibration style)
    cv2.putText(
        image,
        "O(0,0)",
        (ox - 30, oy - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        image,
        "X",
        (ox + axis_length + 10, oy + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        image,
        "Y",
        (ox - 15, oy + axis_length + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )


def find_object_center_and_orientation(object_region, region_offset):
    """Find accurate object center and orientation using contour analysis."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)

        # Threshold to create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get minimum area rectangle for orientation
            rect = cv2.minAreaRect(largest_contour)
            center = rect[0]
            angle = rect[2]

            # Convert to absolute coordinates
            abs_center_x = int(center[0] + region_offset[0])
            abs_center_y = int(center[1] + region_offset[1])

            # Normalize angle
            if rect[1][0] < rect[1][1]:
                angle = angle + 90
            angle = angle % 180

            return abs_center_x, abs_center_y, angle
    except:
        pass

    # Fallback to bounding box center
    h, w = object_region.shape[:2]
    return region_offset[0] + w // 2, region_offset[1] + h // 2, 0


def display_object_measurements(
    image, center, dimensions, origin_point, calibrator, obj_id, angle=0
):
    """Display object measurements on the image."""
    center_x, center_y = center
    width, height = dimensions
    ox, oy = origin_point

    # Calculate relative coordinates
    rel_x_px = center_x - ox
    rel_y_px = center_y - oy

    # Convert to real-world coordinates if calibrated
    if calibrator.calibration_result and calibrator.calibration_result.pixels_per_mm:
        pixels_per_mm = calibrator.calibration_result.pixels_per_mm
        rel_x_mm = rel_x_px / pixels_per_mm
        rel_y_mm = rel_y_px / pixels_per_mm
        width_mm = width / pixels_per_mm
        height_mm = height / pixels_per_mm

        coord_text = f"({rel_x_mm:.1f}, {rel_y_mm:.1f})mm"
        dimensions_text = f"{width_mm:.1f}x{height_mm:.1f}mm"
        unit = "mm"
    else:
        coord_text = f"({rel_x_px}, {rel_y_px})px"
        dimensions_text = f"{width}x{height}px"
        unit = "px"

    # Draw measurements on image
    y_offset = -60
    cv2.putText(
        image,
        f"ID: {obj_id}",
        (center_x - 30, center_y + y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        coord_text,
        (center_x - 60, center_y + y_offset + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        image,
        f"{angle:.1f}°",
        (center_x - 30, center_y + y_offset + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2,
    )
    cv2.putText(
        image,
        dimensions_text,
        (center_x - 40, center_y + y_offset + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        2,
    )


def display_detection_measurements(detections, calibrator, image_shape):
    """Display detection measurements in the sidebar."""
    if not detections or detections.get("num_detections", 0) == 0:
        st.info("No objects detected")
        return

    # Get origin and calibration info
    origin_point = get_calibrated_origin(calibrator, image_shape)
    ox, oy = origin_point

    pixels_per_mm = None
    unit = "px"
    if calibrator.calibration_result and calibrator.calibration_result.pixels_per_mm:
        pixels_per_mm = calibrator.calibration_result.pixels_per_mm
        unit = "mm"
        st.success(f"✅ Calibrated ({pixels_per_mm:.2f} px/mm)")
    else:
        st.warning("⚠️ Uncalibrated")

    st.write(f"**Origin:** ({ox}, {oy}) pixels")
    st.write("---")

    # Display each detection
    for i in range(detections["num_detections"]):
        try:
            # Get detection data
            if "absolute_boxes" in detections and len(detections["absolute_boxes"]) > 0:
                box = detections["absolute_boxes"][i]
            else:
                h, w = image_shape
                box = detections["boxes"][i] * [w, h, w, h]

            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1

            # Calculate measurements
            rel_x_px = center_x - ox
            rel_y_px = center_y - oy

            if pixels_per_mm:
                rel_x = rel_x_px / pixels_per_mm
                rel_y = rel_y_px / pixels_per_mm
                obj_width = width / pixels_per_mm
                obj_height = height / pixels_per_mm
                distance = np.sqrt(rel_x**2 + rel_y**2)
            else:
                rel_x = rel_x_px
                rel_y = rel_y_px
                obj_width = width
                obj_height = height
                distance = np.sqrt(rel_x_px**2 + rel_y_px**2)

            # Get class info
            class_id = detections["classes"][i] if "classes" in detections else 0
            confidence = detections["scores"][i] if "scores" in detections else 0

            # Display object info
            st.write(f"**Object {i+1}:**")
            st.write(f"- Position: ({rel_x:.1f}, {rel_y:.1f}) {unit}")
            st.write(f"- Dimensions: {obj_width:.1f} × {obj_height:.1f} {unit}")
            st.write(f"- Distance from origin: {distance:.1f} {unit}")
            st.write(f"- Confidence: {confidence:.2f}")

            # Show class if available
            if hasattr(detections, "labels") and detections["labels"]:
                st.write(f"- Class: {detections['labels'][i]}")
            elif class_id == 1:
                st.write(f"- Class: Defect")
            else:
                st.write(f"- Class: Normal")

            st.write("---")

        except Exception as e:
            st.error(f"Error displaying object {i+1}: {str(e)}")


if __name__ == "__main__":
    detection_system_page()
