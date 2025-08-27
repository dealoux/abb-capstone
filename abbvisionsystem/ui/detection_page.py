"""Detection system UI page with multi-object support"""

import os
import time
import streamlit as st
import numpy as np
import cv2

from abbvisionsystem.models.defect_detection_model import DefectDetectionModel
from abbvisionsystem.models.yolo_model import YOLODefectModel

from abbvisionsystem.camera.camera import BaslerCamera, WebcamCamera
from abbvisionsystem.preprocessing.preprocessing import (
    prepare_for_detection,
    apply_image_enhancement,
)
from abbvisionsystem.utils.visualization import draw_detection_summary


def detection_system_page():
    """Main detection system interface with multi-object support."""
    st.title("♻️ ABB Vision System")
    st.write("Multi-object defect detection system with camera integration")

    # Check calibration status
    if st.session_state.camera and hasattr(st.session_state.camera, "calibrator"):
        if st.session_state.camera.calibrator.calibration_result:
            st.success("📷 Camera is calibrated")
            scale = st.session_state.camera.calibrator.calibration_result.pixels_per_mm
            if scale:
                st.info(f"Scale: {scale:.2f} pixels/mm")
        else:
            st.warning("⚠️ Camera not calibrated. Go to Camera Calibration page.")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Model selection - Updated for multi-object detection
        model_type = st.selectbox(
            "Select Detection Model",
            [
                "YOLO Defect Detection (Multi-object)",
                "ResNet Defect Classification (Single object)",
            ],
        )

        model_type_map = {
            "YOLO Defect Detection (Multi-object)": "defect_yolo",
            "ResNet Defect Classification (Single object)": "defect_classification",
        }

        # Input selection
        input_option = st.radio(
            "Select Input Source", ["Upload Image", "Camera Integration"]
        )

        # Image enhancement options
        st.subheader("Image Enhancement")
        brightness = st.slider("Brightness", -100, 100, 0)
        contrast = st.slider("Contrast", -100, 100, 0)

        # Detection settings
        st.subheader("Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

        # Multi-object specific settings
        if "YOLO" in model_type:
            iou_threshold = st.slider("IoU Threshold (NMS)", 0.1, 1.0, 0.45, 0.05)
            max_detections = st.slider("Max Detections", 1, 50, 20)

        # Apply enhancements button
        enhance_button = st.button("Apply Settings")

    # Load the selected model - Fixed import approach
    try:
        # Import the model factory function directly
        import sys
        import os

        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Use the model factory logic directly
        model = _get_model_instance(model_type_map[model_type])

        # Show model info
        st.sidebar.success(f"✅ {model_type} loaded")
        if hasattr(model, "model_path"):
            st.sidebar.info(f"Path: {os.path.basename(model.model_path)}")

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("💡 Make sure you've trained a model using the Training Center first!")
        return

    # Main content area with two columns
    col1, col2 = st.columns(2)

    # Input handling
    with col1:
        st.subheader("Input Image")

        if input_option == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png", "bmp"]
            )

            if uploaded_file is not None:
                # Convert uploaded file to image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Display the uploaded image
                st.session_state.image = image
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Uploaded Image",
                    use_container_width=True,
                )

                if enhance_button or st.session_state.detections is None:
                    # Apply enhancements if requested
                    enhanced_image = apply_image_enhancement(
                        image, brightness, contrast
                    )

                    # Prepare for detection
                    detection_image = prepare_for_detection(enhanced_image)

                    # Run detection with progress and YOLO-specific parameters
                    with st.spinner("Running detection..."):
                        if "YOLO" in model_type:
                            # Pass YOLO-specific parameters
                            detections = model.predict(
                                detection_image,
                                conf_threshold=confidence_threshold,
                                iou_threshold=iou_threshold,
                            )
                        else:
                            # Standard prediction for other models
                            detections = model.predict(detection_image)

                    if detections is not None:
                        st.session_state.detections = detections

                        # Show detection count
                        num_detections = detections.get("num_detections", 0)
                        if num_detections > 0:
                            st.success(f"✅ Found {num_detections} object(s)")
                        else:
                            st.info("ℹ️ No objects detected")
                    else:
                        st.error("❌ Failed to perform detection on the image.")

        else:  # Camera option
            _render_camera_interface(
                model,
                brightness,
                contrast,
                confidence_threshold,
                iou_threshold if "YOLO" in model_type else None,
                model_type,
            )

    # Results display
    with col2:
        st.subheader("Detection Results")

        if (
            st.session_state.image is not None
            and st.session_state.detections is not None
        ):
            # Visualize detections on the image
            image_with_boxes = model.visualize_detections(
                cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB),
                st.session_state.detections,
                threshold=confidence_threshold,
            )

            # Display image with detection boxes
            st.image(
                image_with_boxes, caption="Detection Results", use_container_width=True
            )

            # Display detection summary
            _render_detection_summary(
                model, st.session_state.detections, confidence_threshold
            )

            # Save results
            if st.button("💾 Save Results"):
                _save_detection_results(image_with_boxes, st.session_state.detections)


@st.cache_resource
def _get_model_instance(model_type):
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
            "path": "yolo_defect_detector/weights/best.pt",
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

    # Construct full path with multiple fallback options
    model_path = os.path.join(MODEL_BASE_PATH, config["path"])

    # Check if model file exists with enhanced fallback logic
    if not os.path.exists(model_path):
        if model_type == "defect_yolo":
            # Try multiple possible paths for YOLO model from training pipeline
            alt_paths = [
                # From training pipeline output
                os.path.join(
                    MODEL_BASE_PATH, "yolo_defect_detector", "weights", "best.pt"
                ),
                os.path.join(
                    MODEL_BASE_PATH, "yolo_defect_detector", "weights", "last.pt"
                ),
                # Direct path
                os.path.join(MODEL_BASE_PATH, "best.pt"),
                os.path.join(MODEL_BASE_PATH, "yolo_best.pt"),
                # Current directory fallbacks
                "best.pt",
                "yolo_defect_detector/weights/best.pt",
                # Pretrained fallback
                "yolo11s-cls.pt",
            ]

            model_found = False
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    model_found = True
                    st.sidebar.info(f"📍 Using model: {os.path.basename(alt_path)}")
                    break

            if not model_found:
                # Try to download yolov8n.pt as ultimate fallback
                try:
                    from ultralytics import YOLO

                    st.sidebar.warning(
                        "⚠️ No trained model found, using YOLOv8n pretrained"
                    )
                    model_path = "yolov8n.pt"
                    # This will download yolov8n.pt if it doesn't exist
                    YOLO(model_path)
                except Exception as e:
                    raise FileNotFoundError(
                        f"YOLO model not found and cannot download fallback. Tried: {alt_paths}"
                    )
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


def _render_camera_interface(
    model, brightness, contrast, confidence_threshold, iou_threshold=None, model_type=""
):
    """Render camera interface section with YOLO support"""
    st.subheader("Camera Configuration")
    camera_type = st.selectbox("Camera Type", ["Basler", "Webcam"])

    if camera_type == "Basler":
        basler_device_index = st.number_input(
            "Basler Device Index",
            min_value=0,
            max_value=10,
            value=0,
            help="Index of the Basler camera to use (0 for first device)",
        )

        connect_button = st.button("Connect to Basler Camera")

        if connect_button:
            st.session_state.camera = BaslerCamera(
                device_index=int(basler_device_index)
            )

            if st.session_state.camera.connect():
                st.success("Basler camera connected successfully!")
            else:
                st.error(
                    "Failed to connect to Basler camera. Check if the camera is properly connected and Pylon SDK is installed."
                )

    elif camera_type == "Webcam":
        webcam_id = st.number_input("Webcam ID", min_value=0, max_value=10, value=0)
        connect_button = st.button("Connect to Webcam")

        if connect_button:
            st.session_state.camera = WebcamCamera(camera_id=int(webcam_id))
            if st.session_state.camera.connect():
                st.success("Webcam connected successfully!")
            else:
                st.error("Failed to connect to webcam.")

    # Capture button
    if st.session_state.camera and st.session_state.camera.connected:
        if st.button("📸 Capture and Detect"):
            image = st.session_state.camera.capture_image()

            if image is not None:
                # Store and display the captured image
                st.session_state.image = image
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Captured Image",
                    use_container_width=True,
                )

                # Apply enhancements
                enhanced_image = apply_image_enhancement(image, brightness, contrast)

                # Prepare for detection
                detection_image = prepare_for_detection(enhanced_image)

                # Run detection with YOLO-specific parameters
                with st.spinner("Running detection..."):
                    if "YOLO" in model_type:
                        detections = model.predict(
                            detection_image,
                            conf_threshold=confidence_threshold,
                            iou_threshold=iou_threshold,
                        )
                    else:
                        detections = model.predict(detection_image)

                if detections is not None:
                    st.session_state.detections = detections

                    # Show detection count
                    num_detections = detections.get("num_detections", 0)
                    if num_detections > 0:
                        st.success(f"✅ Found {num_detections} object(s)")
                    else:
                        st.info("ℹ️ No objects detected")
                else:
                    st.error("❌ Failed to perform detection on the image.")
            else:
                st.error("Failed to capture image from camera.")


def _render_detection_summary(model, detections, confidence_threshold):
    """Render detection summary with multi-object support"""
    st.subheader("📊 Detection Summary")

    num_detections = detections.get("num_detections", 0)

    if num_detections == 0:
        st.info("No objects detected above confidence threshold")
        return

    # Filter detections by confidence
    valid_detections = []
    for i in range(num_detections):
        if detections["scores"][i] >= confidence_threshold:
            class_id = detections["classes"][i]
            score = detections["scores"][i]

            # Get class name - Updated for better compatibility
            if hasattr(model, "categories"):
                class_name = model.categories.get(class_id, {}).get(
                    "name", f"Class {class_id}"
                )
            elif "labels" in detections and len(detections["labels"]) > i:
                class_name = detections["labels"][i]
            else:
                class_name = f"Class {class_id}"

            valid_detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": score,
                    "box": detections["boxes"][i] if "boxes" in detections else None,
                }
            )

    if not valid_detections:
        st.info("No objects detected above confidence threshold")
        return

    # Summary statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Objects", len(valid_detections))

    with col2:
        defect_count = sum(
            1 for d in valid_detections if d["class_name"].lower() == "defect"
        )
        st.metric("Defects Found", defect_count)

    with col3:
        normal_count = len(valid_detections) - defect_count
        st.metric("Normal Objects", normal_count)

    # Detailed results table
    st.subheader("🔍 Detailed Results")

    results_data = []
    for i, detection in enumerate(valid_detections):
        results_data.append(
            {
                "Object #": i + 1,
                "Class": detection["class_name"],
                "Confidence": f"{detection['confidence']:.3f}",
                "Status": (
                    "⚠️ DEFECT"
                    if detection["class_name"].lower() == "defect"
                    else "✅ NORMAL"
                ),
            }
        )

    st.dataframe(results_data, use_container_width=True)

    # Alert for defects
    if defect_count > 0:
        st.error(f"🚨 {defect_count} defective object(s) detected!")
    else:
        st.success("✅ All objects appear normal")


def _save_detection_results(image_with_boxes, detections):
    """Save detection results to file"""
    try:
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Save image with bounding boxes
        timestamp = int(time.time())
        result_path = os.path.join("results", f"detection_{timestamp}.jpg")

        cv2.imwrite(result_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))

        # Save detection data as JSON
        import json

        # Convert numpy arrays to lists for JSON serialization
        detection_data = {}
        for key, value in detections.items():
            if isinstance(value, np.ndarray):
                detection_data[key] = value.tolist()
            else:
                detection_data[key] = value

        json_path = os.path.join("results", f"detection_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(detection_data, f, indent=2)

        st.success(f"✅ Results saved!")
        st.info(f"📁 Image: {result_path}")
        st.info(f"📄 Data: {json_path}")

    except Exception as e:
        st.error(f"❌ Failed to save results: {str(e)}")
