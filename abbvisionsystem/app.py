import os
import time
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from abbvisionsystem.camera.camera import CognexCamera, BaslerCamera, WebcamCamera
from abbvisionsystem.preprocessing.preprocessing import (
    prepare_for_detection,
    apply_image_enhancement,
)
from abbvisionsystem.models.taco_model import TACOModel
from abbvisionsystem.models.defect_detection_model import (
    DefectClassificationModel,
    ObjectDefectDetectionModel,
)
from abbvisionsystem.utils.visualization import draw_detection_summary
from abbvisionsystem.vision_tools.vision_interface import vision_interface
from abbvisionsystem.camera.camera_interface import camera_calibration_interface

# Set page configuration
st.set_page_config(page_title="ABB Vision System", page_icon="♻️", layout="wide")

# Initialize session state
if "image" not in st.session_state:
    st.session_state.image = None
if "detections" not in st.session_state:
    st.session_state.detections = None
if "camera" not in st.session_state:
    st.session_state.camera = None


def main():
    # Navigation
    page = st.sidebar.selectbox(
        "Choose Application",
        [
            "🏠 Detection System",
            "🔍 Vision Tools",
            "📷 Camera Calibration",
            "📊 Training Center",
        ],
    )

    if page == "🏠 Detection System":
        detection_system_page()
    elif page == "🔍 Vision Tools":
        vision_interface()
    elif page == "📷 Camera Calibration":
        camera_calibration_interface()
    elif page == "📊 Training Center":
        training_center_page()


# Base path for all models
MODEL_BASE_PATH = "trained_models"


# Cache the model loading
@st.cache_resource
def get_model(model_type="taco", detection_mode="whole_image"):
    """Factory function to get appropriate model"""
    # Map of model types to their respective filenames
    model_files = {
        "taco": "ssd_mobilenet_v2_taco_2018_03_29.pb",
        "defect": "defect_classifier.h5",
        "object_defect": "object_defect_classifier.h5",  # For object-level detection
    }

    # Check if model type is supported
    if model_type not in model_files:
        raise ValueError(f"Unknown model type: {model_type}")

    # Get the appropriate filename
    filename = model_files[model_type]

    # Construct full path
    model_path = os.path.join(MODEL_BASE_PATH, filename)

    # Initialize the appropriate model class
    if model_type == "taco":
        model = TACOModel(model_path=model_path)
    elif model_type == "defect":
        class_mapping_path = os.path.join(MODEL_BASE_PATH, "class_mapping.json")
        if detection_mode == "object_level":
            # Use object detection model for individual objects
            model = ObjectDefectDetectionModel(
                classifier_path=model_path, class_mapping_path=class_mapping_path
            )
        else:
            # Use whole image classification
            model = DefectClassificationModel(
                model_path=model_path, class_mapping_path=class_mapping_path
            )
    elif model_type == "object_defect":
        class_mapping_path = os.path.join(MODEL_BASE_PATH, "class_mapping.json")
        model = ObjectDefectDetectionModel(
            classifier_path=model_path, class_mapping_path=class_mapping_path
        )

    # Load the model
    model.load()
    return model


def detection_system_page():
    """Original detection system interface with calibration integration."""
    st.title("♻️ ABB Vision System")
    st.write("Defect Detection system with camera integration")

    # Check calibration status
    if st.session_state.camera and hasattr(st.session_state.camera, "calibrator"):
        if st.session_state.camera.calibrator.calibration_result:
            st.success("📷 Camera is calibrated")
            scale = st.session_state.camera.calibrator.calibration_result.pixels_per_mm
            if scale:
                st.info(f"Scale: {scale:.2f} pixels/mm")
        else:
            st.warning("⚠️ Camera not calibrated. Go to Camera Calibration page.")

    # Create sidebar
    with st.sidebar:
        st.header("Configuration")

        # Model selection
        model_type = st.selectbox(
            "Select Model", ["Defect Detection", "TACO Waste Sorting"]
        )
        model_type_map = {
            "Defect Detection": "defect",
            "TACO Waste Sorting": "taco",
        }

        # Detection mode selection (only for defect detection)
        detection_mode = "whole_image"
        if model_type == "Defect Detection":
            detection_mode = st.radio(
                "Detection Mode",
                ["Whole Image", "Individual Objects"],
                help="Choose whether to classify the entire image or detect and classify individual objects",
            )
            detection_mode = (
                "object_level"
                if detection_mode == "Individual Objects"
                else "whole_image"
            )

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
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

        # Object detection parameters (only for object-level detection)
        if detection_mode == "object_level":
            st.subheader("Object Detection Parameters")
            min_area = st.slider("Min Object Area", 500, 5000, 1000)
            max_area = st.slider("Max Object Area", 10000, 100000, 50000)

        # Apply enhancements button
        enhance_button = st.button("Apply Settings")

    # Load the selected model
    try:
        if model_type == "Defect Detection" and detection_mode == "object_level":
            # Check if object detection model exists
            object_model_path = os.path.join(
                MODEL_BASE_PATH, "object_defect_classifier.h5"
            )
            if os.path.exists(object_model_path):
                model = get_model(model_type="object_defect")
                # Update detection parameters if specified
                if hasattr(model, "min_object_area"):
                    model.min_object_area = min_area
                    model.max_object_area = max_area
                    if model.object_detector:
                        model.object_detector.min_area = min_area
                        model.object_detector.max_area = max_area
            else:
                st.warning(
                    "⚠️ Object detection model not found. Please train the object detection model first."
                )
                st.info("Using whole image classification instead.")
                model = get_model(
                    model_type=model_type_map[model_type], detection_mode="whole_image"
                )
        else:
            model = get_model(
                model_type=model_type_map[model_type], detection_mode=detection_mode
            )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Main content area with two columns
    col1, col2 = st.columns(2)

    # Input handling
    with col1:
        st.subheader("Input Image")

        if input_option == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"]
            )

            if uploaded_file is not None:
                # Convert uploaded file to image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Display the uploaded image
                st.session_state.image = image
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image"
                )

                if enhance_button or st.session_state.detections is None:
                    # Apply enhancements if requested
                    enhanced_image = apply_image_enhancement(
                        image, brightness, contrast
                    )

                    # Prepare for detection
                    detection_image = prepare_for_detection(enhanced_image)

                    # Run detection
                    detections = model.predict(detection_image)

                    if detections is not None:
                        st.session_state.detections = detections
                        st.success("Detection completed successfully!")

                        # Show detection info
                        if detection_mode == "object_level":
                            st.info(
                                f"🎯 Detected {detections['num_detections']} objects"
                            )
                        else:
                            st.info("🖼️ Whole image classification completed")
                    else:
                        st.error("Failed to perform detection on the image.")

        else:  # Camera option
            st.subheader("Camera Configuration")
            camera_type = st.selectbox(
                "Camera Type", ["Cognex", "Basler", "Webcam (Fallback)"]
            )

            if camera_type == "Cognex":
                ip_address = st.text_input("Camera IP Address", "192.168.1.100")
                port = st.text_input("Port", "80")
                username = st.text_input("Username (if required)")
                password = st.text_input("Password (if required)", type="password")

                connect_button = st.button("Connect to Camera")
                if connect_button:
                    st.session_state.camera = CognexCamera(
                        ip_address=ip_address,
                        port=port,
                        username=username if username else None,
                        password=password if password else None,
                    )

                    if st.session_state.camera.connect():
                        st.success("Camera connected successfully!")
                    else:
                        st.error(
                            "Failed to connect to camera. Check settings and try again."
                        )
            elif camera_type == "Basler":
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

            else:  # Webcam
                webcam_id = st.number_input(
                    "Webcam ID", min_value=0, max_value=10, value=0
                )
                connect_button = st.button("Connect to Webcam")

                if connect_button:
                    st.session_state.camera = WebcamCamera(camera_id=int(webcam_id))
                    if st.session_state.camera.connect():
                        st.success("Webcam connected successfully!")
                    else:
                        st.error("Failed to connect to webcam.")

            # Capture button
            if st.session_state.camera and st.session_state.camera.connected:
                if st.button("Capture and Detect"):
                    image = st.session_state.camera.capture_image()

                    if image is not None:
                        # Store and display the captured image
                        st.session_state.image = image
                        st.image(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                            caption="Captured Image",
                        )

                        # Apply enhancements
                        enhanced_image = apply_image_enhancement(
                            image, brightness, contrast
                        )

                        # Prepare for detection
                        detection_image = prepare_for_detection(enhanced_image)

                        # Run detection
                        detections = model.predict(detection_image)

                        if detections is not None:
                            st.session_state.detections = detections
                            st.success("Detection completed successfully!")

                            # Show detection info
                            if detection_mode == "object_level":
                                st.info(
                                    f"🎯 Detected {detections['num_detections']} objects"
                                )
                            else:
                                st.info("🖼️ Whole image classification completed")
                        else:
                            st.error("Failed to perform detection on the image.")
                    else:
                        st.error("Failed to capture image from camera.")

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
            st.image(image_with_boxes, caption="Detection Results")

            # Display detection summary
            draw_detection_summary(
                model, st.session_state.detections, confidence_threshold
            )

            # Enhanced summary for object detection
            if (
                detection_mode == "object_level"
                and st.session_state.detections["num_detections"] > 0
            ):
                st.subheader("🔍 Object Analysis")

                # Create detailed object table
                if "object_details" in st.session_state.detections:
                    import pandas as pd

                    object_data = []
                    for i, obj_detail in enumerate(
                        st.session_state.detections["object_details"]
                    ):
                        if (
                            st.session_state.detections["scores"][i]
                            >= confidence_threshold
                        ):
                            class_id = st.session_state.detections["classes"][i]
                            class_name = model.categories.get(class_id, {}).get(
                                "name", f"Class {class_id}"
                            )

                            object_data.append(
                                {
                                    "Object ID": obj_detail["object_id"],
                                    "Classification": class_name,
                                    "Confidence": f"{st.session_state.detections['scores'][i]:.2%}",
                                    "Area (pixels)": obj_detail["area"],
                                    "Status": (
                                        "✅ Normal" if class_id == 0 else "❌ Defect"
                                    ),
                                }
                            )

                    if object_data:
                        df = pd.DataFrame(object_data)
                        st.dataframe(df, use_container_width=True)

                        # Summary statistics
                        total_objects = len(object_data)
                        defective_objects = sum(
                            1 for obj in object_data if "Defect" in obj["Status"]
                        )

                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        with col_stats1:
                            st.metric("Total Objects", total_objects)
                        with col_stats2:
                            st.metric("Defective", defective_objects)
                        with col_stats3:
                            st.metric(
                                "Pass Rate",
                                f"{((total_objects-defective_objects)/total_objects*100):.1f}%",
                            )

            # Option to save result
            if st.button("Save Results"):
                # Create results directory if it doesn't exist
                os.makedirs("results", exist_ok=True)

                # Save image with bounding boxes
                result_path = os.path.join(
                    "results", f"detection_{int(time.time())}.jpg"
                )
                cv2.imwrite(
                    result_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
                )

                # Save detection data
                if detection_mode == "object_level":
                    import json

                    result_data = {
                        "detection_mode": detection_mode,
                        "num_detections": int(
                            st.session_state.detections["num_detections"]
                        ),
                        "confidence_threshold": confidence_threshold,
                        "objects": [],
                    }

                    if "object_details" in st.session_state.detections:
                        for i, obj_detail in enumerate(
                            st.session_state.detections["object_details"]
                        ):
                            if (
                                st.session_state.detections["scores"][i]
                                >= confidence_threshold
                            ):
                                result_data["objects"].append(
                                    {
                                        "object_id": obj_detail["object_id"],
                                        "class_id": int(
                                            st.session_state.detections["classes"][i]
                                        ),
                                        "confidence": float(
                                            st.session_state.detections["scores"][i]
                                        ),
                                        "bbox_pixels": obj_detail["bbox_pixels"],
                                        "area": obj_detail["area"],
                                    }
                                )

                    json_path = result_path.replace(".jpg", "_data.json")
                    with open(json_path, "w") as f:
                        json.dump(result_data, f, indent=2)

                    st.success(f"Results saved to {result_path} and {json_path}")
                else:
                    st.success(f"Results saved to {result_path}")


# Keep all your existing training functions unchanged...
def training_center_page():
    """Training center for creating and managing vision models."""
    st.title("📊 Vision Training Center")
    st.write("Train and manage computer vision models")

    training_type = st.selectbox(
        "Training Type",
        [
            "Pattern Templates",
            "Defect Classification",
            "Object Detection Training",  # Add this option
            "Blob Detection",
            "Custom Models",
        ],
    )

    if training_type == "Pattern Templates":
        pattern_training_interface()
    elif training_type == "Defect Classification":
        defect_training_interface()
    elif training_type == "Object Detection Training":
        object_detection_training_interface()  # Add this function
    elif training_type == "Blob Detection":
        blob_training_interface()
    elif training_type == "Custom Models":
        custom_model_interface()


def object_detection_training_interface():
    """Interface for training object-level defect detection."""
    st.header("🎯 Object Detection Training")

    st.info("This interface integrates with your training pipeline to:")
    st.write("- Organize data for object-level detection")
    st.write("- Train models to detect and classify individual objects")
    st.write("- Validate object detection performance")
    st.write("- Export models for use in the detection system")

    # Quick training button
    if st.button("Launch Training Pipeline"):
        st.info("💡 To train object detection models:")
        st.code("jupyter notebook pipelinev2.ipynb")
        st.write("This will:")
        st.write("1. Organize your data for object detection")
        st.write("2. Train the object-level classifier")
        st.write("3. Save the model as 'object_defect_classifier.h5'")
        st.write("4. Enable 'Individual Objects' mode in the Detection System")


def pattern_training_interface():
    """Interface for training pattern templates."""
    st.header("🎯 Pattern Template Training")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Management")

        # Upload training images
        uploaded_files = st.file_uploader(
            "Upload Training Images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} images")

            # Show sample images
            for i, file in enumerate(uploaded_files[:3]):  # Show first 3
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption=f"Sample {i+1}",
                    width=200,
                )

    with col2:
        st.subheader("Training Configuration")

        template_name = st.text_input("Template Name", "template_1")

        # ROI definition (simplified)
        st.write("Define template region:")
        roi_method = st.radio("ROI Method", ["Manual Input", "Auto Detect"])

        if roi_method == "Manual Input":
            roi_x = st.number_input("X", min_value=0, value=50)
            roi_y = st.number_input("Y", min_value=0, value=50)
            roi_w = st.number_input("Width", min_value=1, value=100)
            roi_h = st.number_input("Height", min_value=1, value=100)

        # Training parameters
        min_features = st.slider("Min Features", 10, 100, 20)

        if st.button("Train Template"):
            if uploaded_files:
                # Training logic would go here
                st.success(f"Template '{template_name}' trained successfully!")
                st.info("In a full implementation, this would:")
                st.write("- Extract features from all uploaded images")
                st.write("- Create robust template representation")
                st.write("- Save template for future use")
            else:
                st.error("Please upload training images first")


def defect_training_interface():
    """Interface for training defect detection models."""
    st.header("🔍 Defect Detection Training")

    st.info("This interface integrates with your training pipeline:")
    st.write("- Upload normal and defective samples")
    st.write("- Configure training parameters")
    st.write("- Monitor training progress")
    st.write("- Validate model performance")
    st.write("- Export trained models")

    # Quick training button
    if st.button("Launch Training Pipeline"):
        st.info("💡 To train defect detection models:")
        st.code("jupyter notebook pipelinev2.ipynb")


def blob_training_interface():
    """Interface for configuring blob detection."""
    st.header("🔴 Blob Detection Configuration")

    st.info("This interface would provide:")
    st.write("- Upload sample images with known blobs")
    st.write("- Tune detection parameters interactively")
    st.write("- Validate detection accuracy")
    st.write("- Save optimized configurations")


def custom_model_interface():
    """Interface for custom model training."""
    st.header("🤖 Custom Model Training")

    st.info("This interface would provide:")
    st.write("- Upload custom datasets")
    st.write("- Choose model architectures")
    st.write("- Configure training hyperparameters")
    st.write("- Monitor training metrics")
    st.write("- Deploy trained models")


if __name__ == "__main__":
    main()
