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
from abbvisionsystem.models.defect_detection_model import DefectDetectionModel
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
# Update the get_model function to include the new defect detection model
@st.cache_resource
def get_model(model_type="taco"):
    """Factory function to get appropriate model"""
    # Map of model types to their respective filenames
    model_files = {
        "taco": "ssd_mobilenet_v2_taco_2018_03_29.pb",
        "defect": "final_defect_model.h5",
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
        model = DefectDetectionModel(
            model_path=model_path, class_mapping_path=class_mapping_path
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

    # ... (rest of your existing detection system code)
    # Keep all the existing functionality from your current app.py
    # create sidebar
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

        # Apply enhancements button
        enhance_button = st.button("Apply Settings")

    # Load the selected model
    try:
        model = get_model(model_type=model_type_map[model_type])
    except Exception as e:
        if "Custom Model" in model_type:
            st.error(
                "Custom model not available yet. Please use the TACO pre-trained model."
            )
            model = get_model(model_type="taco")
        else:
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

                st.success(f"Results saved to {result_path}")


def training_center_page():
    """Training center for creating and managing vision models."""
    st.title("📊 Vision Training Center")
    st.write("Train and manage computer vision models")

    training_type = st.selectbox(
        "Training Type",
        [
            "Pattern Templates",
            "Defect Classification",
            "Blob Detection",
            "Custom Models",
        ],
    )

    if training_type == "Pattern Templates":
        pattern_training_interface()
    elif training_type == "Defect Classification":
        defect_training_interface()
    elif training_type == "Blob Detection":
        blob_training_interface()
    elif training_type == "Custom Models":
        custom_model_interface()


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

    st.info("This interface would provide:")
    st.write("- Upload normal and defective samples")
    st.write("- Configure training parameters")
    st.write("- Monitor training progress")
    st.write("- Validate model performance")
    st.write("- Export trained models")


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
