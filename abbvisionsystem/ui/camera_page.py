import streamlit as st
import cv2
import numpy as np

from abbvisionsystem.camera.camera import CognexCamera, BaslerCamera, WebcamCamera
from abbvisionsystem.camera.camera_interface import camera_calibration_interface
from abbvisionsystem.models.defect_detection_model import ObjectDefectDetectionModel
from abbvisionsystem.preprocessing.preprocessing import (
    prepare_for_detection,
    apply_image_enhancement,
)
from abbvisionsystem.ui.detection_page import get_model


def camera_page():
    """Camera integration and real-time detection page."""
    st.title("📷 Camera Integration")
    st.write("Connect to cameras and perform real-time object detection")

    # Create tabs for different camera functions
    tab1, tab2, tab3 = st.tabs(
        ["🔌 Camera Connection", "📐 Camera Calibration", "🎯 Real-time Detection"]
    )

    with tab1:
        _camera_connection_interface()

    with tab2:
        st.subheader("📐 Camera Calibration")
        camera_calibration_interface()

    with tab3:
        _realtime_detection_interface()


def _camera_connection_interface():
    """Camera connection interface."""
    st.subheader("🔌 Camera Connection")

    # Check current camera status
    if st.session_state.camera and st.session_state.camera.connected:
        st.success(f"✅ Camera connected: {type(st.session_state.camera).__name__}")

        # Show camera info if available
        if hasattr(st.session_state.camera, "get_camera_info"):
            info = st.session_state.camera.get_camera_info()
            if info:
                st.info(f"📋 Camera Info: {info}")

        # Disconnect button
        if st.button("🔌 Disconnect Camera"):
            st.session_state.camera.disconnect()
            st.session_state.camera = None
            st.rerun()
    else:
        st.warning("⚠️ No camera connected")

    st.markdown("---")

    # Camera selection
    camera_type = st.selectbox(
        "Select Camera Type",
        ["Cognex", "Basler", "Webcam (Fallback)"],
        help="Choose the type of camera to connect",
    )

    if camera_type == "Cognex":
        _cognex_connection_interface()
    elif camera_type == "Basler":
        _basler_connection_interface()
    else:  # Webcam
        _webcam_connection_interface()


def _cognex_connection_interface():
    """Cognex camera connection interface."""
    st.subheader("🏭 Cognex Camera Configuration")

    col1, col2 = st.columns(2)

    with col1:
        ip_address = st.text_input("Camera IP Address", "192.168.1.100")
        port = st.text_input("Port", "80")

    with col2:
        username = st.text_input("Username (optional)")
        password = st.text_input("Password (optional)", type="password")

    if st.button("🔗 Connect to Cognex Camera", use_container_width=True):
        try:
            with st.spinner("Connecting to Cognex camera..."):
                st.session_state.camera = CognexCamera(
                    ip_address=ip_address,
                    port=port,
                    username=username if username else None,
                    password=password if password else None,
                )

                if st.session_state.camera.connect():
                    st.success("✅ Cognex camera connected successfully!")
                    st.rerun()
                else:
                    st.error(
                        "❌ Failed to connect. Check IP address and network connection."
                    )
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")


def _basler_connection_interface():
    """Basler camera connection interface."""
    st.subheader("📹 Basler Camera Configuration")

    device_index = st.number_input(
        "Device Index",
        min_value=0,
        max_value=10,
        value=0,
        help="Index of the Basler camera (0 for first device)",
    )

    if st.button("🔗 Connect to Basler Camera", use_container_width=True):
        try:
            with st.spinner("Connecting to Basler camera..."):
                st.session_state.camera = BaslerCamera(device_index=int(device_index))

                if st.session_state.camera.connect():
                    st.success("✅ Basler camera connected successfully!")
                    st.rerun()
                else:
                    st.error(
                        "❌ Failed to connect. Check camera connection and Pylon SDK installation."
                    )
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")


def _webcam_connection_interface():
    """Webcam connection interface."""
    st.subheader("💻 Webcam Configuration")

    webcam_id = st.number_input(
        "Webcam ID",
        min_value=0,
        max_value=10,
        value=0,
        help="Camera ID (0 for default webcam)",
    )

    if st.button("🔗 Connect to Webcam", use_container_width=True):
        try:
            with st.spinner("Connecting to webcam..."):
                st.session_state.camera = WebcamCamera(camera_id=int(webcam_id))

                if st.session_state.camera.connect():
                    st.success("✅ Webcam connected successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to connect to webcam.")
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")


def _realtime_detection_interface():
    """Real-time detection interface."""
    st.subheader("🎯 Real-time Object Detection")

    if not st.session_state.camera or not st.session_state.camera.connected:
        st.warning("⚠️ Please connect a camera first in the Camera Connection tab")
        return

    # Detection settings
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚙️ Detection Settings")
        model_type = st.selectbox(
            "Model", ["Object Defect Detection", "TACO Waste Sorting"]
        )
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

        # Image enhancement
        brightness = st.slider("Brightness", -100, 100, 0)
        contrast = st.slider("Contrast", -100, 100, 0)

    with col2:
        st.subheader("📐 Object Parameters")
        min_area = st.slider("Min Object Area", 500, 5000, 1000)
        max_area = st.slider("Max Object Area", 10000, 100000, 50000)

        # Auto-capture settings
        auto_capture = st.checkbox(
            "Auto-capture", help="Automatically capture and analyze images"
        )
        if auto_capture:
            capture_interval = st.slider("Capture Interval (seconds)", 1, 10, 3)

    # Load model
    try:
        model_type_map = {
            "Object Defect Detection": "object_defect",
            "TACO Waste Sorting": "taco",
        }
        model = get_model(model_type=model_type_map[model_type])

        # Update detection parameters
        if hasattr(model, "min_object_area"):
            model.min_object_area = min_area
            model.max_object_area = max_area
            if model.object_detector:
                model.object_detector.min_area = min_area
                model.object_detector.max_area = max_area

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return

    # Capture controls
    col_cap1, col_cap2, col_cap3 = st.columns(3)

    with col_cap1:
        if st.button("📸 Capture & Analyze", use_container_width=True):
            _capture_and_analyze(model, brightness, contrast, confidence_threshold)

    with col_cap2:
        if st.button("📹 Live Preview", use_container_width=True):
            _show_live_preview()

    with col_cap3:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.image = None
            st.session_state.detections = None
            st.rerun()

    # Show results if available
    if st.session_state.image is not None and st.session_state.detections is not None:
        st.markdown("---")
        st.subheader("📊 Detection Results")

        # Visualize detections
        image_with_boxes = model.visualize_detections(
            cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB),
            st.session_state.detections,
            threshold=confidence_threshold,
        )

        col_img, col_stats = st.columns([2, 1])

        with col_img:
            st.image(
                image_with_boxes, caption="🎯 Detection Results", use_column_width=True
            )

        with col_stats:
            # Quick stats
            num_objects = st.session_state.detections["num_detections"]
            st.metric("Objects Detected", num_objects)

            if "object_details" in st.session_state.detections and num_objects > 0:
                defective_count = sum(
                    1
                    for i in range(num_objects)
                    if (
                        st.session_state.detections["scores"][i] >= confidence_threshold
                        and st.session_state.detections["classes"][i] == 1
                    )
                )
                st.metric("Defective Objects", defective_count)

                pass_rate = (
                    ((num_objects - defective_count) / num_objects * 100)
                    if num_objects > 0
                    else 0
                )
                st.metric("Pass Rate", f"{pass_rate:.1f}%")


def _capture_and_analyze(model, brightness, contrast, confidence_threshold):
    """Capture image and run analysis."""
    try:
        with st.spinner("📸 Capturing and analyzing..."):
            image = st.session_state.camera.capture_image()

            if image is not None:
                # Store captured image
                st.session_state.image = image

                # Apply enhancements
                enhanced_image = apply_image_enhancement(image, brightness, contrast)

                # Prepare for detection
                detection_image = prepare_for_detection(enhanced_image)

                # Run detection
                detections = model.predict(detection_image)

                if detections is not None:
                    st.session_state.detections = detections
                    st.success(
                        f"✅ Analysis complete! Detected {detections['num_detections']} objects"
                    )
                else:
                    st.error("❌ Failed to analyze image")
            else:
                st.error("❌ Failed to capture image")

    except Exception as e:
        st.error(f"❌ Capture error: {str(e)}")


def _show_live_preview():
    """Show live camera preview."""
    try:
        image = st.session_state.camera.capture_image()
        if image is not None:
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption="📹 Live Preview",
                use_column_width=True,
            )
        else:
            st.error("❌ Failed to get preview")
    except Exception as e:
        st.error(f"❌ Preview error: {str(e)}")
