"""Video processing page for real-time defect detection."""

import streamlit as st
import cv2
import numpy as np
import time
from typing import Optional
import tempfile
import os

from abbvisionsystem.camera.video_processor import (
    VideoProcessor,
    CalibratedVideoProcessor,
    VideoVisualizationHelper,
)
from abbvisionsystem.models.yolo_model import YOLODefectModel
from abbvisionsystem.models.defect_detection_model import DefectDetectionModel
from abbvisionsystem.camera.camera import BaslerCamera, WebcamCamera


def video_detection_page():
    """Video-based defect detection interface."""
    st.title("🎥 Real-time Video Detection")
    st.write("Live defect detection with camera integration and video recording")

    # Sidebar configuration
    with st.sidebar:
        st.header("Video Configuration")

        # Model selection
        model_type = st.selectbox(
            "Select Detection Model",
            [
                "YOLO Defect Detection (Multi-object)",
                "ResNet Defect Classification (Single object)",
            ],
        )

        # Video source selection
        video_source = st.selectbox(
            "Video Source", ["Live Camera", "Upload Video File"]
        )

        # Detection parameters
        st.subheader("Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

        if "YOLO" in model_type:
            iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
            max_detections = st.slider("Max Detections", 1, 50, 20)
        else:
            iou_threshold = 0.45

        # Performance settings
        st.subheader("Performance")
        max_fps = st.slider("Max FPS", 5, 60, 30)
        skip_frames = st.slider("Skip Frames (1=process all)", 1, 10, 1)
        detection_interval = st.slider("Detection Interval", 1, 10, 1)

        # Recording settings
        st.subheader("Recording")
        enable_recording = st.checkbox("Enable Recording")
        record_with_detections = st.checkbox(
            "Record with Detection Overlays", value=True
        )

        # Display settings
        st.subheader("Display")
        show_fps = st.checkbox("Show FPS", value=True)
        show_status = st.checkbox("Show Status Panel", value=True)
        overlay_transparency = st.slider("Overlay Transparency", 0.0, 1.0, 0.7)

    # Load model
    try:
        model = _load_video_model(model_type)
        st.sidebar.success(f"✅ {model_type} loaded")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return

    # Initialize session state
    if "video_processor" not in st.session_state:
        st.session_state.video_processor = None
    if "video_recording" not in st.session_state:
        st.session_state.video_recording = False

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Video Feed")

        # Video display placeholder
        video_placeholder = st.empty()

        # Control buttons
        control_col1, control_col2, control_col3, control_col4 = st.columns(4)

        with control_col1:
            start_button = st.button("▶️ Start Detection")

        with control_col2:
            stop_button = st.button("⏹️ Stop Detection")

        with control_col3:
            if enable_recording:
                record_button = st.button(
                    "🔴 Start Recording"
                    if not st.session_state.video_recording
                    else "⏺️ Stop Recording"
                )
            else:
                record_button = False

        with control_col4:
            snapshot_button = st.button("📸 Snapshot")

    with col2:
        if show_status:
            st.subheader("📊 Detection Statistics")
            stats_placeholder = st.empty()

            st.subheader("🎛️ Live Controls")

            # Live parameter updates
            if st.session_state.video_processor:
                new_confidence = st.slider(
                    "Live Confidence",
                    0.1,
                    1.0,
                    confidence_threshold,
                    0.05,
                    key="live_confidence",
                )

                if "YOLO" in model_type:
                    new_iou = st.slider(
                        "Live IoU", 0.1, 1.0, iou_threshold, 0.05, key="live_iou"
                    )
                else:
                    new_iou = iou_threshold

                # Update processor parameters
                st.session_state.video_processor.update_detection_params(
                    confidence_threshold=new_confidence,
                    iou_threshold=new_iou,
                    skip_frames=skip_frames,
                    detection_interval=detection_interval,
                )

    # Handle video source setup
    if video_source == "Live Camera":
        camera = _setup_camera_source()
        if camera and start_button:
            _start_live_detection(
                camera,
                model,
                video_placeholder,
                stats_placeholder,
                show_fps,
                show_status,
                enable_recording,
                record_with_detections,
            )

    elif video_source == "Upload Video File":
        uploaded_video = st.file_uploader(
            "Choose a video file...", type=["mp4", "avi", "mov", "mkv"]
        )

        if uploaded_video and start_button:
            _start_video_file_detection(
                uploaded_video,
                model,
                video_placeholder,
                stats_placeholder,
                show_fps,
                show_status,
                enable_recording,
                record_with_detections,
            )

    # Handle stop button
    if stop_button and st.session_state.video_processor:
        st.session_state.video_processor.stop_processing()
        if st.session_state.video_recording:
            st.session_state.video_processor.stop_recording()
            st.session_state.video_recording = False
        st.session_state.video_processor = None
        st.success("Video detection stopped")

    # Handle recording button
    if record_button and st.session_state.video_processor:
        if not st.session_state.video_recording:
            # Start recording
            timestamp = int(time.time())
            output_path = f"recordings/detection_video_{timestamp}.mp4"
            os.makedirs("recordings", exist_ok=True)

            if st.session_state.video_processor.start_recording(output_path):
                st.session_state.video_recording = True
                st.success(f"Started recording to: {output_path}")
        else:
            # Stop recording
            st.session_state.video_processor.stop_recording()
            st.session_state.video_recording = False
            st.success("Recording stopped")

    # Handle snapshot
    if snapshot_button and st.session_state.video_processor:
        # This would capture the current frame
        st.info("Snapshot functionality - to be implemented based on current frame")


def _load_video_model(model_type: str):
    """Load the appropriate model for video processing."""
    # Use the same model loading logic as detection_page.py
    if "YOLO" in model_type:
        model_path = "trained_models/yolo_defect_detector/weights/best.pt"
        # Try multiple fallback paths
        fallback_paths = [
            "trained_models/yolo_defect_detector/weights/last.pt",
            "trained_models/best.pt",
            "yolov8n.pt",  # Pretrained fallback
        ]

        for path in [model_path] + fallback_paths:
            if os.path.exists(path) or path == "yolov8n.pt":
                model = YOLODefectModel(model_path=path)
                if model.load():
                    return model

        raise RuntimeError("No YOLO model found")

    else:  # ResNet classification
        model_path = "trained_models/resnet_defect_classifier.keras"
        fallback_paths = [
            "trained_models/resnet_defect_classifier.h5",
            "trained_models/final_defect_model.keras",
            "trained_models/final_defect_model.h5",
        ]

        for path in [model_path] + fallback_paths:
            if os.path.exists(path):
                model = DefectDetectionModel(model_path=path)
                if model.load():
                    return model

        raise RuntimeError("No classification model found")


def _setup_camera_source():
    """Setup camera source from session state or prompt user."""
    if st.session_state.camera and st.session_state.camera.connected:
        return st.session_state.camera

    st.warning(
        "No camera connected. Please connect a camera in the Detection page first."
    )
    return None


def _start_live_detection(
    camera,
    model,
    video_placeholder,
    stats_placeholder,
    show_fps,
    show_status,
    enable_recording,
    record_with_detections,
):
    """Start live detection from camera."""
    try:
        # Create video processor
        if hasattr(camera, "calibrator"):
            processor = CalibratedVideoProcessor(
                model=model, camera=camera, max_fps=30, buffer_size=5
            )
        else:
            processor = VideoProcessor(model=model, max_fps=30, buffer_size=5)

        st.session_state.video_processor = processor
        processor.start_processing()

        st.success("Started live detection")

        # This would start the video loop - in a real implementation,
        # you'd need to handle the video stream in a different way
        # as Streamlit doesn't support true real-time video display

        st.info("Live video processing started. Use stop button to end session.")

    except Exception as e:
        st.error(f"Failed to start live detection: {str(e)}")


def _start_video_file_detection(
    uploaded_video,
    model,
    video_placeholder,
    stats_placeholder,
    show_fps,
    show_status,
    enable_recording,
    record_with_detections,
):
    """Process uploaded video file."""
    try:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.getvalue())
            video_path = tmp_file.name

        # Create video processor
        processor = VideoProcessor(model=model, max_fps=30, buffer_size=10)

        st.session_state.video_processor = processor
        processor.start_processing()

        # Process video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Failed to open video file")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        progress_bar = st.progress(0)
        frame_count = 0

        st.info(f"Processing video: {total_frames} frames at {fps:.1f} FPS")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Add frame to processor
            processor.add_frame(frame)

            # Get result and display
            result = processor.get_result(timeout=0.1)
            if result:
                # Visualize detections
                frame_with_detections = (
                    VideoVisualizationHelper.draw_detections_on_frame(
                        result.frame,
                        result.detections,
                        model,
                        confidence_threshold=processor.confidence_threshold,
                        show_fps=show_fps,
                        fps=(
                            1.0 / result.processing_time
                            if result.processing_time > 0
                            else 0
                        ),
                    )
                )

                # Display frame
                video_placeholder.image(
                    cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )

                # Record if enabled
                if enable_recording and processor.recording:
                    if record_with_detections:
                        processor.record_frame(frame_with_detections)
                    else:
                        processor.record_frame(result.frame)

            # Update progress
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)

            # Update statistics
            if show_status and frame_count % 10 == 0:  # Update every 10 frames
                stats = processor.get_statistics()
                with stats_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Frames Processed", stats["frames_processed"])
                    with col2:
                        st.metric("Detections", stats["detections_made"])
                    with col3:
                        st.metric("Defects Found", stats["defects_detected"])

                    st.metric("Processing FPS", f"{stats['current_fps']:.1f}")
                    st.metric(
                        "Avg Process Time", f"{stats['average_processing_time']:.3f}s"
                    )

            # Control playback speed
            time.sleep(1.0 / fps)  # Maintain original video timing

        cap.release()
        processor.stop_processing()

        # Cleanup
        os.unlink(video_path)

        st.success("Video processing completed!")

    except Exception as e:
        st.error(f"Failed to process video: {str(e)}")


# Helper function to integrate with main app
def add_video_page_to_sidebar():
    """Add video page option to main app sidebar."""
    if st.sidebar.button("🎥 Video Detection"):
        st.session_state.page = "video_detection"


if __name__ == "__main__":
    video_detection_page()
