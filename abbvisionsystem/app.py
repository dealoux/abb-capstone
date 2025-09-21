import os
import streamlit as st
import numpy as np
import cv2

from abbvisionsystem.camera.camera import BaslerCamera, WebcamCamera
from abbvisionsystem.preprocessing.preprocessing import (
    prepare_for_detection,
    apply_image_enhancement,
)
from abbvisionsystem.models.resnet_model import ResnetModel
from abbvisionsystem.models.yolo_model import YOLODefectModel
from abbvisionsystem.utils.visualization import draw_detection_summary

# Import UI pages
from abbvisionsystem.ui.detection_page import detection_system_page
from abbvisionsystem.ui.training_page import training_center_page
from abbvisionsystem.camera.camera_interface import camera_calibration_interface
from abbvisionsystem.vision_tools.vision_interface import vision_interface

if "image" not in st.session_state:
    st.session_state.image = None
if "detections" not in st.session_state:
    st.session_state.detections = None
if "camera" not in st.session_state:
    st.session_state.camera = None

MODEL_BASE_PATH = "trained_models"


def main():
    """Main application entry point"""
    st.set_page_config(page_title="ABB Vision System", page_icon="♻️", layout="wide")

    st.sidebar.title("🤖 ABB Vision System")

    page = st.sidebar.selectbox(
        "Choose Application",
        [
            "🏠 Image Detection",
            "📷 Camera Calibration",
            "📊 Training Center",
            "🔍 Vision Tools",
        ],
    )

    if page == "🏠 Image Detection":
        detection_system_page()
    elif page == "📷 Camera Calibration":
        camera_calibration_interface()
    elif page == "📊 Training Center":
        training_center_page()
    elif page == "🔍 Vision Tools":
        vision_interface()


if __name__ == "__main__":
    main()
