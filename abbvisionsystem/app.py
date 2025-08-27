import os
import streamlit as st
import numpy as np
import cv2

from abbvisionsystem.camera.camera import BaslerCamera, WebcamCamera
from abbvisionsystem.preprocessing.preprocessing import (
    prepare_for_detection,
    apply_image_enhancement,
)
from abbvisionsystem.models.defect_detection_model import DefectDetectionModel
from abbvisionsystem.models.yolo_model import YOLODefectModel
from abbvisionsystem.utils.visualization import draw_detection_summary

# Import UI pages
from abbvisionsystem.ui.detection_page import detection_system_page
from abbvisionsystem.ui.training_page import training_center_page
from abbvisionsystem.ui.video_detection_page import (
    video_detection_page,
)  # Added video page import
from abbvisionsystem.camera.camera_interface import camera_calibration_interface
from abbvisionsystem.vision_tools.vision_interface import vision_interface

if "image" not in st.session_state:
    st.session_state.image = None
if "detections" not in st.session_state:
    st.session_state.detections = None
if "camera" not in st.session_state:
    st.session_state.camera = None

MODEL_BASE_PATH = "trained_models"


@st.cache_resource
def get_model(model_type="taco"):
    """Factory function to get appropriate model - Updated for pipeline compatibility"""
    model_configs = {
        "taco": {
            "path": "ssd_mobilenet_v2_taco_2018_03_29.pb",
            "class": TACOModel,
            "extra_args": {},
        },
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
                "yolo11s.pt",
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
                        "⚠️ No trained model found, using YOLOv11s pretrained"
                    )
                    model_path = "yolo11s.pt"
                    # This will download yolov8n.pt if it doesn't exist
                    YOLO(model_path)
                except Exception as e:
                    raise FileNotFoundError(
                        f"YOLO model not found and cannot download fallback. Tried: {alt_paths}"
                    )
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    model_class = config["class"]
    extra_args = config["extra_args"]

    model = model_class(model_path=model_path, **extra_args)

    if not model.load():
        raise RuntimeError(f"Failed to load {model_type} model from {model_path}")

    return model


def main():
    """Main application entry point"""
    st.set_page_config(page_title="ABB Vision System", page_icon="♻️", layout="wide")

    st.sidebar.title("🤖 ABB Vision System")

    page = st.sidebar.selectbox(
        "Choose Application",
        [
            "🏠 Image Detection",
            "🎥 Video Detection",
            "🔍 Vision Tools",
            "📷 Camera Calibration",
            "📊 Training Center",
        ],
    )

    if page == "🏠 Image Detection":
        detection_system_page()
    elif page == "🎥 Video Detection":
        video_detection_page()
    elif page == "🔍 Vision Tools":
        vision_interface()
    elif page == "📷 Camera Calibration":
        camera_calibration_interface()
    elif page == "📊 Training Center":
        training_center_page()


if __name__ == "__main__":
    main()
