import os
import time
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import json

from abbvisionsystem.models.defect_detection_model import ObjectDefectDetectionModel
from abbvisionsystem.models.taco_model import TACOModel
from abbvisionsystem.preprocessing.preprocessing import (
    prepare_for_detection,
    apply_image_enhancement,
)
from abbvisionsystem.utils.visualization import draw_detection_summary

# Base path for all models
MODEL_BASE_PATH = "trained_models"


@st.cache_resource
def get_model(model_type="object_defect"):
    """Factory function to get appropriate model - only object-based models."""
    model_files = {
        "taco": "ssd_mobilenet_v2_taco_2018_03_29.pb",
        "object_defect": "object_defect_classifier.h5",
    }

    if model_type not in model_files:
        raise ValueError(f"Unknown model type: {model_type}")

    filename = model_files[model_type]
    model_path = os.path.join(MODEL_BASE_PATH, filename)

    if model_type == "taco":
        model = TACOModel(model_path=model_path)
    elif model_type == "object_defect":
        class_mapping_path = os.path.join(MODEL_BASE_PATH, "class_mapping.json")
        model = ObjectDefectDetectionModel(
            classifier_path=model_path, class_mapping_path=class_mapping_path
        )

    model.load()
    return model


def detection_page():
    """Object-based detection page."""
    st.title("🎯 Object-Based Defect Detection")
    st.write("Detect and classify individual objects for defects")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection - only object-based models
        model_type = st.selectbox(
            "Select Model",
            ["Object Defect Detection", "TACO Waste Sorting"],
            help="Choose the detection model to use",
        )
        model_type_map = {
            "Object Defect Detection": "object_defect",
            "TACO Waste Sorting": "taco",
        }

        # Image enhancement options
        st.subheader("🎨 Image Enhancement")
        brightness = st.slider("Brightness", -100, 100, 0)
        contrast = st.slider("Contrast", -100, 100, 0)

        # Detection settings
        st.subheader("🔍 Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

        # Object detection parameters
        st.subheader("📐 Object Parameters")
        min_area = st.slider("Min Object Area", 500, 5000, 1000)
        max_area = st.slider("Max Object Area", 10000, 100000, 50000)

        # Apply settings button
        apply_settings = st.button("🔄 Apply Settings", use_container_width=True)

    # Load the selected model
    try:
        model = get_model(model_type=model_type_map[model_type])

        # Update detection parameters if it's object detection model
        if hasattr(model, "min_object_area"):
            model.min_object_area = min_area
            model.max_object_area = max_area
            if model.object_detector:
                model.object_detector.min_area = min_area
                model.object_detector.max_area = max_area

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Make sure you have trained the model using the training pipeline")
        return

    # Main content area
    col1, col2 = st.columns([1, 1])

    # Input section
    with col1:
        st.subheader("📤 Input")

        # Input options
        input_option = st.radio(
            "Select Input Source",
            ["Upload Image", "Camera Integration"],
            help="Choose how to provide input images",
        )

        if input_option == "Upload Image":
            # Upload image
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Upload an image to detect objects and classify defects",
            )

            if uploaded_file is not None:
                # Convert uploaded file to image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Display the uploaded image
                st.session_state.image = image
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="📥 Uploaded Image",
                    use_column_width=True,
                )

                # Process image when uploaded or settings changed
                if apply_settings or st.session_state.detections is None:
                    with st.spinner("🔄 Processing image..."):
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
                            st.success(
                                f"✅ Detected {detections['num_detections']} objects!"
                            )
                        else:
                            st.error("❌ Failed to perform detection")

        else:  # Camera Integration
            st.info("📷 Camera integration available in Camera Integration page")
            st.write("Use the Camera Integration page to:")
            st.write("- Connect to Cognex, Basler, or Webcam")
            st.write("- Capture images directly")
            st.write("- Run real-time detection")

    # Results section
    with col2:
        st.subheader("📊 Results")

        if (
            st.session_state.image is not None
            and st.session_state.detections is not None
        ):
            # Visualize detections
            image_with_boxes = model.visualize_detections(
                cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB),
                st.session_state.detections,
                threshold=confidence_threshold,
            )

            # Display image with detection boxes
            st.image(
                image_with_boxes, caption="🎯 Detection Results", use_column_width=True
            )

            # Detection summary
            st.subheader("📋 Detection Summary")
            draw_detection_summary(
                model, st.session_state.detections, confidence_threshold
            )

    # Detailed analysis section (full width)
    if (
        st.session_state.image is not None
        and st.session_state.detections is not None
        and st.session_state.detections["num_detections"] > 0
    ):

        st.markdown("---")
        st.subheader("🔍 Detailed Object Analysis")

        # Create detailed object table
        if "object_details" in st.session_state.detections:
            object_data = []
            for i, obj_detail in enumerate(
                st.session_state.detections["object_details"]
            ):
                if st.session_state.detections["scores"][i] >= confidence_threshold:
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
                            "Status": "✅ Normal" if class_id == 0 else "❌ Defect",
                            "Bbox": f"({obj_detail['bbox_pixels'][0]}, {obj_detail['bbox_pixels'][1]}, "
                            f"{obj_detail['bbox_pixels'][2]}, {obj_detail['bbox_pixels'][3]})",
                        }
                    )

            if object_data:
                df = pd.DataFrame(object_data)
                st.dataframe(df, use_container_width=True)

                # Summary statistics
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

                total_objects = len(object_data)
                defective_objects = sum(
                    1 for obj in object_data if "Defect" in obj["Status"]
                )
                normal_objects = total_objects - defective_objects
                pass_rate = (
                    (normal_objects / total_objects * 100) if total_objects > 0 else 0
                )

                with col_stat1:
                    st.metric("Total Objects", total_objects)
                with col_stat2:
                    st.metric("✅ Normal", normal_objects)
                with col_stat3:
                    st.metric("❌ Defective", defective_objects)
                with col_stat4:
                    st.metric("Pass Rate", f"{pass_rate:.1f}%")

        # Save results section
        st.markdown("---")
        col_save1, col_save2, col_save3 = st.columns([1, 1, 1])

        with col_save1:
            if st.button("💾 Save Results", use_container_width=True):
                _save_detection_results(
                    image_with_boxes, st.session_state.detections, confidence_threshold
                )

        with col_save2:
            if st.button("📊 Export Data", use_container_width=True):
                _export_detection_data(
                    st.session_state.detections, confidence_threshold
                )

        with col_save3:
            if st.button("🔄 Clear Results", use_container_width=True):
                st.session_state.detections = None
                st.rerun()


def _save_detection_results(image_with_boxes, detections, confidence_threshold):
    """Save detection results to file."""
    try:
        os.makedirs("results", exist_ok=True)

        # Save image with bounding boxes
        result_path = os.path.join("results", f"detection_{int(time.time())}.jpg")
        cv2.imwrite(result_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))

        # Save detection data
        result_data = {
            "timestamp": time.time(),
            "confidence_threshold": confidence_threshold,
            "num_detections": int(detections["num_detections"]),
            "objects": [],
        }

        if "object_details" in detections:
            for i, obj_detail in enumerate(detections["object_details"]):
                if detections["scores"][i] >= confidence_threshold:
                    result_data["objects"].append(
                        {
                            "object_id": obj_detail["object_id"],
                            "class_id": int(detections["classes"][i]),
                            "confidence": float(detections["scores"][i]),
                            "bbox_pixels": obj_detail["bbox_pixels"],
                            "area": obj_detail["area"],
                        }
                    )

        json_path = result_path.replace(".jpg", "_data.json")
        with open(json_path, "w") as f:
            json.dump(result_data, f, indent=2)

        st.success(f"✅ Results saved to {result_path}")

    except Exception as e:
        st.error(f"❌ Failed to save results: {str(e)}")


def _export_detection_data(detections, confidence_threshold):
    """Export detection data as downloadable file."""
    try:
        if "object_details" in detections:
            export_data = []
            for i, obj_detail in enumerate(detections["object_details"]):
                if detections["scores"][i] >= confidence_threshold:
                    export_data.append(
                        {
                            "object_id": obj_detail["object_id"],
                            "class_id": int(detections["classes"][i]),
                            "confidence": float(detections["scores"][i]),
                            "area": obj_detail["area"],
                            "bbox_x": obj_detail["bbox_pixels"][0],
                            "bbox_y": obj_detail["bbox_pixels"][1],
                            "bbox_width": obj_detail["bbox_pixels"][2],
                            "bbox_height": obj_detail["bbox_pixels"][3],
                        }
                    )

            if export_data:
                df = pd.DataFrame(export_data)
                csv_data = df.to_csv(index=False)

                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"detection_results_{int(time.time())}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("⚠️ No data to export")

    except Exception as e:
        st.error(f"❌ Failed to export data: {str(e)}")
