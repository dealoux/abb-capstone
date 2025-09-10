"""Detection system page with calibration-based coordinate system."""

import streamlit as st
import cv2
import numpy as np
import os
from abbvisionsystem.camera.camera import BaslerCamera, WebcamCamera
from abbvisionsystem.preprocessing.preprocessing import (
    prepare_for_detection,
    apply_image_enhancement,
)
from abbvisionsystem.utils.visualization import draw_detection_summary
from abbvisionsystem.camera.calibration import CameraCalibrator
import logging

logger = logging.getLogger(__name__)


def get_model(model_type="defect_yolo"):
    """Get model instance from main app."""
    from abbvisionsystem.app import get_model as app_get_model

    return app_get_model(model_type)


def detection_system_page():
    """Main detection system interface with calibration-based coordinates."""
    st.title("🏠 Image Detection System")
    st.markdown(
        "Upload images or connect camera for real-time defect detection with calibrated measurements"
    )

    # Load calibrator instance (shared across app)
    if "main_calibrator" not in st.session_state:
        st.session_state.main_calibrator = CameraCalibrator()
        # Try to load existing calibration
        if os.path.exists("calibrations/camera_calibration.json"):
            st.session_state.main_calibrator.load_calibration(
                "calibrations/camera_calibration.json"
            )

    calibrator = st.session_state.main_calibrator

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Detection Settings")

        # Model selection
        model_type = st.selectbox(
            "Detection Model", ["defect_yolo", "defect_classification"], index=0
        )

        # Detection parameters
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

        # Calibration status
        st.subheader("📏 Calibration Status")
        if calibrator.calibration_result:
            st.success("✅ System Calibrated")
            st.info(f"Scale: {calibrator.calibration_result.pixels_per_mm:.2f} px/mm")
            st.info(f"Error: {calibrator.calibration_result.reprojection_error:.3f}")

            # Option to reload calibration
            if st.button("🔄 Reload Calibration"):
                if os.path.exists("calibrations/camera_calibration.json"):
                    if calibrator.load_calibration(
                        "calibrations/camera_calibration.json"
                    ):
                        st.success("Calibration reloaded!")
                        st.experimental_rerun()
        else:
            st.warning("⚠️ System Not Calibrated")
            st.info(
                "Go to Camera Calibration page to calibrate the system for accurate measurements"
            )

        # Coordinate system settings
        st.subheader("🎯 Coordinate Display")
        show_coordinate_system = st.checkbox("Show Coordinate System", value=True)
        show_measurements = st.checkbox("Show Object Measurements", value=True)
        show_origin_lines = st.checkbox("Show Origin Connection Lines", value=True)

    # Main content area
    tab1, tab2 = st.tabs(["📤 Image Upload", "📷 Camera Detection"])

    with tab1:
        image_detection_interface(
            calibrator,
            model_type,
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
            conf_threshold,
            iou_threshold,
            show_coordinate_system,
            show_measurements,
            show_origin_lines,
        )


def image_detection_interface(
    calibrator,
    model_type,
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

                # Load and run model
                try:
                    model = get_model(model_type)

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

                        # Run detection
                        model = get_model(model_type)
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
