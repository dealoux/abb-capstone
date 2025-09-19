"""Detection system page with simplified model management."""

import os
from typing import Optional
import streamlit as st
import cv2
import numpy as np
from abbvisionsystem.camera.camera import BaslerCamera, WebcamCamera
from abbvisionsystem.models.model_factory import ModelFactory
from abbvisionsystem.camera.calibration import CameraCalibrator
from abbvisionsystem.ui.components.model_selector import ModelSelector
from abbvisionsystem.ui.components.calibration_manager import CalibrationManager
from abbvisionsystem.ui.components.coordinate_pickmaster_twin import (
    ABBPickMasterExporter,
    export_detections_for_abb,
)
import tempfile
import zipfile
import logging

logger = logging.getLogger(__name__)


def get_calibrated_origin(calibrator, image_shape):
    """Get the calibrated origin point or fallback to default."""
    if calibrator.origin_point:
        logger.info(f"Using saved origin from calibrator: {calibrator.origin_point}")
        return calibrator.origin_point

    if calibrator.calibration_result and calibrator.calibration_images:
        try:
            first_calib = calibrator.calibration_images[0]
            if first_calib["corners"] is not None and len(first_calib["corners"]) > 0:
                origin_point = tuple(first_calib["corners"][0].ravel().astype(int))
                calibrator.origin_point = origin_point
                logger.info(
                    f"Extracted and saved origin from calibration images: {origin_point}"
                )
                return origin_point
        except Exception as e:
            logger.warning(f"Could not get calibrated origin: {e}")

    height, width = image_shape
    fallback_origin = (width // 4, height // 4)
    logger.warning(f"Using fallback origin: {fallback_origin}")
    return fallback_origin


def detection_system_page():
    """Main detection system interface with calibration-based coordinates."""
    st.title("🏠 Image Detection System")
    st.markdown(
        "Upload images or connect camera for real-time defect detection with calibrated measurements"
    )

    # CRITICAL: Use the same calibrator instance as the calibration page
    if "main_calibrator" not in st.session_state:
        st.session_state.main_calibrator = CameraCalibrator()
        # Auto-load latest calibration if available
        try:
            latest_calibration = CalibrationManager.find_latest_calibration_file()
            if latest_calibration:
                st.session_state.main_calibrator.load_calibration_with_origin(
                    latest_calibration
                )
                st.session_state.loaded_calibration_file = latest_calibration
                st.info(
                    f"🎯 Auto-loaded calibration with origin: {st.session_state.main_calibrator.origin_point}"
                )
        except Exception as e:
            logger.warning(f"Could not auto-load calibration: {e}")

    calibrator = st.session_state.main_calibrator

    # SYNC: If we have calibration data from calibration page, sync it
    if "calibrator" in st.session_state:
        # Use the calibrator from calibration page (session_state.calibrator)
        calibrator = st.session_state.calibrator
        st.session_state.main_calibrator = calibrator
        st.info(f"🔄 Synced with calibration page - Origin: {calibrator.origin_point}")

    # Show current origin status prominently
    if calibrator.origin_point:
        st.success(f"✅ **Using Calibrated Origin**: {calibrator.origin_point}")
        st.info(
            "🎯 This origin is from the checkerboard calibration and will be used for all coordinate calculations"
        )
    else:
        st.warning(
            "⚠️ No calibrated origin available. Go to Camera Calibration page first."
        )

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Detection Settings")

        # Model type selection
        model_type = st.selectbox(
            "Detection Framework", ["defect_yolo", "defect_classification"], index=0
        )

        # Model selection (delegated to component) - FIXED
        selected_model_path = ModelSelector.render_model_selection(model_type)

        # Detection parameters
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

        # Calibration management (delegated to component)
        CalibrationManager.render_calibration_section(calibrator)

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


@st.cache_resource
def get_model_instance(model_type: str, model_path: Optional[str] = None):
    """Get cached model instance."""
    try:
        model = ModelFactory.create_model(model_type, model_path)
        if not model.load():
            raise RuntimeError(f"Failed to load {model_type} model")
        return model
    except Exception as e:
        st.error(f"Failed to create model: {str(e)}")
        raise


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

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Detection Results")

                try:
                    # Get model instance
                    model = get_model_instance(model_type, model_path)

                    # Run detection
                    with st.spinner("Running detection..."):
                        detections = model.predict(
                            image,
                            conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold,
                        )

                    # Process results (keep existing processing logic)
                    result_image = process_detection_results(
                        image,
                        detections,
                        calibrator,
                        show_coordinate_system,
                        show_measurements,
                        show_origin_lines,
                    )

                    st.image(
                        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                        use_container_width=True,  # Fixed: was use_column_width
                    )

                except Exception as e:
                    st.error(f"Detection failed: {str(e)}")
                    st.image(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                        caption="Original Image",
                        use_container_width=True,  # Fixed: was use_column_width
                    )

            with col2:
                st.subheader("📊 Detection Summary")
                if detections and detections.get("num_detections", 0) > 0:
                    display_detection_measurements(
                        detections, calibrator, image.shape[:2]
                    )
                else:
                    st.info("No objects detected")

                # Show calibration info
                st.subheader("📏 Measurement Info")
                if calibrator.calibration_result:
                    st.success("✅ Calibrated System")
                    st.write(
                        f"**Scale:** {calibrator.calibration_result.pixels_per_mm:.2f} pixels/mm"
                    )
                else:
                    st.warning("⚠️ Uncalibrated - showing pixel coordinates")

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
                        model = get_model_instance(model_type, model_path)
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
                    use_container_width=True,  # Fixed: was use_column_width
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

                # FIXED: Calculate actual object center using image analysis
                center_x, center_y, angle = find_accurate_object_center(
                    image, (x1, y1, x2, y2)
                )

                # Calculate bounding box dimensions
                width = x2 - x1
                height = y2 - y1

                # Draw bounding box
                class_id = detections["classes"][i] if "classes" in detections else 0
                color = (
                    (0, 0, 255) if class_id == 1 else (0, 255, 0)
                )  # Red for defect, Green for normal
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

                # Draw object center point - make it more visible
                cv2.circle(
                    result_image, (center_x, center_y), 6, (255, 0, 0), -1
                )  # Blue filled circle for object center
                cv2.circle(
                    result_image, (center_x, center_y), 8, (255, 255, 255), 2
                )  # White border

                # Draw connection line to origin if requested
                if show_origin_lines:
                    cv2.line(
                        result_image,
                        (ox, oy),
                        (center_x, center_y),
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                # Calculate and display measurements using the accurate center
                if show_measurements:
                    display_object_measurements(
                        result_image,
                        (center_x, center_y),
                        (width, height),
                        origin_point,
                        calibrator,
                        i + 1,
                        angle,
                    )

            except Exception as e:
                logger.error(f"Error processing detection {i}: {e}")
                continue

    return result_image


def find_accurate_object_center(image, bbox):
    """Find accurate object center using contour analysis and centroid calculation."""
    x1, y1, x2, y2 = bbox

    # Simple bounding box center as fallback
    bbox_center_x = (x1 + x2) // 2
    bbox_center_y = (y1 + y2) // 2

    try:
        # Extract the object region with padding
        padding = 10
        y_start = max(0, y1 - padding)
        y_end = min(image.shape[0], y2 + padding)
        x_start = max(0, x1 - padding)
        x_end = min(image.shape[1], x2 + padding)

        object_region = image[y_start:y_end, x_start:x_end]

        if object_region.size == 0:
            return bbox_center_x, bbox_center_y, 0

        # Convert to different color spaces for better object detection
        gray = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(object_region, cv2.COLOR_BGR2HSV)

        # Method 1: Use color-based segmentation for food packages
        # Create mask for the object (assuming it's different from background)
        # For your choco pie boxes, they're typically red/brown colors

        # Create mask based on HSV values (works better for colored objects)
        lower_bound = np.array([0, 30, 30])  # Lower HSV threshold
        upper_bound = np.array([180, 255, 255])  # Upper HSV threshold
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Method 2: Use edge detection
        edges = cv2.Canny(gray, 30, 100)

        # Method 3: Use adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )

        # Combine all methods
        combined_mask = cv2.bitwise_or(
            cv2.bitwise_or(color_mask, edges), adaptive_thresh
        )

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Get the largest contour (most likely the main object)
            largest_contour = max(contours, key=cv2.contourArea)

            # Check if contour is large enough
            if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold

                # Calculate centroid using moments
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    # Centroid coordinates relative to object region
                    cx_rel = int(M["m10"] / M["m00"])
                    cy_rel = int(M["m01"] / M["m00"])

                    # Convert to absolute image coordinates
                    center_x = x_start + cx_rel
                    center_y = y_start + cy_rel

                    # Ensure center is within the original bounding box
                    center_x = max(x1, min(x2, center_x))
                    center_y = max(y1, min(y2, center_y))

                    # Calculate orientation using minimum area rectangle
                    rect = cv2.minAreaRect(largest_contour)
                    angle = rect[2]

                    # Normalize angle
                    if rect[1][0] < rect[1][1]:
                        angle = angle + 90
                    angle = angle % 180

                    logger.info(
                        f"Found accurate center at ({center_x}, {center_y}) vs bbox center ({bbox_center_x}, {bbox_center_y})"
                    )
                    return center_x, center_y, angle

        # Method 4: If contours fail, use intensity-based center of mass
        # Invert image so object pixels have higher weight
        inverted = 255 - gray

        # Apply Gaussian blur to smooth out noise
        blurred = cv2.GaussianBlur(inverted, (5, 5), 0)

        # Find center of mass weighted by intensity
        y_indices, x_indices = np.indices(blurred.shape)
        total_intensity = np.sum(blurred)

        if total_intensity > 0:
            center_x_rel = np.sum(x_indices * blurred) / total_intensity
            center_y_rel = np.sum(y_indices * blurred) / total_intensity

            # Convert to absolute coordinates
            center_x = int(x_start + center_x_rel)
            center_y = int(y_start + center_y_rel)

            # Ensure center is within bounding box
            center_x = max(x1, min(x2, center_x))
            center_y = max(y1, min(y2, center_y))

            logger.info(f"Using intensity-based center at ({center_x}, {center_y})")
            return center_x, center_y, 0

    except Exception as e:
        logger.warning(f"Error in accurate center detection: {e}")

    # Fallback to geometric center of bounding box
    logger.warning(f"Using fallback bbox center at ({bbox_center_x}, {bbox_center_y})")
    return bbox_center_x, bbox_center_y, 0


def display_detection_measurements(detections, calibrator, image_shape):
    """Display detection measurements in the sidebar using accurate centers."""
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

    # Create a temporary image for center calculation (we need the actual image)
    # Note: This is a limitation - we need the actual image to calculate accurate centers
    # For now, we'll use the bounding box center but add a note

    st.warning(
        "📍 Note: Measurements shown use bounding box centers. For accurate object centers, view the detection image."
    )

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

            # FIXED: For measurements display, we need to use the same logic as in the image
            # But we don't have access to the actual image here, so we'll use bbox center
            # and add a note that the visual shows the accurate center
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

            # Calculate orientation
            theta = np.arctan2(height, width) * 180 / np.pi
            if theta < 0:
                theta += 180
            theta = theta % 180

            # Get class info
            class_id = detections["classes"][i] if "classes" in detections else 0
            confidence = detections["scores"][i] if "scores" in detections else 0

            # Display object info
            st.write(f"**Object {i+1}:**")
            st.write(f"- Position: ({rel_x:.1f}, {rel_y:.1f}) {unit} ⚠️ *bbox center*")
            st.write(f"- Dimensions: {obj_width:.1f} × {obj_height:.1f} {unit}")
            st.write(f"- Orientation: {theta:.1f}°")
            st.write(f"- Distance from origin: {distance:.1f} {unit}")
            st.write(f"- Confidence: {confidence:.2f}")

            # Show class if available
            if hasattr(detections, "labels") and detections["labels"]:
                st.write(f"- Class: {detections['labels'][i]}")
            elif class_id == 1:
                st.write(f"- Class: Defect")
            else:
                st.write(f"- Class: Normal")

            # Add note about accurate center
            st.caption("💡 Blue dot on image shows accurate object center")
            st.write("---")

        except Exception as e:
            st.error(f"Error displaying object {i+1}: {str(e)}")


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

    # Add axis labels only (removed the O(0,0) label)
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
    """DEPRECATED: Use find_accurate_object_center instead."""
    # This function is kept for backward compatibility but should not be used
    return find_accurate_object_center_legacy(object_region, region_offset)


def find_accurate_object_center_legacy(object_region, region_offset):
    """Legacy function - improved version."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)

        # Use multiple thresholding methods
        methods = [
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU),
            cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY),
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            ),
        ]

        best_center = None
        best_area = 0
        best_angle = 0

        for _, binary in methods:
            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area > best_area:
                    best_area = area

                    # Calculate centroid using moments
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        best_center = (cx, cy)

                        # Get orientation
                        rect = cv2.minAreaRect(largest_contour)
                        best_angle = rect[2]

                        # Normalize angle
                        if rect[1][0] < rect[1][1]:
                            best_angle = best_angle + 90
                        best_angle = best_angle % 180

        if best_center:
            # Convert to absolute coordinates
            abs_center_x = int(best_center[0] + region_offset[0])
            abs_center_y = int(best_center[1] + region_offset[1])
            return abs_center_x, abs_center_y, best_angle

    except Exception as e:
        logger.warning(f"Error in legacy center detection: {e}")

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

            # Calculate orientation
            theta = np.arctan2(height, width) * 180 / np.pi
            if theta < 0:
                theta += 180
            theta = theta % 180

            # Get class info
            class_id = detections["classes"][i] if "classes" in detections else 0
            confidence = detections["scores"][i] if "scores" in detections else 0

            # Display object info
            st.write(f"**Object {i+1}:**")
            st.write(f"- Position: ({rel_x:.1f}, {rel_y:.1f}) {unit}")
            st.write(f"- Dimensions: {obj_width:.1f} × {obj_height:.1f} {unit}")
            st.write(f"- Orientation: {theta:.1f}°")
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

    st.subheader("🤖 ABB Export")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📤 Export for ABB PickMaster"):
            try:
                exported_path = export_detections_for_abb(
                    detections, calibrator, image_shape, origin_point
                )

                if exported_path:
                    st.success(f"✅ Exported: {os.path.basename(exported_path)}")

                    # Read the file for download
                    with open(exported_path, "r", encoding="utf-8") as f:
                        xml_content = f.read()

                    st.download_button(
                        label="⬇️ Download XML",
                        data=xml_content,
                        file_name=os.path.basename(exported_path),
                        mime="application/xml",
                    )
                else:
                    st.error("Failed to export XML file")

            except Exception as e:
                st.error(f"Export error: {str(e)}")

    with col2:
        # Show export preview
        if st.checkbox("🔍 Preview XML"):
            try:
                exporter = ABBPickMasterExporter(calibrator)

                # Create temporary XML for preview
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".xml", delete=False
                ) as tmp_file:
                    if exporter.export_detections_to_xml(
                        detections, image_shape, tmp_file.name, origin_point
                    ):
                        with open(tmp_file.name, "r") as f:
                            xml_preview = f.read()

                        with st.expander("XML Preview"):
                            st.code(xml_preview, language="xml")

                        os.unlink(tmp_file.name)

            except Exception as e:
                st.error(f"Preview error: {str(e)}")


if __name__ == "__main__":
    detection_system_page()
