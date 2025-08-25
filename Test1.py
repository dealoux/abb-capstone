import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import logging
from abbvisionsystem.camera.calibration import CameraCalibrator, CalibrationResult
from abbvisionsystem.models.yolo_model import YOLODefectModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalibrationDetectionUI:
    def __init__(self):
        """Initialize UI components and system."""
        st.set_page_config(page_title="ABB Vision System", layout="wide")
        
        # Initialize calibrator with default pattern size
        if 'calibrator' not in st.session_state:
            calibrator = CameraCalibrator()
            calibrator.set_chessboard_pattern(
                rows=27,  # inner corners
                cols=39,  # inner corners
                square_size_mm=10.0
            )
            st.session_state.calibrator = calibrator
            
        if 'model' not in st.session_state:
            st.session_state.model = YOLODefectModel()
            st.session_state.model.load()

    def calibration_page(self):
        """Render calibration interface."""
        st.header("Camera Calibration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pattern settings
            st.subheader("Calibration Pattern Settings")
            rows = st.number_input("Number of inner corners (rows)", value=27, min_value=1)
            cols = st.number_input("Number of inner corners (cols)", value=39, min_value=1)
            square_size = st.number_input("Square Size (mm)", value=10.0, min_value=0.1)
            
            if st.button("Update Pattern Settings"):
                st.session_state.calibrator.set_chessboard_pattern(rows, cols, square_size)
                st.success("Pattern settings updated!")

            # Upload calibration images
            st.subheader("Calibration Images")
            uploaded_files = st.file_uploader(
                "Upload chessboard pattern images", 
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key="calib_images"
            )

            if uploaded_files:
                if st.button("Start Calibration"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process each calibration image
                    valid_images = 0
                    for i, file in enumerate(uploaded_files):
                        # Convert uploaded file to image
                        image_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                        
                        # Create a copy for visualization
                        vis_image = image.copy()
                        
                        # Draw coordinate system
                        ox, oy = 100, 100  # Default origin point
                        
                        # Draw origin point (Yellow dot)
                        cv2.circle(vis_image, (ox, oy), 5, (255, 255, 0), -1)
                        
                        # Draw coordinate axes
                        axis_length = 100
                        # X-axis (Red)
                        cv2.arrowedLine(vis_image, (ox, oy), (ox + axis_length, oy), 
                                       (0, 0, 255), 2, tipLength=0.1)
                        # Y-axis (Green)
                        cv2.arrowedLine(vis_image, (ox, oy), (ox, oy - axis_length), 
                                       (0, 255, 0), 2, tipLength=0.1)
                        
                        # Add labels
                        cv2.putText(vis_image, "O(0,0)", (ox-20, oy-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        cv2.putText(vis_image, "X", (ox + axis_length + 10, oy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(vis_image, "Y", (ox - 20, oy - axis_length - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Show calibration image with pattern detection and coordinate system
                        pattern_vis = st.session_state.calibrator.visualize_calibration(
                            vis_image, draw_axes=True, draw_cube=True
                        )
                        st.image(pattern_vis, caption=f"Processing {file.name}", channels="BGR")
                        
                        # Add image for calibration (use original image)
                        if st.session_state.calibrator.add_calibration_image(image):
                            valid_images += 1
                            
                            # If this is the first valid image, store its corner as origin
                            if valid_images == 1:
                                corners = st.session_state.calibrator.calibration_images[-1]["corners"]
                                if corners is not None and len(corners) > 0:
                                    origin_point = tuple(corners[0].ravel())
                                    st.session_state.origin_point = origin_point
                
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {i+1}/{len(uploaded_files)} images")

                    # Perform calibration if we have enough images
                    if valid_images >= 3:
                        with st.spinner("Performing calibration..."):
                            result = st.session_state.calibrator.calibrate_camera()
                            if result:
                                st.session_state.calibrator.save_calibration("camera_calibration.json")
                                st.success(f"Calibration successful! Found {valid_images} valid patterns.")
                            else:
                                st.error("Calibration failed")
                    else:
                        st.error(f"Need at least 3 valid images, got {valid_images}")

        with col2:
            # Show calibration status and results
            st.subheader("Calibration Status")
            if st.session_state.calibrator.calibration_result:
                st.success("✅ System Calibrated")
                result = st.session_state.calibrator.calibration_result
                st.write("Calibration Parameters:")
                st.write(f"- Reprojection Error: {result.reprojection_error:.3f}")
                st.write(f"- Scale: {result.pixels_per_mm:.2f} pixels/mm")
                st.write(f"- Image Size: {result.image_size}")
                st.write(f"- Calibration Date: {result.calibration_date}")
            else:
                st.warning("⚠️ System Not Calibrated")
                
            # Show calibration tips
            st.subheader("Calibration Tips")
            tips = st.session_state.calibrator.get_detection_tips()
            for tip in tips:
                st.markdown(f"- {tip}")

    def detection_page(self):
        """Render detection interface."""
        st.header("Object Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Detection settings
            conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
            
            uploaded_file = st.file_uploader("Upload Image for Detection", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                try:
                    # Convert uploaded file to image
                    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        st.error("Failed to load image")
                        return
                    
                    # Show original image
                    st.subheader("Original Image")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
                    # Run detection
                    detections = st.session_state.model.predict(image, conf_threshold=conf_threshold)
                    
                    # Create result image
                    result_image = image.copy()
                    
                    # Always draw coordinate system first
                    ox, oy = 100, 100  # Default origin point if not calibrated
                    if st.session_state.calibrator.calibration_result:
                        try:
                            # Get origin point from calibration
                            origin_point = tuple(st.session_state.calibrator.calibration_images[0]["corners"][0].ravel())
                            ox, oy = map(int, origin_point)
                        except Exception as e:
                            logger.warning(f"Could not get calibrated origin, using default: {e}")
                    
                    # Draw coordinate system
                    # Origin point (Yellow dot)
                    cv2.circle(result_image, (ox, oy), 5, (255, 255, 0), -1)
                    
                    # Coordinate axes
                    axis_length = 100
                    # X-axis (Red)
                    cv2.arrowedLine(result_image, (ox, oy), (ox + axis_length, oy), 
                                    (0, 0, 255), 2, tipLength=0.1)
                    # Y-axis (Green)
                    cv2.arrowedLine(result_image, (ox, oy), (ox, oy - axis_length), 
                                    (0, 255, 0), 2, tipLength=0.1)
                    
                    # Labels
                    cv2.putText(result_image, "O(0,0)", (ox-20, oy-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(result_image, "X", (ox + axis_length + 10, oy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(result_image, "Y", (ox - 20, oy - axis_length - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Run detection after drawing coordinate system
                    if detections and detections["num_detections"] > 0:
                        for i in range(detections["num_detections"]):
                            box = detections["absolute_boxes"][i]
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Get the object region
                            object_region = image[y1:y2, x1:x2]
                            if object_region.size == 0:
                                continue
                                
                            # Convert to grayscale for contour detection
                            gray = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)
                            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                            
                            # Find contours
                            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            

                            if contours:
                                # Get the largest contour
                                largest_contour = max(contours, key=cv2.contourArea)
                                
                                # Shift contour points to original image coordinates
                                shifted_contour = largest_contour + np.array([x1, y1])
                                
                                # Fit minimum area rectangle
                                rect = cv2.minAreaRect(shifted_contour)
                                center = tuple(map(int, rect[0]))
                                size = tuple(map(int, rect[1]))
                                angle = rect[2]
                                
                                # Normalize angle to 0-180 degrees
                                if size[0] < size[1]:
                                    angle = angle + 90
                                    
                                # Convert angle to positive range (0-180)
                                angle = angle % 180
                                
                                # Draw rotated rectangle
                                box_points = cv2.boxPoints(rect)
                                box_points = box_points.astype(np.int_)
                                cv2.drawContours(result_image, [box_points], 0, (255,0,255), 2)
                                
                                # Draw orientation line
                                line_length = max(size) // 2
                                orientation_x = center[0] + line_length * np.cos(np.radians(angle))
                                orientation_y = center[1] + line_length * np.sin(np.radians(angle))
                                cv2.arrowedLine(result_image, 
                                                  center,
                                                  (int(orientation_x), int(orientation_y)),
                                                  (255, 0, 255), 2, tipLength=0.3)
                                
                                # Draw center point
                                cv2.circle(result_image, center, 4, (0,0,255), -1)
                                
                                # Calculate coordinates relative to origin if calibrated
                                if st.session_state.calibrator.calibration_result:
                                    rel_x = center[0] - ox
                                    rel_y = oy - center[1]
                                    world_x, world_y = st.session_state.calibrator.pixel_to_world_coordinates(
                                        rel_x, rel_y
                                    )
                                    coord_text = f"({world_x:.1f}, {world_y:.1f})mm\n{angle:.1f}°"
                                    
                                    # Draw line from origin to object center
                                    cv2.line(result_image, (ox, oy), center,
                                            (255, 255, 0), 1, cv2.LINE_AA)
                                else:
                                    coord_text = f"({center[0]}, {center[1]})px\n{angle:.1f}°"
                                
                                # Draw text with both coordinates and angle
                                cv2.putText(result_image, coord_text.split('\n')[0],
                                          (center[0]-45, center[1]-20),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                                cv2.putText(result_image, coord_text.split('\n')[1],
                                          (center[0]-45, center[1]+20),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

                        # Show detection results
                        st.subheader("Detection Results")
                        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                        
                        # Update measurements in col2
                        with col2:
                            st.subheader("Measurements")
                            for i in range(detections["num_detections"]):
                                st.write(f"Object {i+1}:")
                                if st.session_state.calibrator.calibration_result:
                                    st.write(f"Position: {coord_text.split('\n')[0]}")
                                st.write(f"Orientation: {angle:.1f}°")
                                st.write(f"Confidence: {detections['scores'][i]:.2f}")
                    else:
                        st.warning("No objects detected")
                        with col2:
                            st.info("No objects detected in the image")

                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    logger.error(f"Detection error: {str(e)}", exc_info=True)
            else:
                with col2:
                    st.info("Upload an image to see measurements")

    def main(self):
        """Main UI layout and navigation."""
        st.title("ABB Vision System")
        
        # Navigation
        tab1, tab2 = st.tabs(["Calibration", "Detection"])
        
        with tab1:
            self.calibration_page()
            
        with tab2:
            self.detection_page()

if __name__ == "__main__":
    ui = CalibrationDetectionUI()
    ui.main()