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
                rows=8,  # inner corners
                cols=5,  # inner corners
                square_size_mm=25.0
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
                        
                        # Add image for calibration first
                        if st.session_state.calibrator.add_calibration_image(image):
                            valid_images += 1
                            
                            # Get the detected corners from the latest calibration image
                            latest_calib_data = st.session_state.calibrator.calibration_images[-1]
                            corners = latest_calib_data["corners"]
                            
                            if corners is not None and len(corners) > 0:
                                # Set origin to the first corner (top-left of chessboard pattern)
                                origin_point = tuple(corners[0].ravel().astype(int))
                                
                                # Store origin from first valid image
                                if valid_images == 1:
                                    st.session_state.origin_point = origin_point
                                
                                # Create visualization with actual origin
                                vis_image = image.copy()
                                ox, oy = origin_point
                                
                                # Draw origin point at actual first corner
                                cv2.circle(vis_image, (ox, oy), 8, (255, 255, 0), -1)
                                
                                # Draw coordinate axes
                                axis_length = 100
                                # X-axis (Red) - along chessboard row
                                cv2.arrowedLine(vis_image, (ox, oy), (ox + axis_length, oy), 
                                               (0, 0, 255), 3, tipLength=0.1)
                                # Y-axis (Green) - along chessboard column  
                                cv2.arrowedLine(vis_image, (ox, oy), (ox, oy + axis_length), 
                                               (0, 255, 0), 3, tipLength=0.1)
                                
                                # Add labels
                                cv2.putText(vis_image, "O(0,0)", (ox-30, oy-15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                                cv2.putText(vis_image, "X", (ox + axis_length + 10, oy + 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                cv2.putText(vis_image, "Y", (ox - 15, oy + axis_length + 25), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                
                                # Show pattern detection visualization
                                pattern_vis = st.session_state.calibrator.visualize_calibration(
                                    vis_image, draw_axes=True, draw_cube=True
                                )
                                st.image(pattern_vis, caption=f"✅ {file.name} - Origin at first corner", channels="BGR")
                            else:
                                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                       caption=f"❌ {file.name} - Pattern not detected")
                        else:
                            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                   caption=f"❌ {file.name} - Invalid pattern")
            
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
                                
                                # Display origin information
                                if hasattr(st.session_state, 'origin_point'):
                                    st.info(f"Origin set at first corner: {st.session_state.origin_point}")
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
                    
                    # Use calibrated origin point - FIXED
                    ox, oy = 100, 100  # Default fallback
                    if hasattr(st.session_state, 'origin_point') and st.session_state.origin_point:
                        ox, oy = st.session_state.origin_point
                        st.info(f"Using calibrated origin: ({ox}, {oy})")
                    elif st.session_state.calibrator.calibration_result:
                        try:
                            # Get origin from first calibration image
                            first_calib = st.session_state.calibrator.calibration_images[0]
                            if first_calib["corners"] is not None:
                                origin_point = tuple(first_calib["corners"][0].ravel().astype(int))
                                ox, oy = origin_point
                                st.session_state.origin_point = origin_point
                                st.info(f"Origin set from calibration: ({ox}, {oy})")
                        except Exception as e:
                            logger.warning(f"Could not get calibrated origin: {e}")
                            st.warning("Using default origin (100, 100)")
                    else:
                        st.warning("System not calibrated - using default origin")
                    
                    # Get calibration data for unit conversion
                    pixels_per_mm = None
                    if st.session_state.calibrator.calibration_result:
                        pixels_per_mm = st.session_state.calibrator.calibration_result.pixels_per_mm
                        st.info(f"Scale factor: {pixels_per_mm:.2f} pixels/mm")
                    
                    # Draw coordinate system - match calibration style
                    # Origin point (Yellow dot) - larger like in calibration
                    cv2.circle(result_image, (ox, oy), 8, (255, 255, 0), -1)
                    
                    # Coordinate axes - match calibration
                    axis_length = 100
                    # X-axis (Red) - along chessboard row
                    cv2.arrowedLine(result_image, (ox, oy), (ox + axis_length, oy), 
                                   (0, 0, 255), 3, tipLength=0.1)
                    # Y-axis (Green) - along chessboard column  
                    cv2.arrowedLine(result_image, (ox, oy), (ox, oy + axis_length), 
                                   (0, 255, 0), 3, tipLength=0.1)
                    
                    # Labels - match calibration style
                    cv2.putText(result_image, "O(0,0)", (ox-30, oy-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(result_image, "X", (ox + axis_length + 10, oy + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(result_image, "Y", (ox - 15, oy + axis_length + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Run detection after drawing coordinate system
                    if detections and detections["num_detections"] > 0:
                        detection_results = []  # Store results for measurements display
                        
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
                                
                                # Calculate coordinates relative to calibrated origin (in pixels)
                                rel_x_px = center[0] - ox
                                rel_y_px = center[1] - oy
                                
                                # Convert to millimeters if calibrated
                                if pixels_per_mm:
                                    rel_x_mm = rel_x_px / pixels_per_mm
                                    rel_y_mm = rel_y_px / pixels_per_mm
                                    coord_text = f"({rel_x_mm:.1f}, {rel_y_mm:.1f})mm"
                                    unit = "mm"
                                    
                                    # Calculate object dimensions in mm
                                    width_mm = size[0] / pixels_per_mm
                                    height_mm = size[1] / pixels_per_mm
                                    dimensions_text = f"{width_mm:.1f}x{height_mm:.1f}mm"
                                else:
                                    rel_x_mm = rel_x_px
                                    rel_y_mm = rel_y_px
                                    coord_text = f"({rel_x_px}, {rel_y_px})px"
                                    unit = "px"
                                    dimensions_text = f"{size[0]}x{size[1]}px"
                                
                                # Store results for measurements display
                                detection_results.append({
                                    'id': i + 1,
                                    'x': rel_x_mm,
                                    'y': rel_y_mm,
                                    'angle': angle,
                                    'confidence': detections['scores'][i],
                                    'width': width_mm if pixels_per_mm else size[0],
                                    'height': height_mm if pixels_per_mm else size[1],
                                    'unit': unit
                                })
                                
                                # Draw line from origin to object center
                                cv2.line(result_image, (ox, oy), center,
                                        (255, 255, 0), 1, cv2.LINE_AA)
                                
                                # Draw text with coordinates, angle, and dimensions
                                cv2.putText(result_image, coord_text,
                                          (center[0]-60, center[1]-30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                                cv2.putText(result_image, f"{angle:.1f}°",
                                          (center[0]-30, center[1]-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
                                cv2.putText(result_image, dimensions_text,
                                          (center[0]-40, center[1]+30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 2)

                        # Show detection results
                        st.subheader("Detection Results")
                        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                        
                        # Update measurements in col2
                        with col2:
                            st.subheader("Measurements")
                            if pixels_per_mm:
                                st.success(f"✅ Calibrated System (Scale: {pixels_per_mm:.2f} pixels/mm)")
                            else:
                                st.warning("⚠️ Uncalibrated - showing pixel coordinates")
                            
                            for result in detection_results:
                                st.write(f"**Object {result['id']}:**")
                                st.write(f"- Position: ({result['x']:.1f}, {result['y']:.1f}) {result['unit']}")
                                st.write(f"- Orientation: {result['angle']:.1f}°")
                                st.write(f"- Dimensions: {result['width']:.1f} x {result['height']:.1f} {result['unit']}")
                                st.write(f"- Confidence: {result['confidence']:.2f}")
                                
                                # Calculate distance from origin
                                if pixels_per_mm:
                                    distance_mm = np.sqrt(result['x']**2 + result['y']**2)
                                    st.write(f"- Distance from origin: {distance_mm:.1f} mm")
                                else:
                                    distance_px = np.sqrt(result['x']**2 + result['y']**2)
                                    st.write(f"- Distance from origin: {distance_px:.1f} px")
                                st.write("---")
                    else:
                        # Show image with coordinate system even if no objects detected
                        st.subheader("Detection Results")
                        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
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