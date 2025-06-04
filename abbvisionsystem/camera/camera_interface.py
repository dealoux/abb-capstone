"""Streamlit interface for camera calibration."""

import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from abbvisionsystem.camera.camera import BaslerCamera
from abbvisionsystem.camera.calibration import CameraCalibrator


def camera_calibration_interface():
    """Camera calibration interface for Basler cameras."""
    st.title("📷 Camera Calibration")
    st.write(
        "Calibrate your Basler camera for accurate measurements and defect detection"
    )

    # Initialize calibrator in session state
    if "calibrator" not in st.session_state:
        st.session_state.calibrator = CameraCalibrator()

    calibrator = st.session_state.calibrator

    # Sidebar for calibration settings
    with st.sidebar:
        st.header("Calibration Settings")

        # Chessboard pattern configuration
        st.subheader("Chessboard Pattern")
        cols = st.number_input(
            "Columns (inner corners)", min_value=4, max_value=50, value=39
        )
        rows = st.number_input(
            "Rows (inner corners)", min_value=4, max_value=50, value=27
        )
        square_size = st.number_input(
            "Square Size (mm)", min_value=1.0, max_value=100.0, value=10.0, step=0.1
        )

        if st.button("Update Pattern Settings"):
            calibrator.set_chessboard_pattern(rows, cols, square_size)
            st.success("Pattern settings updated")

        # Display current settings
        st.info(
            f"Current pattern: {calibrator.chessboard_size[0]}×{calibrator.chessboard_size[1]}"
        )
        st.info(f"Square size: {calibrator.square_size_mm}mm")

        # Visualization options
        st.subheader("Visualization Options")
        show_axes = st.checkbox("Show Coordinate Axes", value=True)
        show_cube = st.checkbox("Show 3D Cube", value=False)
        show_numbering = st.checkbox("Show Corner Numbers", value=False)
        show_camera_frame = st.checkbox("Show Camera Pinhole Frame", value=False)

        # Camera connection
        st.subheader("Camera Connection")
        if "camera" not in st.session_state:
            st.session_state.camera = None

        camera_index = st.number_input(
            "Camera Index", min_value=0, max_value=5, value=0
        )

        if st.button("Connect Camera"):
            camera = BaslerCamera(device_index=camera_index)
            if camera.connect():
                st.session_state.camera = camera
                st.success("Camera connected successfully")
            else:
                st.error("Failed to connect to camera")

        if st.session_state.camera and st.session_state.camera.connected:
            st.success("📷 Camera Connected")

            # Camera parameters
            st.subheader("Camera Parameters")
            exposure = st.slider("Exposure Time (µs)", 100, 50000, 10000, 100)
            gain = st.slider("Gain", 0.0, 10.0, 1.0, 0.1)

            if st.button("Update Camera Settings"):
                st.session_state.camera.set_exposure_time(exposure)
                st.session_state.camera.set_gain(gain)
                st.success("Camera settings updated")

    # Main calibration interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Image Capture & Collection")

        # Live preview with enhanced visualization
        if st.session_state.camera and st.session_state.camera.connected:
            col_preview1, col_preview2 = st.columns(2)

            with col_preview1:
                if st.button("📸 Capture Live Preview"):
                    image = st.session_state.camera.capture_image()
                    if image is not None:
                        st.session_state.preview_image = image
                        st.success("Image captured!")
                    else:
                        st.error("Failed to capture image")

            with col_preview2:
                if st.button("🎯 Visualize Pattern"):
                    if hasattr(st.session_state, "preview_image"):
                        # Create enhanced visualization using draw function
                        vis_image = calibrator.draw_chessboard_pattern(
                            st.session_state.preview_image,
                            draw_axes=show_axes,
                            draw_cube=show_cube,
                            draw_camera_frame=show_camera_frame,
                        )

                        # Store visualization for display
                        st.session_state.visualization_image = vis_image
                        st.success("Pattern visualized!")
                    else:
                        st.warning("Capture an image first")

        # Display images side by side
        if hasattr(st.session_state, "preview_image"):
            st.subheader("Image Preview & Pattern Detection")

            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(
                ["Original", "Pattern Detection", "Enhanced Visualization"]
            )

            with tab1:
                st.image(
                    cv2.cvtColor(st.session_state.preview_image, cv2.COLOR_BGR2RGB),
                    caption="Original Captured Image",
                    use_container_width=True,
                )

            with tab2:
                # Basic pattern detection visualization
                basic_vis = calibrator.visualize_calibration(
                    st.session_state.preview_image,
                    draw_axes=False,
                    draw_cube=False,
                    draw_camera_frame=show_camera_frame,
                )
                st.image(
                    cv2.cvtColor(basic_vis, cv2.COLOR_BGR2RGB),
                    caption="Basic Pattern Detection",
                    use_container_width=True,
                )

            with tab3:
                # Enhanced visualization with all features
                if hasattr(st.session_state, "visualization_image"):
                    st.image(
                        cv2.cvtColor(
                            st.session_state.visualization_image, cv2.COLOR_BGR2RGB
                        ),
                        caption="Enhanced Pattern Visualization with 3D Elements",
                        use_container_width=True,
                    )
                else:
                    st.info("Click 'Visualize Pattern' to see enhanced visualization")

        # Add calibration image
        if st.button("✅ Add Calibration Image"):
            if hasattr(st.session_state, "preview_image"):
                success = calibrator.add_calibration_image(
                    st.session_state.preview_image
                )
                if success:
                    st.success(f"Added image {len(calibrator.calibration_images)}")

                    # Show the added image with visualization
                    st.subheader("Added Image Visualization")
                    added_vis = calibrator.draw_chessboard_pattern(
                        st.session_state.preview_image, draw_axes=True, draw_cube=False
                    )
                    st.image(
                        cv2.cvtColor(added_vis, cv2.COLOR_BGR2RGB),
                        caption=f"Calibration Image {len(calibrator.calibration_images)} - Successfully Added",
                        use_container_width=True,
                    )
                else:
                    st.error("Could not detect chessboard in image")

                    # Show failed detection visualization
                    failed_vis = calibrator.draw_chessboard_pattern(
                        st.session_state.preview_image, draw_axes=False, draw_cube=False
                    )
                    st.image(
                        cv2.cvtColor(failed_vis, cv2.COLOR_BGR2RGB),
                        caption="Failed Detection - Check pattern settings and image quality",
                        use_container_width=True,
                    )
            else:
                st.warning("Capture a preview image first")

        # Manual image upload with visualization
        st.subheader("Manual Image Upload")
        uploaded_files = st.file_uploader(
            "Upload Calibration Images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.subheader("Uploaded Images Processing")
            for i, uploaded_file in enumerate(uploaded_files):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                success = calibrator.add_calibration_image(image)

                # Create visualization for each uploaded image
                upload_vis = calibrator.draw_chessboard_pattern(
                    image, draw_axes=show_axes, draw_cube=show_cube
                )

                # Show result with status
                status_text = (
                    "✅ Successfully Added" if success else "❌ Detection Failed"
                )
                status_color = "green" if success else "red"

                with st.expander(f"{uploaded_file.name} - {status_text}"):
                    col_orig, col_vis = st.columns(2)

                    with col_orig:
                        st.image(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                            caption="Original",
                            use_container_width=True,
                        )

                    with col_vis:
                        st.image(
                            cv2.cvtColor(upload_vis, cv2.COLOR_BGR2RGB),
                            caption=f"Pattern Detection - {status_text}",
                            use_container_width=True,
                        )

        # Current status
        st.subheader("Calibration Status")
        num_images = len(calibrator.calibration_images)
        st.metric("Calibration Images", num_images)

        if num_images < 5:
            st.warning("Need at least 5 images for calibration")
        else:
            st.success("Ready for calibration!")

        # Show collected images gallery
        if num_images > 0:
            with st.expander(f"View Collected Images ({num_images})"):
                cols_per_row = 3
                for i in range(0, num_images, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < num_images:
                            with cols[j]:
                                calib_data = calibrator.calibration_images[i + j]
                                vis_img = calibrator.draw_chessboard_pattern(
                                    calib_data["image"],
                                    draw_axes=False,
                                    draw_cube=False,
                                )
                                st.image(
                                    cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB),
                                    caption=f"Image {i + j + 1}",
                                    use_container_width=True,
                                )

        # Clear images
        if st.button("🗑️ Clear All Images"):
            calibrator.calibration_images.clear()
            st.success("Cleared all calibration images")

    with col2:
        st.subheader("Calibration & Results")

        # Perform calibration
        if st.button("🔧 Calibrate Camera"):
            if len(calibrator.calibration_images) >= 5:
                with st.spinner("Performing camera calibration..."):
                    result = calibrator.calibrate_camera()

                if result:
                    st.success("✅ Calibration successful!")

                    # Display calibration results
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.metric(
                            "Reprojection Error", f"{result.reprojection_error:.4f}"
                        )
                        st.metric(
                            "Image Size",
                            f"{result.image_size[0]}×{result.image_size[1]}",
                        )

                    with col_b:
                        if result.pixels_per_mm:
                            st.metric("Scale", f"{result.pixels_per_mm:.2f} px/mm")
                            mm_per_pixel = 1.0 / result.pixels_per_mm
                            st.metric("Resolution", f"{mm_per_pixel:.4f} mm/px")

                    # Camera matrix
                    st.subheader("Camera Matrix")
                    st.code(
                        f"""
fx = {result.camera_matrix[0,0]:.2f}
fy = {result.camera_matrix[1,1]:.2f}
cx = {result.camera_matrix[0,2]:.2f}
cy = {result.camera_matrix[1,2]:.2f}
                    """
                    )

                    # Distortion coefficients
                    st.subheader("Distortion Coefficients")
                    st.code(
                        f"""
k1 = {result.distortion_coefficients[0,0]:.6f}
k2 = {result.distortion_coefficients[0,1]:.6f}
p1 = {result.distortion_coefficients[0,2]:.6f}
p2 = {result.distortion_coefficients[0,3]:.6f}
k3 = {result.distortion_coefficients[0,4]:.6f}
                    """
                    )

                    # Visualization of calibration results
                    st.subheader("🎯 Calibration Visualization Results")

                    if hasattr(st.session_state, "preview_image"):
                        # Show calibration visualization with all features
                        calib_vis = calibrator.draw_chessboard_pattern(
                            st.session_state.preview_image,
                            draw_axes=True,
                            draw_cube=True,
                        )

                        st.image(
                            cv2.cvtColor(calib_vis, cv2.COLOR_BGR2RGB),
                            caption="Calibration Result: 3D Coordinate System and Cube Projection",
                            use_container_width=True,
                        )

                        # Quality assessment visualization
                        st.info(
                            f"""
                        **Calibration Quality Assessment:**
                        - Reprojection Error: {result.reprojection_error:.4f} pixels
                        - Quality: {'Excellent' if result.reprojection_error < 0.5 else 'Good' if result.reprojection_error < 1.0 else 'Poor - Consider Recalibrating'}
                        - Scale Accuracy: {result.pixels_per_mm:.2f} pixels/mm
                        """
                        )

                else:
                    st.error("❌ Calibration failed")
            else:
                st.error("Need at least 5 calibration images")

        # Calibration result visualization section
        if calibrator.calibration_result:
            st.subheader("🎨 Interactive Calibration Visualization")

            # Visualization controls
            col_vis1, col_vis2 = st.columns(2)

            with col_vis1:
                vis_axes = st.checkbox(
                    "Show Coordinate Axes", value=True, key="result_axes"
                )
                vis_cube = st.checkbox("Show 3D Cube", value=True, key="result_cube")

            with col_vis2:
                if st.button("🔄 Update Visualization"):
                    if hasattr(st.session_state, "preview_image"):
                        result_vis = calibrator.draw_chessboard_pattern(
                            st.session_state.preview_image,
                            draw_axes=vis_axes,
                            draw_cube=vis_cube,
                        )

                        st.image(
                            cv2.cvtColor(result_vis, cv2.COLOR_BGR2RGB),
                            caption="Updated Calibration Visualization",
                            use_container_width=True,
                        )

        # Save/Load calibration
        st.subheader("Save/Load Calibration")

        # Save calibration
        if calibrator.calibration_result:
            calibration_name = st.text_input(
                "Calibration Name",
                f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

            if st.button("💾 Save Calibration"):
                filepath = f"calibrations/{calibration_name}.json"
                os.makedirs("calibrations", exist_ok=True)

                if calibrator.save_calibration(filepath):
                    st.success(f"Calibration saved to {filepath}")
                else:
                    st.error("Failed to save calibration")

        # Load calibration
        st.subheader("Load Existing Calibration")

        # List available calibrations
        if os.path.exists("calibrations"):
            calibration_files = [
                f for f in os.listdir("calibrations") if f.endswith(".json")
            ]

            if calibration_files:
                selected_calibration = st.selectbox(
                    "Select Calibration File", calibration_files
                )

                if st.button("📂 Load Calibration"):
                    filepath = f"calibrations/{selected_calibration}"
                    if calibrator.load_calibration(filepath):
                        st.success(f"Calibration loaded from {filepath}")
                        # Display loaded calibration info
                        if calibrator.calibration_result:
                            st.metric(
                                "Loaded - Reprojection Error",
                                f"{calibrator.calibration_result.reprojection_error:.4f}",
                            )
                            if calibrator.calibration_result.pixels_per_mm:
                                st.metric(
                                    "Loaded - Scale",
                                    f"{calibrator.calibration_result.pixels_per_mm:.2f} px/mm",
                                )

                            # Show visualization of loaded calibration
                            if hasattr(st.session_state, "preview_image"):
                                loaded_vis = calibrator.draw_chessboard_pattern(
                                    st.session_state.preview_image,
                                    draw_axes=True,
                                    draw_cube=True,
                                )
                                st.image(
                                    cv2.cvtColor(loaded_vis, cv2.COLOR_BGR2RGB),
                                    caption="Loaded Calibration Visualization",
                                    use_container_width=True,
                                )
                    else:
                        st.error("Failed to load calibration")
            else:
                st.info("No saved calibrations found")

        # Test undistortion with visualization
        if calibrator.calibration_result and hasattr(st.session_state, "preview_image"):
            st.subheader("🔄 Distortion Correction Test")

            if st.button("Test Undistortion"):
                original = st.session_state.preview_image
                undistorted = calibrator.undistort_image(original)

                # Create visualizations for both images
                original_vis = calibrator.draw_chessboard_pattern(
                    original, draw_axes=True, draw_cube=False
                )
                undistorted_vis = calibrator.draw_chessboard_pattern(
                    undistorted, draw_axes=True, draw_cube=False
                )

                # Show before/after comparison with pattern visualization
                tab_orig, tab_undist, tab_overlay = st.tabs(
                    ["Original", "Undistorted", "Overlay Comparison"]
                )

                with tab_orig:
                    st.image(
                        cv2.cvtColor(original_vis, cv2.COLOR_BGR2RGB),
                        caption="Original (With Distortion)",
                        use_container_width=True,
                    )

                with tab_undist:
                    st.image(
                        cv2.cvtColor(undistorted_vis, cv2.COLOR_BGR2RGB),
                        caption="Corrected (Undistorted)",
                        use_container_width=True,
                    )

                with tab_overlay:
                    # Create side-by-side comparison
                    comparison = np.hstack([original_vis, undistorted_vis])
                    st.image(
                        cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB),
                        caption="Side-by-Side: Original (Left) vs Undistorted (Right)",
                        use_container_width=True,
                    )

    # Instructions and tips
    with st.expander("📋 Calibration Instructions & Tips"):
        st.markdown(
            """
        ### How to Calibrate Your Camera:
        
        1. **Print Chessboard Pattern**: 
           - Use a high-quality printer on flat, rigid paper
           - Ensure squares are exactly the size you specified
           - Mount on a flat surface (e.g., clipboard)
        
        2. **Capture Images**:
           - Take 10-20 images of the chessboard
           - Vary the position and orientation of the chessboard
           - Cover different areas of the image
           - Keep the chessboard flat and in focus
        
        3. **Image Quality Tips**:
           - Ensure good, even lighting
           - Avoid reflections on the chessboard
           - All corners should be visible
           - Images should be sharp (not blurry)
        
        4. **For Top-Down Setup**:
           - Place chessboard on the inspection surface
           - Capture at different positions across the field of view
           - Include some tilted orientations for better calibration
        
        ### Visualization Features:
        - **Coordinate Axes**: Red=X, Green=Y, Blue=Z axes showing camera orientation
        - **3D Cube**: Virtual cube projected onto the chessboard for depth visualization
        - **Corner Numbering**: Shows detection order for debugging
        - **Pattern Detection**: Highlights all detected corners
        
        ### Quality Indicators:
        - **Reprojection Error < 0.5**: Excellent
        - **Reprojection Error 0.5-1.0**: Good
        - **Reprojection Error > 1.0**: Poor (recalibrate)
        
        ### Troubleshooting:
        - If pattern not detected: Check pattern size settings
        - Poor quality: Improve lighting and reduce reflections
        - High reprojection error: Add more varied calibration images
        """
        )

    # Integration with main camera object
    if calibrator.calibration_result and st.session_state.camera:
        # Apply calibration to the camera object
        st.session_state.camera.calibrator = calibrator
        st.info("✅ Calibration applied to camera object")

    # Detection tips specific to current settings
    with st.expander("🔍 Pattern Detection Tips"):
        tips = calibrator.get_detection_tips()
        for i, tip in enumerate(tips, 1):
            st.write(f"{i}. {tip}")


if __name__ == "__main__":
    camera_calibration_interface()
