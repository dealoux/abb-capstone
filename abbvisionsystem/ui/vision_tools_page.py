import streamlit as st
import cv2
import numpy as np
from PIL import Image

from abbvisionsystem.vision_tools.vision_interface import vision_interface


def vision_tools_page():
    """Vision tools and utilities page."""
    st.title("🔍 Vision Tools & Utilities")
    st.write("Advanced computer vision tools for analysis and debugging")

    # Create tabs for different vision tools
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔍 Vision Interface",
            "📐 Measurement Tools",
            "🖼️ Image Processing",
            "🔧 Debug Tools",
        ]
    )

    with tab1:
        st.subheader("🔍 Advanced Vision Interface")
        vision_interface()

    with tab2:
        _measurement_tools_interface()

    with tab3:
        _image_processing_interface()

    with tab4:
        _debug_tools_interface()


def _measurement_tools_interface():
    """Measurement and calibration tools."""
    st.subheader("📐 Measurement Tools")

    # Check if camera is calibrated
    if st.session_state.camera and hasattr(st.session_state.camera, "calibrator"):
        if st.session_state.camera.calibrator.calibration_result:
            scale = st.session_state.camera.calibrator.calibration_result.pixels_per_mm
            st.success(f"✅ Camera calibrated: {scale:.2f} pixels/mm")

            # Measurement interface
            if st.session_state.image is not None:
                _measurement_interface(scale)
            else:
                st.info("📷 Capture or upload an image to enable measurements")
        else:
            st.warning("⚠️ Camera not calibrated")
            st.info("Go to Camera Integration → Camera Calibration to calibrate")
    else:
        st.warning("⚠️ No camera connected")
        st.info("Connect a camera to enable measurement tools")


def _measurement_interface(pixels_per_mm):
    """Interactive measurement interface."""
    st.subheader("📏 Interactive Measurements")

    # Measurement mode
    measurement_mode = st.selectbox(
        "Measurement Mode",
        ["Point to Point", "Area Measurement", "Angle Measurement", "Multi-point"],
    )

    if measurement_mode == "Point to Point":
        st.info("💡 Click two points on the image to measure distance")

        # Simple point-to-point measurement
        if st.button("📏 Enable Point Measurement"):
            st.info("Feature coming soon: Interactive point selection")

    elif measurement_mode == "Area Measurement":
        st.info("💡 Select a region to measure area")

        # Area measurement settings
        area_shape = st.selectbox("Shape", ["Rectangle", "Circle", "Polygon"])

        if st.button("📐 Enable Area Measurement"):
            st.info("Feature coming soon: Interactive area selection")

    elif measurement_mode == "Angle Measurement":
        st.info("💡 Select three points to measure angle")

        if st.button("📐 Enable Angle Measurement"):
            st.info("Feature coming soon: Interactive angle measurement")

    # Measurement history
    st.subheader("📊 Measurement History")

    # Placeholder for measurement history
    if "measurements" not in st.session_state:
        st.session_state.measurements = []

    if st.session_state.measurements:
        for i, measurement in enumerate(st.session_state.measurements):
            st.write(f"{i+1}. {measurement}")
    else:
        st.info("No measurements recorded yet")


def _image_processing_interface():
    """Image processing and enhancement tools."""
    st.subheader("🖼️ Image Processing Tools")

    if st.session_state.image is not None:
        # Create columns for controls and preview
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("🎛️ Controls")

            # Basic adjustments
            st.write("**Basic Adjustments**")
            brightness = st.slider("Brightness", -100, 100, 0)
            contrast = st.slider("Contrast", -100, 100, 0)
            gamma = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)

            # Color adjustments
            st.write("**Color Adjustments**")
            saturation = st.slider("Saturation", -100, 100, 0)
            hue_shift = st.slider("Hue Shift", -180, 180, 0)

            # Filters
            st.write("**Filters**")
            blur_kernel = st.slider("Blur", 0, 15, 0, 2)
            sharpen = st.checkbox("Sharpen")
            edge_detection = st.checkbox("Edge Detection")

            # Noise reduction
            denoise = st.checkbox("Denoise")

            # Apply processing
            processed_image = _apply_image_processing(
                st.session_state.image,
                brightness,
                contrast,
                gamma,
                saturation,
                hue_shift,
                blur_kernel,
                sharpen,
                edge_detection,
                denoise,
            )

        with col2:
            st.subheader("📸 Processed Image")

            # Show before/after
            comparison_mode = st.radio("View Mode", ["Processed", "Side by Side"])

            if comparison_mode == "Side by Side":
                col_orig, col_proc = st.columns(2)
                with col_orig:
                    st.write("Original")
                    st.image(cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB))
                with col_proc:
                    st.write("Processed")
                    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            else:
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

            # Save processed image
            if st.button("💾 Save Processed Image"):
                _save_processed_image(processed_image)

    else:
        st.info("📷 Upload or capture an image to enable processing tools")


def _apply_image_processing(
    image,
    brightness,
    contrast,
    gamma,
    saturation,
    hue_shift,
    blur_kernel,
    sharpen,
    edge_detection,
    denoise,
):
    """Apply various image processing operations."""
    result = image.copy()

    try:
        # Brightness and contrast
        if brightness != 0 or contrast != 0:
            result = cv2.convertScaleAbs(
                result, alpha=(100 + contrast) / 100, beta=brightness
            )

        # Gamma correction
        if gamma != 1.0:
            gamma_table = np.array(
                [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]
            ).astype("uint8")
            result = cv2.LUT(result, gamma_table)

        # Color adjustments (convert to HSV)
        if saturation != 0 or hue_shift != 0:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

            if hue_shift != 0:
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

            if saturation != 0:
                hsv[:, :, 1] = cv2.add(hsv[:, :, 1], saturation)

            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Blur
        if blur_kernel > 0:
            kernel_size = blur_kernel * 2 + 1
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)

        # Sharpen
        if sharpen:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            result = cv2.filter2D(result, -1, kernel)

        # Edge detection
        if edge_detection:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Denoise
        if denoise:
            result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)

        return result

    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
        return image


def _save_processed_image(processed_image):
    """Save the processed image."""
    try:
        import os
        import time

        os.makedirs("processed_images", exist_ok=True)
        filename = f"processed_image_{int(time.time())}.jpg"
        filepath = os.path.join("processed_images", filename)

        cv2.imwrite(filepath, processed_image)
        st.success(f"✅ Image saved: {filepath}")

    except Exception as e:
        st.error(f"❌ Save failed: {str(e)}")


def _debug_tools_interface():
    """Debug and analysis tools."""
    st.subheader("🔧 Debug & Analysis Tools")

    if st.session_state.image is not None:
        # Image information
        st.subheader("📊 Image Information")

        height, width = st.session_state.image.shape[:2]
        channels = (
            st.session_state.image.shape[2]
            if len(st.session_state.image.shape) > 2
            else 1
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Width", f"{width} px")
        with col2:
            st.metric("Height", f"{height} px")
        with col3:
            st.metric("Channels", channels)

        # Histogram analysis
        st.subheader("📈 Histogram Analysis")

        if st.checkbox("Show Histograms"):
            _show_histograms(st.session_state.image)

        # Object detection debug
        if st.session_state.detections is not None:
            st.subheader("🎯 Detection Debug Info")

            # Detection statistics
            num_detections = st.session_state.detections.get("num_detections", 0)
            st.write(f"**Total Detections:** {num_detections}")

            if num_detections > 0:
                # Show detection details
                with st.expander("🔍 Detection Details"):
                    for i in range(num_detections):
                        score = st.session_state.detections["scores"][i]
                        class_id = st.session_state.detections["classes"][i]

                        st.write(f"**Object {i+1}:**")
                        st.write(f"  - Class ID: {class_id}")
                        st.write(f"  - Confidence: {score:.3f}")

                        if "object_details" in st.session_state.detections:
                            details = st.session_state.detections["object_details"][i]
                            st.write(f"  - Area: {details.get('area', 'N/A')} pixels")
                            st.write(f"  - Bbox: {details.get('bbox_pixels', 'N/A')}")

        # Performance metrics
        st.subheader("⚡ Performance Metrics")

        if st.button("🔍 Analyze Processing Time"):
            _analyze_processing_performance()

    else:
        st.info("📷 Upload or capture an image to enable debug tools")


def _show_histograms(image):
    """Show color histograms for the image."""
    try:
        import matplotlib.pyplot as plt

        # Calculate histograms
        colors = ["blue", "green", "red"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            axes[i].plot(hist, color=color)
            axes[i].set_title(f"{color.capitalize()} Channel")
            axes[i].set_xlabel("Pixel Value")
            axes[i].set_ylabel("Frequency")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Histogram error: {str(e)}")


def _analyze_processing_performance():
    """Analyze processing performance."""
    try:
        import time

        if st.session_state.image is not None:
            # Simulate processing performance analysis
            st.info("🔄 Analyzing processing performance...")

            # Measure basic operations
            operations = {
                "Color Conversion": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                "Gaussian Blur": lambda img: cv2.GaussianBlur(img, (15, 15), 0),
                "Edge Detection": lambda img: cv2.Canny(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150
                ),
                "Resize": lambda img: cv2.resize(img, (224, 224)),
            }

            results = {}

            for op_name, operation in operations.items():
                start_time = time.time()
                _ = operation(st.session_state.image)
                end_time = time.time()
                results[op_name] = (end_time - start_time) * 1000  # ms

            # Display results
            st.subheader("⚡ Performance Results")

            for op_name, duration in results.items():
                st.metric(op_name, f"{duration:.2f} ms")

    except Exception as e:
        st.error(f"❌ Performance analysis failed: {str(e)}")
