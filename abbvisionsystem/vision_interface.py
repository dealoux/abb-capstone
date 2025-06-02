"""Streamlit interface for the vision system tools."""

import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import json

from abbvisionsystem.vision_tools.pattern_matching import (
    AdvancedPatMax,
    SearchStrategy,
    AcceptanceLevel,
)
from abbvisionsystem.vision_tools.blob_analysis import BlobAnalyzer
from abbvisionsystem.vision_tools.edge_detection import EdgeDetector
from abbvisionsystem.vision_tools.line_max import (
    SmartLineDetector,
    ExpectedLine,
    LineType,
    LineQuality,
)


def vision_interface():
    """Main interface for vision system tools."""
    st.title("🔍 Advanced Vision System")
    st.write(
        "Industrial vision tools for pattern matching, blob analysis, and defect detection"
    )

    # Sidebar for tool selection
    with st.sidebar:
        st.header("Vision Tools")
        tool_selection = st.radio(
            "Select Tool",
            [
                "PatMax/PatQuick Matching",
                "SmartLine/LineMax Detection",
                "Blob Analysis",
                "Edge Detection",
                "Measurement Tools",
                "Defect Analysis Suite",
            ],
        )

        # Global image upload
        st.header("Image Input")
        uploaded_file = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_file is not None:
            # Convert uploaded file to image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state.current_image = image

            # Display uploaded image
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input Image", width=200
            )

    # Main content area
    if "current_image" not in st.session_state:
        st.info("Please upload an image to begin analysis")
        return

    image = st.session_state.current_image

    # Tool-specific interfaces
    if tool_selection == "PatMax/PatQuick Matching":
        patmax_interface(image)
    elif tool_selection == "SmartLine/LineMax Detection":
        smartline_interface(image)
    elif tool_selection == "Blob Analysis":
        blob_analysis_interface(image)
    elif tool_selection == "Edge Detection":
        edge_detection_interface(image)
    elif tool_selection == "Measurement Tools":
        measurement_tools_interface(image)
    elif tool_selection == "Defect Analysis Suite":
        defect_analysis_interface(image)


def patmax_interface(image: np.ndarray):
    """Advanced PatMax/PatQuick pattern matching interface."""
    st.header("🎯 PatMax/PatQuick Pattern Matching")

    # Initialize PatMax matcher
    if "patmax_matcher" not in st.session_state:
        st.session_state.patmax_matcher = AdvancedPatMax()

    matcher = st.session_state.patmax_matcher

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pattern Training")

        # Pattern ID and search strategy
        pattern_id = st.text_input("Pattern ID", "pattern_1")
        search_strategy = st.selectbox(
            "Search Strategy",
            [
                SearchStrategy.PATMAX.value,
                SearchStrategy.PATQUICK.value,
                SearchStrategy.FEATURE_BASED.value,
            ],
        )

        # ROI definition
        st.write("Define Region of Interest:")
        roi_x = st.number_input(
            "X", min_value=0, max_value=image.shape[1] - 1, value=100
        )
        roi_y = st.number_input(
            "Y", min_value=0, max_value=image.shape[0] - 1, value=100
        )
        roi_w = st.number_input(
            "Width", min_value=1, max_value=image.shape[1], value=100
        )
        roi_h = st.number_input(
            "Height", min_value=1, max_value=image.shape[0], value=100
        )

        # Advanced parameters
        with st.expander("Advanced Training Parameters"):
            acceptance_threshold = st.slider(
                "Acceptance Threshold", 0.1, 1.0, 0.7, 0.05
            )

            # Expected locations for defect detection
            st.write("Expected Pattern Locations (for defect detection):")
            num_expected = st.number_input(
                "Number of Expected Locations", min_value=0, max_value=10, value=0
            )
            expected_locations = []

            for i in range(num_expected):
                col_a, col_b = st.columns(2)
                with col_a:
                    exp_x = st.number_input(f"Expected X {i+1}", value=200 + i * 50)
                with col_b:
                    exp_y = st.number_input(f"Expected Y {i+1}", value=200 + i * 50)
                expected_locations.append((exp_x, exp_y))

        if st.button("Train PatMax Pattern"):
            roi = (roi_x, roi_y, roi_w, roi_h)
            search_strat = SearchStrategy(search_strategy)

            success = matcher.train_patmax_template(
                image,
                roi,
                pattern_id,
                expected_locations=expected_locations,
                search_strategy=search_strat,
                acceptance_threshold=acceptance_threshold,
            )

            if success:
                st.success(f"PatMax pattern '{pattern_id}' trained successfully!")
                st.info(f"Strategy: {search_strategy}")
                st.info(f"Expected locations: {len(expected_locations)}")
            else:
                st.error("Failed to train pattern")

        # Show trained patterns
        if matcher.templates:
            st.write("Trained Patterns:")
            for pid, template in matcher.templates.items():
                st.write(f"- {pid} ({template.search_strategy.value})")

    with col2:
        st.subheader("Pattern Detection")

        # Detection parameters
        detect_defects = st.checkbox("Enable Defect Detection", value=True)

        if st.button("Find PatMax Patterns"):
            if matcher.templates:
                matches = matcher.find_patmax_patterns(
                    image, detect_defects=detect_defects
                )

                if matches:
                    st.success(f"Found {len(matches)} pattern matches")

                    # Visualize results
                    result_image = image.copy()
                    defect_count = 0

                    for match in matches:
                        # Choose color based on match quality and defect status
                        if match.score == 0:  # Missing pattern
                            color = (255, 0, 255)  # Magenta for missing
                            defect_count += 1
                        elif not match.is_expected:
                            color = (0, 0, 255)  # Red for unexpected
                            defect_count += 1
                        elif match.acceptance_level == AcceptanceLevel.HIGH:
                            color = (0, 255, 0)  # Green for high quality
                        elif match.acceptance_level == AcceptanceLevel.MEDIUM:
                            color = (0, 255, 255)  # Yellow for medium
                        else:
                            color = (128, 128, 128)  # Gray for low quality

                        center = (int(match.x), int(match.y))

                        if match.score > 0:  # Only draw if not missing
                            cv2.circle(result_image, center, 10, color, 3)
                            cv2.putText(
                                result_image,
                                f"{match.template_id}: {match.score:.2f}",
                                (center[0] + 15, center[1]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2,
                            )
                        else:
                            # Draw X for missing pattern
                            cv2.drawMarker(
                                result_image,
                                center,
                                color,
                                cv2.MARKER_TILTED_CROSS,
                                20,
                                3,
                            )
                            cv2.putText(
                                result_image,
                                f"MISSING: {match.template_id}",
                                (center[0] + 15, center[1]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2,
                            )

                    st.image(
                        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                        caption=f"PatMax Results - {defect_count} defects detected",
                    )

                    # Detailed results
                    for i, match in enumerate(matches):
                        with st.expander(f"Match {i+1}: {match.template_id}"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"Position: ({match.x:.1f}, {match.y:.1f})")
                                st.write(f"Score: {match.score:.3f}")
                                st.write(f"Angle: {match.angle:.1f}°")
                                st.write(f"Scale: {match.scale:.2f}")
                            with col_b:
                                st.write(f"Strategy: {match.search_strategy.value}")
                                st.write(f"Quality: {match.acceptance_level.value}")
                                st.write(f"Expected: {match.is_expected}")
                                st.write(
                                    f"Deviation: {match.deviation_from_expected:.1f}"
                                )
                                if match.score == 0:
                                    st.error("MISSING PATTERN")
                                elif not match.is_expected:
                                    st.warning("UNEXPECTED PATTERN")
                else:
                    st.warning("No patterns found")

                # Defect analysis summary
                if detect_defects:
                    for template_id in matcher.templates.keys():
                        defect_analysis = matcher.detect_pattern_defects(
                            image, template_id
                        )
                        if defect_analysis.get("total_defects", 0) > 0:
                            st.error(
                                f"Defects detected for {template_id}: {defect_analysis['total_defects']}"
                            )
                            with st.expander(f"Defect Details - {template_id}"):
                                st.json(defect_analysis)
            else:
                st.warning("No patterns trained yet")


def smartline_interface(image: np.ndarray):
    """SmartLine/LineMax detection interface."""
    st.header("📏 SmartLine/LineMax Detection")

    # Initialize SmartLine detector
    if "smartline_detector" not in st.session_state:
        st.session_state.smartline_detector = SmartLineDetector()

    detector = st.session_state.smartline_detector

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Expected Line Configuration")

        # Add expected lines for defect detection
        with st.expander("Add Expected Line"):
            line_id = st.text_input("Line ID", "line_1")

            # Expected line coordinates
            st.write("Expected Line Coordinates:")
            exp_start_x = st.number_input("Start X", value=100)
            exp_start_y = st.number_input("Start Y", value=100)
            exp_end_x = st.number_input("End X", value=300)
            exp_end_y = st.number_input("End Y", value=100)

            # Tolerances
            st.write("Tolerances:")
            tolerance_width = st.slider("Position Tolerance (pixels)", 1, 50, 10)
            tolerance_angle = st.slider("Angle Tolerance (degrees)", 1, 45, 5)
            tolerance_length = st.slider("Length Tolerance (%)", 1, 50, 10)
            min_edge_strength = st.slider("Min Edge Strength", 0.1, 1.0, 0.3, 0.05)

            line_type = st.selectbox(
                "Line Type",
                [LineType.STRAIGHT.value, LineType.EDGE.value, LineType.RIDGE.value],
            )

            if st.button("Add Expected Line"):
                expected_line = ExpectedLine(
                    id=line_id,
                    expected_start=(exp_start_x, exp_start_y),
                    expected_end=(exp_end_x, exp_end_y),
                    tolerance_width=tolerance_width,
                    tolerance_angle=tolerance_angle,
                    tolerance_length=tolerance_length,
                    min_edge_strength=min_edge_strength,
                    line_type=LineType(line_type),
                )
                detector.add_expected_line(expected_line)
                st.success(f"Added expected line: {line_id}")

        # Show expected lines
        if detector.expected_lines:
            st.write("Expected Lines:")
            for line_id, exp_line in detector.expected_lines.items():
                st.write(f"- {line_id} ({exp_line.line_type.value})")

        # Detection parameters
        st.subheader("Detection Parameters")
        with st.expander("Configure Detection"):
            canny_low = st.slider("Canny Low Threshold", 1, 200, 50)
            canny_high = st.slider("Canny High Threshold", 1, 300, 150)
            hough_threshold = st.slider("Hough Threshold", 1, 200, 100)
            min_line_length = st.slider("Min Line Length", 1, 200, 30)
            max_line_gap = st.slider("Max Line Gap", 1, 50, 10)

            detector.configure_detection(
                canny_low=canny_low,
                canny_high=canny_high,
                hough_threshold=hough_threshold,
                min_line_length=min_line_length,
                max_line_gap=max_line_gap,
            )

    with col2:
        st.subheader("Line Detection Results")

        # ROI selection
        use_roi = st.checkbox("Use Region of Interest")
        roi = None
        if use_roi:
            roi_x = st.number_input(
                "ROI X", min_value=0, max_value=image.shape[1] - 1, value=0
            )
            roi_y = st.number_input(
                "ROI Y", min_value=0, max_value=image.shape[0] - 1, value=0
            )
            roi_w = st.number_input(
                "ROI Width", min_value=1, max_value=image.shape[1], value=image.shape[1]
            )
            roi_h = st.number_input(
                "ROI Height",
                min_value=1,
                max_value=image.shape[0],
                value=image.shape[0],
            )
            roi = (roi_x, roi_y, roi_w, roi_h)

        show_defects_only = st.checkbox("Show Defects Only", value=False)

        if st.button("Detect SmartLines"):
            lines = detector.detect_smart_lines(image, roi=roi)

            if lines:
                defect_lines = [line for line in lines if line.is_defect]
                st.success(f"Detected {len(lines)} lines ({len(defect_lines)} defects)")

                # Visualize results
                result_image = detector.visualize_smart_lines(
                    image, lines, show_defects_only=show_defects_only
                )
                st.image(
                    cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                    caption=f"SmartLine Results",
                )

                # Line statistics
                st.subheader("Line Analysis")

                # Quality distribution
                quality_counts = {}
                for line in lines:
                    quality = line.quality.value
                    quality_counts[quality] = quality_counts.get(quality, 0) + 1

                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("Quality Distribution:")
                    for quality, count in quality_counts.items():
                        st.write(f"- {quality}: {count}")

                with col_b:
                    st.write("Line Types:")
                    type_counts = {}
                    for line in lines:
                        line_type = line.line_type.value
                        type_counts[line_type] = type_counts.get(line_type, 0) + 1
                    for line_type, count in type_counts.items():
                        st.write(f"- {line_type}: {count}")

                # Detailed line information
                for i, line in enumerate(lines):
                    if show_defects_only and not line.is_defect:
                        continue

                    with st.expander(
                        f"Line {i+1}: {line.line_type.value} {'(DEFECT)' if line.is_defect else ''}"
                    ):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(
                                f"Start: ({line.start_point[0]:.1f}, {line.start_point[1]:.1f})"
                            )
                            st.write(
                                f"End: ({line.end_point[0]:.1f}, {line.end_point[1]:.1f})"
                            )
                            st.write(f"Length: {line.length:.1f}")
                            st.write(f"Angle: {line.angle:.1f}°")
                        with col_b:
                            st.write(f"Straightness: {line.straightness:.3f}")
                            st.write(f"Edge Strength: {line.edge_strength:.3f}")
                            st.write(f"Quality: {line.quality.value}")
                            if line.expected_line_id:
                                st.write(f"Expected: {line.expected_line_id}")
                                st.write(
                                    f"Deviation: {line.deviation_from_expected:.1f}"
                                )
                            if line.is_defect:
                                st.error("DEFECT DETECTED")
            else:
                st.warning("No lines detected")


def blob_analysis_interface(image: np.ndarray):
    """Enhanced blob analysis interface."""
    st.header("🔴 Blob Analysis")

    # Initialize blob analyzer
    if "blob_analyzer" not in st.session_state:
        st.session_state.blob_analyzer = BlobAnalyzer()

    analyzer = st.session_state.blob_analyzer

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detection Parameters")

        # Detection method
        detection_method = st.selectbox(
            "Detection Method",
            ["Simple Blob Detector", "Contour Analysis", "Hough Circles"],
        )

        # Common parameters
        min_area = st.number_input("Min Area", min_value=1, value=50)
        max_area = st.number_input("Max Area", min_value=1, value=5000)

        if detection_method == "Simple Blob Detector":
            min_circularity = st.slider("Min Circularity", 0.0, 1.0, 0.1, 0.05)
            min_convexity = st.slider("Min Convexity", 0.0, 1.0, 0.5, 0.05)
            min_inertia = st.slider("Min Inertia", 0.0, 1.0, 0.01, 0.01)

            analyzer.configure_detector(
                min_area, max_area, min_circularity, min_convexity, min_inertia
            )

        elif detection_method == "Contour Analysis":
            binary_threshold = st.slider("Binary Threshold", 0, 255, 127)

        elif detection_method == "Hough Circles":
            min_radius = st.number_input("Min Radius", min_value=1, value=10)
            max_radius = st.number_input("Max Radius", min_value=1, value=100)
            min_distance = st.number_input("Min Distance", min_value=1, value=50)

        # Filtering options
        with st.expander("Advanced Filtering"):
            filter_enabled = st.checkbox("Enable Filtering")
            if filter_enabled:
                filter_min_area = st.number_input(
                    "Filter Min Area", min_value=1, value=min_area
                )
                filter_max_area = st.number_input(
                    "Filter Max Area", min_value=1, value=max_area
                )
                filter_min_circularity = st.slider(
                    "Filter Min Circularity", 0.0, 1.0, 0.0, 0.05
                )
                filter_max_aspect_ratio = st.slider(
                    "Filter Max Aspect Ratio", 1.0, 10.0, 5.0, 0.1
                )

    with col2:
        st.subheader("Results")

        if st.button("Detect Blobs"):
            # Perform detection based on selected method
            if detection_method == "Simple Blob Detector":
                blobs = analyzer.detect_blobs_simple(image)
            elif detection_method == "Contour Analysis":
                blobs = analyzer.detect_blobs_advanced(
                    image, binary_threshold, min_area, max_area
                )
            elif detection_method == "Hough Circles":
                blobs = analyzer.detect_circles_hough(
                    image, min_radius, max_radius, min_distance
                )

            # Apply additional filtering if enabled
            if filter_enabled:
                blobs = analyzer.filter_blobs(
                    blobs,
                    min_area=filter_min_area,
                    max_area=filter_max_area,
                    min_circularity=filter_min_circularity,
                    max_aspect_ratio=filter_max_aspect_ratio,
                )

            if blobs:
                st.success(f"Detected {len(blobs)} blobs")

                # Visualize results
                result_image = analyzer.visualize_blobs(image, blobs)
                st.image(
                    cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                    caption=f"Detected Blobs ({len(blobs)})",
                )

                # Blob statistics
                st.subheader("Blob Statistics")

                # Summary statistics
                areas = [blob.area for blob in blobs]
                circularities = [blob.circularity for blob in blobs]
                aspect_ratios = [blob.aspect_ratio for blob in blobs]

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Blobs", len(blobs))
                    st.metric("Avg Area", f"{np.mean(areas):.1f}")
                with col_b:
                    st.metric("Min Area", f"{np.min(areas):.1f}")
                    st.metric("Max Area", f"{np.max(areas):.1f}")
                with col_c:
                    st.metric("Avg Circularity", f"{np.mean(circularities):.3f}")
                    st.metric("Avg Aspect Ratio", f"{np.mean(aspect_ratios):.2f}")

                # Individual blob details
                for i, blob in enumerate(blobs):
                    with st.expander(f"Blob {i+1}"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(
                                f"Center: ({blob.center_x:.1f}, {blob.center_y:.1f})"
                            )
                            st.write(f"Area: {blob.area:.1f}")
                            st.write(f"Perimeter: {blob.perimeter:.1f}")
                        with col_b:
                            st.write(f"Circularity: {blob.circularity:.3f}")
                            st.write(f"Aspect Ratio: {blob.aspect_ratio:.2f}")
                            st.write(f"Angle: {blob.angle:.1f}°")
                            st.write(f"Size: {blob.width:.1f} x {blob.height:.1f}")

                # Inter-blob measurements
                if len(blobs) > 1:
                    distances = analyzer.measure_blob_distances(blobs)
                    st.subheader("Inter-blob Distances")
                    for dist_info in distances[:10]:  # Show first 10 distances
                        st.write(
                            f"Blob {dist_info['blob1_index']+1} ↔ Blob {dist_info['blob2_index']+1}: {dist_info['distance']:.1f} pixels"
                        )
            else:
                st.warning("No blobs detected")


def edge_detection_interface(image: np.ndarray):
    """Enhanced edge detection interface."""
    st.header("📏 Edge Detection")

    # Initialize edge detector
    if "edge_detector" not in st.session_state:
        st.session_state.edge_detector = EdgeDetector()

    detector = st.session_state.edge_detector

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detection Parameters")

        # Edge detection method
        detection_method = st.selectbox(
            "Detection Method", ["Canny Edges", "Hough Lines", "Contour Edges"]
        )

        if detection_method == "Canny Edges":
            low_threshold = st.slider("Low Threshold", 0, 255, 50)
            high_threshold = st.slider("High Threshold", 0, 255, 150)
            blur_kernel = st.slider("Blur Kernel", 0, 15, 5, 2)
            detector.configure_canny(low_threshold, high_threshold)

        elif detection_method == "Hough Lines":
            hough_threshold = st.slider("Hough Threshold", 1, 200, 100)
            min_line_length = st.slider("Min Line Length", 1, 200, 50)
            max_line_gap = st.slider("Max Line Gap", 1, 50, 10)
            detector.configure_hough(hough_threshold, min_line_length, max_line_gap)

        elif detection_method == "Contour Edges":
            binary_threshold = st.slider("Binary Threshold", 0, 255, 127)

        # Analysis options
        with st.expander("Analysis Options"):
            find_parallel = st.checkbox("Find Parallel Lines")
            find_perpendicular = st.checkbox("Find Perpendicular Lines")
            if find_parallel or find_perpendicular:
                angle_tolerance = st.slider("Angle Tolerance", 1.0, 15.0, 5.0, 0.5)

    with col2:
        st.subheader("Results")

        if st.button("Detect Edges/Lines"):
            if detection_method == "Canny Edges":
                edges_image = detector.detect_edges_canny(image, blur_kernel)
                st.image(edges_image, caption="Detected Edges", cmap="gray")

                # Edge statistics
                edge_pixels = np.sum(edges_image > 0)
                total_pixels = edges_image.shape[0] * edges_image.shape[1]
                edge_density = edge_pixels / total_pixels
                st.metric("Edge Density", f"{edge_density:.4f}")
                st.metric("Edge Pixels", f"{edge_pixels:,}")

            elif detection_method == "Hough Lines":
                lines = detector.detect_lines_hough(image)
                if lines:
                    st.success(f"Detected {len(lines)} lines")
                    result_image = detector.visualize_lines(image, lines)
                    st.image(
                        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                        caption=f"Detected Lines ({len(lines)})",
                    )

                    # Line analysis
                    if find_parallel:
                        parallel_groups = detector.find_parallel_lines(
                            lines, angle_tolerance
                        )
                        if parallel_groups:
                            st.subheader("Parallel Line Groups")
                            for i, group in enumerate(parallel_groups):
                                st.write(f"Group {i+1}: Lines {group}")

                    if find_perpendicular:
                        perp_pairs = detector.find_perpendicular_lines(
                            lines, angle_tolerance
                        )
                        if perp_pairs:
                            st.subheader("Perpendicular Line Pairs")
                            for pair in perp_pairs:
                                st.write(f"Lines {pair[0]+1} ⊥ {pair[1]+1}")

                    # Line statistics
                    lengths = [line.length for line in lines]
                    angles = [line.angle_degrees for line in lines]

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Total Lines", len(lines))
                        st.metric("Avg Length", f"{np.mean(lengths):.1f}")
                        st.metric("Min Length", f"{np.min(lengths):.1f}")
                    with col_b:
                        st.metric("Max Length", f"{np.max(lengths):.1f}")
                        st.metric(
                            "Angle Range",
                            f"{np.min(angles):.1f}° to {np.max(angles):.1f}°",
                        )

                    # Individual line details
                    for i, line in enumerate(lines[:10]):  # Show first 10 lines
                        with st.expander(f"Line {i+1}"):
                            st.write(
                                f"Start: ({line.start_point[0]:.1f}, {line.start_point[1]:.1f})"
                            )
                            st.write(
                                f"End: ({line.end_point[0]:.1f}, {line.end_point[1]:.1f})"
                            )
                            st.write(f"Length: {line.length:.1f}")
                            st.write(f"Angle: {line.angle_degrees:.1f}°")
                else:
                    st.warning("No lines detected")

            elif detection_method == "Contour Edges":
                edges = detector.detect_edges_contour(image, binary_threshold)
                if edges:
                    st.success(f"Detected {len(edges)} edges")
                    result_image = detector.visualize_edges(image, edges)
                    st.image(
                        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                        caption=f"Detected Edges ({len(edges)})",
                    )

                    # Edge statistics
                    edge_lengths = [edge.length for edge in edges]
                    edge_angles = [edge.angle for edge in edges]

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Total Edges", len(edges))
                        st.metric("Avg Length", f"{np.mean(edge_lengths):.1f}")
                    with col_b:
                        st.metric("Total Length", f"{np.sum(edge_lengths):.1f}")
                        st.metric("Angle Spread", f"{np.std(edge_angles):.1f}°")

                    # Individual edge details
                    for i, edge in enumerate(edges[:10]):  # Show first 10 edges
                        with st.expander(f"Edge {i+1}"):
                            st.write(
                                f"Start: ({edge.start_point[0]:.1f}, {edge.start_point[1]:.1f})"
                            )
                            st.write(
                                f"End: ({edge.end_point[0]:.1f}, {edge.end_point[1]:.1f})"
                            )
                            st.write(f"Length: {edge.length:.1f}")
                            st.write(f"Angle: {edge.angle:.1f}°")
                            st.write(f"Strength: {edge.strength:.3f}")
                else:
                    st.warning("No edges detected")


def measurement_tools_interface(image: np.ndarray):
    """Enhanced measurement tools interface."""
    st.header("📐 Measurement Tools")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Calibration")
        pixels_per_mm = st.number_input(
            "Pixels per mm", min_value=0.1, value=1.0, step=0.1
        )
        pixels_per_inch = pixels_per_mm * 25.4

        st.write(
            f"Scale: {pixels_per_mm:.2f} pixels/mm ({pixels_per_inch:.1f} pixels/inch)"
        )

        # Coordinate system
        st.subheader("Coordinate System")
        origin_x = st.number_input("Origin X", value=0)
        origin_y = st.number_input("Origin Y", value=0)

        # Unit selection
        unit = st.selectbox("Measurement Unit", ["pixels", "mm", "inches"])

    with col2:
        st.subheader("Point-to-Point Measurement")

        # Point 1
        st.write("Point 1:")
        x1 = st.number_input("X1", value=100) - origin_x
        y1 = st.number_input("Y1", value=100) - origin_y

        # Point 2
        st.write("Point 2:")
        x2 = st.number_input("X2", value=200) - origin_x
        y2 = st.number_input("Y2", value=200) - origin_y

        # Calculate measurements
        pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if unit == "mm":
            real_distance = pixel_distance / pixels_per_mm
            unit_label = "mm"
        elif unit == "inches":
            real_distance = pixel_distance / pixels_per_inch
            unit_label = "in"
        else:
            real_distance = pixel_distance
            unit_label = "pixels"

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Display results
        st.metric("Distance", f"{real_distance:.2f} {unit_label}")
        st.metric("Angle", f"{angle:.1f}°")

        # Additional measurements
        st.subheader("Additional Measurements")
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if unit == "mm":
            dx_real = dx / pixels_per_mm
            dy_real = dy / pixels_per_mm
        elif unit == "inches":
            dx_real = dx / pixels_per_inch
            dy_real = dy / pixels_per_inch
        else:
            dx_real = dx
            dy_real = dy

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("ΔX", f"{dx_real:.2f} {unit_label}")
        with col_b:
            st.metric("ΔY", f"{dy_real:.2f} {unit_label}")

    # Visualize measurement
    result_image = image.copy()

    # Adjust coordinates back to image space
    img_x1, img_y1 = x1 + origin_x, y1 + origin_y
    img_x2, img_y2 = x2 + origin_x, y2 + origin_y

    cv2.line(
        result_image,
        (int(img_x1), int(img_y1)),
        (int(img_x2), int(img_y2)),
        (0, 255, 0),
        2,
    )
    cv2.circle(result_image, (int(img_x1), int(img_y1)), 5, (255, 0, 0), -1)
    cv2.circle(result_image, (int(img_x2), int(img_y2)), 5, (0, 0, 255), -1)

    # Add measurement text
    mid_x = int((img_x1 + img_x2) / 2)
    mid_y = int((img_y1 + img_y2) / 2)
    cv2.putText(
        result_image,
        f"{real_distance:.1f} {unit_label}",
        (mid_x, mid_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Draw origin
    cv2.drawMarker(
        result_image,
        (int(origin_x), int(origin_y)),
        (255, 0, 255),
        cv2.MARKER_CROSS,
        20,
        2,
    )

    st.image(
        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
        caption="Measurement Visualization",
    )


def defect_analysis_interface(image: np.ndarray):
    """Comprehensive defect analysis interface."""
    st.header("🔍 Defect Analysis Suite")

    st.info(
        "This interface combines all vision tools for comprehensive defect detection"
    )

    # Analysis modes
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Pattern Defects", "Line Defects", "Blob Anomalies", "Full Inspection"],
    )

    if analysis_mode == "Pattern Defects":
        st.subheader("Pattern-based Defect Detection")
        st.write("Use this mode to detect:")
        st.write("- Missing patterns")
        st.write("- Displaced patterns")
        st.write("- Distorted patterns")
        st.write("- Unexpected patterns")

        if st.button("Run Pattern Defect Analysis"):
            if (
                "patmax_matcher" in st.session_state
                and st.session_state.patmax_matcher.templates
            ):
                matcher = st.session_state.patmax_matcher

                # Run defect analysis for all trained patterns
                total_defects = 0
                defect_summary = {}

                for template_id in matcher.templates.keys():
                    defects = matcher.detect_pattern_defects(image, template_id)
                    defect_summary[template_id] = defects
                    total_defects += defects.get("total_defects", 0)

                if total_defects > 0:
                    st.error(f"DEFECTS DETECTED: {total_defects} total defects found")
                else:
                    st.success("No pattern defects detected")

                # Display detailed results
                for template_id, defects in defect_summary.items():
                    with st.expander(
                        f"Pattern {template_id} - {defects.get('total_defects', 0)} defects"
                    ):
                        if defects.get("missing_patterns"):
                            st.error(
                                f"Missing patterns: {len(defects['missing_patterns'])}"
                            )
                        if defects.get("unexpected_patterns"):
                            st.warning(
                                f"Unexpected patterns: {len(defects['unexpected_patterns'])}"
                            )
                        if defects.get("distorted_patterns"):
                            st.warning(
                                f"Distorted patterns: {len(defects['distorted_patterns'])}"
                            )
                        if defects.get("displaced_patterns"):
                            st.warning(
                                f"Displaced patterns: {len(defects['displaced_patterns'])}"
                            )

                        st.json(defects)
            else:
                st.warning(
                    "No trained patterns available. Please train patterns first in PatMax interface."
                )

    elif analysis_mode == "Line Defects":
        st.subheader("Line-based Defect Detection")
        st.write("Use this mode to detect:")
        st.write("- Missing expected lines")
        st.write("- Broken or discontinuous lines")
        st.write("- Lines with poor quality")
        st.write("- Unexpected lines")

        if st.button("Run Line Defect Analysis"):
            if "smartline_detector" in st.session_state:
                detector = st.session_state.smartline_detector
                lines = detector.detect_smart_lines(image)

                defective_lines = [line for line in lines if line.is_defect]
                missing_lines = [line for line in lines if line.length == 0]
                poor_quality = [
                    line
                    for line in lines
                    if line.quality in [LineQuality.POOR, LineQuality.FAIR]
                ]

                st.write(f"Total lines detected: {len(lines)}")
                st.write(f"Defective lines: {len(defective_lines)}")
                st.write(f"Missing expected lines: {len(missing_lines)}")
                st.write(f"Poor quality lines: {len(poor_quality)}")

                if defective_lines:
                    st.error(
                        f"LINE DEFECTS DETECTED: {len(defective_lines)} defective lines"
                    )

                    # Visualize only defects
                    result_image = detector.visualize_smart_lines(
                        image, lines, show_defects_only=True
                    )
                    st.image(
                        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                        caption="Line Defects",
                    )
                else:
                    st.success("No line defects detected")
            else:
                st.warning(
                    "SmartLine detector not initialized. Please configure expected lines first."
                )

    elif analysis_mode == "Blob Anomalies":
        st.subheader("Blob Anomaly Detection")
        st.write("Use this mode to detect:")
        st.write("- Unexpected blobs/particles")
        st.write("- Missing expected features")
        st.write("- Size/shape anomalies")
        st.write("- Position deviations")

        # Expected blob configuration
        with st.expander("Configure Expected Blobs"):
            expected_count = st.number_input(
                "Expected Blob Count", min_value=0, value=0
            )
            if expected_count > 0:
                count_tolerance = st.slider("Count Tolerance", 0, 10, 1)
                size_tolerance = st.slider("Size Tolerance (%)", 1, 100, 20)

        if st.button("Run Blob Anomaly Analysis"):
            if "blob_analyzer" in st.session_state:
                analyzer = st.session_state.blob_analyzer
                blobs = analyzer.detect_blobs_advanced(image)

                anomalies = []

                # Check blob count
                if expected_count > 0:
                    count_diff = abs(len(blobs) - expected_count)
                    if count_diff > count_tolerance:
                        anomalies.append(
                            f"Blob count anomaly: Expected {expected_count}, found {len(blobs)}"
                        )

                # Check for size anomalies
                if blobs:
                    areas = [blob.area for blob in blobs]
                    mean_area = np.mean(areas)
                    std_area = np.std(areas)

                    for i, blob in enumerate(blobs):
                        deviation = abs(blob.area - mean_area) / mean_area * 100
                        if deviation > size_tolerance:
                            anomalies.append(
                                f"Blob {i+1} size anomaly: {deviation:.1f}% deviation"
                            )

                if anomalies:
                    st.error(f"BLOB ANOMALIES DETECTED: {len(anomalies)} anomalies")
                    for anomaly in anomalies:
                        st.warning(anomaly)
                else:
                    st.success("No blob anomalies detected")

                # Show blob visualization
                if blobs:
                    result_image = analyzer.visualize_blobs(image, blobs)
                    st.image(
                        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                        caption=f"Blob Analysis - {len(blobs)} blobs",
                    )
            else:
                st.warning("Blob analyzer not initialized.")

    elif analysis_mode == "Full Inspection":
        st.subheader("Comprehensive Inspection")
        st.write("Runs all defect detection algorithms")

        if st.button("Run Full Inspection"):
            inspection_results = {
                "pattern_defects": 0,
                "line_defects": 0,
                "blob_anomalies": 0,
                "edge_issues": 0,
            }

            # Pattern analysis
            if (
                "patmax_matcher" in st.session_state
                and st.session_state.patmax_matcher.templates
            ):
                matcher = st.session_state.patmax_matcher
                for template_id in matcher.templates.keys():
                    defects = matcher.detect_pattern_defects(image, template_id)
                    inspection_results["pattern_defects"] += defects.get(
                        "total_defects", 0
                    )

            # Line analysis
            if "smartline_detector" in st.session_state:
                detector = st.session_state.smartline_detector
                lines = detector.detect_smart_lines(image)
                inspection_results["line_defects"] = len(
                    [line for line in lines if line.is_defect]
                )

            # Blob analysis
            if "blob_analyzer" in st.session_state:
                analyzer = st.session_state.blob_analyzer
                blobs = analyzer.detect_blobs_advanced(image)
                # Simple anomaly check - could be more sophisticated
                if blobs:
                    areas = [blob.area for blob in blobs]
                    mean_area = np.mean(areas)
                    anomalous_blobs = [
                        blob
                        for blob in blobs
                        if abs(blob.area - mean_area) / mean_area > 0.5
                    ]
                    inspection_results["blob_anomalies"] = len(anomalous_blobs)

            # Summary
            total_issues = sum(inspection_results.values())

            if total_issues > 0:
                st.error(f"INSPECTION FAILED: {total_issues} total issues detected")
            else:
                st.success("INSPECTION PASSED: No defects detected")

            # Detailed breakdown
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pattern Defects", inspection_results["pattern_defects"])
            with col2:
                st.metric("Line Defects", inspection_results["line_defects"])
            with col3:
                st.metric("Blob Anomalies", inspection_results["blob_anomalies"])
            with col4:
                st.metric("Edge Issues", inspection_results["edge_issues"])


if __name__ == "__main__":
    vision_interface()
