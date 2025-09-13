"""Calibration management component for the UI."""

import streamlit as st
import json
import os
import glob
import numpy as np
from typing import Optional
from datetime import datetime
from abbvisionsystem.camera.calibration import CameraCalibrator
import logging

logger = logging.getLogger(__name__)


class CalibrationManager:
    """Component for handling calibration management in the UI."""

    @staticmethod
    def render_calibration_section(calibrator: CameraCalibrator):
        """Render the complete calibration management section."""
        st.subheader("📏 Calibration Management")

        # Current calibration status
        CalibrationManager._show_calibration_status(calibrator)

        # Calibration file upload
        CalibrationManager._render_calibration_upload(calibrator)

        # Available calibration files browser
        CalibrationManager._render_calibration_browser(calibrator)

        # Clear calibration option
        CalibrationManager._render_clear_option(calibrator)

    @staticmethod
    def _show_calibration_status(calibrator: CameraCalibrator):
        """Show current calibration status."""
        if calibrator.calibration_result:
            st.success("✅ System Calibrated")

            # Show loaded calibration file info
            if hasattr(st.session_state, "loaded_calibration_file"):
                filename = os.path.basename(st.session_state.loaded_calibration_file)
                st.info(f"📁 Loaded: {filename}")

            st.info(f"Scale: {calibrator.calibration_result.pixels_per_mm:.2f} px/mm")
            st.info(f"Error: {calibrator.calibration_result.reprojection_error:.3f}")

            # Show calibration date if available
            if calibrator.calibration_result.calibration_date:
                st.info(f"Date: {calibrator.calibration_result.calibration_date}")
        else:
            st.warning("⚠️ System Not Calibrated")
            st.info("Upload a calibration file or go to Camera Calibration page")

    @staticmethod
    def _render_calibration_upload(calibrator: CameraCalibrator):
        """Render calibration file upload section."""
        st.subheader("📤 Upload Calibration")

        uploaded_calibration = st.file_uploader(
            "Choose a calibration file",
            type=["json"],
            key="calibration_upload",
            help="Upload a calibration JSON file (calibration_YYYYMMDD_HHMMSS.json)",
        )

        if uploaded_calibration is not None:
            try:
                # Read and parse the uploaded file
                calibration_content = uploaded_calibration.read()
                calibration_text = calibration_content.decode("utf-8")

                # Check if the JSON is complete
                if not calibration_text.strip().endswith("}"):
                    st.error("❌ Incomplete JSON file - file appears to be truncated")
                    st.warning(
                        "The calibration file is missing data at the end. Please check the original file."
                    )

                    # Show preview of the file content
                    with st.expander("🔍 File Content Preview"):
                        st.text(calibration_text[-200:])  # Show last 200 characters
                    return

                # Try to parse JSON with better error handling
                try:
                    calibration_data = json.loads(calibration_text)
                except json.JSONDecodeError as json_err:
                    st.error(f"❌ Invalid JSON format: {json_err}")

                    # Provide more specific error information
                    error_line = getattr(json_err, "lineno", "unknown")
                    error_col = getattr(json_err, "colno", "unknown")
                    error_pos = getattr(json_err, "pos", "unknown")

                    st.warning(
                        f"Error at line {error_line}, column {error_col} (position {error_pos})"
                    )

                    # Show the problematic area
                    lines = calibration_text.split("\n")
                    if isinstance(error_line, int) and error_line <= len(lines):
                        st.code(f"Problem line: {lines[error_line-1]}")

                    # Suggest fixes for common issues
                    st.info("💡 Common fixes:")
                    st.write("- Ensure the file ends with a closing brace `}`")
                    st.write("- Check for missing commas between fields")
                    st.write("- Verify all string values are in quotes")
                    st.write("- Make sure numeric values don't have trailing commas")

                    return

                # Validate calibration file format
                if CalibrationManager._validate_calibration_format(calibration_data):
                    # Check for missing optional fields and add defaults
                    calibration_data = CalibrationManager._fix_calibration_data(
                        calibration_data
                    )

                    # Create temporary file to load calibration
                    temp_calibration_path = (
                        f"temp_calibration_{uploaded_calibration.name}"
                    )

                    # Save temporary file
                    with open(temp_calibration_path, "w") as f:
                        json.dump(calibration_data, f, indent=2)

                    # Load calibration
                    if calibrator.load_calibration(temp_calibration_path):
                        st.success(
                            f"✅ Calibration loaded: {uploaded_calibration.name}"
                        )
                        st.session_state.loaded_calibration_file = (
                            uploaded_calibration.name
                        )

                        # Display calibration info
                        CalibrationManager._display_calibration_info(calibration_data)

                        # Clean up temporary file
                        if os.path.exists(temp_calibration_path):
                            os.remove(temp_calibration_path)

                        # Force refresh
                        st.rerun()
                    else:
                        st.error("Failed to load calibration file")
                        if os.path.exists(temp_calibration_path):
                            os.remove(temp_calibration_path)
                else:
                    st.error("Invalid calibration file format")
                    st.warning(
                        "The file doesn't contain the required calibration fields"
                    )

            except UnicodeDecodeError:
                st.error("❌ Cannot read file - invalid text encoding")
            except Exception as e:
                st.error(f"❌ Unexpected error loading calibration: {str(e)}")
                logger.error(f"Calibration upload error: {e}")

    @staticmethod
    def _render_calibration_browser(calibrator: CameraCalibrator):
        """Render available calibration files browser."""
        st.subheader("📂 Available Calibrations")

        calibrations_dir = "calibrations"
        if os.path.exists(calibrations_dir):
            available_calibrations = []
            for filename in os.listdir(calibrations_dir):
                if filename.startswith("calibration_") and filename.endswith(".json"):
                    filepath = os.path.join(calibrations_dir, filename)
                    if os.path.isfile(filepath):
                        # Check if file is valid JSON before adding to list
                        try:
                            with open(filepath, "r") as f:
                                json.load(f)
                            # Get file modification time
                            mtime = os.path.getmtime(filepath)
                            available_calibrations.append((filename, filepath, mtime))
                        except (json.JSONDecodeError, Exception):
                            # Skip invalid files but show warning
                            st.warning(f"⚠️ Skipping invalid file: {filename}")
                            continue

            if available_calibrations:
                # Sort by modification time (most recent first)
                available_calibrations.sort(key=lambda x: x[2], reverse=True)

                selected_calibration = st.selectbox(
                    "Select calibration file:",
                    options=["None"] + [f[0] for f in available_calibrations],
                    key="calibration_selector",
                )

                if selected_calibration != "None":
                    selected_path = next(
                        f[1]
                        for f in available_calibrations
                        if f[0] == selected_calibration
                    )

                    # Show file info
                    file_size = os.path.getsize(selected_path) / 1024  # KB
                    mod_time = os.path.getmtime(selected_path)
                    mod_date = datetime.fromtimestamp(mod_time).strftime(
                        "%Y-%m-%d %H:%M"
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Size:** {file_size:.1f} KB")
                    with col2:
                        st.write(f"**Modified:** {mod_date}")

                    if st.button("🔄 Load Selected Calibration"):
                        if calibrator.load_calibration(selected_path):
                            st.success(f"✅ Loaded: {selected_calibration}")
                            st.session_state.loaded_calibration_file = selected_path
                            st.rerun()
                        else:
                            st.error("Failed to load calibration")
            else:
                st.info("No calibration files found in calibrations directory")
        else:
            st.info("Calibrations directory not found")

    @staticmethod
    def _render_clear_option(calibrator: CameraCalibrator):
        """Render clear calibration option."""
        if st.button("🗑️ Clear Current Calibration"):
            calibrator.calibration_result = None
            if hasattr(st.session_state, "loaded_calibration_file"):
                del st.session_state.loaded_calibration_file
            st.success("Calibration cleared")
            st.rerun()

    @staticmethod
    def _validate_calibration_format(calibration_data: dict) -> bool:
        """Validate that the uploaded file has the correct calibration format."""
        required_fields = [
            "camera_matrix",
            "distortion_coefficients",
            "reprojection_error",
            "image_size",
        ]

        try:
            # Check required fields exist
            for field in required_fields:
                if field not in calibration_data:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Validate camera matrix format
            camera_matrix = calibration_data["camera_matrix"]
            if not isinstance(camera_matrix, list) or len(camera_matrix) != 3:
                return False

            for row in camera_matrix:
                if not isinstance(row, list) or len(row) != 3:
                    return False

            # Validate distortion coefficients
            dist_coeffs = calibration_data["distortion_coefficients"]
            if not isinstance(dist_coeffs, list) or len(dist_coeffs) == 0:
                return False

            # Validate numeric fields
            if not isinstance(calibration_data["reprojection_error"], (int, float)):
                return False

            # Validate image size
            image_size = calibration_data["image_size"]
            if not isinstance(image_size, list) or len(image_size) != 2:
                return False

            return True

        except Exception as e:
            logger.error(f"Calibration validation error: {e}")
            return False

    @staticmethod
    def _fix_calibration_data(calibration_data: dict) -> dict:
        """Fix calibration data by adding missing optional fields with defaults."""
        # Add default calibration date if missing
        if "calibration_date" not in calibration_data:
            calibration_data["calibration_date"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        # Add default pixels_per_mm if missing (will be calculated later if needed)
        if "pixels_per_mm" not in calibration_data:
            calibration_data["pixels_per_mm"] = None

        # Add default chessboard pattern info if missing
        if "chessboard_size" not in calibration_data:
            calibration_data["chessboard_size"] = [9, 6]  # Default pattern

        if "square_size_mm" not in calibration_data:
            calibration_data["square_size_mm"] = 10.0  # Default square size

        return calibration_data

    @staticmethod
    def _display_calibration_info(calibration_data: dict):
        """Display information about the loaded calibration."""
        st.subheader("📊 Calibration Details")

        # Basic info
        col1, col2 = st.columns(2)

        with col1:
            st.write(
                f"**Reprojection Error:** {calibration_data['reprojection_error']:.4f}"
            )
            st.write(
                f"**Image Size:** {calibration_data['image_size'][0]} × {calibration_data['image_size'][1]}"
            )

            if (
                "pixels_per_mm" in calibration_data
                and calibration_data["pixels_per_mm"]
            ):
                st.write(f"**Scale:** {calibration_data['pixels_per_mm']:.2f} px/mm")
                st.write(
                    f"**Resolution:** {1/calibration_data['pixels_per_mm']:.4f} mm/px"
                )

        with col2:
            if "calibration_date" in calibration_data:
                st.write(f"**Date:** {calibration_data['calibration_date']}")

            if "chessboard_size" in calibration_data:
                size = calibration_data["chessboard_size"]
                st.write(f"**Pattern:** {size[0]} × {size[1]}")

            if "square_size_mm" in calibration_data:
                st.write(f"**Square Size:** {calibration_data['square_size_mm']} mm")

        # Camera matrix (collapsible)
        with st.expander("🔍 Camera Matrix"):
            camera_matrix = np.array(calibration_data["camera_matrix"])
            st.text(f"fx: {camera_matrix[0,0]:.2f}")
            st.text(f"fy: {camera_matrix[1,1]:.2f}")
            st.text(f"cx: {camera_matrix[0,2]:.2f}")
            st.text(f"cy: {camera_matrix[1,2]:.2f}")

        # Distortion coefficients (collapsible)
        with st.expander("📐 Distortion Coefficients"):
            dist_coeffs = calibration_data["distortion_coefficients"][0]
            st.text(f"k1: {dist_coeffs[0]:.6f}")
            st.text(f"k2: {dist_coeffs[1]:.6f}")
            if len(dist_coeffs) > 2:
                st.text(f"p1: {dist_coeffs[2]:.6f}")
            if len(dist_coeffs) > 3:
                st.text(f"p2: {dist_coeffs[3]:.6f}")
            if len(dist_coeffs) > 4:
                st.text(f"k3: {dist_coeffs[4]:.6f}")

    @staticmethod
    def find_latest_calibration_file() -> Optional[str]:
        """Find the latest calibration file in the calibrations directory."""
        calibrations_dir = "calibrations"

        if not os.path.exists(calibrations_dir):
            return None

        # Look for calibration files with the pattern calibration_YYYYMMDD_HHMMSS.json
        calibration_files = []
        for filename in os.listdir(calibrations_dir):
            if filename.startswith("calibration_") and filename.endswith(".json"):
                filepath = os.path.join(calibrations_dir, filename)
                if os.path.isfile(filepath):
                    # Extract timestamp from filename
                    try:
                        # Extract date part from filename like calibration_20250910_145755.json
                        timestamp_part = filename.replace("calibration_", "").replace(
                            ".json", ""
                        )
                        calibration_files.append((filepath, timestamp_part))
                    except Exception:
                        continue

        if not calibration_files:
            return None

        # Sort by timestamp (most recent first)
        calibration_files.sort(key=lambda x: x[1], reverse=True)
        return calibration_files[0][0]

    @staticmethod
    def get_calibration_summary(calibrator: CameraCalibrator) -> dict:
        """Get a summary of the current calibration status."""
        if not calibrator.calibration_result:
            return {
                "calibrated": False,
                "status": "Not calibrated",
                "message": "No calibration data available",
            }

        return {
            "calibrated": True,
            "status": "Calibrated",
            "pixels_per_mm": calibrator.calibration_result.pixels_per_mm,
            "reprojection_error": calibrator.calibration_result.reprojection_error,
            "image_size": calibrator.calibration_result.image_size,
            "calibration_date": calibrator.calibration_result.calibration_date,
            "loaded_file": getattr(
                st.session_state, "loaded_calibration_file", "Unknown"
            ),
        }
