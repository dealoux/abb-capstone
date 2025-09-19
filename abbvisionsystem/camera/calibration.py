"""Camera calibration utilities for accurate measurements and distortion correction."""

import os
import json
import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Camera calibration result container."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_vectors: List[np.ndarray]
    translation_vectors: List[np.ndarray]
    reprojection_error: float
    image_size: Tuple[int, int]
    pixels_per_mm: Optional[float] = None
    calibration_date: Optional[str] = None


class CameraCalibrator:
    """Camera calibration class for Basler cameras in top-down configuration."""

    def __init__(self):
        self.calibration_images = []
        self.calibration_result = None
        self.chessboard_size = (39, 27)  # Default chessboard pattern
        self.square_size_mm = 10.0  # Default square size in mm
        self.origin_point = None  # Store the origin point
        self.pattern_rows = 27
        self.pattern_cols = 39
        self.square_size = 10.0

    def set_chessboard_pattern(self, rows: int, cols: int, square_size_mm: float):
        """Set chessboard calibration pattern parameters."""
        self.chessboard_size = (cols, rows)  # OpenCV expects (cols, rows)
        self.square_size_mm = square_size_mm
        # Also update the individual attributes for consistency
        self.pattern_rows = rows
        self.pattern_cols = cols
        self.square_size = square_size_mm
        logger.info(
            f"Chessboard pattern set to {cols}x{rows}, square size: {square_size_mm}mm"
        )

    def _preprocess_image_for_detection(self, image: np.ndarray) -> List[np.ndarray]:
        """Preprocess image with multiple techniques to improve corner detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Create multiple preprocessed versions
        processed_images = []

        # 1. Original grayscale
        processed_images.append(gray)

        # 2. Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        processed_images.append(blurred)

        # 3. Histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray)
        processed_images.append(equalized)

        # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(gray)
        processed_images.append(clahe_enhanced)

        # 5. Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        morph_close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        processed_images.append(morph_close)

        # 6. Bilateral filter for edge preservation
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        processed_images.append(bilateral)

        return processed_images

    def add_calibration_image(self, image: np.ndarray) -> bool:
        """Add a calibration image and set origin from first valid image."""
        # Get multiple preprocessed versions of the image
        processed_images = self._preprocess_image_for_detection(image)

        corners = None
        best_image = None
        detection_method = None

        # Try different flag combinations for findChessboardCorners
        flag_combinations = [
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FILTER_QUADS,
            cv2.CALIB_CB_ADAPTIVE_THRESH,
            cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_FAST_CHECK,
            0,  # No flags
        ]

        # Try each preprocessed image with each flag combination
        for i, proc_img in enumerate(processed_images):
            for j, flags in enumerate(flag_combinations):
                try:
                    ret, temp_corners = cv2.findChessboardCorners(
                        proc_img, self.chessboard_size, flags=flags
                    )

                    if ret and temp_corners is not None:
                        corners = temp_corners
                        best_image = proc_img
                        detection_method = f"Preprocessing {i}, Flags {j}"
                        logger.info(f"Chessboard detected using: {detection_method}")
                        break

                except Exception as e:
                    logger.debug(
                        f"Detection failed for preprocessing {i}, flags {j}: {e}"
                    )
                    continue

            if corners is not None:
                break

        if corners is None:
            logger.warning("Could not find chessboard corners with any method")
            return False

        try:
            # Refine corner positions with multiple criteria
            criteria_sets = [
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01),
            ]

            refined_corners = None
            for criteria in criteria_sets:
                try:
                    refined_corners = cv2.cornerSubPix(
                        best_image, corners, (11, 11), (-1, -1), criteria
                    )
                    break
                except:
                    continue

            if refined_corners is None:
                refined_corners = corners  # Use unrefined if refinement fails

            # Validate corner quality
            if self._validate_corners(refined_corners, best_image.shape):
                self.calibration_images.append(
                    {
                        "image": image.copy(),
                        "gray": best_image,
                        "corners": refined_corners,
                        "detection_method": detection_method,
                    }
                )

                # CRITICAL: Set origin from first valid image - with proper type conversion
                if len(self.calibration_images) == 1:
                    # Set origin to the first corner (top-left of chessboard pattern)
                    # Convert numpy types to regular Python int for JSON serialization
                    origin_raw = refined_corners[0].ravel()
                    self.origin_point = (int(origin_raw[0]), int(origin_raw[1]))
                    logger.info(f"🎯 ORIGIN SET from first calibration image: {self.origin_point}")
                    logger.info(f"🎯 Origin type: {type(self.origin_point)} with elements: {[type(x) for x in self.origin_point]}")
                    
                    # Store in a way that can be accessed from session state
                    if hasattr(self, '_session_sync'):
                        self._session_sync['origin_point'] = self.origin_point

                logger.info(
                    f"Added calibration image {len(self.calibration_images)} using {detection_method}"
                )
                return True
            else:
                logger.warning("Corner validation failed - poor corner quality")
                return False

        except Exception as e:
            logger.error(f"Error refining corners: {e}")
            return False

    def _validate_corners(
        self, corners: np.ndarray, image_shape: Tuple[int, int]
    ) -> bool:
        """Validate the quality of detected corners."""
        if corners is None or len(corners) == 0:
            return False

        # Check if we have the expected number of corners
        expected_corners = self.chessboard_size[0] * self.chessboard_size[1]
        if len(corners) != expected_corners:
            logger.warning(f"Expected {expected_corners} corners, got {len(corners)}")
            return False

        # Check if corners are within image bounds
        corners_2d = corners.reshape(-1, 2)
        height, width = image_shape

        if (
            np.any(corners_2d < 0)
            or np.any(corners_2d[:, 0] >= width)
            or np.any(corners_2d[:, 1] >= height)
        ):
            logger.warning("Some corners are outside image bounds")
            return False

        # Check corner distribution (should cover reasonable area)
        min_x, min_y = np.min(corners_2d, axis=0)
        max_x, max_y = np.max(corners_2d, axis=0)

        coverage_x = (max_x - min_x) / width
        coverage_y = (max_y - min_y) / height

        if coverage_x < 0.2 or coverage_y < 0.2:
            logger.warning(
                f"Chessboard covers too small area: {coverage_x:.2f}x{coverage_y:.2f}"
            )
            return False

        # Check for reasonable corner spacing
        corners_reshaped = corners_2d.reshape(
            self.chessboard_size[1], self.chessboard_size[0], 2
        )

        # Calculate distances between adjacent corners
        horizontal_distances = []
        vertical_distances = []

        for i in range(self.chessboard_size[1]):
            for j in range(self.chessboard_size[0] - 1):
                dist = np.linalg.norm(
                    corners_reshaped[i, j] - corners_reshaped[i, j + 1]
                )
                horizontal_distances.append(dist)

        for i in range(self.chessboard_size[1] - 1):
            for j in range(self.chessboard_size[0]):
                dist = np.linalg.norm(
                    corners_reshaped[i, j] - corners_reshaped[i + 1, j]
                )
                vertical_distances.append(dist)

        # Check if spacing is reasonably consistent
        if horizontal_distances:
            h_mean = np.mean(horizontal_distances)
            h_std = np.std(horizontal_distances)
            if h_std / h_mean > 0.3:  # 30% variation is too much
                logger.warning(
                    f"Inconsistent horizontal spacing: std/mean = {h_std/h_mean:.3f}"
                )
                return False

        if vertical_distances:
            v_mean = np.mean(vertical_distances)
            v_std = np.std(vertical_distances)
            if v_std / v_mean > 0.3:  # 30% variation is too much
                logger.warning(
                    f"Inconsistent vertical spacing: std/mean = {v_std/v_mean:.3f}"
                )
                return False

        return True

    def calibrate_camera(self) -> Optional[CalibrationResult]:
        """Perform camera calibration using collected images."""
        if len(self.calibration_images) < 5:
            logger.error("Need at least 5 calibration images")
            return None

        # Prepare object points - matching OpenCV tutorial format
        objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
        ].T.reshape(-1, 2)
        objp *= self.square_size_mm

        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        image_size = None

        for calib_data in self.calibration_images:
            objpoints.append(objp)
            imgpoints.append(calib_data["corners"])

            if image_size is None:
                image_size = calib_data["gray"].shape[::-1]

        # Use more robust calibration flags
        calibration_flags = (
            cv2.CALIB_RATIONAL_MODEL  # Use rational distortion model
            + cv2.CALIB_THIN_PRISM_MODEL  # Include thin prism distortion
            + cv2.CALIB_TILTED_MODEL  # Include tilted sensor model
        )

        try:
            # Perform calibration with robust flags
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, image_size, None, None, flags=calibration_flags
            )
        except:
            # Fallback to basic calibration if advanced model fails
            logger.warning("Advanced calibration model failed, using basic model")
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, image_size, None, None
            )

        if ret:
            # Calculate reprojection error
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(
                    objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(
                    imgpoints2
                )
                total_error += error

            mean_error = total_error / len(objpoints)

            # Calculate pixels per mm (for top-down view)
            pixels_per_mm = self._calculate_pixels_per_mm(
                camera_matrix, objpoints[0], imgpoints[0]
            )

            self.calibration_result = CalibrationResult(
                camera_matrix=camera_matrix,
                distortion_coefficients=dist_coeffs,
                rotation_vectors=rvecs,
                translation_vectors=tvecs,
                reprojection_error=mean_error,
                image_size=image_size,
                pixels_per_mm=pixels_per_mm,
                calibration_date=str(np.datetime64("now")),
            )

            logger.info(
                f"Camera calibration successful. Reprojection error: {mean_error:.3f}"
            )
            logger.info(f"Scale: {pixels_per_mm:.2f} pixels/mm")

            return self.calibration_result
        else:
            logger.error("Camera calibration failed")
            return None

    def _calculate_pixels_per_mm(
        self, camera_matrix: np.ndarray, objpoints: np.ndarray, imgpoints: np.ndarray
    ) -> float:
        """Calculate pixels per mm scale factor."""
        try:
            # Take two adjacent points from the chessboard
            obj_p1 = objpoints[0]  # First corner
            obj_p2 = objpoints[1]  # Adjacent corner

            img_p1 = imgpoints[0][0]  # Corresponding image point
            img_p2 = imgpoints[1][0]  # Corresponding image point

            # Calculate real-world distance (should be square_size_mm)
            real_distance = np.linalg.norm(obj_p2[:2] - obj_p1[:2])

            # Calculate pixel distance
            pixel_distance = np.linalg.norm(img_p2 - img_p1)

            # Calculate scale
            pixels_per_mm = pixel_distance / real_distance

            return pixels_per_mm

        except Exception as e:
            logger.warning(f"Could not calculate pixels per mm: {e}")
            return 1.0  # Default fallback

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from image."""
        if self.calibration_result is None:
            logger.warning("No calibration data available")
            return image

        return cv2.undistort(
            image,
            self.calibration_result.camera_matrix,
            self.calibration_result.distortion_coefficients,
        )

    def pixel_to_world_coordinates(
        self, pixel_x: float, pixel_y: float, world_z: float = 0
    ) -> Tuple[float, float]:
        """Convert pixel coordinates to real-world coordinates (mm)."""
        if (
            self.calibration_result is None
            or self.calibration_result.pixels_per_mm is None
        ):
            logger.warning("No calibration data available")
            return pixel_x, pixel_y

        # For top-down camera, assume Z=0 (flat surface)
        world_x = pixel_x / self.calibration_result.pixels_per_mm
        world_y = pixel_y / self.calibration_result.pixels_per_mm

        return world_x, world_y

    def draw_chessboard_pattern(
        self,
        image: np.ndarray,
        draw_axes: bool = True,
        draw_cube: bool = False,
        draw_camera_frame: bool = False,
    ) -> np.ndarray:
        """
        Draw the detected chessboard pattern with enhanced visualization INCLUDING ORIGIN.
        """
        # Get multiple preprocessed versions of the image
        processed_images = self._preprocess_image_for_detection(image)

        result_image = image.copy()
        corners = None
        detection_info = "No chessboard detected"

        # Try to detect with the same robust method
        flag_combinations = [
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FILTER_QUADS,
            cv2.CALIB_CB_ADAPTIVE_THRESH,
            0,
        ]

        for i, proc_img in enumerate(processed_images):
            for j, flags in enumerate(flag_combinations):
                try:
                    ret, temp_corners = cv2.findChessboardCorners(
                        proc_img, self.chessboard_size, flags=flags
                    )

                    if ret:
                        # Refine corners like in OpenCV tutorial
                        criteria = (
                            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30,
                            0.001,
                        )
                        corners = cv2.cornerSubPix(
                            proc_img, temp_corners, (11, 11), (-1, -1), criteria
                        )
                        detection_info = f"Detected (Method {i}.{j})"
                        break
                except:
                    continue
            if corners is not None:
                break

        if corners is not None:
            # Draw chessboard corners - matching OpenCV tutorial
            cv2.drawChessboardCorners(result_image, self.chessboard_size, corners, True)

            # CRITICAL: Draw the origin point (first corner) prominently
            origin_point = tuple(corners[0].ravel().astype(int))
            
            # Draw large yellow circle for origin
            cv2.circle(result_image, origin_point, 12, (0, 255, 255), -1)  # Yellow filled circle
            cv2.circle(result_image, origin_point, 12, (0, 0, 0), 2)        # Black border
            
            # Draw origin label
            cv2.putText(
                result_image, "ORIGIN", 
                (origin_point[0] - 30, origin_point[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            cv2.putText(
                result_image, "(0,0)", 
                (origin_point[0] - 20, origin_point[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            # Draw simple coordinate axes from origin
            axis_length = 80
            # X-axis (Red arrow)
            cv2.arrowedLine(
                result_image, origin_point, 
                (origin_point[0] + axis_length, origin_point[1]), 
                (0, 0, 255), 3, tipLength=0.1
            )
            # Y-axis (Green arrow)
            cv2.arrowedLine(
                result_image, origin_point, 
                (origin_point[0], origin_point[1] + axis_length), 
                (0, 255, 0), 3, tipLength=0.1
            )
            
            # Axis labels
            cv2.putText(
                result_image, "X", 
                (origin_point[0] + axis_length + 10, origin_point[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            cv2.putText(
                result_image, "Y", 
                (origin_point[0] - 10, origin_point[1] + axis_length + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            # Show origin coordinates
            detection_info += f" - Origin: {origin_point}"

            # Draw coordinate axes if calibration is available and requested
            if draw_axes and self.calibration_result is not None:
                self._draw_coordinate_axes(result_image, corners, processed_images[0])

            # Draw 3D cube if requested and calibration is available
            if draw_cube and self.calibration_result is not None:
                self._draw_3d_cube(result_image, corners, processed_images[0])

            if draw_camera_frame:
                self._draw_camera_pinhole_frame(
                    result_image, corners, processed_images[0]
                )

            # Validate corners and show quality
            is_valid = self._validate_corners(corners, processed_images[0].shape)
            quality_text = "✓ Good Quality" if is_valid else "⚠ Poor Quality"
            detection_info += f" - {quality_text}"

            # Show if this is the stored origin
            if self.origin_point and tuple(self.origin_point) == origin_point:
                cv2.putText(
                    result_image, "SAVED ORIGIN", 
                    (origin_point[0] - 40, origin_point[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

        # Add comprehensive info text
        info_texts = [detection_info]

        if self.calibration_result:
            info_texts.extend(
                [
                    f"Calibrated: ✓",
                    f"Error: {self.calibration_result.reprojection_error:.3f}",
                    f"Scale: {self.calibration_result.pixels_per_mm:.2f} px/mm",
                ]
            )

        # Show stored origin info
        if self.origin_point:
            info_texts.append(f"Stored Origin: {self.origin_point}")

        # Draw pattern info
        info_texts.append(
            f"Pattern: {self.chessboard_size[0]}×{self.chessboard_size[1]}"
        )
        info_texts.append(f"Square: {self.square_size_mm}mm")

        for i, text in enumerate(info_texts):
            color = (
                (0, 255, 0)
                if "✓" in text
                else (0, 165, 255) if "⚠" in text else (255, 255, 255)
            )
            cv2.putText(
                result_image,
                text,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return result_image

    def visualize_calibration(
        self,
        image: np.ndarray,
        draw_axes: bool = True,
        draw_cube: bool = False,
        draw_camera_frame: bool = False,
    ) -> np.ndarray:
        """
        Enhanced visualization with pattern drawing capabilities INCLUDING ORIGIN.
        This is the main method to call for visualization.
        """
        return self.draw_chessboard_pattern(
            image, draw_axes, draw_cube, draw_camera_frame
        )

    def draw_origin_on_image(self, image: np.ndarray) -> np.ndarray:
        """Draw the stored origin point on any image."""
        result_image = image.copy()
        
        if self.origin_point:
            ox, oy = self.origin_point
            
            # Draw large yellow circle for origin
            cv2.circle(result_image, (ox, oy), 12, (0, 255, 255), -1)  # Yellow filled circle
            cv2.circle(result_image, (ox, oy), 12, (0, 0, 0), 2)        # Black border
            
            # Draw origin label
            cv2.putText(
                result_image, "ORIGIN", 
                (ox - 30, oy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            cv2.putText(
                result_image, "(0,0)", 
                (ox - 20, oy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            # Draw coordinate axes
            axis_length = 80
            # X-axis (Red arrow)
            cv2.arrowedLine(
                result_image, (ox, oy), 
                (ox + axis_length, oy), 
                (0, 0, 255), 3, tipLength=0.1
            )
            # Y-axis (Green arrow)
            cv2.arrowedLine(
                result_image, (ox, oy), 
                (ox, oy + axis_length), 
                (0, 255, 0), 3, tipLength=0.1
            )
            
            # Axis labels
            cv2.putText(
                result_image, "X", 
                (ox + axis_length + 10, oy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            cv2.putText(
                result_image, "Y", 
                (ox - 10, oy + axis_length + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            
            # Add coordinates text
            cv2.putText(
                result_image, f"Origin: ({ox}, {oy})", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
        
        return result_image

    def save_calibration(self, filepath: str) -> bool:
        """Save calibration data to file."""
        if self.calibration_result is None:
            logger.error("No calibration data to save")
            return False

        try:
            calibration_data = {
                "camera_matrix": self.calibration_result.camera_matrix.tolist(),
                "distortion_coefficients": self.calibration_result.distortion_coefficients.tolist(),
                "reprojection_error": float(self.calibration_result.reprojection_error),
                "image_size": list(self.calibration_result.image_size),
                "pixels_per_mm": (
                    float(self.calibration_result.pixels_per_mm)
                    if self.calibration_result.pixels_per_mm is not None
                    else None
                ),
                "calibration_date": (
                    str(self.calibration_result.calibration_date)
                    if self.calibration_result.calibration_date is not None
                    else None
                ),
                "chessboard_size": list(self.chessboard_size),
                "square_size_mm": float(self.square_size_mm),
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, "w") as f:
                json.dump(calibration_data, f, indent=2)

            logger.info(f"Calibration saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False

    def load_calibration(self, filepath: str) -> bool:
        """Load calibration data from file."""
        try:
            with open(filepath, "r") as f:
                calibration_data = json.load(f)

            self.calibration_result = CalibrationResult(
                camera_matrix=np.array(calibration_data["camera_matrix"]),
                distortion_coefficients=np.array(
                    calibration_data["distortion_coefficients"]
                ),
                rotation_vectors=[],  # Not saved/loaded for simplicity
                translation_vectors=[],
                reprojection_error=calibration_data["reprojection_error"],
                image_size=tuple(calibration_data["image_size"]),
                pixels_per_mm=calibration_data.get("pixels_per_mm"),
                calibration_date=calibration_data.get("calibration_date"),
            )

            self.chessboard_size = tuple(
                calibration_data.get("chessboard_size", (9, 6))
            )
            self.square_size_mm = calibration_data.get("square_size_mm", 25.0)

            logger.info(f"Calibration loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False

    def save_calibration_with_origin(self, filepath: str) -> bool:
        """Save calibration results including origin point to JSON file."""
        if not self.calibration_result:
            logger.error("No calibration result to save")
            return False

        try:
            import json
            from datetime import datetime

            # Helper function to convert numpy types to Python types
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32, np.integer)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32, np.floating)):
                    return float(obj)
                elif isinstance(obj, tuple):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            # Prepare calibration data including origin - with proper type conversion
            calibration_data = {
                "camera_matrix": convert_numpy_types(self.calibration_result.camera_matrix),
                "distortion_coefficients": convert_numpy_types(self.calibration_result.distortion_coefficients),
                "reprojection_error": convert_numpy_types(self.calibration_result.reprojection_error),
                "image_size": convert_numpy_types(self.calibration_result.image_size),
                "pixels_per_mm": convert_numpy_types(self.calibration_result.pixels_per_mm) if self.calibration_result.pixels_per_mm else None,
                "calibration_date": str(self.calibration_result.calibration_date) if self.calibration_result.calibration_date else datetime.now().isoformat(),
                "chessboard_size": convert_numpy_types(self.chessboard_size),
                "square_size_mm": convert_numpy_types(self.square_size_mm),
                "origin_point": convert_numpy_types(self.origin_point) if self.origin_point else None,
                "pattern_rows": convert_numpy_types(self.pattern_rows),
                "pattern_cols": convert_numpy_types(self.pattern_cols),
                "square_size": convert_numpy_types(self.square_size),
                "num_calibration_images": convert_numpy_types(len(self.calibration_images))
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save to file with proper error handling
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2, default=convert_numpy_types)

            logger.info(f"Calibration with origin saved to {filepath}")
            logger.info(f"Origin point saved: {self.origin_point}")
            return True

        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            logger.error(f"Error details: {str(e)}")
            
            # Additional debugging for JSON serialization issues
            if "JSON serializable" in str(e):
                logger.error("JSON Serialization issue - checking data types:")
                logger.error(f"Origin point type: {type(self.origin_point)}")
                if self.origin_point:
                    logger.error(f"Origin elements types: {[type(x) for x in self.origin_point]}")
                logger.error(f"Pattern rows type: {type(self.pattern_rows)}")
                logger.error(f"Pattern cols type: {type(self.pattern_cols)}")
        
        return False

    def load_calibration_with_origin(self, filepath: str) -> bool:
        """Load calibration results including origin point from JSON file."""
        try:
            import json
            import numpy as np

            # Check if file exists
            if not os.path.exists(filepath):
                logger.error(f"Calibration file does not exist: {filepath}")
                return False

            # Check file size
            if os.path.getsize(filepath) == 0:
                logger.error(f"Calibration file is empty: {filepath}")
                return False

            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in calibration file {filepath}: {e}")
                    logger.error(f"JSON error at line {e.lineno}, column {e.colno}")
                    return False

            # Validate required fields
            required_fields = ["camera_matrix", "distortion_coefficients", "reprojection_error", "image_size"]
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field '{field}' in calibration file")
                    return False

            # Restore calibration result
            self.calibration_result = CalibrationResult(
                camera_matrix=np.array(data["camera_matrix"]),
                distortion_coefficients=np.array(data["distortion_coefficients"]),
                rotation_vectors=[],  # These aren't typically saved/needed
                translation_vectors=[],  # These aren't typically saved/needed
                reprojection_error=float(data["reprojection_error"]),
                image_size=tuple(data["image_size"]),
                pixels_per_mm=float(data["pixels_per_mm"]) if data.get("pixels_per_mm") else None,
                calibration_date=str(data.get("calibration_date"))
            )

            # Restore pattern settings with defaults
            self.chessboard_size = tuple(data.get("chessboard_size", [39, 27]))
            self.square_size_mm = float(data.get("square_size_mm", 10.0))
            
            # Restore individual pattern attributes
            self.pattern_rows = int(data.get("pattern_rows", self.chessboard_size[1]))
            self.pattern_cols = int(data.get("pattern_cols", self.chessboard_size[0]))
            self.square_size = float(data.get("square_size", self.square_size_mm))

            # CRITICAL: Restore origin point with type conversion
            if data.get("origin_point"):
                origin_data = data["origin_point"]
                self.origin_point = (int(origin_data[0]), int(origin_data[1]))
            else:
                self.origin_point = None

            logger.info(f"Calibration with origin loaded from {filepath}")
            logger.info(f"Origin point loaded: {self.origin_point}")
            logger.info(f"Pattern size: {self.chessboard_size}, Square size: {self.square_size_mm}mm")
            
            return True

        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            logger.error(f"File path: {filepath}")
            return False

    def load_origin_only(self, filepath: str) -> bool:
        """Load just the origin point from a saved file."""
        try:
            import json
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if data.get('type') == 'origin_only':
                self.origin_point = tuple(data["origin_point"])
                
                # Also load pattern settings if available
                if 'pattern_size' in data:
                    self.chessboard_size = tuple(data["pattern_size"])
                if 'square_size_mm' in data:
                    self.square_size_mm = data["square_size_mm"]
                    self.square_size = data["square_size_mm"]
                
                logger.info(f"Origin point loaded: {self.origin_point}")
                return True
            else:
                logger.error("File is not an origin-only file")
                return False
            
        except Exception as e:
            logger.error(f"Failed to load origin: {e}")
            return False

    def _draw_coordinate_axes(self, image: np.ndarray, corners: np.ndarray, gray_image: np.ndarray) -> None:
        """Draw 3D coordinate axes on the detected chessboard."""
        if self.calibration_result is None:
            return
        
        try:
            # Define 3D points for coordinate axes (in chessboard coordinate system)
            axis_length = 3  # Length in chessboard squares
            axis_points = np.float32([
                [0, 0, 0],  # Origin
                [axis_length, 0, 0],  # X-axis
                [0, axis_length, 0],  # Y-axis
                [0, 0, -axis_length]  # Z-axis (negative for top-down view)
            ]).reshape(-1, 3)
            
            # Scale by square size
            axis_points *= self.square_size_mm
            
            # Project 3D points to image plane
            rvec = np.zeros((3, 1))  # Assume chessboard is flat (no rotation)
            tvec = np.zeros((3, 1))  # Translation will be estimated
            
            # Use solvePnP to get pose
            object_points = self._generate_object_points()
            success, rvec, tvec = cv2.solvePnP(
                object_points, corners, 
                self.calibration_result.camera_matrix,
                self.calibration_result.distortion_coefficients
            )
            
            if success:
                # Project axis points
                axis_img_points, _ = cv2.projectPoints(
                    axis_points, rvec, tvec, 
                    self.calibration_result.camera_matrix,
                    self.calibration_result.distortion_coefficients
                )
                
                axis_img_points = axis_img_points.reshape(-1, 2).astype(int)
                
                # Draw axes
                origin = tuple(axis_img_points[0])
                x_end = tuple(axis_img_points[1])
                y_end = tuple(axis_img_points[2])
                z_end = tuple(axis_img_points[3])
                
                # Draw axes with different colors
                cv2.arrowedLine(image, origin, x_end, (0, 0, 255), 3, tipLength=0.1)  # Red X
                cv2.arrowedLine(image, origin, y_end, (0, 255, 0), 3, tipLength=0.1)  # Green Y
                cv2.arrowedLine(image, origin, z_end, (255, 0, 0), 3, tipLength=0.1)  # Blue Z
                
                # Add labels
                cv2.putText(image, "X", (x_end[0] + 10, x_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, "Y", (y_end[0] + 10, y_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, "Z", (z_end[0] + 10, z_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
        except Exception as e:
            logger.warning(f"Failed to draw coordinate axes: {e}")

    def _draw_3d_cube(self, image: np.ndarray, corners: np.ndarray, gray_image: np.ndarray) -> None:
        """Draw a 3D cube on the detected chessboard."""
        if self.calibration_result is None:
            return
        
        try:
            # Define 3D points for a cube (in chessboard coordinate system)
            cube_size = 3  # Size in chessboard squares
            cube_points = np.float32([
                [0, 0, 0], [cube_size, 0, 0], [cube_size, cube_size, 0], [0, cube_size, 0],  # Bottom face
                [0, 0, -cube_size], [cube_size, 0, -cube_size], [cube_size, cube_size, -cube_size], [0, cube_size, -cube_size]  # Top face
            ]).reshape(-1, 3)
            
            # Scale by square size
            cube_points *= self.square_size_mm
            
            # Get pose using solvePnP
            object_points = self._generate_object_points()
            success, rvec, tvec = cv2.solvePnP(
                object_points, corners,
                self.calibration_result.camera_matrix,
                self.calibration_result.distortion_coefficients
            )
            
            if success:
                # Project cube points
                cube_img_points, _ = cv2.projectPoints(
                    cube_points, rvec, tvec, 
                    self.calibration_result.camera_matrix,
                    self.calibration_result.distortion_coefficients
                )
                
                cube_img_points = cube_img_points.reshape(-1, 2).astype(int)
                
                # Draw cube edges
                # Bottom face
                cv2.drawContours(image, [cube_img_points[:4]], -1, (0, 255, 0), 2)
                # Top face
                cv2.drawContours(image, [cube_img_points[4:8]], -1, (0, 255, 0), 2)
                # Vertical edges
                for i in range(4):
                    cv2.line(image, tuple(cube_img_points[i]), tuple(cube_img_points[i+4]), (0, 255, 0), 2)
                
        except Exception as e:
            logger.warning(f"Failed to draw 3D cube: {e}")

    def _draw_camera_pinhole_frame(self, image: np.ndarray, corners: np.ndarray, gray_image: np.ndarray) -> None:
        """Draw camera pinhole model visualization."""
        if self.calibration_result is None:
            return
        
        try:
            # Get camera center and principal point
            cx = self.calibration_result.camera_matrix[0, 2]
            cy = self.calibration_result.camera_matrix[1, 2]
            fx = self.calibration_result.camera_matrix[0, 0]
            fy = self.calibration_result.camera_matrix[1, 1]
            
            # Draw principal point
            cv2.circle(image, (int(cx), int(cy)), 5, (255, 255, 0), -1)
            cv2.putText(image, "Principal Point", (int(cx) + 10, int(cy)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw field of view indicators
            height, width = image.shape[:2]
            cv2.rectangle(image, (0, 0), (width-1, height-1), (255, 255, 0), 2)
            
            # Add focal length info
            cv2.putText(image, f"fx: {fx:.1f}", (10, height-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(image, f"fy: {fy:.1f}", (10, height-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
        except Exception as e:
            logger.warning(f"Failed to draw camera frame: {e}")

    def _generate_object_points(self) -> np.ndarray:
        """Generate 3D object points for the chessboard pattern."""
        # Create 3D points for chessboard corners
        object_points = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        object_points[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        object_points *= self.square_size_mm
        return object_points

    def _validate_corners(self, corners: np.ndarray, image_shape: tuple) -> bool:
        """Validate the quality of detected corners."""
        if corners is None or len(corners) == 0:
            return False
        
        try:
            # Check if corners are within image bounds
            height, width = image_shape
            corners_2d = corners.reshape(-1, 2)
            
            if np.any(corners_2d < 0) or np.any(corners_2d[:, 0] >= width) or np.any(corners_2d[:, 1] >= height):
                return False
            
            # Check corner distribution (should be spread across image)
            x_range = np.max(corners_2d[:, 0]) - np.min(corners_2d[:, 0])
            y_range = np.max(corners_2d[:, 1]) - np.min(corners_2d[:, 1])
            
            # Pattern should cover reasonable portion of image
            min_coverage = 0.3  # 30% of image
            if x_range < width * min_coverage or y_range < height * min_coverage:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Corner validation failed: {e}")
            return False

    def get_detection_tips(self) -> List[str]:
        """Get tips for improving pattern detection."""
        tips = [
            "Ensure the chessboard pattern is flat and not warped",
            "Use good, even lighting without shadows or reflections",
            "Make sure all corners of the pattern are visible",
            "Keep the camera steady and in focus",
            "Try different angles and positions for the chessboard",
            "Ensure the pattern size settings match your physical chessboard",
            "Use a high-contrast chessboard (black and white squares)",
            "Avoid motion blur by keeping the pattern stationary during capture"
        ]
        return tips

    def debug_chessboard_detection(self, image: np.ndarray) -> dict:
        """Debug chessboard detection with detailed information."""
        debug_info = {
            "image_shape": image.shape,
            "chessboard_size": self.chessboard_size,
            "detection_attempts": [],
            "preprocessing_results": []
        }
        
        # Get preprocessed images
        processed_images = self._preprocess_image_for_detection(image)
        
        for i, proc_img in enumerate(processed_images):
            debug_info["preprocessing_results"].append({
                "method": f"Preprocessing_{i}",
                "mean_intensity": float(np.mean(proc_img)),
                "std_intensity": float(np.std(proc_img)),
                "min_intensity": int(np.min(proc_img)),
                "max_intensity": int(np.max(proc_img))
            })
        
        # Try detection with different methods
        flag_combinations = [
            (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE, "ADAPTIVE_THRESH + NORMALIZE"),
            (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS, "ADAPTIVE + NORMALIZE + FILTER"),
            (cv2.CALIB_CB_ADAPTIVE_THRESH, "ADAPTIVE_THRESH"),
            (cv2.CALIB_CB_NORMALIZE_IMAGE, "NORMALIZE"),
            (cv2.CALIB_CB_FAST_CHECK, "FAST_CHECK"),
            (0, "NO_FLAGS"),
        ]
        
        for i, proc_img in enumerate(processed_images):
            for flags, flag_name in flag_combinations:
                try:
                    ret, corners = cv2.findChessboardCorners(proc_img, self.chessboard_size, flags=flags)
                    debug_info["detection_attempts"].append({
                        "preprocessing": i,
                        "flags": flag_name,
                        "success": bool(ret),
                        "corners_found": len(corners) if ret else 0
                    })
                except Exception as e:
                    debug_info["detection_attempts"].append({
                        "preprocessing": i,
                        "flags": flag_name,
                        "success": False,
                        "error": str(e)
                    })
        
        return debug_info

def detection_page():
    """Detection page that uses the calibrated origin."""
    st.title("🔍 Object Detection")
    
    # Get calibration data from session state
    calibration_data = st.session_state.get('calibration_data', {})
    camera = st.session_state.get('camera')
    
    # Show calibration status at the top
    st.subheader("📐 Calibration Status")
    
    if calibration_data.get('has_calibration', False):
        # ...existing status display...
        
        # Add origin source information
        origin = calibration_data.get('origin_point')
        if origin:
            st.info(f"🎯 **Using Saved Origin**: {origin} (from checkerboard calibration)")
            st.success("✅ This origin point was saved from the checkerboard pattern and can be used without the checkerboard present")
        
        # ...rest of existing code...
