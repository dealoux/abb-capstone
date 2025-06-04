"""Camera calibration utilities for accurate measurements and distortion correction."""

import cv2
import numpy as np
import json
import os
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

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

    def set_chessboard_pattern(self, rows: int, cols: int, square_size_mm: float):
        """Set chessboard calibration pattern parameters."""
        self.chessboard_size = (cols, rows)  # OpenCV expects (cols, rows)
        self.square_size_mm = square_size_mm
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
        """Add a calibration image to the dataset with robust corner detection."""
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
        Draw the detected chessboard pattern with enhanced visualization.
        Similar to OpenCV tutorial but with additional features.
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

            # Draw coordinate axes if calibration is available and requested
            if draw_axes and self.calibration_result is not None:
                self._draw_coordinate_axes(result_image, corners, proc_img)

            # Draw 3D cube if requested and calibration is available
            if draw_cube and self.calibration_result is not None:
                self._draw_3d_cube(result_image, corners, proc_img)

            if draw_camera_frame:
                self._draw_camera_pinhole_frame(
                    result_image, corners, processed_images[0]
                )

            # Validate corners and show quality
            is_valid = self._validate_corners(corners, processed_images[0].shape)
            quality_text = "✓ Good Quality" if is_valid else "⚠ Poor Quality"
            detection_info += f" - {quality_text}"

            # Add corner numbering for debugging
            self._add_corner_numbering(result_image, corners)

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

    def _draw_coordinate_axes(
        self, image: np.ndarray, corners: np.ndarray, gray: np.ndarray
    ):
        """Draw coordinate axes on the chessboard origin."""
        if self.calibration_result is None:
            return

        try:
            # Define axis points (origin + 3 axis endpoints)
            axis_length = 3 * self.square_size_mm  # 3 squares length
            axis_points = np.float32(
                [
                    [0, 0, 0],  # Origin
                    [axis_length, 0, 0],  # X-axis (red)
                    [0, axis_length, 0],  # Y-axis (green)
                    [0, 0, -axis_length],  # Z-axis (blue)
                ]
            ).reshape(-1, 3)

            # Project 3D points to image plane
            # Use the first image's rotation and translation vectors
            if (
                self.calibration_result.rotation_vectors
                and self.calibration_result.translation_vectors
            ):
                rvec = self.calibration_result.rotation_vectors[0]
                tvec = self.calibration_result.translation_vectors[0]
            else:
                # Estimate pose for current image
                objp = np.zeros(
                    (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
                )
                objp[:, :2] = np.mgrid[
                    0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
                ].T.reshape(-1, 2)
                objp *= self.square_size_mm

                _, rvec, tvec = cv2.solvePnP(
                    objp,
                    corners,
                    self.calibration_result.camera_matrix,
                    self.calibration_result.distortion_coefficients,
                )

            # Project axis points
            axis_img_points, _ = cv2.projectPoints(
                axis_points,
                rvec,
                tvec,
                self.calibration_result.camera_matrix,
                self.calibration_result.distortion_coefficients,
            )

            # Convert to integer coordinates
            origin = tuple(axis_img_points[0].ravel().astype(int))
            x_end = tuple(axis_img_points[1].ravel().astype(int))
            y_end = tuple(axis_img_points[2].ravel().astype(int))
            z_end = tuple(axis_img_points[3].ravel().astype(int))

            # Draw axes with different colors
            cv2.arrowedLine(image, origin, x_end, (0, 0, 255), 3)  # X-axis: Red
            cv2.arrowedLine(image, origin, y_end, (0, 255, 0), 3)  # Y-axis: Green
            cv2.arrowedLine(image, origin, z_end, (255, 0, 0), 3)  # Z-axis: Blue

            # Add axis labels
            cv2.putText(
                image, "X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            cv2.putText(
                image, "Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                image, "Z", z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
            )

        except Exception as e:
            logger.debug(f"Could not draw coordinate axes: {e}")

    def _draw_3d_cube(self, image: np.ndarray, corners: np.ndarray, gray: np.ndarray):
        """Draw a 3D cube on the chessboard for visualization."""
        if self.calibration_result is None:
            return

        try:
            # Define cube points
            cube_size = 3 * self.square_size_mm
            cube_points = np.float32(
                [
                    [0, 0, 0],
                    [0, cube_size, 0],
                    [cube_size, cube_size, 0],
                    [cube_size, 0, 0],  # Bottom face
                    [0, 0, -cube_size],
                    [0, cube_size, -cube_size],
                    [cube_size, cube_size, -cube_size],
                    [cube_size, 0, -cube_size],  # Top face
                ]
            ).reshape(-1, 3)

            # Estimate pose for current image
            objp = np.zeros(
                (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
            )
            objp[:, :2] = np.mgrid[
                0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
            ].T.reshape(-1, 2)
            objp *= self.square_size_mm

            _, rvec, tvec = cv2.solvePnP(
                objp,
                corners,
                self.calibration_result.camera_matrix,
                self.calibration_result.distortion_coefficients,
            )

            # Project cube points
            cube_img_points, _ = cv2.projectPoints(
                cube_points,
                rvec,
                tvec,
                self.calibration_result.camera_matrix,
                self.calibration_result.distortion_coefficients,
            )

            cube_img_points = np.int32(cube_img_points).reshape(-1, 2)

            # Draw bottom face in green
            cv2.drawContours(image, [cube_img_points[:4]], -1, (0, 255, 0), -2)

            # Draw top face in blue
            cv2.drawContours(image, [cube_img_points[4:8]], -1, (255, 0, 0), -2)

            # Draw vertical edges in red
            for i, j in zip(range(4), range(4, 8)):
                cv2.line(
                    image,
                    tuple(cube_img_points[i]),
                    tuple(cube_img_points[j]),
                    (0, 0, 255),
                    2,
                )

        except Exception as e:
            logger.debug(f"Could not draw 3D cube: {e}")

    def _add_corner_numbering(
        self, image: np.ndarray, corners: np.ndarray, max_numbers: int = 20
    ):
        """Add numbering to corners for debugging purposes."""
        corners_2d = corners.reshape(-1, 2)

        # Only show numbering for first few corners to avoid clutter
        num_to_show = min(max_numbers, len(corners_2d))

        for i in range(num_to_show):
            center = tuple(corners_2d[i].astype(int))
            cv2.putText(
                image,
                str(i),
                (center[0] + 5, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )

    def visualize_calibration(
        self,
        image: np.ndarray,
        draw_axes: bool = True,
        draw_cube: bool = False,
        draw_camera_frame: bool = False,
    ) -> np.ndarray:
        """
        Enhanced visualization with pattern drawing capabilities.
        This is the main method to call for visualization.
        """
        return self.draw_chessboard_pattern(
            image, draw_axes, draw_cube, draw_camera_frame
        )

    def save_calibration(self, filepath: str) -> bool:
        """Save calibration data to file."""
        if self.calibration_result is None:
            logger.error("No calibration data to save")
            return False

        try:
            calibration_data = {
                "camera_matrix": self.calibration_result.camera_matrix.tolist(),
                "distortion_coefficients": self.calibration_result.distortion_coefficients.tolist(),
                "reprojection_error": self.calibration_result.reprojection_error,
                "image_size": self.calibration_result.image_size,
                "pixels_per_mm": self.calibration_result.pixels_per_mm,
                "calibration_date": self.calibration_result.calibration_date,
                "chessboard_size": self.chessboard_size,
                "square_size_mm": self.square_size_mm,
            }

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

    def get_detection_tips(self) -> List[str]:
        """Get tips for better chessboard detection."""
        return [
            "Ensure the chessboard is printed with high contrast (pure black/white)",
            "Use matte paper to avoid reflections",
            "Make sure the chessboard is completely flat",
            "Provide even lighting without shadows or reflections",
            "Ensure all corners are visible in the image",
            "Keep the chessboard pattern sharp (avoid motion blur)",
            "Try different angles and positions",
            "Check that the pattern size matches your settings",
            "For the pattern shown: use 19x13 inner corners",
            "Square size should match your actual printed size",
        ]

    def process_calibration_images_with_display(
        self, image_paths: List[str]
    ) -> List[np.ndarray]:
        """
        Process multiple calibration images and return visualization results.
        Similar to the OpenCV tutorial approach.
        """
        visualization_results = []

        for i, image_path in enumerate(image_paths):
            try:
                # Read image
                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue

                # Try to add as calibration image
                success = self.add_calibration_image(img)

                # Create visualization
                vis_img = self.draw_chessboard_pattern(
                    img, draw_axes=True, draw_cube=False
                )

                # Add status text
                status = "✓ Added" if success else "✗ Failed"
                cv2.putText(
                    vis_img,
                    f"Image {i+1}: {status}",
                    (10, vis_img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if success else (0, 0, 255),
                    2,
                )

                visualization_results.append(vis_img)

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue

        return visualization_results
