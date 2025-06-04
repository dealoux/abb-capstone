"""LineMax and SmartLine equivalent tools for defect detection."""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LineType(Enum):
    """Types of lines that can be detected."""

    STRAIGHT = "straight"
    CURVED = "curved"
    EDGE = "edge"
    RIDGE = "ridge"


class LineQuality(Enum):
    """Quality levels for detected lines."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class SmartLineResult:
    """Enhanced line detection result for defect analysis."""

    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    center_point: Tuple[float, float]
    length: float
    angle: float
    straightness: float  # How straight the line is (0-1)
    edge_strength: float  # Strength of the edge
    line_type: LineType
    quality: LineQuality
    expected_line_id: Optional[str]  # If this matches an expected line
    deviation_from_expected: float
    is_defect: bool


@dataclass
class ExpectedLine:
    """Definition of an expected line for defect detection."""

    id: str
    expected_start: Tuple[float, float]
    expected_end: Tuple[float, float]
    tolerance_width: float  # How far from expected position is acceptable
    tolerance_angle: float  # Angular tolerance in degrees
    tolerance_length: float  # Length tolerance as percentage
    min_edge_strength: float
    line_type: LineType


class SmartLineDetector:
    """Advanced line detection for defect analysis (LineMax/SmartLine equivalent)."""

    def __init__(self):
        self.expected_lines = {}
        self.detection_params = {
            "canny_low": 50,
            "canny_high": 150,
            "hough_threshold": 100,
            "min_line_length": 30,
            "max_line_gap": 10,
            "edge_strength_threshold": 0.3,
        }

    def add_expected_line(self, line_def: ExpectedLine):
        """Add an expected line definition for defect detection."""
        self.expected_lines[line_def.id] = line_def
        logger.info(f"Added expected line: {line_def.id}")

    def configure_detection(self, **params):
        """Configure detection parameters."""
        self.detection_params.update(params)

    def detect_smart_lines(
        self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[SmartLineResult]:
        """Detect lines using smart line detection algorithms."""
        if roi:
            x, y, w, h = roi
            search_image = image[y : y + h, x : x + w].copy()
            offset_x, offset_y = x, y
        else:
            search_image = image.copy()
            offset_x, offset_y = 0, 0

        if len(search_image.shape) == 3:
            gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = search_image.copy()

        # Detect lines using multiple methods
        straight_lines = self._detect_straight_lines(gray)
        edge_lines = self._detect_edge_lines(gray)
        ridge_lines = self._detect_ridge_lines(gray)

        # Combine and analyze all detected lines
        all_lines = straight_lines + edge_lines + ridge_lines

        # Adjust coordinates for ROI offset
        for line in all_lines:
            line.start_point = (
                line.start_point[0] + offset_x,
                line.start_point[1] + offset_y,
            )
            line.end_point = (
                line.end_point[0] + offset_x,
                line.end_point[1] + offset_y,
            )
            line.center_point = (
                line.center_point[0] + offset_x,
                line.center_point[1] + offset_y,
            )

        # Analyze for defects
        analyzed_lines = self._analyze_lines_for_defects(all_lines)

        return analyzed_lines

    def _detect_straight_lines(self, image: np.ndarray) -> List[SmartLineResult]:
        """Detect straight lines using advanced Hough transform."""
        lines = []

        # Apply edge detection
        edges = cv2.Canny(
            image,
            self.detection_params["canny_low"],
            self.detection_params["canny_high"],
        )

        # Probabilistic Hough Line Transform
        detected_lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            self.detection_params["hough_threshold"],
            minLineLength=self.detection_params["min_line_length"],
            maxLineGap=self.detection_params["max_line_gap"],
        )

        if detected_lines is not None:
            for line in detected_lines:
                x1, y1, x2, y2 = line[0]

                # Calculate line properties
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                # Calculate edge strength along the line
                edge_strength = self._calculate_edge_strength_along_line(
                    image, (x1, y1), (x2, y2)
                )

                # Calculate straightness (for straight lines, this is always high)
                straightness = 0.95  # Hough lines are inherently straight

                # Determine quality based on edge strength and length
                if edge_strength > 0.8 and length > 50:
                    quality = LineQuality.EXCELLENT
                elif edge_strength > 0.6 and length > 30:
                    quality = LineQuality.GOOD
                elif edge_strength > 0.4:
                    quality = LineQuality.FAIR
                else:
                    quality = LineQuality.POOR

                lines.append(
                    SmartLineResult(
                        start_point=(float(x1), float(y1)),
                        end_point=(float(x2), float(y2)),
                        center_point=center,
                        length=length,
                        angle=angle,
                        straightness=straightness,
                        edge_strength=edge_strength,
                        line_type=LineType.STRAIGHT,
                        quality=quality,
                        expected_line_id=None,
                        deviation_from_expected=0.0,
                        is_defect=False,
                    )
                )

        return lines

    def _detect_edge_lines(self, image: np.ndarray) -> List[SmartLineResult]:
        """Detect edge lines with sub-pixel accuracy."""
        lines = []

        # Use Sobel operators for gradient calculation
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        # Find strong edges
        strong_edges = magnitude > np.percentile(magnitude, 90)

        # Find contours of strong edge regions
        edges_uint8 = (strong_edges * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            edges_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < 10:  # Skip very small contours
                continue

            # Fit line to contour
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

            # Calculate line endpoints
            lefty = int((-x * vy / vx) + y)
            righty = int(((image.shape[1] - x) * vy / vx) + y)

            # Bound the line to image dimensions
            start_point = (0, lefty)
            end_point = (image.shape[1] - 1, righty)

            # Calculate properties
            length = np.sqrt(
                (end_point[0] - start_point[0]) ** 2
                + (end_point[1] - start_point[1]) ** 2
            )
            angle = np.degrees(np.arctan2(vy[0], vx[0]))
            center = (
                (start_point[0] + end_point[0]) / 2,
                (start_point[1] + end_point[1]) / 2,
            )

            # Calculate straightness based on how well points fit the line
            straightness = self._calculate_line_straightness(contour)

            # Calculate average edge strength
            edge_strength = np.mean(magnitude[contour[:, 0, 1], contour[:, 0, 0]])
            edge_strength = edge_strength / 255.0  # Normalize

            # Determine quality
            if straightness > 0.9 and edge_strength > 0.7:
                quality = LineQuality.EXCELLENT
            elif straightness > 0.8 and edge_strength > 0.5:
                quality = LineQuality.GOOD
            elif straightness > 0.6:
                quality = LineQuality.FAIR
            else:
                quality = LineQuality.POOR

            lines.append(
                SmartLineResult(
                    start_point=start_point,
                    end_point=end_point,
                    center_point=center,
                    length=length,
                    angle=angle,
                    straightness=straightness,
                    edge_strength=edge_strength,
                    line_type=LineType.EDGE,
                    quality=quality,
                    expected_line_id=None,
                    deviation_from_expected=0.0,
                    is_defect=False,
                )
            )

        return lines

    def _detect_ridge_lines(self, image: np.ndarray) -> List[SmartLineResult]:
        """Detect ridge lines (bright lines on dark background)."""
        lines = []

        # Apply morphological operations to enhance ridges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))  # Vertical lines
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))  # Horizontal lines
        opened2 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # Combine both directions
        ridge_enhanced = cv2.addWeighted(opened, 0.5, opened2, 0.5, 0)

        # Threshold to get ridge regions
        _, binary = cv2.threshold(
            ridge_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Skip small regions
                continue

            # Fit line to contour
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

            # Get bounding box to determine line endpoints
            rect = cv2.boundingRect(contour)

            # Calculate line endpoints within the bounding box
            if abs(vx[0]) > abs(vy[0]):  # More horizontal
                start_point = (
                    float(rect[0]),
                    float(y[0] - (rect[0] - x[0]) * vy[0] / vx[0]),
                )
                end_point = (
                    float(rect[0] + rect[2]),
                    float(y[0] + (rect[0] + rect[2] - x[0]) * vy[0] / vx[0]),
                )
            else:  # More vertical
                start_point = (
                    float(x[0] - (rect[1] - y[0]) * vx[0] / vy[0]),
                    float(rect[1]),
                )
                end_point = (
                    float(x[0] + (rect[1] + rect[3] - y[0]) * vx[0] / vy[0]),
                    float(rect[1] + rect[3]),
                )

            # Calculate properties
            length = np.sqrt(
                (end_point[0] - start_point[0]) ** 2
                + (end_point[1] - start_point[1]) ** 2
            )
            angle = np.degrees(np.arctan2(vy[0], vx[0]))
            center = (
                (start_point[0] + end_point[0]) / 2,
                (start_point[1] + end_point[1]) / 2,
            )

            # Calculate straightness
            straightness = self._calculate_line_straightness(contour)

            # Calculate edge strength (for ridges, use intensity along the line)
            edge_strength = self._calculate_ridge_strength(
                image, start_point, end_point
            )

            quality = (
                LineQuality.GOOD
                if straightness > 0.8 and edge_strength > 0.6
                else LineQuality.FAIR
            )

            lines.append(
                SmartLineResult(
                    start_point=start_point,
                    end_point=end_point,
                    center_point=center,
                    length=length,
                    angle=angle,
                    straightness=straightness,
                    edge_strength=edge_strength,
                    line_type=LineType.RIDGE,
                    quality=quality,
                    expected_line_id=None,
                    deviation_from_expected=0.0,
                    is_defect=False,
                )
            )

        return lines

    def _analyze_lines_for_defects(
        self, detected_lines: List[SmartLineResult]
    ) -> List[SmartLineResult]:
        """Analyze detected lines for defects compared to expected lines."""
        analyzed_lines = []
        used_expected_lines = set()

        for line in detected_lines:
            best_match_id = None
            min_deviation = float("inf")

            # Find the best matching expected line
            for exp_id, expected_line in self.expected_lines.items():
                if exp_id in used_expected_lines:
                    continue

                # Calculate deviation from expected line
                deviation = self._calculate_line_deviation(line, expected_line)

                if (
                    deviation < min_deviation
                    and deviation < expected_line.tolerance_width
                ):
                    min_deviation = deviation
                    best_match_id = exp_id

            # Update line properties based on analysis
            if best_match_id:
                line.expected_line_id = best_match_id
                line.deviation_from_expected = min_deviation
                used_expected_lines.add(best_match_id)

                # Check if this constitutes a defect
                expected_line = self.expected_lines[best_match_id]
                line.is_defect = (
                    min_deviation > expected_line.tolerance_width
                    or line.edge_strength < expected_line.min_edge_strength
                    or abs(line.angle - self._calculate_expected_angle(expected_line))
                    > expected_line.tolerance_angle
                )
            else:
                # Unexpected line - potential defect
                line.is_defect = True
                line.deviation_from_expected = min_deviation

            analyzed_lines.append(line)

        # Check for missing expected lines
        for exp_id, expected_line in self.expected_lines.items():
            if exp_id not in used_expected_lines:
                # Add a "missing line" entry
                center = (
                    (expected_line.expected_start[0] + expected_line.expected_end[0])
                    / 2,
                    (expected_line.expected_start[1] + expected_line.expected_end[1])
                    / 2,
                )

                missing_line = SmartLineResult(
                    start_point=expected_line.expected_start,
                    end_point=expected_line.expected_end,
                    center_point=center,
                    length=0,  # Missing line has no actual length
                    angle=self._calculate_expected_angle(expected_line),
                    straightness=0,
                    edge_strength=0,
                    line_type=expected_line.line_type,
                    quality=LineQuality.POOR,
                    expected_line_id=exp_id,
                    deviation_from_expected=0,
                    is_defect=True,  # Missing line is a defect
                )
                analyzed_lines.append(missing_line)
                logger.warning(f"Missing expected line: {exp_id}")

        return analyzed_lines

    def _calculate_edge_strength_along_line(
        self, image: np.ndarray, start: Tuple[float, float], end: Tuple[float, float]
    ) -> float:
        """Calculate average edge strength along a line."""
        # Create line profile
        num_points = int(np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2))
        if num_points < 2:
            return 0.0

        x_coords = np.linspace(start[0], end[0], num_points).astype(int)
        y_coords = np.linspace(start[1], end[1], num_points).astype(int)

        # Ensure coordinates are within image bounds
        valid_mask = (
            (x_coords >= 0)
            & (x_coords < image.shape[1])
            & (y_coords >= 0)
            & (y_coords < image.shape[0])
        )

        if not np.any(valid_mask):
            return 0.0

        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        # Calculate gradient along the line
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Sample magnitude along the line
        line_magnitudes = magnitude[y_coords, x_coords]

        # Return normalized average magnitude
        return np.mean(line_magnitudes) / 255.0

    def _calculate_line_straightness(self, contour: np.ndarray) -> float:
        """Calculate how straight a line is based on its contour points."""
        if len(contour) < 3:
            return 0.0

        # Fit a line to the points
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

        # Calculate distances from points to the fitted line
        distances = []
        for point in contour:
            px, py = point[0]
            # Distance from point to line
            distance = abs(
                (vy[0] * px - vx[0] * py + vx[0] * y[0] - vy[0] * x[0])
                / np.sqrt(vx[0] ** 2 + vy[0] ** 2)
            )
            distances.append(distance)

        # Calculate straightness as inverse of average distance
        avg_distance = np.mean(distances)
        max_expected_distance = 5.0  # pixels

        straightness = max(0.0, 1.0 - avg_distance / max_expected_distance)
        return straightness

    def _calculate_ridge_strength(
        self, image: np.ndarray, start: Tuple[float, float], end: Tuple[float, float]
    ) -> float:
        """Calculate ridge strength along a line."""
        # Sample intensity along the line
        num_points = int(np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2))
        if num_points < 2:
            return 0.0

        x_coords = np.linspace(start[0], end[0], num_points).astype(int)
        y_coords = np.linspace(start[1], end[1], num_points).astype(int)

        # Ensure coordinates are within bounds
        valid_mask = (
            (x_coords >= 0)
            & (x_coords < image.shape[1])
            & (y_coords >= 0)
            & (y_coords < image.shape[0])
        )

        if not np.any(valid_mask):
            return 0.0

        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        # Sample intensities
        intensities = image[y_coords, x_coords]

        # Ridge strength is based on how much brighter the line is compared to surroundings
        avg_intensity = np.mean(intensities)

        # Sample surrounding region for comparison
        surrounding_intensities = []
        for i in range(len(x_coords)):
            x, y = x_coords[i], y_coords[i]
            # Sample in perpendicular direction
            for offset in [-3, -2, -1, 1, 2, 3]:
                # Calculate perpendicular direction
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    perp_x = int(x - dy * offset / length)
                    perp_y = int(y + dx * offset / length)

                    if 0 <= perp_x < image.shape[1] and 0 <= perp_y < image.shape[0]:
                        surrounding_intensities.append(image[perp_y, perp_x])

        if surrounding_intensities:
            avg_surrounding = np.mean(surrounding_intensities)
            ridge_strength = (avg_intensity - avg_surrounding) / 255.0
            return max(0.0, min(1.0, ridge_strength))

        return avg_intensity / 255.0

    def _calculate_line_deviation(
        self, detected_line: SmartLineResult, expected_line: ExpectedLine
    ) -> float:
        """Calculate how much a detected line deviates from expected line."""
        # Calculate distance from line endpoints to expected line
        start_dist = self._point_to_line_distance(
            detected_line.start_point,
            expected_line.expected_start,
            expected_line.expected_end,
        )

        end_dist = self._point_to_line_distance(
            detected_line.end_point,
            expected_line.expected_start,
            expected_line.expected_end,
        )

        # Average distance represents line deviation
        return (start_dist + end_dist) / 2

    def _point_to_line_distance(
        self,
        point: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float],
    ) -> float:
        """Calculate distance from a point to a line segment."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Calculate distance using the formula for point to line distance
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        if denominator == 0:
            return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

        return numerator / denominator

    def _calculate_expected_angle(self, expected_line: ExpectedLine) -> float:
        """Calculate the expected angle of a line."""
        dx = expected_line.expected_end[0] - expected_line.expected_start[0]
        dy = expected_line.expected_end[1] - expected_line.expected_start[1]
        return np.degrees(np.arctan2(dy, dx))

    def visualize_smart_lines(
        self,
        image: np.ndarray,
        lines: List[SmartLineResult],
        show_defects_only: bool = False,
    ) -> np.ndarray:
        """Visualize detected lines with defect highlighting."""
        result_image = image.copy()

        for line in lines:
            if show_defects_only and not line.is_defect:
                continue

            # Choose color based on line status
            if line.is_defect:
                color = (0, 0, 255)  # Red for defects
            elif line.quality == LineQuality.EXCELLENT:
                color = (0, 255, 0)  # Green for excellent
            elif line.quality == LineQuality.GOOD:
                color = (0, 255, 255)  # Yellow for good
            else:
                color = (128, 128, 128)  # Gray for fair/poor

            # Draw line
            if line.length > 0:  # Only draw if it's not a missing line
                start = (int(line.start_point[0]), int(line.start_point[1]))
                end = (int(line.end_point[0]), int(line.end_point[1]))
                cv2.line(result_image, start, end, color, 2)

                # Draw endpoints
                cv2.circle(result_image, start, 3, color, -1)
                cv2.circle(result_image, end, 3, color, -1)

            # Add text information
            center = (int(line.center_point[0]), int(line.center_point[1]))
            if line.is_defect:
                text = f"DEFECT: {line.line_type.value}"
                cv2.putText(
                    result_image,
                    text,
                    (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
            elif line.expected_line_id:
                text = f"{line.expected_line_id}: {line.quality.value}"
                cv2.putText(
                    result_image,
                    text,
                    (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

        return result_image
