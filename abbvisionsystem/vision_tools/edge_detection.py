"""Edge detection and line measurement tools."""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Edge:
    """Represents a detected edge."""

    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    length: float
    angle: float
    strength: float
    points: List[Tuple[float, float]]


@dataclass
class Line:
    """Represents a detected line."""

    rho: float
    theta: float
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    length: float
    angle_degrees: float


class EdgeDetector:
    """Advanced edge detection and line measurement."""

    def __init__(self):
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 100
        self.min_line_length = 50
        self.max_line_gap = 10

    def configure_canny(self, low_threshold: int = 50, high_threshold: int = 150):
        """Configure Canny edge detection parameters."""
        self.canny_low = low_threshold
        self.canny_high = high_threshold

    def configure_hough(
        self, threshold: int = 100, min_line_length: int = 50, max_line_gap: int = 10
    ):
        """Configure Hough line detection parameters."""
        self.hough_threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def detect_edges_canny(self, image: np.ndarray, blur_kernel: int = 5) -> np.ndarray:
        """Detect edges using Canny edge detector."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        if blur_kernel > 0:
            blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        else:
            blurred = gray

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        return edges

    def detect_lines_hough(self, image: np.ndarray) -> List[Line]:
        """Detect lines using Hough Line Transform."""
        edges = self.detect_edges_canny(image)

        # Standard Hough Line Transform
        lines_std = cv2.HoughLines(edges, 1, np.pi / 180, self.hough_threshold)

        # Probabilistic Hough Line Transform for line segments
        lines_p = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        detected_lines = []

        # Process probabilistic lines (more useful for measurements)
        if lines_p is not None:
            for line in lines_p:
                x1, y1, x2, y2 = line[0]

                # Calculate line properties
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)

                # Convert to polar coordinates for consistency
                rho = x1 * np.cos(angle_rad) + y1 * np.sin(angle_rad)
                theta = angle_rad

                detected_lines.append(
                    Line(
                        rho=rho,
                        theta=theta,
                        start_point=(float(x1), float(y1)),
                        end_point=(float(x2), float(y2)),
                        length=length,
                        angle_degrees=angle_deg,
                    )
                )

        return detected_lines

    def detect_edges_contour(
        self, image: np.ndarray, binary_threshold: int = 127
    ) -> List[Edge]:
        """Detect edges using contour analysis."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply threshold
        _, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        edges = []
        for contour in contours:
            if len(contour) < 2:
                continue

            # Approximate contour to get dominant edges
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Extract edges from approximated contour
            for i in range(len(approx)):
                start_idx = i
                end_idx = (i + 1) % len(approx)

                start_point = (
                    float(approx[start_idx][0][0]),
                    float(approx[start_idx][0][1]),
                )
                end_point = (float(approx[end_idx][0][0]), float(approx[end_idx][0][1]))

                # Calculate edge properties
                length = np.sqrt(
                    (end_point[0] - start_point[0]) ** 2
                    + (end_point[1] - start_point[1]) ** 2
                )
                angle = np.degrees(
                    np.arctan2(
                        end_point[1] - start_point[1], end_point[0] - start_point[0]
                    )
                )

                # Extract points along the edge
                points = []
                num_points = max(2, int(length / 5))  # Point every 5 pixels
                for t in np.linspace(0, 1, num_points):
                    x = start_point[0] + t * (end_point[0] - start_point[0])
                    y = start_point[1] + t * (end_point[1] - start_point[1])
                    points.append((x, y))

                edges.append(
                    Edge(
                        start_point=start_point,
                        end_point=end_point,
                        length=length,
                        angle=angle,
                        strength=1.0,  # Could calculate based on gradient
                        points=points,
                    )
                )

        return edges

    def measure_distances(self, points: List[Tuple[float, float]]) -> List[float]:
        """Measure distances between consecutive points."""
        distances = []
        for i in range(len(points) - 1):
            dist = np.sqrt(
                (points[i + 1][0] - points[i][0]) ** 2
                + (points[i + 1][1] - points[i][1]) ** 2
            )
            distances.append(dist)
        return distances

    def measure_angles(self, lines: List[Line]) -> List[float]:
        """Measure angles between consecutive lines."""
        angles = []
        for i in range(len(lines) - 1):
            angle_diff = abs(lines[i + 1].angle_degrees - lines[i].angle_degrees)
            # Normalize to 0-180 range
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            angles.append(angle_diff)
        return angles

    def find_parallel_lines(
        self, lines: List[Line], angle_tolerance: float = 5.0
    ) -> List[List[int]]:
        """Find groups of parallel lines."""
        parallel_groups = []
        used_indices = set()

        for i, line1 in enumerate(lines):
            if i in used_indices:
                continue

            group = [i]
            used_indices.add(i)

            for j, line2 in enumerate(lines):
                if j <= i or j in used_indices:
                    continue

                # Calculate angle difference
                angle_diff = abs(line1.angle_degrees - line2.angle_degrees)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                if angle_diff <= angle_tolerance:
                    group.append(j)
                    used_indices.add(j)

            if len(group) > 1:
                parallel_groups.append(group)

        return parallel_groups

    def find_perpendicular_lines(
        self, lines: List[Line], angle_tolerance: float = 5.0
    ) -> List[Tuple[int, int]]:
        """Find pairs of perpendicular lines."""
        perpendicular_pairs = []

        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines):
                if j <= i:
                    continue

                # Calculate angle difference
                angle_diff = abs(line1.angle_degrees - line2.angle_degrees)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                # Check if perpendicular (90 degrees)
                if abs(angle_diff - 90) <= angle_tolerance:
                    perpendicular_pairs.append((i, j))

        return perpendicular_pairs

    def visualize_edges(
        self, image: np.ndarray, edges: List[Edge], show_info: bool = True
    ) -> np.ndarray:
        """Visualize detected edges on the image."""
        result_image = image.copy()

        for i, edge in enumerate(edges):
            # Draw edge line
            start = (int(edge.start_point[0]), int(edge.start_point[1]))
            end = (int(edge.end_point[0]), int(edge.end_point[1]))
            cv2.line(result_image, start, end, (0, 255, 0), 2)

            # Draw start and end points
            cv2.circle(result_image, start, 3, (255, 0, 0), -1)
            cv2.circle(result_image, end, 3, (0, 0, 255), -1)

            # Add information text
            if show_info:
                mid_x = int((edge.start_point[0] + edge.end_point[0]) / 2)
                mid_y = int((edge.start_point[1] + edge.end_point[1]) / 2)
                info_text = f"#{i+1} L:{edge.length:.1f} A:{edge.angle:.1f}°"
                cv2.putText(
                    result_image,
                    info_text,
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

        return result_image

    def visualize_lines(
        self, image: np.ndarray, lines: List[Line], show_info: bool = True
    ) -> np.ndarray:
        """Visualize detected lines on the image."""
        result_image = image.copy()

        for i, line in enumerate(lines):
            # Draw line
            start = (int(line.start_point[0]), int(line.start_point[1]))
            end = (int(line.end_point[0]), int(line.end_point[1]))
            cv2.line(result_image, start, end, (0, 255, 255), 2)

            # Add information text
            if show_info:
                mid_x = int((line.start_point[0] + line.end_point[0]) / 2)
                mid_y = int((line.start_point[1] + line.end_point[1]) / 2)
                info_text = f"#{i+1} L:{line.length:.1f} A:{line.angle_degrees:.1f}°"
                cv2.putText(
                    result_image,
                    info_text,
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1,
                )

        return result_image
