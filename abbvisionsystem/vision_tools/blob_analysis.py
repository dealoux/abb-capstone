"""Blob analysis for detecting and measuring circular/elliptical features."""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Blob:
    """Represents a detected blob."""

    center_x: float
    center_y: float
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    angle: float
    width: float
    height: float
    contour: np.ndarray
    bounding_rect: Tuple[int, int, int, int]


class BlobAnalyzer:
    """Advanced blob detection and analysis."""

    def __init__(self):
        self.detector_params = cv2.SimpleBlobDetector_Params()
        self._setup_default_params()
        self.detector = cv2.SimpleBlobDetector_create(self.detector_params)

    def _setup_default_params(self):
        """Setup default blob detector parameters."""
        # Filter by Area
        self.detector_params.filterByArea = True
        self.detector_params.minArea = 50
        self.detector_params.maxArea = 50000

        # Filter by Circularity
        self.detector_params.filterByCircularity = True
        self.detector_params.minCircularity = 0.1

        # Filter by Convexity
        self.detector_params.filterByConvexity = True
        self.detector_params.minConvexity = 0.5

        # Filter by Inertia
        self.detector_params.filterByInertia = True
        self.detector_params.minInertiaRatio = 0.01

    def configure_detector(
        self,
        min_area: float = 50,
        max_area: float = 50000,
        min_circularity: float = 0.1,
        min_convexity: float = 0.5,
        min_inertia: float = 0.01,
    ) -> None:
        """Configure blob detector parameters."""
        self.detector_params.minArea = min_area
        self.detector_params.maxArea = max_area
        self.detector_params.minCircularity = min_circularity
        self.detector_params.minConvexity = min_convexity
        self.detector_params.minInertiaRatio = min_inertia

        # Recreate detector with new parameters
        self.detector = cv2.SimpleBlobDetector_create(self.detector_params)

    def detect_blobs_simple(self, image: np.ndarray) -> List[Blob]:
        """Simple blob detection using OpenCV's SimpleBlobDetector."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detect blobs
        keypoints = self.detector.detect(gray)

        blobs = []
        for kp in keypoints:
            # Create a simple blob representation
            blob = Blob(
                center_x=kp.pt[0],
                center_y=kp.pt[1],
                area=np.pi * (kp.size / 2) ** 2,
                perimeter=2 * np.pi * (kp.size / 2),
                circularity=1.0,  # Assume circular for SimpleBlobDetector
                aspect_ratio=1.0,
                angle=kp.angle,
                width=kp.size,
                height=kp.size,
                contour=np.array([]),  # Empty for simple detection
                bounding_rect=(
                    int(kp.pt[0] - kp.size / 2),
                    int(kp.pt[1] - kp.size / 2),
                    int(kp.size),
                    int(kp.size),
                ),
            )
            blobs.append(blob)

        return blobs

    def detect_blobs_advanced(
        self,
        image: np.ndarray,
        binary_threshold: int = 127,
        min_area: float = 50,
        max_area: float = 50000,
    ) -> List[Blob]:
        """Advanced blob detection using contour analysis."""
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

        blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            # Calculate blob properties
            perimeter = cv2.arcLength(contour, True)

            # Circularity
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0

            # Moments for center calculation
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x = M["m10"] / M["m00"]
                center_y = M["m01"] / M["m00"]
            else:
                center_x, center_y = 0, 0

            # Fit ellipse for orientation and dimensions
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (center_x_ellipse, center_y_ellipse), (width, height), angle = ellipse
                center_x, center_y = center_x_ellipse, center_y_ellipse
                aspect_ratio = max(width, height) / max(min(width, height), 1)
            else:
                # Use bounding rectangle as fallback
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w / 2, y + h / 2
                width, height = w, h
                angle = 0
                aspect_ratio = max(w, h) / max(min(w, h), 1)

            # Bounding rectangle
            bounding_rect = cv2.boundingRect(contour)

            blob = Blob(
                center_x=center_x,
                center_y=center_y,
                area=area,
                perimeter=perimeter,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                angle=angle,
                width=width,
                height=height,
                contour=contour,
                bounding_rect=bounding_rect,
            )
            blobs.append(blob)

        return blobs

    def detect_circles_hough(
        self,
        image: np.ndarray,
        min_radius: int = 10,
        max_radius: int = 100,
        min_distance: int = 50,
    ) -> List[Blob]:
        """Detect circular blobs using Hough Circle Transform."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_distance,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        blobs = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for x, y, r in circles:
                area = np.pi * r * r
                perimeter = 2 * np.pi * r

                blob = Blob(
                    center_x=float(x),
                    center_y=float(y),
                    area=area,
                    perimeter=perimeter,
                    circularity=1.0,  # Perfect circle
                    aspect_ratio=1.0,
                    angle=0.0,
                    width=2 * r,
                    height=2 * r,
                    contour=np.array(
                        []
                    ),  # Could generate approximate contour if needed
                    bounding_rect=(x - r, y - r, 2 * r, 2 * r),
                )
                blobs.append(blob)

        return blobs

    def filter_blobs(
        self,
        blobs: List[Blob],
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
        min_circularity: Optional[float] = None,
        max_aspect_ratio: Optional[float] = None,
    ) -> List[Blob]:
        """Filter blobs based on various criteria."""
        filtered_blobs = []

        for blob in blobs:
            # Apply filters
            if min_area is not None and blob.area < min_area:
                continue
            if max_area is not None and blob.area > max_area:
                continue
            if min_circularity is not None and blob.circularity < min_circularity:
                continue
            if max_aspect_ratio is not None and blob.aspect_ratio > max_aspect_ratio:
                continue

            filtered_blobs.append(blob)

        return filtered_blobs

    def visualize_blobs(
        self, image: np.ndarray, blobs: List[Blob], show_info: bool = True
    ) -> np.ndarray:
        """Draw detected blobs on the image."""
        result_image = image.copy()

        for i, blob in enumerate(blobs):
            # Draw center point
            center = (int(blob.center_x), int(blob.center_y))
            cv2.circle(result_image, center, 3, (0, 255, 0), -1)

            # Draw bounding rectangle
            x, y, w, h = blob.bounding_rect
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw contour if available
            if len(blob.contour) > 0:
                cv2.drawContours(result_image, [blob.contour], -1, (0, 255, 255), 2)

            # Add text information
            if show_info:
                info_text = f"#{i+1} A:{blob.area:.0f} C:{blob.circularity:.2f}"
                cv2.putText(
                    result_image,
                    info_text,
                    (int(blob.center_x + 10), int(blob.center_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        return result_image

    def measure_blob_distances(self, blobs: List[Blob]) -> List[Dict]:
        """Calculate distances between all blob pairs."""
        distances = []

        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                blob1, blob2 = blobs[i], blobs[j]

                # Calculate Euclidean distance
                distance = np.sqrt(
                    (blob1.center_x - blob2.center_x) ** 2
                    + (blob1.center_y - blob2.center_y) ** 2
                )

                distances.append(
                    {
                        "blob1_index": i,
                        "blob2_index": j,
                        "distance": distance,
                        "blob1_center": (blob1.center_x, blob1.center_y),
                        "blob2_center": (blob2.center_x, blob2.center_y),
                    }
                )

        return distances
