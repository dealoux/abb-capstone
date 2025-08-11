"""Object detection utilities for industrial vision."""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Traditional computer vision object detector."""

    def __init__(self, min_area=1000, max_area=50000):
        self.min_area = min_area
        self.max_area = max_area

    def detect_objects(self, image):
        """Detect objects using contour detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold to create binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected_objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Add padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)

                detected_objects.append(
                    {"id": i, "bbox": (x, y, w, h), "contour": contour, "area": area}
                )

        return detected_objects

    def extract_object_roi(self, image, bbox):
        """Extract region of interest from image."""
        x, y, w, h = bbox
        return image[y : y + h, x : x + w]

    def visualize_detections(self, image, objects):
        """Visualize detected objects on image."""
        result = image.copy()

        for obj in objects:
            x, y, w, h = obj["bbox"]
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                result,
                f"ID: {obj['id']}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return result
