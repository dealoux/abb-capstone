"""YOLO-based defect detection model for app integration."""

import os
import json
import logging
import numpy as np
import cv2
from abbvisionsystem.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class YOLODefectModel(BaseModel):
    """YOLO-based model for multi-object defect detection."""

    def __init__(
        self, model_path="trained_models/yolo_defect_detector/weights/best.pt"
    ):
        super().__init__(model_path)
        self.model = None
        self.categories = {
            0: {"name": "Normal", "id": 0},
            1: {"name": "Defect", "id": 1},
        }

    def load(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO

            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False

            self.model = YOLO(self.model_path)
            self.loaded = True
            logger.info(f"YOLO model loaded from: {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            return False

    def predict(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """Run prediction with YOLO model - Updated to match training pipeline format."""
        if not self.loaded:
            logger.warning("Model not loaded. Call load() first.")
            return None

        try:
            # Handle different input types
            if isinstance(image, str):
                # If image is a path, load it
                image_array = cv2.imread(image)
            elif isinstance(image, np.ndarray):
                # If image is already an array, use it directly
                image_array = image
            else:
                logger.error("Invalid image input type")
                return None

            # Run YOLO inference with specified thresholds
            results = self.model(
                image_array, conf=conf_threshold, iou=iou_threshold, verbose=False
            )[0]

            # Process results - Format compatible with detection page
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                scores = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)

                # Convert to relative coordinates for compatibility
                h, w = image_array.shape[:2]
                boxes_rel = boxes / [w, h, w, h]  # Normalize to [0, 1]

                return {
                    "boxes": boxes_rel,
                    "scores": scores,
                    "classes": classes,
                    "num_detections": len(boxes),
                    "labels": [self.categories[cls]["name"] for cls in classes],
                    "absolute_boxes": boxes,  # Keep absolute coordinates too
                }
            else:
                return {
                    "boxes": np.array([]),
                    "scores": np.array([]),
                    "classes": np.array([]),
                    "num_detections": 0,
                    "labels": [],
                    "absolute_boxes": np.array([]),
                }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None

    def visualize_detections(self, image, detections, threshold=0.25):
        """Draw detection results on image - Updated for compatibility."""
        if isinstance(image, str):
            image_array = cv2.imread(image)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_array = image.copy()

        h, w = image_array.shape[:2]

        if detections["num_detections"] > 0:
            for i in range(detections["num_detections"]):
                if detections["scores"][i] < threshold:
                    continue

                # Use absolute boxes if available, otherwise convert from relative
                if (
                    "absolute_boxes" in detections
                    and len(detections["absolute_boxes"]) > 0
                ):
                    box = detections["absolute_boxes"][i]
                else:
                    # Convert from relative to absolute coordinates
                    box = detections["boxes"][i] * [w, h, w, h]

                x1, y1, x2, y2 = box.astype(int)

                class_id = detections["classes"][i]
                score = detections["scores"][i]

                # Color based on class
                color = (255, 0, 0) if class_id == 1 else (0, 255, 0)  # Red for defect

                # Draw bounding box
                cv2.rectangle(image_array, (x1, y1), (x2, y2), color, 2)

                # Draw label
                class_name = self.categories.get(class_id, {}).get(
                    "name", f"Class {class_id}"
                )
                label = f"{class_name}: {score:.2f}"

                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    image_array,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1,
                )
                cv2.putText(
                    image_array,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        return image_array
