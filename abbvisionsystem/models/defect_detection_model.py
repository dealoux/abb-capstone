"""Object-based defect detection model - simplified version."""

import os
import json
import logging
import numpy as np
import tensorflow as tf
import cv2

logger = logging.getLogger(__name__)


class ObjectDefectDetectionModel:
    """Object detection + defect classification model."""

    def __init__(
        self,
        classifier_path="trained_models/object_defect_classifier.h5",
        class_mapping_path="trained_models/class_mapping.json",
        min_object_area=1000,
        max_object_area=50000,
    ):
        self.model_path = classifier_path
        self.class_mapping_path = class_mapping_path
        self.classifier = None
        self.class_mapping = {}
        self.min_object_area = min_object_area
        self.max_object_area = max_object_area
        self.loaded = False

        # Import object detector - use lazy import to avoid circular imports
        self.object_detector = None

        # Default categories
        self.categories = {
            0: {"name": "Normal", "id": 0},
            1: {"name": "Defect", "id": 1},
        }

    def _init_object_detector(self):
        """Initialize object detector with lazy loading to avoid circular imports."""
        if self.object_detector is None:
            from abbvisionsystem.model_training.object_detector import ObjectDetector

            self.object_detector = ObjectDetector(
                self.min_object_area, self.max_object_area
            )

    def load(self):
        """Load the object classification model."""
        try:
            # Initialize object detector
            self._init_object_detector()

            # Load classifier
            keras_path = self.model_path.replace(".h5", ".keras")
            if os.path.exists(keras_path):
                self.classifier = tf.keras.models.load_model(keras_path)
                logger.info(f"Loaded model from: {keras_path}")
            else:
                self.classifier = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded model from: {self.model_path}")

            # Load class mapping
            if os.path.exists(self.class_mapping_path):
                with open(self.class_mapping_path, "r") as f:
                    self.class_mapping = json.load(f)

                for class_id, class_name in self.class_mapping.items():
                    self.categories[int(class_id)] = {
                        "name": class_name,
                        "id": int(class_id),
                    }

            self.loaded = True
            logger.info("Object defect detection model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def predict(self, image):
        """Run full pipeline: detect objects + classify each object."""
        if not self.loaded:
            logger.warning("Model not loaded. Call load() first.")
            return None

        try:
            # Ensure object detector is initialized
            self._init_object_detector()

            # Step 1: Detect objects
            detected_objects = self.object_detector.detect_objects(image)

            if not detected_objects:
                return {
                    "boxes": np.array([]),
                    "scores": np.array([]),
                    "classes": np.array([]),
                    "num_detections": 0,
                    "object_details": [],
                }

            # Step 2: Classify each detected object
            boxes = []
            scores = []
            classes = []
            object_details = []

            height, width = image.shape[:2]

            for obj in detected_objects:
                # Extract object ROI
                x, y, w, h = obj["bbox"]
                object_roi = image[y : y + h, x : x + w]

                # Classify object
                classification = self._classify_object(object_roi)

                # Convert bbox to normalized coordinates
                norm_bbox = [
                    y / height,  # y1
                    x / width,  # x1
                    (y + h) / height,  # y2
                    (x + w) / width,  # x2
                ]

                boxes.append(norm_bbox)
                scores.append(classification["confidence"])
                classes.append(classification["class_id"])

                object_details.append(
                    {
                        "object_id": obj["id"],
                        "bbox_pixels": obj["bbox"],
                        "bbox_normalized": norm_bbox,
                        "classification": classification,
                        "area": obj["area"],
                    }
                )

            return {
                "boxes": np.array(boxes),
                "scores": np.array(scores),
                "classes": np.array(classes),
                "num_detections": len(detected_objects),
                "object_details": object_details,
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None

    def _classify_object(self, object_roi):
        """Classify a single object ROI."""
        try:
            # Resize and preprocess
            roi_resized = cv2.resize(object_roi, (224, 224))
            roi_normalized = roi_resized.astype("float32") / 255.0
            roi_batch = np.expand_dims(roi_normalized, axis=0)

            # Predict
            prediction = self.classifier.predict(roi_batch, verbose=0)
            score = float(prediction[0][0])
            class_id = 1 if score > 0.5 else 0

            return {
                "class_id": class_id,
                "confidence": score if class_id == 1 else 1.0 - score,
                "raw_score": score,
            }

        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return {"class_id": 0, "confidence": 0.0, "raw_score": 0.0}

    def visualize_detections(self, image, detections, threshold=0.5):
        """Draw object detection and classification results."""
        result = image.copy()

        if detections["num_detections"] == 0:
            return result

        for i in range(detections["num_detections"]):
            score = detections["scores"][i]
            if score < threshold:
                continue

            class_id = detections["classes"][i]

            # Get pixel coordinates
            if "object_details" in detections and i < len(detections["object_details"]):
                x, y, w, h = detections["object_details"][i]["bbox_pixels"]
            else:
                # Fallback: convert from normalized coordinates
                height, width = image.shape[:2]
                y1, x1, y2, x2 = detections["boxes"][i]
                x = int(x1 * width)
                y = int(y1 * height)
                w = int((x2 - x1) * width)
                h = int((y2 - y1) * height)

            # Color based on classification
            color = (0, 0, 255) if class_id == 1 else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)

            # Add labels
            class_name = self.categories.get(class_id, {}).get(
                "name", f"Class {class_id}"
            )
            label = f"{class_name}: {score:.2f}"

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                result, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1
            )

            # Draw label text
            cv2.putText(
                result,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Add object ID
            cv2.putText(
                result,
                f"ID: {i}",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return result
