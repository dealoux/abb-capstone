import os
import json
import logging
import numpy as np
import cv2
import glob
from abbvisionsystem.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class YOLODefectModel(BaseModel):
    """YOLO-based model for multi-object defect detection."""

    def __init__(self, model_path=None):
        # Auto-discover the best trained model if no path provided
        if model_path is None:
            model_path = self._find_best_trained_model()

        super().__init__(model_path)
        self.model = None
        self.categories = {
            0: {"name": "Normal", "id": 0},
            1: {"name": "Defect", "id": 1},
        }

    def _find_best_trained_model(self):
        """Automatically find the best trained YOLO model with enhanced fallback."""
        # Search paths in order of preference - prioritize working models
        search_paths = [
            # Legacy working models first (from old implementation)
            "trained_models/yolo_defect_detector/weights/best.pt",
            "trained_models/yolo_defect_detector/weights/last.pt",
            # From new training pipeline (yolo11s training)
            "yolo11_trained_models/yolo11s_defect_detector/weights/best.pt",
            "yolo11_trained_models/yolo11n_defect_detector/weights/best.pt",
            "yolo11_trained_models/yolo11m_defect_detector/weights/best.pt",
            # Other legacy paths
            "trained_models/yolo11s_defect_detector/weights/best.pt",
            "trained_models/yolo11n_defect_detector/weights/best.pt",
            "trained_models/yolo11m_defect_detector/weights/best.pt",
            # CPU training paths
            "cpu_trained_models/yolo11s_cpu_defect_detector/weights/best.pt",
            "cpu_trained_models/yolo11n_cpu_defect_detector/weights/best.pt",
            # Direct paths
            "best.pt",
            "yolo_best.pt",
            "yolo11s_best.pt",
            # Fallback to pretrained (these should work reliably)
            "yolov8n.pt",  # More stable than yolo11s
            "yolo11s.pt",
        ]

        for path in search_paths:
            if os.path.exists(path):
                logger.info(f"Found trained model: {path}")
                return path

        # If no trained model found, return default path that will trigger download
        logger.warning("No trained model found, will attempt to use pretrained model")
        return "yolov8n.pt"  # Use more stable v8 as fallback

    def get_available_models(self):
        """Get list of all available trained models."""
        models = []

        # Search for all possible trained models
        search_patterns = [
            "trained_models/*/weights/best.pt",  # Legacy models first
            "yolo11_trained_models/*/weights/best.pt",
            "cpu_trained_models/*/weights/best.pt",
        ]

        for pattern in search_patterns:
            found_models = glob.glob(pattern)
            for model_path in found_models:
                if os.path.exists(model_path):
                    # Extract model name from path
                    model_name = model_path.split("/")[-3]  # Get directory name
                    models.append(
                        {
                            "name": model_name,
                            "path": model_path,
                            "size_mb": os.path.getsize(model_path) / (1024 * 1024),
                            "modified": os.path.getmtime(model_path),
                        }
                    )

        # Sort by modification time (newest first)
        models.sort(key=lambda x: x["modified"], reverse=True)
        return models

    def load(self):
        """Load the YOLO model with enhanced error handling."""
        try:
            from ultralytics import YOLO

            logger.info(f"Attempting to load YOLO model from: {self.model_path}")

            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")

                # Try to find alternative model
                alternative_path = self._find_best_trained_model()
                if alternative_path != self.model_path and os.path.exists(
                    alternative_path
                ):
                    logger.info(f"Using alternative model: {alternative_path}")
                    self.model_path = alternative_path
                elif self.model_path in ["yolov8n.pt", "yolo11s.pt"]:
                    # For pretrained models, let YOLO download them
                    logger.info(f"Downloading pretrained model: {self.model_path}")
                else:
                    logger.error(f"No valid model found. Tried: {self.model_path}")
                    return False

            self.model = YOLO(self.model_path)
            self.loaded = True

            # Log model info
            if os.path.exists(self.model_path):
                model_size = os.path.getsize(self.model_path) / (1024 * 1024)
                logger.info(f"YOLO model loaded successfully!")
                logger.info(f"  Path: {self.model_path}")
                logger.info(f"  Size: {model_size:.1f} MB")
            else:
                logger.info(f"YOLO model downloaded and loaded: {self.model_path}")

            # Show available models for reference
            available_models = self.get_available_models()
            if available_models:
                logger.info(f"  Available trained models: {len(available_models)}")
                for model in available_models[:3]:  # Show top 3 newest
                    logger.info(f"    - {model['name']} ({model['size_mb']:.1f} MB)")

            return True

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            return False

    def predict(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """Run prediction with YOLO model - Enhanced for multi-object detection."""
        if not self.loaded:
            logger.warning("Model not loaded. Call load() first.")
            return None

        try:
            # Handle different input types with better error checking
            if isinstance(image, str):
                if not os.path.exists(image):
                    logger.error(f"Image file not found: {image}")
                    return self._empty_result()
                image_array = cv2.imread(image)
                if image_array is None:
                    logger.error(f"Could not load image from path: {image}")
                    return self._empty_result()
            elif isinstance(image, np.ndarray):
                image_array = image.copy()
            else:
                logger.error(f"Invalid image input type: {type(image)}")
                return self._empty_result()

            # Ensure image is valid
            if image_array.size == 0:
                logger.error("Empty image provided")
                return self._empty_result()

            # Run YOLO inference with enhanced error handling
            try:
                results = self.model(
                    image_array,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                    device="cpu",  # Force CPU for compatibility
                )[0]
            except Exception as inference_error:
                logger.error(f"YOLO inference failed: {inference_error}")
                return self._empty_result()

            # Process results with enhanced validation
            if results.boxes is not None and len(results.boxes) > 0:
                try:
                    boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    scores = results.boxes.conf.cpu().numpy()
                    classes = results.boxes.cls.cpu().numpy().astype(int)

                    # Validate arrays
                    if len(boxes) == 0 or len(scores) == 0 or len(classes) == 0:
                        logger.warning("Empty detection arrays")
                        return self._empty_result()

                    # Convert to relative coordinates for compatibility
                    h, w = image_array.shape[:2]
                    if h == 0 or w == 0:
                        logger.error("Invalid image dimensions")
                        return self._empty_result()

                    boxes_rel = boxes / [w, h, w, h]  # Normalize to [0, 1]

                    # Create labels with validation
                    labels = []
                    for cls in classes:
                        if cls in self.categories:
                            labels.append(self.categories[cls]["name"])
                        else:
                            labels.append(f"Class {cls}")
                            logger.warning(f"Unknown class ID: {cls}")

                    detection_result = {
                        "boxes": boxes_rel,
                        "scores": scores,
                        "classes": classes,
                        "num_detections": len(boxes),
                        "labels": labels,
                        "absolute_boxes": boxes,
                    }

                    logger.info(f"Successfully detected {len(boxes)} objects")
                    return detection_result

                except Exception as processing_error:
                    logger.error(f"Result processing failed: {processing_error}")
                    return self._empty_result()
            else:
                logger.info("No objects detected")
                return self._empty_result()

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return self._empty_result()

    def _empty_result(self):
        """Return empty detection result."""
        return {
            "boxes": np.array([]),
            "scores": np.array([]),
            "classes": np.array([]),
            "num_detections": 0,
            "labels": [],
            "absolute_boxes": np.array([]),
        }

    def visualize_detections(self, image, detections, threshold=0.25):
        """Draw detection results on image - Updated for compatibility."""
        if isinstance(image, str):
            image_array = cv2.imread(image)
            if image_array is not None:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                logger.error(f"Could not load image from path: {image}")
                return None
        else:
            image_array = image.copy()

        h, w = image_array.shape[:2]

        # Handle case where detections might be None or empty
        if detections is None or detections.get("num_detections", 0) == 0:
            return image_array

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

    def get_model_info(self):
        """Get information about the currently loaded model."""
        info = {
            "model_path": self.model_path,
            "loaded": self.loaded,
            "categories": self.categories,
        }

        if os.path.exists(self.model_path):
            info["file_size_mb"] = os.path.getsize(self.model_path) / (1024 * 1024)
            info["file_exists"] = True
        else:
            info["file_exists"] = False

        return info
