import os
import logging
from typing import Dict, List
from datetime import datetime
import numpy as np
import cv2
import glob
from abbvisionsystem.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class YOLODefectModel(BaseModel):
    """YOLO-based model for multi-object defect detection."""

    def __init__(self, model_path=None):
        self.model = None
        self.categories = {
            0: {"name": "Normal", "id": 0},
            1: {"name": "Defect", "id": 1},
        }
        super().__init__(model_path)

    @classmethod
    def get_available_models(cls) -> List[Dict]:
        """Get list of all available YOLO models."""
        models = []

        search_patterns = cls.get_default_search_patterns()
        found_models = set()  # Avoid duplicates

        for pattern in search_patterns:
            for model_path in glob.glob(pattern):
                if os.path.isfile(model_path) and model_path not in found_models:
                    try:
                        model_info = cls._extract_model_info(model_path)
                        models.append(model_info)
                        found_models.add(model_path)
                    except Exception as e:
                        logger.warning(f"Error processing model {model_path}: {e}")
                        continue

        # Sort by modification time (newest first)
        models.sort(key=lambda x: x.get("modified_timestamp", 0), reverse=True)
        return models

    @classmethod
    def get_default_search_patterns(cls) -> List[str]:
        """Get default search patterns for YOLO models."""
        return [
            "trained_models/*/weights/best.pt",
            "trained_models/*/weights/last.pt",
            "trained_models/*.pt",
            "models/*/best.pt",
            "models/*/last.pt",
            "weights/*.pt",
            "best.pt",
            "last.pt",
        ]

    @classmethod
    def get_fallback_models(cls) -> List[str]:
        """Get fallback YOLO models."""
        return [
            "yolov8n.pt",  # Most stable
            "yolov8s.pt",
            "yolo11s.pt",
        ]

    @classmethod
    def get_pretrained_models(cls) -> List[Dict]:
        """Get available pretrained YOLO models."""
        return [
            {
                "name": "YOLOv8n",
                "path": "yolov8n.pt",
                "description": "Nano - Fastest, lowest accuracy",
                "category": "pretrained",
            },
            {
                "name": "YOLOv8s",
                "path": "yolov8s.pt",
                "description": "Small - Good balance of speed and accuracy",
                "category": "pretrained",
            },
            {
                "name": "YOLOv8m",
                "path": "yolov8m.pt",
                "description": "Medium - Higher accuracy, slower",
                "category": "pretrained",
            },
            {
                "name": "YOLOv8l",
                "path": "yolov8l.pt",
                "description": "Large - High accuracy, slower",
                "category": "pretrained",
            },
            {
                "name": "YOLOv8x",
                "path": "yolov8x.pt",
                "description": "Extra Large - Highest accuracy, slowest",
                "category": "pretrained",
            },
            {
                "name": "YOLO11s",
                "path": "yolo11s.pt",
                "description": "YOLO11 Small - Latest version",
                "category": "pretrained",
            },
            {
                "name": "YOLO11m",
                "path": "yolo11m.pt",
                "description": "YOLO11 Medium - Latest version",
                "category": "pretrained",
            },
        ]

    @classmethod
    def _extract_model_info(cls, model_path: str) -> Dict:
        """Extract information about a model file."""
        model_name = cls._extract_model_name(model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        modified_time = os.path.getmtime(model_path)
        date_str = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M")

        # Determine model type from path/name
        model_type = "Custom"
        if "enhanced" in model_path.lower():
            model_type = "Enhanced YOLO"
        elif "defect" in model_path.lower():
            model_type = "Defect Detection"
        elif "best" in model_path.lower():
            model_type = "Best Checkpoint"
        elif "last" in model_path.lower():
            model_type = "Latest Checkpoint"

        # Try to extract training info
        accuracy, epochs = cls._extract_training_info(model_path)

        return {
            "name": model_name,
            "path": model_path,
            "size_mb": model_size_mb,
            "modified_timestamp": modified_time,
            "date_str": date_str,
            "model_type": model_type,
            "accuracy": accuracy,
            "epochs": epochs,
            "category": "trained",
        }

    @classmethod
    def _extract_model_name(cls, model_path: str) -> str:
        """Extract a readable model name from the file path."""
        dir_name = os.path.basename(os.path.dirname(model_path))
        file_name = os.path.basename(model_path)

        if dir_name and dir_name not in ["weights", "models", "."]:
            model_name = dir_name.replace("_", " ").title()
            if file_name != "best.pt":
                model_name += f" ({file_name})"
        else:
            model_name = file_name.replace(".pt", "").replace("_", " ").title()

        return model_name

    @classmethod
    def _extract_training_info(cls, model_path: str) -> tuple:
        """Try to extract training information from model file or associated files."""
        accuracy = "Unknown"
        epochs = "Unknown"

        try:
            # Look for results files
            model_dir = os.path.dirname(model_path)
            parent_dir = os.path.dirname(model_dir)

            for results_file in ["results.csv", "results.txt"]:
                results_path = os.path.join(parent_dir, results_file)
                if os.path.exists(results_path):
                    try:
                        with open(results_path, "r") as f:
                            lines = f.readlines()
                            if lines:
                                last_line = lines[-1].strip()
                                if "," in last_line:
                                    parts = last_line.split(",")
                                    if len(parts) > 1:
                                        epochs = parts[0].strip()
                                    if len(parts) > 6:
                                        accuracy = f"{float(parts[6].strip()):.3f}"
                        break
                    except Exception:
                        continue

            # Try to load model metadata
            try:
                import torch

                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location="cpu")
                    if isinstance(checkpoint, dict):
                        if "epoch" in checkpoint:
                            epochs = str(checkpoint["epoch"])
                        if "best_fitness" in checkpoint:
                            accuracy = f"{checkpoint['best_fitness']:.3f}"
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Could not extract training info for {model_path}: {e}")

        return accuracy, epochs

    @classmethod
    def validate_model(cls, model_path: str) -> bool:
        """Validate that a file is a valid YOLO model."""
        try:
            if not model_path.endswith(".pt"):
                return False

            file_size = os.path.getsize(model_path)
            if file_size < 1024 * 1024 or file_size > 500 * 1024 * 1024:
                return False

            import torch

            try:
                checkpoint = torch.load(model_path, map_location="cpu")
                return isinstance(checkpoint, dict)
            except:
                return False

        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False

    def load(self):
        """Load the YOLO model with enhanced error handling."""
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model from: {self.model_path}")

            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")

                # Try to find alternative
                alternative_path = self._auto_discover_model()
                if alternative_path != self.model_path and os.path.exists(
                    alternative_path
                ):
                    logger.info(f"Using alternative model: {alternative_path}")
                    self.model_path = alternative_path
                elif self.model_path in self.get_fallback_models():
                    logger.info(f"Downloading pretrained model: {self.model_path}")
                else:
                    logger.error(f"No valid model found: {self.model_path}")
                    return False

            self.model = YOLO(self.model_path)
            self.loaded = True

            logger.info(f"YOLO model loaded successfully: {self.model_path}")
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
