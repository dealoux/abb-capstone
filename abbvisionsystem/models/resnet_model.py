"""Defect detection model implementation for industrial inspection."""

import os
import glob
import json
import logging
from datetime import datetime
import numpy as np
import tensorflow as tf
import cv2
from abbvisionsystem.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ResnetModel(BaseModel):
    """Model for industrial defect detection/classification."""

    def __init__(
        self,
        model_path: str | None = None,
        class_mapping_path: str = "trained_models/class_mapping.json",
    ):
        """Initialize classification model.

        If a model_path is provided and exists, use it. Otherwise, trigger
        BaseModel auto-discovery to locate the best available classification
        model in the workspace.
        """
        # Trigger auto-discovery when path is None or doesn't exist
        use_path = model_path if (model_path and os.path.exists(model_path)) else None
        super().__init__(use_path)

        # If BaseModel auto-discovery selected a path, prefer that
        resolved_path = (
            self.model_path if getattr(self, "model_path", None) else model_path
        )

        self.class_mapping_path = class_mapping_path
        self.model = None
        self.class_mapping = {}
        # Compute model_info_path from resolved model path or fallback directory
        info_base_dir = (
            os.path.dirname(resolved_path)
            if resolved_path
            else os.path.dirname(class_mapping_path)
        )
        info_base_name = os.path.splitext(
            os.path.basename(resolved_path or "resnet_defect_classifier")
        )[0]
        self.model_info_path = os.path.join(
            info_base_dir, f"{info_base_name}_info.json"
        )

        # Default categories (will be overridden by class_mapping if available)
        self.categories = {
            0: {"name": "Normal", "id": 0},
            1: {"name": "Defect", "id": 1},
        }

    # ---- BaseModel abstract methods (discovery helpers) ----
    @classmethod
    def get_default_search_patterns(cls):
        """Default locations/patterns for classification models."""
        return [
            # Preferred enhanced classifier in repo
            "trained_models/enhanced_resnet_defect_classifier.keras",
            "trained_models/enhanced_resnet_defect_classifier.h5",
            # Generic patterns
            "trained_models/*.keras",
            "trained_models/*.h5",
            "trained_models/*/*.keras",
            "trained_models/*/*.h5",
            # Possible root-level artifacts
            "*.keras",
            "*.h5",
        ]

    @classmethod
    def get_fallback_models(cls):
        """Fallback models to try when discovery finds nothing."""
        return [
            "trained_models/enhanced_resnet_defect_classifier.keras",
            "trained_models/enhanced_resnet_defect_classifier.h5",
            "trained_models/final_defect_model.h5",
        ]

    @classmethod
    def get_available_models(cls):
        """Discover available classification models.

        Returns a list of dicts with at least: name, path, size_mb,
        modified_timestamp, date_str, model_type.
        """
        models: list[dict] = []
        found = set()
        for pattern in cls.get_default_search_patterns():
            for path in glob.glob(pattern):
                if (
                    os.path.isfile(path)
                    and path not in found
                    and path.endswith((".keras", ".h5"))
                ):
                    try:
                        size_mb = os.path.getsize(path) / (1024 * 1024)
                        mtime = os.path.getmtime(path)
                        date_str = datetime.fromtimestamp(mtime).strftime(
                            "%Y-%m-%d %H:%M"
                        )
                        name = os.path.basename(path).replace("_", " ")
                        models.append(
                            {
                                "name": name,
                                "path": path,
                                "size_mb": size_mb,
                                "modified_timestamp": mtime,
                                "date_str": date_str,
                                "model_type": "ResNet Classification",
                            }
                        )
                        found.add(path)
                    except Exception as e:
                        logger.debug(f"Skipping candidate model {path}: {e}")
                        continue

        # Sort newest first
        models.sort(key=lambda x: x.get("modified_timestamp", 0), reverse=True)
        return models

    def load(self):
        """Load the defect detection model supporting both .h5 and .keras formats."""
        try:
            # First check if model_info.json exists and contains keras_format_path
            keras_model_path = None
            if os.path.exists(self.model_info_path):
                try:
                    with open(self.model_info_path, "r") as f:
                        model_info = json.load(f)
                        if "keras_format_path" in model_info:
                            keras_model_path = model_info["keras_format_path"]
                            logger.info(
                                f"Found keras model path in info file: {keras_model_path}"
                            )
                except Exception as e:
                    logger.warning(f"Could not parse model info file: {str(e)}")

            # Try loading model in order of preference:
            # 1. keras_format_path from model_info if available
            # 2. original path with .keras extension
            # 3. original h5 path
            model_loaded = False

            # Try the keras path from model_info first
            if keras_model_path and os.path.exists(keras_model_path):
                try:
                    logger.info(
                        f"Attempting to load model from keras path: {keras_model_path}"
                    )
                    self.model = tf.keras.models.load_model(
                        keras_model_path, compile=False
                    )
                    model_loaded = True
                    logger.info(f"Successfully loaded model from: {keras_model_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to load keras model from {keras_model_path}: {str(e)}"
                    )

            # Try with .keras extension if original path has .h5
            if not model_loaded and self.model_path.endswith(".h5"):
                keras_path = self.model_path.replace(".h5", ".keras")
                if os.path.exists(keras_path):
                    try:
                        logger.info(f"Attempting to load model from: {keras_path}")
                        self.model = tf.keras.models.load_model(
                            keras_path, compile=False
                        )
                        model_loaded = True
                        logger.info(f"Successfully loaded model from: {keras_path}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to load keras model from {keras_path}: {str(e)}"
                        )

            # Fall back to original path
            if not model_loaded:
                if not os.path.exists(self.model_path):
                    logger.error(f"Model file not found: {self.model_path}")
                    return False

                logger.info(
                    f"Attempting to load model from original path: {self.model_path}"
                )
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                logger.info(f"Successfully loaded model from: {self.model_path}")

            # Load class mapping if available
            if os.path.exists(self.class_mapping_path):
                with open(self.class_mapping_path, "r") as f:
                    self.class_mapping = json.load(f)

                # Update categories
                for class_id, class_name in self.class_mapping.items():
                    self.categories[int(class_id)] = {
                        "name": class_name,
                        "id": int(class_id),
                    }

            self.loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def predict(self, image, conf_threshold=None, iou_threshold=None, **kwargs):
        """Run prediction with the defect detection model."""
        if not self.loaded:
            logger.warning("Model not loaded. Call load() first.")
            return None

        try:
            # Resize image to model's expected input size
            img = cv2.resize(image, (224, 224))

            # Preprocess image - normalize to [0,1]
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # Make prediction
            predictions = self.model.predict(img)

            # For binary classification
            score = float(predictions[0][0])
            class_id = 1 if score > 0.5 else 0

            # For compatibility with the TACO model return format
            # We're creating a pseudo detection with a single box covering most of the image
            # This allows the app to display the result using the same visualization logic
            return {
                "boxes": np.array(
                    [[0.1, 0.1, 0.9, 0.9]]
                ),  # Single box covering most of image
                "scores": np.array(
                    [score if class_id == 1 else 1.0 - score]
                ),  # Confidence score
                "classes": np.array([class_id]),  # Class ID (0=normal, 1=defect)
                "num_detections": 1,  # Only one detection for the whole image
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None

    def visualize_detections(self, image, detections, threshold=0.5):
        """Draw classification results on image."""
        image_with_result = image.copy()
        height, width, _ = image.shape

        # Only proceed if we have detections
        if detections["num_detections"] > 0:
            # Get class and score
            class_id = detections["classes"][0]
            score = detections["scores"][0]

            if score > threshold:
                # Draw border color based on class
                border_color = (
                    (0, 0, 255) if class_id == 1 else (0, 255, 0)
                )  # Red for defect, Green for normal
                border_thickness = 10

                # Draw colored border around the entire image
                cv2.rectangle(
                    image_with_result,
                    (border_thickness, border_thickness),
                    (width - border_thickness, height - border_thickness),
                    border_color,
                    border_thickness,
                )

                # Get class name
                class_name = self.categories.get(class_id, {}).get(
                    "name", f"Class {class_id}"
                )

                # Add label with large font at the top
                label = f"{class_name}: {score:.2f}"
                font_scale = 1.5
                font_thickness = 2

                # Get text size for centering
                text_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )[0]

                # Position text in the top center
                text_x = (width - text_size[0]) // 2
                text_y = text_size[1] + 20

                # Draw text with background
                cv2.putText(
                    image_with_result,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    border_color,
                    font_thickness,
                )

        return image_with_result
