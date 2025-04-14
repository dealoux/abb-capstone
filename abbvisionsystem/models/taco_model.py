"""TACO (Trash Annotations in Context) pre-trained model implementation."""

import os
import logging
import numpy as np
import tensorflow as tf
import cv2
from abbvisionsystem.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class TACOModel(BaseModel):
    """Interface for pre-trained TACO waste detection model."""

    def __init__(self, model_path="train_models/ssd_mobilenet_v2_taco_2018_03_29.pb"):
        super().__init__(model_path)
        self.detection_graph = None
        self.session = None
        self.input_tensor = None
        self.output_tensors = None

        # TACO dataset categories - update based on your specific model
        self.categories = {
            0: {"name": "Background", "id": 0},
            1: {"name": "Plastic bag & wrapper", "id": 1},
            2: {"name": "Plastic bottle", "id": 2},
            3: {"name": "Bottle cap", "id": 3},
            4: {"name": "Metal can", "id": 4},
            5: {"name": "Cardboard", "id": 5},
            6: {"name": "Cup", "id": 6},
            7: {"name": "Lid", "id": 7},
            8: {"name": "Paper", "id": 8},
            9: {"name": "Straw", "id": 9},
            10: {"name": "Paper bag", "id": 10},
            11: {"name": "Tupperware", "id": 11},
            12: {"name": "Disposable plastic container", "id": 12},
            13: {"name": "Disposable food container", "id": 13},
            14: {"name": "Foam food container", "id": 14},
            15: {"name": "Other plastic", "id": 15},
            16: {"name": "Plastic utensils", "id": 16},
            17: {"name": "Pop tab", "id": 17},
            18: {"name": "Rope & strings", "id": 18},
            19: {"name": "Shoe", "id": 19},
            20: {"name": "Six pack rings", "id": 20},
            21: {"name": "Plastic utensils", "id": 21},
            22: {"name": "Cigarette", "id": 22},
            23: {"name": "Glass bottle", "id": 23},
            24: {"name": "Broken glass", "id": 24},
            25: {"name": "Food waste", "id": 25},
            26: {"name": "Aluminium foil", "id": 26},
            27: {"name": "Battery", "id": 27},
            28: {"name": "Scrap metal", "id": 28},
            29: {"name": "Clothing", "id": 29},
            30: {"name": "Fabric, cloth", "id": 30},
            31: {"name": "Dry waste", "id": 31},
            32: {"name": "Glass shard", "id": 32},
            33: {"name": "Other plastic wrap", "id": 33},
            34: {"name": "Paper cup", "id": 34},
            35: {"name": "Styrofoam piece", "id": 35},
            36: {"name": "Unlabeled litter", "id": 36},
            37: {"name": "Cigarette butt", "id": 37},
            38: {"name": "Paper straw", "id": 38},
            39: {"name": "Plastic film", "id": 39},
            40: {"name": "Plastic straw", "id": 40},
            41: {"name": "Paper bag & wrap", "id": 41},
            42: {"name": "Blister packaging", "id": 42},
            43: {"name": "Broken glass bottle", "id": 43},
            44: {"name": "Aerosol can", "id": 44},
            45: {"name": "Drink can", "id": 45},
            46: {"name": "Food can", "id": 46},
            47: {"name": "Corrugated carton", "id": 47},
            48: {"name": "Drink carton", "id": 48},
            49: {"name": "Egg carton", "id": 49},
            50: {"name": "Toilet tube", "id": 50},
            51: {"name": "Disposable plastic cup", "id": 51},
            52: {"name": "Foam cup", "id": 52},
            53: {"name": "Glass cup", "id": 53},
            54: {"name": "Other plastic cup", "id": 54},
            55: {"name": "Paper cup", "id": 55},
            56: {"name": "Plastic cup", "id": 56},
            57: {"name": "Plastic lid", "id": 57},
            58: {"name": "Newspaper & magazine", "id": 58},
            59: {"name": "Normal paper", "id": 59},
            60: {"name": "Paper sheet", "id": 60},
        }

    def load(self):
        """Load the TensorFlow frozen graph model."""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False

            # Load frozen graph
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.compat.v1.GraphDef()
                with tf.io.gfile.GFile(self.model_path, "rb") as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name="")

            # Create TensorFlow session
            self.session = tf.compat.v1.Session(graph=self.detection_graph)

            # Get input and output tensors
            self.input_tensor = self.detection_graph.get_tensor_by_name(
                "image_tensor:0"
            )

            # Detection outputs - adjust these based on your model's actual output tensors
            self.output_tensors = {
                "detection_boxes": self.detection_graph.get_tensor_by_name(
                    "detection_boxes:0"
                ),
                "detection_scores": self.detection_graph.get_tensor_by_name(
                    "detection_scores:0"
                ),
                "detection_classes": self.detection_graph.get_tensor_by_name(
                    "detection_classes:0"
                ),
                "num_detections": self.detection_graph.get_tensor_by_name(
                    "num_detections:0"
                ),
            }

            self.loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def predict(self, image):
        """Run object detection on an image."""
        if not self.loaded:
            logger.warning("Model not loaded. Call load() first.")
            return None

        try:
            # The model expects a batch of images, so add an axis
            image_expanded = np.expand_dims(image, axis=0)

            # Run inference
            outputs = self.session.run(
                self.output_tensors, feed_dict={self.input_tensor: image_expanded}
            )

            # Convert to numpy arrays
            boxes = outputs["detection_boxes"][0]
            scores = outputs["detection_scores"][0]
            classes = outputs["detection_classes"][0].astype(np.int64)
            num_detections = int(outputs["num_detections"][0])

            return {
                "boxes": boxes[:num_detections],
                "scores": scores[:num_detections],
                "classes": classes[:num_detections],
                "num_detections": num_detections,
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None

    def visualize_detections(self, image, detections, threshold=0.5):
        """Draw bounding boxes on image for visualizing detections."""
        image_with_boxes = image.copy()
        height, width, _ = image.shape

        for i in range(detections["num_detections"]):
            if detections["scores"][i] > threshold:
                # Get box coordinates
                box = detections["boxes"][i]
                y_min, x_min, y_max, x_max = box

                # Convert normalized coordinates to pixel values
                x_min = int(x_min * width)
                x_max = int(x_max * width)
                y_min = int(y_min * height)
                y_max = int(y_max * height)

                # Get class and score
                class_id = detections["classes"][i]
                score = detections["scores"][i]

                # Draw bounding box
                cv2.rectangle(
                    image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
                )

                # Add label
                class_name = self.categories.get(class_id, {}).get(
                    "name", f"Class {class_id}"
                )
                label = f"{class_name}: {score:.2f}"
                cv2.putText(
                    image_with_boxes,
                    label,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return image_with_boxes
