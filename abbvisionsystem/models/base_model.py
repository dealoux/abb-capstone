"""Abstract base class for all models."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all model implementations."""

    def __init__(self, model_path):
        """Initialize model.

        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path
        self.loaded = False

    @abstractmethod
    def load(self):
        """Load the model from disk.

        Returns:
            bool: Success status
        """
        pass

    @abstractmethod
    def predict(self, image):
        """Make predictions on an input image.

        Args:
            image: Input image (numpy array)

        Returns:
            dict: Prediction results
        """
        pass

    @abstractmethod
    def visualize_detections(self, image, detections, threshold=0.5):
        """Visualize detection results on an image.

        Args:
            image: Input image (numpy array)
            detections: Detection results from predict()
            threshold: Confidence threshold for displaying detections

        Returns:
            numpy.ndarray: Image with visualized detections
        """
        pass
