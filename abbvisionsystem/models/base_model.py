"""Abstract base class for all models with enhanced discovery and management."""

import logging
import os
import glob
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all model implementations with discovery capabilities."""

    def __init__(self, model_path=None):
        """Initialize model with optional custom path.

        Args:
            model_path: Custom path to model file. If None, will auto-discover.
        """
        self.model_path = model_path
        self.loaded = False

        # Auto-discover if no path provided
        if self.model_path is None:
            self.model_path = self._auto_discover_model()

    @abstractmethod
    def load(self):
        """Load the model from disk."""
        pass

    @abstractmethod
    def predict(self, image):
        """Make predictions on an input image."""
        pass

    @abstractmethod
    def visualize_detections(self, image, detections, threshold=0.5):
        """Visualize detection results on an image."""
        pass

    @classmethod
    @abstractmethod
    def get_available_models(cls) -> List[Dict]:
        """Get list of all available models of this type."""
        pass

    @classmethod
    @abstractmethod
    def get_default_search_patterns(cls) -> List[str]:
        """Get default search patterns for this model type."""
        pass

    @classmethod
    @abstractmethod
    def get_fallback_models(cls) -> List[str]:
        """Get fallback model paths if no trained models found."""
        pass

    def _auto_discover_model(self) -> str:
        """Auto-discover the best available model."""
        available_models = self.get_available_models()

        if available_models:
            # Return the most recent trained model
            return available_models[0]["path"]

        # Fall back to default models
        fallback_models = self.get_fallback_models()
        for fallback in fallback_models:
            if os.path.exists(fallback):
                logger.info(f"Using fallback model: {fallback}")
                return fallback

        # Return first fallback (might be downloadable)
        if fallback_models:
            logger.info(f"Will attempt to use/download: {fallback_models[0]}")
            return fallback_models[0]

        raise FileNotFoundError("No models available and no fallback options")

    @classmethod
    def create_with_path(cls, model_path: str):
        """Create model instance with specific path."""
        return cls(model_path=model_path)

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        info = {
            "model_path": self.model_path,
            "loaded": self.loaded,
            "model_type": self.__class__.__name__,
        }

        if self.model_path and os.path.exists(self.model_path):
            info.update(
                {
                    "file_size_mb": os.path.getsize(self.model_path) / (1024 * 1024),
                    "file_exists": True,
                    "modified": datetime.fromtimestamp(
                        os.path.getmtime(self.model_path)
                    ).strftime("%Y-%m-%d %H:%M"),
                }
            )
        else:
            info["file_exists"] = False

        return info
