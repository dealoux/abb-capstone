"""Model factory for creating and managing different model types."""

import logging
from typing import Dict, Optional
from abbvisionsystem.models.yolo_model import YOLODefectModel
from abbvisionsystem.models.defect_detection_model import DefectDetectionModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating and managing models."""

    MODEL_CLASSES = {
        "defect_yolo": YOLODefectModel,
        "defect_classification": DefectDetectionModel,
    }

    @classmethod
    def create_model(cls, model_type: str, model_path: Optional[str] = None):
        """Create a model instance.

        Args:
            model_type: Type of model to create
            model_path: Optional custom model path

        Returns:
            Model instance
        """
        if model_type not in cls.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = cls.MODEL_CLASSES[model_type]

        if model_path:
            return model_class.create_with_path(model_path)
        else:
            return model_class()

    @classmethod
    def get_available_models(cls, model_type: str) -> Dict:
        """Get available models for a specific type."""
        if model_type not in cls.MODEL_CLASSES:
            return {"trained": [], "pretrained": []}

        model_class = cls.MODEL_CLASSES[model_type]

        result = {"trained": model_class.get_available_models(), "pretrained": []}

        # Add pretrained models if available
        if hasattr(model_class, "get_pretrained_models"):
            result["pretrained"] = model_class.get_pretrained_models()

        return result

    @classmethod
    def validate_model(cls, model_type: str, model_path: str) -> bool:
        """Validate a model file for a specific type."""
        if model_type not in cls.MODEL_CLASSES:
            return False

        model_class = cls.MODEL_CLASSES[model_type]

        if hasattr(model_class, "validate_model"):
            return model_class.validate_model(model_path)

        # Default validation - just check if file exists
        import os

        return os.path.exists(model_path)
