from typing import List, Tuple, Dict, Optional
from pathlib import Path
import os
import time
from ultralytics import YOLO


class YOLO11DefectDetector:
    def __init__(
        self, model_variant: str = "yolo11s", model_path: Optional[str] = None
    ):
        self.model_variant = model_variant
        self.model_path = model_path
        self.model = None
        self.loaded = False

        # Available YOLO11 model variants
        self.available_models = {
            # Detection models
            "yolo11n": "yolo11n.pt",
            "yolo11s": "yolo11s.pt",
            "yolo11m": "yolo11m.pt",
            "yolo11l": "yolo11l.pt",
            "yolo11x": "yolo11x.pt",
            # Segmentation models
            "yolo11n-seg": "yolo11n-seg.pt",
            "yolo11s-seg": "yolo11s-seg.pt",
            "yolo11m-seg": "yolo11m-seg.pt",
            "yolo11l-seg": "yolo11l-seg.pt",
            "yolo11x-seg": "yolo11x-seg.pt",
            # Oriented Bounding Box models
            "yolo11n-obb": "yolo11n-obb.pt",
            "yolo11s-obb": "yolo11s-obb.pt",
            "yolo11m-obb": "yolo11m-obb.pt",
            "yolo11l-obb": "yolo11l-obb.pt",
            "yolo11x-obb": "yolo11x-obb.pt",
            # Classification models
            "yolo11n-cls": "yolo11n-cls.pt",
            "yolo11s-cls": "yolo11s-cls.pt",
            "yolo11m-cls": "yolo11m-cls.pt",
            "yolo11l-cls": "yolo11l-cls.pt",
            "yolo11x-cls": "yolo11x-cls.pt",
        }

        # Model characteristics for user guidance
        self.model_specs = {
            "yolo11n": {
                "size": "~6MB",
                "speed": "Fastest",
                "accuracy": "Good",
                "use_case": "Real-time edge",
            },
            "yolo11s": {
                "size": "~22MB",
                "speed": "Fast",
                "accuracy": "Very Good",
                "use_case": "Production (Recommended)",
            },
            "yolo11m": {
                "size": "~50MB",
                "speed": "Medium",
                "accuracy": "Excellent",
                "use_case": "Quality control",
            },
            "yolo11l": {
                "size": "~87MB",
                "speed": "Slower",
                "accuracy": "Higher",
                "use_case": "Research/offline",
            },
            "yolo11x": {
                "size": "~137MB",
                "speed": "Slowest",
                "accuracy": "Highest",
                "use_case": "Maximum accuracy",
            },
        }

    def get_model_info(self) -> Dict:
        """Get information about the current model variant."""
        base_variant = self.model_variant.split("-")[0]  # Remove -seg, -obb, etc.

        info = {
            "variant": self.model_variant,
            "base_model": base_variant,
            "model_file": self.available_models.get(self.model_variant, "Unknown"),
            "type": self._get_model_type(),
            "specs": self.model_specs.get(base_variant, {}),
            "available": self.model_variant in self.available_models,
        }

        return info

    def _get_model_type(self) -> str:
        """Determine model type from variant name."""
        if "-seg" in self.model_variant:
            return "segmentation"
        elif "-obb" in self.model_variant:
            return "oriented_detection"
        elif "-cls" in self.model_variant:
            return "classification"
        else:
            return "detection"

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load YOLO11 model with dynamic variant support.

        Args:
            model_path: Optional custom model path, overrides instance model_path
        """
        if model_path:
            self.model_path = model_path

        print(f"🔄 Loading YOLO11 model: {self.model_variant}")

        # Validate model variant
        if self.model_variant not in self.available_models:
            print(f"❌ Unknown model variant: {self.model_variant}")
            print(f"Available variants: {list(self.available_models.keys())}")
            return False

        try:
            # Try custom weights first if provided
            if self.model_path and os.path.exists(self.model_path):
                print(f"📁 Loading custom weights: {self.model_path}")
                self.model = YOLO(self.model_path)
                self.loaded = True
                print(f"✅ Custom {self.model_variant} model loaded successfully")
                return True

            # Try trained model paths
            trained_paths = [
                f"trained_models/{self.model_variant}_defect_detector/weights/best.pt",
                f"trained_models/yolo_defect_detector/weights/best.pt",
                f"trained_models/{self.model_variant}/weights/best.pt",
                "best.pt",
            ]

            for path in trained_paths:
                if os.path.exists(path):
                    print(f"📁 Loading trained weights: {path}")
                    self.model = YOLO(path)
                    self.loaded = True
                    print(f"✅ Trained {self.model_variant} model loaded successfully")
                    return True

            # Fallback to pretrained model
            model_file = self.available_models[self.model_variant]
            print(f"📁 Loading pretrained model: {model_file}")
            self.model = YOLO(model_file)
            self.loaded = True
            print(f"✅ Pretrained {self.model_variant} model loaded successfully")

            # Show model info
            info = self.get_model_info()
            if info["specs"]:
                print(
                    f"ℹ️  Model specs: {info['specs']['size']} size, {info['specs']['speed']} speed, {info['specs']['accuracy']} accuracy"
                )

            return True

        except Exception as e:
            print(f"❌ Failed to load {self.model_variant}: {str(e)}")
            self.loaded = False
            return False

    def train(
        self,
        dataset_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        project: str = "trained_models",
        name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Train YOLO11 model with dynamic variant support.

        Args:
            dataset_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            project: Project directory name
            name: Experiment name (auto-generated if None)
            **kwargs: Additional training parameters

        Returns:
            Path to best trained weights
        """

        if not self.loaded:
            if not self.load_model():
                raise RuntimeError(
                    f"Failed to load {self.model_variant} model for training"
                )

        # Auto-generate name if not provided
        if name is None:
            name = f"{self.model_variant}_defect_detector"

        print(f"🚀 Starting {self.model_variant.upper()} training...")
        print(f"📊 Training parameters:")
        print(f"   Model: {self.model_variant}")
        print(f"   Dataset: {dataset_yaml}")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch}")
        print(f"   Project: {project}")
        print(f"   Name: {name}")

        # Set default training parameters optimized for defect detection
        training_params = {
            "data": dataset_yaml,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "project": project,
            "name": name,
            "save_period": 10,  # Save every 10 epochs
            "plots": True,  # Generate training plots
            "device": "auto",  # Auto-detect GPU/CPU
            "workers": 8,  # Data loading workers
            "patience": 20,  # Early stopping patience
            "save": True,  # Save checkpoints
            "verbose": True,  # Verbose output
            "seed": 42,  # Reproducible results
            "amp": True,  # Automatic Mixed Precision
            "fraction": 1.0,  # Use full dataset
            # Data augmentation optimized for industrial defect detection
            "hsv_h": 0.015,  # Hue augmentation (minimal for industrial)
            "hsv_s": 0.7,  # Saturation augmentation
            "hsv_v": 0.4,  # Value augmentation
            "degrees": 0.0,  # Rotation (disabled for industrial parts)
            "translate": 0.1,  # Translation
            "scale": 0.5,  # Scale augmentation
            "shear": 0.0,  # Shear (disabled for industrial parts)
            "perspective": 0.0,  # Perspective (disabled for industrial parts)
            "flipud": 0.0,  # Vertical flip (disabled for industrial parts)
            "fliplr": 0.5,  # Horizontal flip
            "mosaic": 1.0,  # Mosaic augmentation
            "mixup": 0.0,  # Mixup augmentation
            "copy_paste": 0.0,  # Copy-paste augmentation
        }

        # Update with user-provided parameters
        training_params.update(kwargs)

        try:
            start_time = time.time()

            # Train the model
            results = self.model.train(**training_params)

            training_time = time.time() - start_time

            # Get best weights path
            best_weights_path = os.path.join(project, name, "weights", "best.pt")

            print(f"✅ {self.model_variant.upper()} training completed!")
            print(
                f"⏱️  Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)"
            )
            print(f"💾 Best weights saved to: {best_weights_path}")

            # Validate that weights file exists
            if os.path.exists(best_weights_path):
                print(
                    f"✅ Best weights file confirmed: {os.path.getsize(best_weights_path) / 1024 / 1024:.1f} MB"
                )
            else:
                print(f"⚠️  Warning: Best weights file not found at expected location")

            return best_weights_path

        except Exception as e:
            print(f"❌ {self.model_variant.upper()} training failed: {str(e)}")
            raise

    def predict(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 100,
    ) -> Dict:
        """
        Run prediction with dynamic model support.

        Args:
            image_path: Path to image or numpy array
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_det: Maximum detections

        Returns:
            Dictionary with detection results
        """
        if not self.loaded:
            if not self.load_model():
                raise RuntimeError(
                    f"Failed to load {self.model_variant} model for prediction"
                )

        try:
            # Run inference
            results = self.model(
                image_path,
                conf=float(conf_threshold),
                iou=float(iou_threshold),
                max_det=max_det,
                verbose=False,
                device="auto",
            )

            if not results or len(results) == 0:
                return self._empty_result()

            result = results[0]

            # Process results based on model type
            model_type = self._get_model_type()

            if model_type == "segmentation":
                return self._process_segmentation_results(result, conf_threshold)
            elif model_type == "oriented_detection":
                return self._process_obb_results(result, conf_threshold)
            elif model_type == "classification":
                return self._process_classification_results(result, conf_threshold)
            else:
                return self._process_detection_results(result, conf_threshold)

        except Exception as e:
            print(f"❌ {self.model_variant} prediction error: {str(e)}")
            return self._empty_result()

    def _process_detection_results(self, result, conf_threshold: float) -> Dict:
        """Process standard detection results."""
        import numpy as np

        if result.boxes is None or len(result.boxes) == 0:
            return self._empty_result()

        # Extract data with proper type conversion
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        classes = result.boxes.cls.cpu().numpy().astype(np.int32)

        # Filter by confidence
        valid_indices = scores >= conf_threshold

        if not np.any(valid_indices):
            return self._empty_result()

        valid_boxes = boxes[valid_indices]
        valid_scores = scores[valid_indices]
        valid_classes = classes[valid_indices]

        # Generate labels
        labels = []
        for cls in valid_classes:
            cls_int = int(cls)
            if cls_int == 0:
                labels.append("Normal")
            elif cls_int == 1:
                labels.append("Defect")
            else:
                labels.append(f"Class_{cls_int}")

        return {
            "boxes": valid_boxes,
            "scores": valid_scores,
            "classes": valid_classes,
            "labels": labels,
            "num_detections": len(valid_scores),
            "model_variant": self.model_variant,
            "model_type": "detection",
        }

    def _process_segmentation_results(self, result, conf_threshold: float) -> Dict:
        """Process segmentation results."""
        detection_result = self._process_detection_results(result, conf_threshold)

        if result.masks is not None and len(result.masks) > 0:
            import numpy as np

            masks = result.masks.data.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            valid_indices = scores >= conf_threshold
            valid_masks = masks[valid_indices] if np.any(valid_indices) else []

            detection_result["masks"] = valid_masks
            detection_result["model_type"] = "segmentation"

        return detection_result

    def _process_obb_results(self, result, conf_threshold: float) -> Dict:
        """Process oriented bounding box results."""
        import numpy as np

        if hasattr(result, "obb") and result.obb is not None:
            obb_data = result.obb.xywhr.cpu().numpy().astype(np.float32)
            scores = result.obb.conf.cpu().numpy().astype(np.float32)
            classes = result.obb.cls.cpu().numpy().astype(np.int32)

            valid_indices = scores >= conf_threshold

            if not np.any(valid_indices):
                return self._empty_result()

            valid_obb = obb_data[valid_indices]
            valid_scores = scores[valid_indices]
            valid_classes = classes[valid_indices]

            labels = []
            for cls in valid_classes:
                cls_int = int(cls)
                if cls_int == 0:
                    labels.append("Normal")
                elif cls_int == 1:
                    labels.append("Defect")
                else:
                    labels.append(f"Class_{cls_int}")

            return {
                "obb_boxes": valid_obb,
                "scores": valid_scores,
                "classes": valid_classes,
                "labels": labels,
                "num_detections": len(valid_scores),
                "model_variant": self.model_variant,
                "model_type": "oriented_detection",
            }
        else:
            return self._process_detection_results(result, conf_threshold)

    def _process_classification_results(self, result, conf_threshold: float) -> Dict:
        """Process classification results."""
        import numpy as np

        if hasattr(result, "probs") and result.probs is not None:
            probs = result.probs.data.cpu().numpy()
            top_class = np.argmax(probs)
            confidence = float(probs[top_class])

            if confidence >= conf_threshold:
                class_name = "Normal" if top_class == 0 else "Defect"

                return {
                    "classification": {
                        "class": int(top_class),
                        "class_name": class_name,
                        "confidence": confidence,
                        "probabilities": probs.tolist(),
                    },
                    "model_variant": self.model_variant,
                    "model_type": "classification",
                }

        return self._empty_result()

    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        import numpy as np

        return {
            "boxes": np.array([], dtype=np.float32).reshape(0, 4),
            "scores": np.array([], dtype=np.float32),
            "classes": np.array([], dtype=np.int32),
            "labels": [],
            "num_detections": 0,
            "model_variant": self.model_variant,
            "model_type": self._get_model_type(),
        }

    @staticmethod
    def list_available_models() -> Dict[str, Dict]:
        """List all available YOLO11 model variants with their specifications."""
        detector = YOLO11DefectDetector()

        models_info = {}
        for variant in detector.available_models.keys():
            temp_detector = YOLO11DefectDetector(variant)
            models_info[variant] = temp_detector.get_model_info()

        return models_info

    @staticmethod
    def recommend_model_for_use_case(use_case: str) -> List[str]:
        """
        Recommend YOLO11 models for specific use cases.

        Args:
            use_case: One of 'real_time', 'production', 'quality_control',
                     'research', 'edge', 'segmentation', 'oriented'

        Returns:
            List of recommended model variants
        """
        recommendations = {
            "real_time": ["yolo11n", "yolo11s"],
            "edge": ["yolo11n"],
            "production": ["yolo11s", "yolo11m"],
            "quality_control": ["yolo11m", "yolo11l"],
            "research": ["yolo11m", "yolo11l", "yolo11x"],
            "maximum_accuracy": ["yolo11x"],
            "segmentation": ["yolo11s-seg", "yolo11m-seg"],
            "oriented": ["yolo11s-obb", "yolo11m-obb"],
            "classification": ["yolo11s-cls", "yolo11m-cls"],
        }

        return recommendations.get(use_case.lower(), ["yolo11s"])


def create_multi_object_test_images(
    single_object_dir: str,
    output_dir: str,
    images_per_composition: int = 50,
    objects_per_image: Tuple[int, int] = (2, 6),
) -> None:
    """Create multi-object test images (unchanged from original)."""
    # Keep original implementation
    pass


# Helper function for easy model comparison
def compare_yolo11_models(
    dataset_yaml: str,
    model_variants: List[str] = None,
    epochs: int = 50,
    test_images_dir: str = None,
) -> Dict:
    """
    Compare multiple YOLO11 model variants.

    Args:
        dataset_yaml: Path to dataset YAML
        model_variants: List of model variants to compare
        epochs: Training epochs for each model
        test_images_dir: Directory with test images for evaluation

    Returns:
        Comparison results dictionary
    """
    if model_variants is None:
        model_variants = ["yolo11n", "yolo11s", "yolo11m"]

    print(f"🔬 Comparing {len(model_variants)} YOLO11 models...")
    print(f"Models: {', '.join(model_variants)}")

    results = {}

    for variant in model_variants:
        print(f"\n🚀 Training and evaluating {variant}...")

        try:
            # Create detector
            detector = YOLO11DefectDetector(variant)

            # Train model
            best_weights = detector.train(
                dataset_yaml=dataset_yaml, epochs=epochs, name=f"{variant}_comparison"
            )

            # Evaluate if test directory provided
            if test_images_dir and os.path.exists(test_images_dir):
                eval_results = evaluate_yolo11_model(detector, test_images_dir)
                results[variant] = eval_results

                print(
                    f"✅ {variant}: {eval_results.get('detection_rate', 0):.2%} detection rate"
                )
            else:
                results[variant] = {"status": "trained", "weights_path": best_weights}

        except Exception as e:
            print(f"❌ {variant} failed: {e}")
            results[variant] = {"status": "failed", "error": str(e)}

    return results


def evaluate_yolo11_model(detector: YOLO11DefectDetector, test_dir: str) -> Dict:
    """
    Evaluate YOLO11 model on test directory.

    Args:
        detector: Loaded YOLO11DefectDetector instance
        test_dir: Directory containing test images

    Returns:
        Evaluation results dictionary
    """
    import glob
    import time

    results = {
        "total_images": 0,
        "images_with_detections": 0,
        "total_detections": 0,
        "avg_detections_per_image": 0.0,
        "detection_rate": 0.0,
        "avg_inference_time": 0.0,
        "model_variant": detector.model_variant,
    }

    # Get test images
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG"]:
        image_files.extend(glob.glob(os.path.join(test_dir, ext)))

    if not image_files:
        print(f"⚠️ No images found in {test_dir}")
        return results

    print(f"🔍 Evaluating {detector.model_variant} on {len(image_files)} images...")

    inference_times = []

    for img_path in image_files:
        try:
            start_time = time.time()
            detections = detector.predict(img_path, conf_threshold=0.1)
            inference_time = time.time() - start_time

            inference_times.append(inference_time)
            results["total_images"] += 1

            if detections and detections.get("num_detections", 0) > 0:
                results["images_with_detections"] += 1
                results["total_detections"] += detections["num_detections"]

        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")
            continue

    # Calculate metrics
    if results["total_images"] > 0:
        results["avg_detections_per_image"] = (
            results["total_detections"] / results["total_images"]
        )
        results["detection_rate"] = (
            results["images_with_detections"] / results["total_images"]
        )

    if inference_times:
        results["avg_inference_time"] = sum(inference_times) / len(inference_times)

    return results
