"""YOLOv8-based defect detection for multi-object scenarios."""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class YOLODefectDetector:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the YOLO detector."""
        self.model = None
        self.model_path = model_path
        self.class_names = {0: "normal", 1: "defect"}

    def load_model(self, model_path: Optional[str] = None) -> bool:
        try:
            from ultralytics import YOLO

            path = model_path or self.model_path or "yolo11s-cls.pt"
            self.model = YOLO(path)
            self.model_path = path
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def train(
        self,
        dataset_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        project: str = "trained_models",
        name: str = "yolo_defect_detector",
        **kwargs,
    ):
        """Train YOLOv8 model with enhanced parameters optimized for defect detection with rich datasets."""
        if not self.model:
            # Try multiple model variants for better compatibility
            model_variants = ["yolo11s-cls.pt"]
            loaded = False
            for variant in model_variants:
                if self.load_model(variant):
                    print(f"✅ Successfully loaded {variant}")
                    loaded = True
                    break

            if not loaded:
                print("❌ Failed to load any YOLO model variant")
                return None

        enhanced_params = {
            "data": dataset_yaml,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "project": project,
            "name": name,
            "save": True,
            "plots": True,
            "device": "cpu",  # Ensure CPU compatibility
            "verbose": True,
            "patience": 50,  # Early stopping patience
            "save_period": 10,  # Save checkpoint every 10 epochs
            # Enhanced data augmentation for rich dataset
            "hsv_h": 0.02,  # Slightly increased hue augmentation
            "hsv_s": 0.7,  # Saturation augmentation
            "hsv_v": 0.4,  # Value augmentation
            "degrees": 15.0,  # Increased rotation for better generalization
            "translate": 0.1,  # Translation augmentation
            "scale": 0.3,  # Increased scale variation
            "shear": 3.0,  # Increased shear augmentation
            "perspective": 0.0,  # Keep perspective at 0
            "flipud": 0.0,  # No vertical flip for industrial objects
            "fliplr": 0.5,  # Horizontal flip
            "mosaic": 1.0,  # Mosaic augmentation
            "mixup": 0.1,  # Mixup augmentation
            "copy_paste": 0.1,  # Copy-paste augmentation
            # Optimized training parameters for enhanced dataset
            "lr0": 0.01,  # Initial learning rate
            "lrf": 0.01,  # Final learning rate
            "momentum": 0.937,  # SGD momentum
            "weight_decay": 0.0005,  # Weight decay
            "warmup_epochs": 3,  # Warmup epochs
            "warmup_momentum": 0.8,  # Warmup momentum
            "warmup_bias_lr": 0.1,  # Warmup bias learning rate
            # Enhanced loss weights for defect detection
            "box": 0.05,  # Box loss gain
            "cls": 0.5,  # Class loss gain
            "dfl": 1.5,  # DFL loss gain
            # Enhanced optimization for rich dataset
            "optimizer": "SGD",  # Use SGD for stability
            "close_mosaic": 10,  # Disable mosaic in last 10 epochs
        }

        print(f"🚀 Starting Enhanced YOLOv8 training...")
        print(f"   📊 Dataset: {dataset_yaml}")
        print(f"   🔄 Epochs: {epochs}")
        print(f"   📐 Image size: {imgsz}")
        print(f"   📦 Batch size: {batch}")
        print(f"   🎯 Enhanced augmentation: ON")
        print(f"   ⚙️  Optimized for rich dataset: ON")

        try:
            # Train the model with enhanced parameters
            results = self.model.train(**enhanced_params)

            # Update model path to best weights
            best_weights = os.path.join(project, name, "weights", "best.pt")
            if os.path.exists(best_weights):
                self.model_path = best_weights
                self.load_model(best_weights)
                print(f"✅ Enhanced training completed! Best weights: {best_weights}")

                # Print training summary
                print(f"📈 Training Summary:")
                print(f"   📊 Model: {self.model_path}")
                print(
                    f"   🎯 Enhanced features: Multi-object detection, Rich backgrounds"
                )
                print(f"   🔄 Augmentations: Enhanced for industrial scenarios")

            else:
                # Fallback to last weights
                last_weights = os.path.join(project, name, "weights", "last.pt")
                if os.path.exists(last_weights):
                    self.model_path = last_weights
                    self.load_model(last_weights)
                    print(f"✅ Training completed! Using last weights: {last_weights}")

            return self.model_path

        except Exception as e:
            print(f"❌ Enhanced training failed: {str(e)}")
            print("💡 Trying with reduced parameters for compatibility...")

            # Fallback with reduced parameters but still enhanced
            fallback_params = {
                "data": dataset_yaml,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": max(4, batch // 2),  # Reduce batch size
                "project": project,
                "name": name + "_fallback",
                "device": "cpu",
                "verbose": True,
                "patience": 30,
                # Keep essential augmentations
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 10.0,
                "scale": 0.2,
                "fliplr": 0.5,
                "mosaic": 1.0,
            }

            try:
                print(f"🔄 Retrying with fallback parameters...")
                print(f"   📦 Reduced batch size: {fallback_params['batch']}")
                results = self.model.train(**fallback_params)

                best_weights = os.path.join(
                    project, name + "_fallback", "weights", "best.pt"
                )
                if os.path.exists(best_weights):
                    self.model_path = best_weights
                    self.load_model(best_weights)
                    print(f"✅ Fallback training successful: {best_weights}")
                return self.model_path

            except Exception as e2:
                print(f"❌ Fallback training also failed: {str(e2)}")
                print("💡 Consider:")
                print("   - Reducing batch size further (batch=2)")
                print("   - Reducing image size (imgsz=320)")
                print("   - Checking available memory")
                raise e2

    def predict(
        self, image_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45
    ) -> Dict:
        """Predict defects in image with multiple objects."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Run inference
            results = self.model(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
                device="cpu",  # Force CPU
            )[0]

            # Process results
            detections = {"boxes": [], "scores": [], "classes": [], "labels": []}

            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                scores = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)

                detections["boxes"] = boxes
                detections["scores"] = scores
                detections["classes"] = classes
                detections["labels"] = [
                    self.class_names.get(cls, f"class_{cls}") for cls in classes
                ]

            return detections

        except Exception as e:
            print(f"❌ Prediction failed for {image_path}: {str(e)}")
            return {"boxes": [], "scores": [], "classes": [], "labels": []}

    def predict_batch(
        self, image_paths: List[str], conf_threshold: float = 0.25
    ) -> List[Dict]:
        """Predict defects on multiple images."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")

        results = []
        for img_path in image_paths:
            try:
                detection = self.predict(img_path, conf_threshold)
                results.append(
                    {"image_path": img_path, "detections": detection, "success": True}
                )
            except Exception as e:
                results.append(
                    {
                        "image_path": img_path,
                        "detections": {
                            "boxes": [],
                            "scores": [],
                            "classes": [],
                            "labels": [],
                        },
                        "success": False,
                        "error": str(e),
                    }
                )

        return results

    def visualize_detections(
        self, image_path: str, detections: Dict, save_path: Optional[str] = None
    ) -> np.ndarray:
        """Visualize detections on image."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw detections
        if len(detections["boxes"]) > 0:
            for i, (box, score, class_id, label) in enumerate(
                zip(
                    detections["boxes"],
                    detections["scores"],
                    detections["classes"],
                    detections["labels"],
                )
            ):
                x1, y1, x2, y2 = box.astype(int)

                # Color based on class
                color = (
                    (255, 0, 0) if class_id == 1 else (0, 255, 0)
                )  # Red for defect, green for normal

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label_text = f"{label}: {score:.2f}"
                label_size = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )[0]
                cv2.rectangle(
                    img,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1,
                )
                cv2.putText(
                    img,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        if save_path:
            try:
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.title(f"Detections: {len(detections['boxes'])} objects found")
                plt.axis("off")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"❌ Could not save visualization: {str(e)}")

        return img

    def evaluate_on_test_set(
        self, test_images_dir: str, conf_threshold: float = 0.25
    ) -> Dict:
        """Evaluate model performance on test set."""
        normal_dir = os.path.join(test_images_dir, "normal")
        defect_dir = os.path.join(test_images_dir, "defect")

        results = {
            "total_images": 0,
            "normal_images": 0,
            "defect_images": 0,
            "correct_predictions": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

        # Test normal images
        if os.path.exists(normal_dir):
            normal_files = [
                f
                for f in os.listdir(normal_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]

            for img_file in normal_files:
                img_path = os.path.join(normal_dir, img_file)
                detections = self.predict(img_path, conf_threshold)

                # Count defect detections
                defect_count = sum(1 for cls in detections["classes"] if cls == 1)

                results["total_images"] += 1
                results["normal_images"] += 1

                if defect_count == 0:  # Correctly identified as normal
                    results["correct_predictions"] += 1
                else:  # False positive
                    results["false_positives"] += 1

        # Test defect images
        if os.path.exists(defect_dir):
            defect_files = [
                f
                for f in os.listdir(defect_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]

            for img_file in defect_files:
                img_path = os.path.join(defect_dir, img_file)
                detections = self.predict(img_path, conf_threshold)

                # Count defect detections
                defect_count = sum(1 for cls in detections["classes"] if cls == 1)

                results["total_images"] += 1
                results["defect_images"] += 1

                if defect_count > 0:  # Correctly identified defect
                    results["correct_predictions"] += 1
                else:  # False negative
                    results["false_negatives"] += 1

        # Calculate metrics
        if results["total_images"] > 0:
            results["accuracy"] = (
                results["correct_predictions"] / results["total_images"]
            )

        tp = results["defect_images"] - results["false_negatives"]
        fp = results["false_positives"]
        fn = results["false_negatives"]

        if tp + fp > 0:
            results["precision"] = tp / (tp + fp)
        if tp + fn > 0:
            results["recall"] = tp / (tp + fn)
        if results["precision"] + results["recall"] > 0:
            results["f1_score"] = (
                2
                * (results["precision"] * results["recall"])
                / (results["precision"] + results["recall"])
            )

        return results


def create_multi_object_test_images(
    single_object_dir: str,
    output_dir: str,
    images_per_composition: int = 50,
    objects_per_image: Tuple[int, int] = (2, 6),
) -> None:
    """Create test images with multiple objects for realistic evaluation."""
    os.makedirs(output_dir, exist_ok=True)

    # Get source images
    normal_dir = os.path.join(single_object_dir, "normal")
    defect_dir = os.path.join(single_object_dir, "defect")

    normal_files = []
    defect_files = []

    if os.path.exists(normal_dir):
        normal_files = [
            f
            for f in os.listdir(normal_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

    if os.path.exists(defect_dir):
        defect_files = [
            f
            for f in os.listdir(defect_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

    if not normal_files and not defect_files:
        print(f"❌ No images found in {single_object_dir}")
        return

    print(f"📝 Creating {images_per_composition} multi-object test images...")
    print(f"   Normal images available: {len(normal_files)}")
    print(f"   Defect images available: {len(defect_files)}")

    for i in range(images_per_composition):
        # Create canvas
        canvas_size = (1024, 768)
        canvas = (
            np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 240
        )  # Light gray background

        # Random number of objects
        num_objects = np.random.randint(objects_per_image[0], objects_per_image[1] + 1)

        annotations = []  # For YOLO format
        placed_objects = 0

        for j in range(num_objects):
            # Choose normal or defect (70% normal, 30% defect)
            is_defect = np.random.random() < 0.3
            source_files = defect_files if is_defect else normal_files
            source_subdir = "defect" if is_defect else "normal"

            if not source_files:
                continue

            # Load random source image
            img_file = np.random.choice(source_files)
            img_path = os.path.join(single_object_dir, source_subdir, img_file)
            obj_img = cv2.imread(img_path)

            if obj_img is None:
                continue

            obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)

            # Resize object (random scale)
            scale = np.random.uniform(0.3, 0.8)
            new_size = (int(obj_img.shape[1] * scale), int(obj_img.shape[0] * scale))

            if new_size[0] <= 0 or new_size[1] <= 0:
                continue

            obj_img = cv2.resize(obj_img, new_size)

            # Random position (ensure object fits)
            max_x = canvas_size[0] - new_size[0]
            max_y = canvas_size[1] - new_size[1]

            if max_x > 0 and max_y > 0:
                x = np.random.randint(0, max_x)
                y = np.random.randint(0, max_y)

                # Place object on canvas
                canvas[y : y + new_size[1], x : x + new_size[0]] = obj_img

                # Create annotation
                x_center = (x + new_size[0] / 2) / canvas_size[0]
                y_center = (y + new_size[1] / 2) / canvas_size[1]
                width = new_size[0] / canvas_size[0]
                height = new_size[1] / canvas_size[1]

                class_id = 1 if is_defect else 0
                annotations.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
                placed_objects += 1

        # Save image
        img_name = f"multi_object_{i:04d}.jpg"
        img_path = os.path.join(output_dir, img_name)
        success = cv2.imwrite(img_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        if not success:
            print(f"❌ Failed to save image: {img_path}")
            continue

        # Save annotation
        ann_path = os.path.join(output_dir, f"multi_object_{i:04d}.txt")
        try:
            with open(ann_path, "w") as f:
                f.write("\n".join(annotations))
        except Exception as e:
            print(f"❌ Failed to save annotation: {ann_path}, error: {str(e)}")

    print(
        f"✅ Created {images_per_composition} multi-object test images in {output_dir}"
    )
