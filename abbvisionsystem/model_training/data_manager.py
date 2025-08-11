"""Data organization and preprocessing utilities for defect detection."""

import os
import shutil
import random
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class DataManager:
    """Handles data organization, preprocessing, and augmentation."""

    def __init__(self, source_dir, output_dir):
        self.source_dir = source_dir
        self.output_dir = output_dir

    def organize_object_detection_data(
        self, min_area=1000, max_area=50000, train_ratio=0.7, val_ratio=0.15
    ):
        """Organize data for object-level detection and classification."""
        from abbvisionsystem.model_training.object_detector import ObjectDetector

        # Create directory structure
        for split in ["train", "validation", "test"]:
            for class_name in ["normal", "defect"]:
                os.makedirs(
                    f"{self.output_dir}_objects/{split}/{class_name}", exist_ok=True
                )

        detector = ObjectDetector(min_area=min_area, max_area=max_area)

        # Process images and extract objects from all source folders
        all_objects = []

        # Process good images (normal class)
        good_dir = os.path.join(self.source_dir, "good")
        if os.path.exists(good_dir):
            normal_objects = self._extract_objects_from_dir(
                good_dir, "normal", detector
            )
            all_objects.extend(normal_objects)
            print(f"Extracted {len(normal_objects)} normal objects from 'good' folder")

        # Process defect images (defect class)
        defect_dir = os.path.join(self.source_dir, "defect")
        if os.path.exists(defect_dir):
            defect_objects = self._extract_objects_from_dir(
                defect_dir, "defect", detector
            )
            all_objects.extend(defect_objects)
            print(
                f"Extracted {len(defect_objects)} defect objects from 'defect' folder"
            )

        # Process both images (mixed class - classify each object individually)
        both_dir = os.path.join(self.source_dir, "both")
        if os.path.exists(both_dir):
            both_objects = self._extract_objects_from_dir(both_dir, "mixed", detector)
            all_objects.extend(both_objects)
            print(f"Extracted {len(both_objects)} mixed objects from 'both' folder")
            print(
                "Note: Objects from 'both' folder will need manual labeling or advanced classification"
            )

        # Split and save objects
        all_metadata = self._split_and_save_objects_advanced(
            all_objects, train_ratio, val_ratio
        )

        # Save metadata
        os.makedirs(f"{self.output_dir}_objects", exist_ok=True)
        with open(f"{self.output_dir}_objects/metadata.json", "w") as f:
            json.dump(all_metadata, f, indent=2)

        return all_metadata

    def _extract_objects_from_dir(self, image_dir, class_label, detector):
        """Extract objects from all images in directory."""
        extracted_objects = []
        image_files = self._get_image_files(image_dir)

        for img_file in image_files:
            # Load image safely
            image = self._load_image_safe(str(img_file))
            if image is None:
                continue

            # Detect objects
            objects = detector.detect_objects(image)

            for i, obj in enumerate(objects):
                # Extract object ROI
                roi = detector.extract_object_roi(image, obj["bbox"])

                object_filename = f"{img_file.stem}_obj_{i}.jpg"
                extracted_objects.append(
                    {
                        "roi": roi,
                        "filename": object_filename,
                        "bbox": obj["bbox"],
                        "source_image": img_file.name,
                        "class": class_label,
                        "area": obj["area"],
                        "source_folder": os.path.basename(image_dir),
                    }
                )

        return extracted_objects

    def _split_and_save_objects_advanced(self, objects, train_ratio, val_ratio):
        """Split objects into train/val/test with intelligent class handling."""
        # Separate objects by class
        normal_objects = [obj for obj in objects if obj["class"] == "normal"]
        defect_objects = [obj for obj in objects if obj["class"] == "defect"]
        mixed_objects = [obj for obj in objects if obj["class"] == "mixed"]

        # Shuffle each class
        random.shuffle(normal_objects)
        random.shuffle(defect_objects)
        random.shuffle(mixed_objects)

        metadata = []

        # Split normal objects
        if normal_objects:
            metadata.extend(
                self._split_class_objects(
                    normal_objects, "normal", train_ratio, val_ratio
                )
            )

        # Split defect objects
        if defect_objects:
            metadata.extend(
                self._split_class_objects(
                    defect_objects, "defect", train_ratio, val_ratio
                )
            )

        # Handle mixed objects - for now, distribute them equally between normal and defect
        # In a real scenario, you'd want manual labeling or more sophisticated classification
        if mixed_objects:
            print(
                "⚠️ Mixed objects found. Distributing equally between normal and defect classes."
            )
            print("💡 Consider manual labeling for better accuracy.")

            half = len(mixed_objects) // 2
            mixed_as_normal = mixed_objects[:half]
            mixed_as_defect = mixed_objects[half:]

            # Update class labels
            for obj in mixed_as_normal:
                obj["class"] = "normal"
                obj["original_class"] = "mixed"
            for obj in mixed_as_defect:
                obj["class"] = "defect"
                obj["original_class"] = "mixed"

            metadata.extend(
                self._split_class_objects(
                    mixed_as_normal, "normal", train_ratio, val_ratio
                )
            )
            metadata.extend(
                self._split_class_objects(
                    mixed_as_defect, "defect", train_ratio, val_ratio
                )
            )

        return metadata

    def _split_class_objects(self, objects, class_name, train_ratio, val_ratio):
        """Split objects of a single class into train/val/test."""
        total = len(objects)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits = {
            "train": objects[:train_end],
            "validation": objects[train_end:val_end],
            "test": objects[val_end:],
        }

        metadata = []

        for split_name, split_objects in splits.items():
            split_dir = f"{self.output_dir}_objects/{split_name}/{class_name}"

            for obj in split_objects:
                # Save object image
                object_path = os.path.join(split_dir, obj["filename"])
                cv2.imwrite(object_path, cv2.cvtColor(obj["roi"], cv2.COLOR_RGB2BGR))

                # Save metadata
                metadata_entry = {
                    "filename": obj["filename"],
                    "bbox": obj["bbox"],
                    "source_image": obj["source_image"],
                    "class": obj["class"],
                    "split": split_name,
                    "area": obj["area"],
                    "source_folder": obj["source_folder"],
                }

                # Add original class info if it was mixed
                if "original_class" in obj:
                    metadata_entry["original_class"] = obj["original_class"]

                metadata.append(metadata_entry)

        return metadata

    def _get_image_files(self, directory):
        """Get all image files from directory."""
        extensions = [
            ".heic",
            ".HEIC",
            ".jpg",
            ".jpeg",
            ".JPG",
            ".JPEG",
            ".png",
            ".PNG",
            ".bmp",
            ".BMP",
            ".tiff",
            ".TIFF",
        ]

        files = []
        if os.path.exists(directory):
            for ext in extensions:
                files.extend(list(Path(directory).glob(f"*{ext}")))
        return files

    def _load_image_safe(self, image_path):
        """Safely load image handling different formats."""
        try:
            # Try with OpenCV first
            img = cv2.imread(image_path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Try with PIL for HEIC and other formats
            with Image.open(image_path) as pil_img:
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                return np.array(pil_img)

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            return None

    def _copy_and_convert_if_needed(self, src_path, dest_dir):
        """Copy file and convert HEIC to JPG if needed."""
        src_path = Path(src_path)

        if src_path.suffix.lower() == ".heic":
            # Convert HEIC to JPG
            dest_filename = src_path.stem + ".jpg"
            dest_path = os.path.join(dest_dir, dest_filename)

            try:
                with Image.open(src_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(dest_path, "JPEG", quality=95)
                return True
            except Exception as e:
                logger.error(f"Failed to convert {src_path}: {str(e)}")
                return False
        else:
            # Regular copy for other formats
            dest_path = os.path.join(dest_dir, src_path.name)
            try:
                shutil.copy(src_path, dest_path)
                return True
            except Exception as e:
                logger.error(f"Failed to copy {src_path}: {str(e)}")
                return False


class SyntheticDataGenerator:
    """Generate synthetic defect data from normal images."""

    @staticmethod
    def generate_synthetic_defects(normal_dir, output_dir, num_defects_per_image=1):
        """Generate synthetic defects from normal images."""
        os.makedirs(output_dir, exist_ok=True)

        normal_images = [
            f
            for f in os.listdir(normal_dir)
            if f.lower().endswith((".bmp", ".jpg", ".jpeg", ".png"))
        ]

        total_generated = 0

        for img_file in normal_images:
            img_path = os.path.join(normal_dir, img_file)
            img = cv2.imread(img_path)

            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue

            base_name = os.path.splitext(img_file)[0]

            for i in range(num_defects_per_image):
                defect_img = SyntheticDataGenerator._add_synthetic_defect(img.copy())

                output_path = os.path.join(
                    output_dir, f"{base_name}_synthetic_defect_{i+1}.jpg"
                )
                cv2.imwrite(output_path, defect_img)
                total_generated += 1

        print(f"Generated {total_generated} synthetic defect images")
        return total_generated

    @staticmethod
    def _add_synthetic_defect(img):
        """Add a random synthetic defect to an image."""
        defect_type = np.random.randint(1, 5)

        if defect_type == 1:
            # Add scratch
            x1, y1 = np.random.randint(0, img.shape[1]), np.random.randint(
                0, img.shape[0]
            )
            x2, y2 = np.random.randint(0, img.shape[1]), np.random.randint(
                0, img.shape[0]
            )
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), np.random.randint(1, 4))

        elif defect_type == 2:
            # Add spot/blob
            cx, cy = np.random.randint(0, img.shape[1]), np.random.randint(
                0, img.shape[0]
            )
            radius = np.random.randint(5, 25)
            color = (0, 0, np.random.randint(150, 255))
            cv2.circle(img, (cx, cy), radius, color, -1)

        elif defect_type == 3:
            # Add noise region
            x = np.random.randint(0, max(1, img.shape[1] - 50))
            y = np.random.randint(0, max(1, img.shape[0] - 50))
            w = np.random.randint(30, min(50, img.shape[1] - x))
            h = np.random.randint(30, min(50, img.shape[0] - y))

            roi = img[y : y + h, x : x + w]
            noise = np.random.randint(0, 50, roi.shape, dtype=np.uint8)
            img[y : y + h, x : x + w] = cv2.add(roi, noise)

        else:
            # Add missing part (black rectangle)
            x = np.random.randint(0, max(1, img.shape[1] - 40))
            y = np.random.randint(0, max(1, img.shape[0] - 40))
            w = np.random.randint(10, min(40, img.shape[1] - x))
            h = np.random.randint(10, min(40, img.shape[0] - y))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

        return img
