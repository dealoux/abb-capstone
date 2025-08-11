"""Data preparation utilities for defect detection."""

import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


def organize_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    """Organize images into train/validation/test splits."""
    # Create directory structure
    for split in ["train", "validation", "test"]:
        for class_name in ["normal", "defect"]:
            os.makedirs(f"{output_dir}/{split}/{class_name}", exist_ok=True)

    # Process normal and defect images
    for class_name in ["good", "defect"]:
        class_dir = os.path.join(
            source_dir, "good" if class_name == "good" else "defect"
        )
        output_class = "normal" if class_name == "good" else "defect"

        # Get image files
        image_files = list(Path(class_dir).glob("*.JPG"))
        if not image_files:
            image_files = list(Path(class_dir).glob("*.jpg"))

        random.shuffle(image_files)

        # Split data
        count = len(image_files)
        train_end = int(count * train_ratio)
        val_end = train_end + int(count * val_ratio)

        # Copy files to appropriate directories
        for i, file_path in enumerate(image_files):
            if i < train_end:
                dest_dir = f"{output_dir}/train/{output_class}"
            elif i < val_end:
                dest_dir = f"{output_dir}/validation/{output_class}"
            else:
                dest_dir = f"{output_dir}/test/{output_class}"
            shutil.copy(file_path, f"{dest_dir}/{file_path.name}")

    print(f"Dataset organized into {output_dir}")


def prepare_yolo_dataset(
    classification_dataset_dir: str, output_dir: str = "yolo_dataset"
) -> str:
    """Convert classification dataset to YOLO format for multi-object detection."""
    # Create YOLO directory structure
    for split in ["train", "val", "test"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    # Process each split
    for split in ["train", "validation", "test"]:
        yolo_split = "val" if split == "validation" else split

        # Process normal images (class 0 = normal, no defect annotations)
        normal_dir = os.path.join(classification_dataset_dir, split, "normal")
        if os.path.exists(normal_dir):
            for img_file in os.listdir(normal_dir):
                if img_file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".BMP")):
                    # Copy image
                    src_path = os.path.join(normal_dir, img_file)
                    dst_path = os.path.join(
                        f"{output_dir}/images/{yolo_split}",
                        f"{os.path.splitext(img_file)[0]}.jpg",
                    )

                    img = cv2.imread(src_path)
                    if img is not None:
                        cv2.imwrite(dst_path, img)

                        # Create empty label file (no defects to annotate)
                        label_path = os.path.join(
                            f"{output_dir}/labels/{yolo_split}",
                            f"{os.path.splitext(img_file)[0]}.txt",
                        )
                        with open(label_path, "w") as f:
                            pass  # Empty file for normal images

        # Process defect images (class 1 = defect, annotate entire object as defective)
        defect_dir = os.path.join(classification_dataset_dir, split, "defect")
        if os.path.exists(defect_dir):
            for img_file in os.listdir(defect_dir):
                if img_file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".BMP")):
                    # Copy image
                    src_path = os.path.join(defect_dir, img_file)
                    dst_path = os.path.join(
                        f"{output_dir}/images/{yolo_split}",
                        f"{os.path.splitext(img_file)[0]}.jpg",
                    )

                    img = cv2.imread(src_path)
                    if img is not None:
                        cv2.imwrite(dst_path, img)

                        # Create label file with defect annotation
                        # Format: <class> <x_center> <y_center> <width> <height> (normalized)
                        label_path = os.path.join(
                            f"{output_dir}/labels/{yolo_split}",
                            f"{os.path.splitext(img_file)[0]}.txt",
                        )
                        with open(label_path, "w") as f:
                            # Annotate entire object as defective (covers 80% of image)
                            f.write("1 0.5 0.5 0.8 0.8\n")  # class 1 for defect

    # Create dataset.yaml for YOLO
    yaml_content = f"""
# YOLOv8 dataset configuration
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: normal
  1: defect

# Number of classes
nc: 2
"""

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"YOLO dataset prepared in {output_dir}")
    return yaml_path


def generate_synthetic_defects(
    normal_dir: str, output_dir: str, num_defects_per_image: int = 2
) -> None:
    """Generate synthetic defects for data augmentation."""
    os.makedirs(output_dir, exist_ok=True)

    normal_images = [
        f
        for f in os.listdir(normal_dir)
        if f.endswith((".BMP", ".bmp", ".jpg", ".png"))
    ]

    for img_file in normal_images:
        img_path = os.path.join(normal_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        base_name = os.path.splitext(img_file)[0]

        for i in range(num_defects_per_image):
            defect_img = img.copy()
            defect_type = np.random.randint(1, 5)

            if defect_type == 1:  # Scratches
                x1, y1 = np.random.randint(0, img.shape[1]), np.random.randint(
                    0, img.shape[0]
                )
                x2, y2 = np.random.randint(0, img.shape[1]), np.random.randint(
                    0, img.shape[0]
                )
                cv2.line(defect_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            elif defect_type == 2:  # Spots
                cx, cy = np.random.randint(0, img.shape[1]), np.random.randint(
                    0, img.shape[0]
                )
                radius = np.random.randint(5, 20)
                cv2.circle(defect_img, (cx, cy), radius, (0, 0, 255), -1)

            elif defect_type == 3:  # Noise
                x, y = np.random.randint(0, img.shape[1] - 50), np.random.randint(
                    0, img.shape[0] - 50
                )
                w, h = np.random.randint(30, 50), np.random.randint(30, 50)
                roi = defect_img[y : y + h, x : x + w]
                noise = np.random.randint(0, 50, roi.shape, dtype=np.uint8)
                defect_img[y : y + h, x : x + w] = cv2.add(roi, noise)

            else:  # Missing parts
                x, y = np.random.randint(0, img.shape[1] - 40), np.random.randint(
                    0, img.shape[0] - 40
                )
                w, h = np.random.randint(10, 40), np.random.randint(10, 40)
                cv2.rectangle(defect_img, (x, y), (x + w, y + h), (0, 0, 0), -1)

            output_path = os.path.join(output_dir, f"{base_name}_defect_{i+1}.jpg")
            cv2.imwrite(output_path, defect_img)

    print(f"Generated {len(normal_images) * num_defects_per_image} synthetic defects")
