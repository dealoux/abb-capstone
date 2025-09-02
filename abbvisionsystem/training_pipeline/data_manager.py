import os
import shutil
import random
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from collections import defaultdict


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
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping...")
            continue

        image_files = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".bmp", ".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(image_files)

        total_images = len(image_files)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)

        splits = {
            "train": image_files[:train_end],
            "validation": image_files[train_end:val_end],
            "test": image_files[val_end:],
        }

        output_class = "normal" if class_name == "good" else "defect"

        for split, files in splits.items():
            for img_file in files:
                src_path = os.path.join(class_dir, img_file)
                dst_path = os.path.join(output_dir, split, output_class, img_file)
                shutil.copy2(src_path, dst_path)

    print(f"Dataset organized into {output_dir}")


def augment_with_backgrounds(
    object_images_dir: str,
    output_dir: str,
    background_images: List[str] = None,
    objects_per_image: Tuple[int, int] = (1, 4),
    images_per_object: int = 3,
    multi_object_scenes: int = 100,
) -> str:
    """Create training images with objects placed on realistic backgrounds."""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    print(f"🎨 Creating realistic training data with backgrounds...")

    # Get object files
    good_files = []
    defect_files = []

    good_dir = os.path.join(object_images_dir, "good")
    defect_dir = os.path.join(object_images_dir, "defect")

    if os.path.exists(good_dir):
        good_files = [
            f
            for f in os.listdir(good_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

    if os.path.exists(defect_dir):
        defect_files = [
            f
            for f in os.listdir(defect_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

    image_count = 0

    # Create single-object training images
    print("  Creating single-object training images...")
    for category, files in [("good", good_files), ("defect", defect_files)]:
        for img_file in files:
            for i in range(images_per_object):
                obj_path = os.path.join(object_images_dir, category, img_file)
                obj_img = cv2.imread(obj_path)

                if obj_img is None:
                    continue

                background = create_random_background((640, 640))
                is_defect = category == "defect"

                result_img, annotations = place_object_on_background(
                    obj_img, background, is_defect
                )

                # Save image and label
                img_name = f"{category}_{os.path.splitext(img_file)[0]}_{i:03d}.jpg"
                cv2.imwrite(os.path.join(output_dir, "images", img_name), result_img)

                label_name = f"{category}_{os.path.splitext(img_file)[0]}_{i:03d}.txt"
                with open(os.path.join(output_dir, "labels", label_name), "w") as f:
                    f.write("\n".join(annotations))

                image_count += 1

    # Create multi-object scenes
    print("  Creating multi-object training scenes...")
    for i in range(multi_object_scenes):
        background = create_random_background((640, 640))
        num_objects = random.randint(objects_per_image[0], objects_per_image[1])

        result_img, annotations = place_multiple_objects(
            background, good_files, defect_files, object_images_dir, num_objects
        )

        # Save image and label
        img_name = f"multi_scene_{i:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, "images", img_name), result_img)

        label_name = f"multi_scene_{i:04d}.txt"
        with open(os.path.join(output_dir, "labels", label_name), "w") as f:
            f.write("\n".join(annotations))

        image_count += 1

    print(f"✅ Generated {image_count} realistic training images with backgrounds")
    return output_dir


def create_random_background(size: Tuple[int, int]) -> np.ndarray:
    """Create realistic backgrounds similar to industrial environments."""
    background_types = [
        create_gradient_background,
        create_texture_background,
        create_noisy_background,
        create_industrial_background,
    ]

    background_func = random.choice(background_types)
    return background_func(size)


def create_gradient_background(size: Tuple[int, int]) -> np.ndarray:
    """Create gradient background similar to industrial surfaces."""
    h, w = size[1], size[0]
    background = np.zeros((h, w, 3), dtype=np.uint8)

    direction = random.choice(["horizontal", "vertical", "diagonal"])
    start_color = random.randint(180, 220)
    end_color = random.randint(200, 240)

    if direction == "horizontal":
        for i in range(w):
            color_val = int(start_color + (end_color - start_color) * (i / w))
            background[:, i] = [color_val, color_val, color_val]
    elif direction == "vertical":
        for i in range(h):
            color_val = int(start_color + (end_color - start_color) * (i / h))
            background[i, :] = [color_val, color_val, color_val]
    else:
        for i in range(h):
            for j in range(w):
                progress = (i + j) / (h + w)
                color_val = int(start_color + (end_color - start_color) * progress)
                background[i, j] = [color_val, color_val, color_val]

    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return background


def create_texture_background(size: Tuple[int, int]) -> np.ndarray:
    """Create textured background simulating various surfaces."""
    h, w = size[1], size[0]
    base_color = random.randint(160, 240)
    background = np.full((h, w, 3), base_color, dtype=np.uint8)

    texture_type = random.choice(["fabric", "metal", "concrete"])

    if texture_type == "fabric":
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                intensity = random.randint(-5, 5)
                background[i : i + 2, j : j + 2] = np.clip(
                    background[i : i + 2, j : j + 2].astype(np.int16) + intensity,
                    0,
                    255,
                ).astype(np.uint8)

    elif texture_type == "metal":
        lines = random.randint(5, 15)
        for _ in range(lines):
            y = random.randint(0, h - 1)
            intensity = random.randint(-15, 15)
            background[y, :] = np.clip(
                background[y, :].astype(np.int16) + intensity, 0, 255
            ).astype(np.uint8)

    else:  # concrete
        for _ in range(20):
            x, y = random.randint(0, w - 5), random.randint(0, h - 5)
            size = random.randint(2, 4)
            intensity = random.randint(-10, 10)
            background[y : y + size, x : x + size] = np.clip(
                background[y : y + size, x : x + size].astype(np.int16) + intensity,
                0,
                255,
            ).astype(np.uint8)

    return background


def create_noisy_background(size: Tuple[int, int]) -> np.ndarray:
    """Create noisy background similar to real industrial environments."""
    h, w = size[1], size[0]

    base_colors = [
        [200, 200, 200],
        [180, 180, 180],
        [220, 220, 220],
        [190, 195, 200],
    ]

    base_color = random.choice(base_colors)
    background = np.full((h, w, 3), base_color, dtype=np.uint8)

    noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return background


def create_industrial_background(size: Tuple[int, int]) -> np.ndarray:
    """Create background that mimics industrial conveyor belts or surfaces."""
    h, w = size[1], size[0]

    industrial_colors = [
        [200, 200, 210],
        [190, 190, 190],
        [210, 210, 220],
        [180, 185, 200],
    ]

    base_color = random.choice(industrial_colors)
    background = np.full((h, w, 3), base_color, dtype=np.uint8)

    pattern_type = random.choice(["lines", "dots", "clean"])

    if pattern_type == "lines":
        for _ in range(random.randint(3, 8)):
            y = random.randint(0, h - 1)
            intensity = random.randint(-8, 8)
            background[y, :] = np.clip(
                background[y, :].astype(np.int16) + intensity, 0, 255
            ).astype(np.uint8)

    elif pattern_type == "dots":
        for _ in range(random.randint(10, 30)):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            intensity = random.randint(-5, 5)
            if 0 <= x < w and 0 <= y < h:
                background[y, x] = np.clip(
                    background[y, x].astype(np.int16) + intensity, 0, 255
                ).astype(np.uint8)

    noise = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return background


def place_object_on_background(
    obj_img: np.ndarray,
    background: np.ndarray,
    is_defect: bool,
    scale_range: Tuple[float, float] = (0.2, 0.6),
) -> Tuple[np.ndarray, List[str]]:
    """Place single object on background with proper scaling and positioning."""
    bg_h, bg_w = background.shape[:2]
    obj_h, obj_w = obj_img.shape[:2]

    scale_factor = random.uniform(scale_range[0], scale_range[1])
    new_w = max(10, int(obj_w * scale_factor))
    new_h = max(10, int(obj_h * scale_factor))

    new_w = min(new_w, bg_w - 10)
    new_h = min(new_h, bg_h - 10)

    if new_w <= 0 or new_h <= 0:
        return background, []

    obj_resized = cv2.resize(obj_img, (new_w, new_h))

    if len(obj_resized.shape) != len(background.shape):
        if len(obj_resized.shape) == 2:
            obj_resized = cv2.cvtColor(obj_resized, cv2.COLOR_GRAY2BGR)

    max_x = max(0, bg_w - new_w)
    max_y = max(0, bg_h - new_h)
    x = random.randint(0, max_x) if max_x > 0 else 0
    y = random.randint(0, max_y) if max_y > 0 else 0

    result_img = background.copy()

    try:
        alpha = random.uniform(0.8, 0.95)
        roi = result_img[y : y + new_h, x : x + new_w]
        blended = cv2.addWeighted(obj_resized, alpha, roi, 1 - alpha, 0)
        result_img[y : y + new_h, x : x + new_w] = blended
    except Exception as e:
        print(f"Error blending object: {e}")
        return background, []

    x_center = (x + new_w / 2) / bg_w
    y_center = (y + new_h / 2) / bg_h
    width = new_w / bg_w
    height = new_h / bg_h

    class_id = 1 if is_defect else 0
    annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    return result_img, [annotation]


def place_multiple_objects(
    background: np.ndarray,
    good_files: List[str],
    defect_files: List[str],
    source_dir: str,
    num_objects: int,
    max_attempts: int = 50,
) -> Tuple[np.ndarray, List[str]]:
    """Place multiple objects on background avoiding overlap."""
    result_img = background.copy()
    annotations = []
    placed_boxes = []

    bg_h, bg_w = background.shape[:2]

    for obj_idx in range(num_objects):
        placed = False

        for attempt in range(max_attempts):
            is_defect = random.random() < 0.3
            files = defect_files if is_defect else good_files

            if not files:
                continue

            category = "defect" if is_defect else "good"
            img_file = random.choice(files)
            obj_path = os.path.join(source_dir, category, img_file)
            obj_img = cv2.imread(obj_path)

            if obj_img is None:
                continue

            scale_factor = random.uniform(0.1, 0.3)
            obj_h, obj_w = obj_img.shape[:2]
            new_w = max(15, int(obj_w * scale_factor))
            new_h = max(15, int(obj_h * scale_factor))

            new_w = min(new_w, bg_w - 20)
            new_h = min(new_h, bg_h - 20)

            if new_w <= 0 or new_h <= 0:
                continue

            obj_resized = cv2.resize(obj_img, (new_w, new_h))

            max_x = max(0, bg_w - new_w)
            max_y = max(0, bg_h - new_h)
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0

            new_box = (x, y, x + new_w, y + new_h)

            overlap = any(
                boxes_overlap(new_box, existing, 10) for existing in placed_boxes
            )

            if not overlap:
                try:
                    alpha = random.uniform(0.8, 0.9)
                    roi = result_img[y : y + new_h, x : x + new_w]
                    blended = cv2.addWeighted(obj_resized, alpha, roi, 1 - alpha, 0)
                    result_img[y : y + new_h, x : x + new_w] = blended

                    x_center = (x + new_w / 2) / bg_w
                    y_center = (y + new_h / 2) / bg_h
                    width = new_w / bg_w
                    height = new_h / bg_h

                    class_id = 1 if is_defect else 0
                    annotations.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )

                    placed_boxes.append(new_box)
                    placed = True
                    break

                except Exception:
                    continue

    return result_img, annotations


def boxes_overlap(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
    min_distance: int = 5,
) -> bool:
    """Check if two boxes overlap or are too close."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_1 -= min_distance
    y1_1 -= min_distance
    x2_1 += min_distance
    y2_1 += min_distance

    return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)


def prepare_yolo_dataset(
    classification_dataset_dir: str, output_dir: str = "yolo_dataset"
) -> str:
    """Convert classification dataset to YOLO format for multi-object detection."""
    for split in ["train", "val", "test"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    for split in ["train", "validation", "test"]:
        yolo_split = "val" if split == "validation" else split

        for class_name in ["normal", "defect"]:
            class_dir = os.path.join(classification_dataset_dir, split, class_name)

            if not os.path.exists(class_dir):
                continue

            image_files = [
                f
                for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]

            for img_file in image_files:
                src_path = os.path.join(class_dir, img_file)
                dst_path = os.path.join(output_dir, "images", yolo_split, img_file)
                shutil.copy2(src_path, dst_path)

                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(output_dir, "labels", yolo_split, label_file)

                with open(label_path, "w") as f:
                    if class_name == "defect":
                        f.write("1 0.5 0.5 0.8 0.8\n")

    yaml_content = f"""# YOLOv8 dataset configuration
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


def prepare_yolo_dataset_from_realistic(
    realistic_data_dir: str,
    output_dir: str = "yolo_dataset_realistic",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> str:
    """Prepare YOLO dataset from realistic training data with backgrounds."""

    for split in ["train", "val", "test"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    images_dir = os.path.join(realistic_data_dir, "images")
    labels_dir = os.path.join(realistic_data_dir, "labels")

    image_files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(image_files)

    total_images = len(image_files)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:],
    }

    for split, files in splits.items():
        for img_file in files:
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_dir, "images", split, img_file)
            shutil.copy2(src_img, dst_img)

            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(output_dir, "labels", split, label_file)

            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                with open(dst_label, "w") as f:
                    f.write("")

    yaml_content = f"""# YOLOv8 dataset configuration for realistic data
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

    print(f"✅ Realistic YOLO dataset prepared in {output_dir}")
    print(f"   Train: {len(splits['train'])} images")
    print(f"   Val: {len(splits['val'])} images")
    print(f"   Test: {len(splits['test'])} images")

    return yaml_path


def generate_synthetic_defects(
    normal_dir: str,
    output_dir: str,
    num_defects_per_image: int = 2,
    defect_types: List[str] = None,
) -> None:
    """Generate synthetic defects for data augmentation."""
    if defect_types is None:
        defect_types = ["scratch", "dent", "discoloration", "crack"]

    os.makedirs(output_dir, exist_ok=True)

    normal_images = [
        f
        for f in os.listdir(normal_dir)
        if f.lower().endswith((".bmp", ".jpg", ".jpeg", ".png"))
    ]

    for img_file in normal_images:
        img_path = os.path.join(normal_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        for i in range(num_defects_per_image):
            defected_img = img.copy()

            for _ in range(random.randint(1, 3)):
                defect_type = random.choice(defect_types)
                if defect_type == "scratch":
                    defected_img = add_scratch(defected_img)
                elif defect_type == "dent":
                    defected_img = add_dent(defected_img)
                elif defect_type == "discoloration":
                    defected_img = add_discoloration(defected_img)
                elif defect_type == "crack":
                    defected_img = add_crack(defected_img)

            output_filename = (
                f"synthetic_{defect_type}_{os.path.splitext(img_file)[0]}_{i}.jpg"
            )
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, defected_img)

    total_generated = len(normal_images) * num_defects_per_image
    print(f"Generated {total_generated} synthetic defect images in {output_dir}")


def add_scratch(img: np.ndarray) -> np.ndarray:
    """Add scratch defect to image."""
    h, w = img.shape[:2]
    start_point = (random.randint(0, w), random.randint(0, h))
    end_point = (random.randint(0, w), random.randint(0, h))
    color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    thickness = random.randint(1, 3)
    cv2.line(img, start_point, end_point, color, thickness)
    return img


def add_dent(img: np.ndarray) -> np.ndarray:
    """Add dent defect to image."""
    h, w = img.shape[:2]
    center = (random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4))
    radius = random.randint(10, 30)
    color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
    cv2.circle(img, center, radius, color, -1)
    return img


def add_discoloration(img: np.ndarray) -> np.ndarray:
    """Add discoloration defect to image."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4))
    radius = random.randint(20, 50)
    cv2.circle(mask, center, radius, 255, -1)

    color_shift = np.random.randint(-50, 50, size=3)
    img_shifted = img.astype(np.int16) + color_shift
    img_shifted = np.clip(img_shifted, 0, 255).astype(np.uint8)

    img = np.where(mask[..., None], img_shifted, img)
    return img


def add_crack(img: np.ndarray) -> np.ndarray:
    """Add crack defect to image."""
    h, w = img.shape[:2]
    points = []
    for i in range(5):
        x = random.randint(0, w)
        y = random.randint(0, h)
        points.append((x, y))

    color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, 1)

    return img


def validate_dataset(dataset_dir: str) -> dict:
    """Validate dataset structure and return statistics."""
    stats = {"total_images": 0, "splits": {}, "classes": {}, "issues": []}

    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if os.path.exists(split_dir):
            stats["splits"][split] = {}
            for class_name in ["normal", "defect"]:
                class_dir = os.path.join(split_dir, class_name)
                if os.path.exists(class_dir):
                    image_files = [
                        f
                        for f in os.listdir(class_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                    ]
                    count = len(image_files)
                    stats["splits"][split][class_name] = count
                    stats["total_images"] += count

                    if class_name not in stats["classes"]:
                        stats["classes"][class_name] = 0
                    stats["classes"][class_name] += count
                else:
                    stats["issues"].append(f"Missing directory: {class_dir}")

    return stats


def print_dataset_statistics(stats: dict) -> None:
    """Print dataset statistics in a formatted way."""
    print("\n📊 Dataset Statistics:")
    print("=" * 40)
    print(f"Total Images: {stats['total_images']}")

    print("\nBy Split:")
    for split, classes in stats["splits"].items():
        total_split = sum(classes.values())
        print(f"  {split}: {total_split} images")
        for class_name, count in classes.items():
            print(f"    {class_name}: {count}")

    print("\nBy Class:")
    for class_name, count in stats["classes"].items():
        print(f"  {class_name}: {count} images")

    if stats["issues"]:
        print("\n⚠️  Issues Found:")
        for issue in stats["issues"]:
            print(f"  - {issue}")


class EnhancedDataManager:
    """Enhanced data manager for handling diverse dataset structures while maintaining backward compatibility."""

    def __init__(self, source_data_dir: str):
        """Initialize with source data directory."""
        self.source_data_dir = source_data_dir
        self.data_categories = self._analyze_data_structure()

    def _analyze_data_structure(self) -> Dict:
        """Analyze the data structure and categorize available datasets."""
        categories = {
            "cropped": {"good": [], "defect": []},
            "background": {"good": [], "defect": []},
            "test": {"mixed": []},
            "stats": {},
        }

        # Define folder mappings
        folder_mappings = {
            "good": {"type": "good", "has_background": False},
            "defect": {"type": "defect", "has_background": False},
            "good_colorless": {"type": "good", "has_background": True},
            "defect_colorless": {"type": "defect", "has_background": True},
            "defect_colorless_deform": {
                "type": "defect",
                "has_background": True,
                "subtype": "deformed",
            },
            "defect_colorless_nowords": {
                "type": "defect",
                "has_background": True,
                "subtype": "no_words",
            },
            "both": {"type": "mixed", "has_background": True},
        }

        # Scan folders and categorize images
        for folder_name, props in folder_mappings.items():
            folder_path = os.path.join(self.source_data_dir, folder_name)

            if not os.path.exists(folder_path):
                continue

            # Handle nested structure in some folders
            image_files = []
            if os.path.isdir(folder_path):
                # Check for nested folders first
                subdirs = [
                    d
                    for d in os.listdir(folder_path)
                    if os.path.isdir(os.path.join(folder_path, d))
                ]

                if subdirs:  # Has subdirectories
                    for subdir in subdirs:
                        subdir_path = os.path.join(folder_path, subdir)
                        sub_files = self._get_image_files(subdir_path)
                        for f in sub_files:
                            image_files.append(os.path.join(subdir, f))
                else:  # Direct image files
                    image_files = self._get_image_files(folder_path)

            # Categorize images
            category_info = {
                "folder": folder_name,
                "path": folder_path,
                "files": image_files,
                "count": len(image_files),
                "has_background": props["has_background"],
                "subtype": props.get("subtype", "standard"),
            }

            if props["type"] == "mixed":
                categories["test"]["mixed"] = category_info
            elif props["has_background"]:
                categories["background"][props["type"]].append(category_info)
            else:
                categories["cropped"][props["type"]].append(category_info)

        # Calculate statistics
        for category in ["cropped", "background"]:
            for class_type in ["good", "defect"]:
                total_count = sum(
                    cat["count"] for cat in categories[category][class_type]
                )
                categories["stats"][f"{category}_{class_type}"] = total_count

        categories["stats"]["test_mixed"] = categories["test"]["mixed"].get("count", 0)

        return categories

    def _get_image_files(self, directory: str) -> List[str]:
        """Get all image files from a directory."""
        if not os.path.exists(directory):
            return []

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".JPG",
            ".JPEG",
            ".PNG",
            ".BMP",
        }
        return [
            f
            for f in os.listdir(directory)
            if any(f.endswith(ext) for ext in image_extensions)
        ]

    def print_data_summary(self):
        """Print a comprehensive summary of available data."""
        print("🗂️  Enhanced Data Structure Analysis")
        print("=" * 50)

        print("\n📋 Cropped Images (no background):")
        for class_type in ["good", "defect"]:
            total = self.data_categories["stats"].get(f"cropped_{class_type}", 0)
            print(f"  {class_type.capitalize():>7}: {total:>4} images")
            for cat in self.data_categories["cropped"][class_type]:
                print(f"    - {cat['folder']}: {cat['count']} images")

        print("\n🌄 Background Images (with background):")
        for class_type in ["good", "defect"]:
            categories = self.data_categories["background"][class_type]
            if categories:
                total = sum(cat["count"] for cat in categories)
                print(f"  {class_type.capitalize():>7}: {total:>4} images")
                for cat in categories:
                    subtype_info = (
                        f" ({cat['subtype']})" if cat["subtype"] != "standard" else ""
                    )
                    print(f"    - {cat['folder']}: {cat['count']} images{subtype_info}")

        print("\n🧪 Test Data:")
        test_data = self.data_categories["test"]["mixed"]
        if test_data:
            print(f"  Mixed images: {test_data['count']} images")
            print(f"    - {test_data['folder']}: {test_data['count']} images")

        # Overall statistics
        total_good = self.data_categories["stats"].get("cropped_good", 0) + sum(
            cat["count"] for cat in self.data_categories["background"]["good"]
        )
        total_defect = self.data_categories["stats"].get("cropped_defect", 0) + sum(
            cat["count"] for cat in self.data_categories["background"]["defect"]
        )
        total_test = self.data_categories["stats"].get("test_mixed", 0)

        print(f"\n📊 Total Summary:")
        print(f"  Good samples: {total_good:>4}")
        print(f"  Defect samples: {total_defect:>4}")
        print(f"  Test samples: {total_test:>4}")
        print(f"  Grand total: {total_good + total_defect + total_test:>4}")


# ========================================
# ENHANCED WRAPPER FUNCTIONS (BACKWARD COMPATIBLE)
# ========================================


def organize_enhanced_dataset(
    source_data_dir: str,
    output_dir: str,
    strategy: str = "mixed",  # "cropped_only", "background_only", "mixed", "legacy"
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    balance_classes: bool = True,
) -> str:
    """Enhanced dataset organization with backward compatibility.

    Args:
        strategy:
            - "legacy": Use original organize_dataset function (backward compatible)
            - "cropped_only": Use only cropped images without backgrounds
            - "background_only": Use only images with backgrounds
            - "mixed": Use all available data (recommended)
    """

    if strategy == "legacy":
        # Use original function for backward compatibility
        organize_dataset(
            source_data_dir, output_dir, train_ratio, val_ratio, test_ratio
        )
        return output_dir

    # Use enhanced functionality
    manager = EnhancedDataManager(source_data_dir)
    manager.print_data_summary()

    # Create directory structure
    for split in ["train", "validation", "test"]:
        for class_name in ["normal", "defect"]:
            os.makedirs(f"{output_dir}/{split}/{class_name}", exist_ok=True)

    # Collect all available images based on strategy
    all_images = {"good": [], "defect": []}

    # Add images based on strategy
    if strategy in ["cropped_only", "mixed"]:
        all_images = _add_cropped_images(manager, all_images)

    if strategy in ["background_only", "mixed"]:
        all_images = _add_background_images(manager, all_images)

    # Balance classes if requested
    if balance_classes:
        all_images = _balance_classes(all_images)

    # Split and copy images
    _split_and_copy_images(all_images, output_dir, train_ratio, val_ratio, test_ratio)

    print(f"✅ Enhanced dataset created in {output_dir}")
    return output_dir


def _add_cropped_images(manager: EnhancedDataManager, all_images: Dict) -> Dict:
    """Add cropped images to the collection."""
    print("  📸 Adding cropped images...")

    for class_type in ["good", "defect"]:
        for cat in manager.data_categories["cropped"][class_type]:
            folder_path = cat["path"]
            for img_file in cat["files"]:
                img_path = os.path.join(folder_path, img_file)
                all_images[class_type].append(
                    {
                        "path": img_path,
                        "type": "cropped",
                        "source": cat["folder"],
                        "augment": True,
                    }
                )

    print(
        f"    Good: {len([img for img in all_images['good'] if img['type'] == 'cropped'])}"
    )
    print(
        f"    Defect: {len([img for img in all_images['defect'] if img['type'] == 'cropped'])}"
    )

    return all_images


def _add_background_images(manager: EnhancedDataManager, all_images: Dict) -> Dict:
    """Add background images to the collection."""
    print("  🌄 Adding background images...")

    for class_type in ["good", "defect"]:
        for cat in manager.data_categories["background"][class_type]:
            folder_path = cat["path"]

            for img_file in cat["files"]:
                if "/" in img_file or "\\" in img_file:  # Nested file
                    img_path = os.path.join(folder_path, img_file)
                else:  # Direct file
                    img_path = os.path.join(folder_path, img_file)

                all_images[class_type].append(
                    {
                        "path": img_path,
                        "type": "background",
                        "source": cat["folder"],
                        "subtype": cat["subtype"],
                        "augment": False,
                    }
                )

    good_bg_count = len(
        [img for img in all_images["good"] if img["type"] == "background"]
    )
    defect_bg_count = len(
        [img for img in all_images["defect"] if img["type"] == "background"]
    )
    print(f"    Good: {good_bg_count}")
    print(f"    Defect: {defect_bg_count}")

    return all_images


def _balance_classes(all_images: Dict) -> Dict:
    """Balance classes by upsampling minority class."""
    good_count = len(all_images["good"])
    defect_count = len(all_images["defect"])

    print(f"  ⚖️  Balancing classes: Good={good_count}, Defect={defect_count}")

    if good_count == defect_count:
        print("    Classes already balanced!")
        return all_images

    # Determine which class to upsample
    if good_count < defect_count:
        minority_class = "good"
        target_count = defect_count
    else:
        minority_class = "defect"
        target_count = good_count

    # Upsample minority class
    minority_images = all_images[minority_class].copy()
    while len(all_images[minority_class]) < target_count:
        img_to_duplicate = random.choice(minority_images)
        all_images[minority_class].append(img_to_duplicate.copy())

    print(
        f"    After balancing: Good={len(all_images['good'])}, Defect={len(all_images['defect'])}"
    )

    return all_images


def _split_and_copy_images(
    all_images: Dict,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    """Split images and copy to appropriate directories."""
    print("  📂 Splitting and copying images...")

    for class_type in ["good", "defect"]:
        images = all_images[class_type]
        random.shuffle(images)

        # Calculate split indices
        total_count = len(images)
        train_end = int(total_count * train_ratio)
        val_end = train_end + int(total_count * val_ratio)

        # Split images
        splits = {
            "train": images[:train_end],
            "validation": images[train_end:val_end],
            "test": images[val_end:],
        }

        output_class = "normal" if class_type == "good" else "defect"

        for split_name, split_images in splits.items():
            split_dir = os.path.join(output_dir, split_name, output_class)

            for i, img_info in enumerate(split_images):
                src_path = img_info["path"]

                if not os.path.exists(src_path):
                    print(f"    ⚠️  File not found: {src_path}")
                    continue

                # Generate unique filename
                base_name = f"{img_info['source']}_{img_info['type']}_{i:04d}"
                if img_info.get("subtype", "standard") != "standard":
                    base_name += f"_{img_info['subtype']}"

                dst_filename = f"{base_name}.jpg"
                dst_path = os.path.join(split_dir, dst_filename)

                try:
                    # Copy and optionally process image
                    img = cv2.imread(src_path)
                    if img is not None:
                        # Apply augmentation if specified
                        if img_info.get("augment", False):
                            img = _apply_basic_augmentation(img)

                        # Ensure consistent format
                        cv2.imwrite(dst_path, img)
                    else:
                        print(f"    ⚠️  Could not read image: {src_path}")

                except Exception as e:
                    print(f"    ❌ Error copying {src_path}: {str(e)}")

    # Print final statistics
    _print_split_statistics(output_dir)


def _apply_basic_augmentation(img: np.ndarray) -> np.ndarray:
    """Apply basic augmentation to cropped images."""
    # Random brightness adjustment
    if random.random() < 0.5:
        brightness = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)

    # Random rotation (small angles)
    if random.random() < 0.3:
        angle = random.uniform(-10, 10)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))

    # Random noise
    if random.random() < 0.2:
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    return img


def _print_split_statistics(output_dir: str):
    """Print statistics about the created dataset splits."""
    print(f"\n  📊 Final Dataset Statistics:")

    for split in ["train", "validation", "test"]:
        for class_name in ["normal", "defect"]:
            split_dir = os.path.join(output_dir, split, class_name)
            if os.path.exists(split_dir):
                count = len(
                    [
                        f
                        for f in os.listdir(split_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                    ]
                )
                print(f"    {split:>10}/{class_name:>6}: {count:>4} images")


def prepare_enhanced_yolo_dataset(
    source_data_dir: str,
    classification_dataset_dir: str,
    output_dir: str = "enhanced_yolo_dataset",
    multi_object_scenes: int = 200,
    use_legacy: bool = False,
) -> str:
    """Enhanced YOLO dataset preparation with backward compatibility.

    Args:
        use_legacy: If True, uses original prepare_yolo_dataset function
    """

    if use_legacy:
        # Use original function for backward compatibility
        return prepare_yolo_dataset(classification_dataset_dir, output_dir)

    print(f"🎯 Creating Enhanced YOLO Dataset")
    print(f"   Source: {source_data_dir}")
    print(f"   Classification data: {classification_dataset_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Multi-object scenes: {multi_object_scenes}")

    # Create enhanced data manager
    manager = EnhancedDataManager(source_data_dir)

    # Create YOLO directory structure
    for split in ["train", "val", "test"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    # First, create basic YOLO dataset from classification data
    _convert_classification_to_yolo(classification_dataset_dir, output_dir)

    # Add realistic multi-object scenes using enhanced manager
    _add_realistic_multi_object_scenes(manager, output_dir, multi_object_scenes)

    # Create dataset.yaml
    yaml_path = _create_yolo_config(output_dir)

    # Print dataset statistics
    _print_yolo_statistics(output_dir)

    print(f"✅ Enhanced YOLO dataset created: {yaml_path}")
    return yaml_path


def _convert_classification_to_yolo(classification_dir: str, output_dir: str):
    """Convert classification dataset to YOLO format."""
    print("  🔄 Converting classification data to YOLO format...")

    for split in ["train", "validation", "test"]:
        yolo_split = "val" if split == "validation" else split

        for class_name in ["normal", "defect"]:
            class_dir = os.path.join(classification_dir, split, class_name)

            if not os.path.exists(class_dir):
                continue

            image_files = [
                f
                for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]

            for img_file in image_files:
                src_path = os.path.join(class_dir, img_file)

                # Copy image
                dst_img_path = os.path.join(output_dir, "images", yolo_split, img_file)
                shutil.copy2(src_path, dst_img_path)

                # Create label
                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(output_dir, "labels", yolo_split, label_file)

                with open(label_path, "w") as f:
                    if class_name == "defect":
                        # For defect images, create a bounding box covering most of the image
                        f.write("1 0.5 0.5 0.8 0.8\n")  # class 1 for defect
                    # For normal images, leave empty (no defects to annotate)


def _add_realistic_multi_object_scenes(
    manager: EnhancedDataManager, output_dir: str, num_scenes: int
):
    """Add realistic multi-object scenes to the YOLO dataset."""
    print(f"  🌄 Creating {num_scenes} realistic multi-object scenes...")

    # Use background images and cropped objects to create realistic scenes
    background_images = []
    cropped_objects = {"good": [], "defect": []}

    # Collect background images
    for class_type in ["good", "defect"]:
        for cat in manager.data_categories["background"][class_type]:
            folder_path = cat["path"]
            for img_file in cat["files"][:10]:  # Limit for backgrounds
                if "/" in img_file or "\\" in img_file:
                    img_path = os.path.join(folder_path, img_file)
                else:
                    img_path = os.path.join(folder_path, img_file)
                background_images.append(img_path)

    # Collect cropped objects
    for class_type in ["good", "defect"]:
        for cat in manager.data_categories["cropped"][class_type]:
            folder_path = cat["path"]
            for img_file in cat["files"]:
                img_path = os.path.join(folder_path, img_file)
                cropped_objects[class_type].append(img_path)

    # Create multi-object scenes
    created_scenes = 0
    for i in range(num_scenes):
        try:
            # Choose random background or create synthetic one
            if background_images and random.random() < 0.3:
                bg_path = random.choice(background_images)
                background = cv2.imread(bg_path)
                if background is None:
                    background = create_random_background((640, 640))
                else:
                    background = cv2.resize(background, (640, 640))
            else:
                background = create_random_background((640, 640))

            # Place multiple objects
            scene_img, annotations = _create_multi_object_scene(
                background, cropped_objects, num_objects=(1, 4)
            )

            # Save to train split (80% train, 20% val)
            split = "train" if random.random() < 0.8 else "val"

            # Save image
            img_name = f"enhanced_multi_scene_{i:04d}.jpg"
            img_path = os.path.join(output_dir, "images", split, img_name)
            cv2.imwrite(img_path, scene_img)

            # Save annotations
            label_name = f"enhanced_multi_scene_{i:04d}.txt"
            label_path = os.path.join(output_dir, "labels", split, label_name)
            with open(label_path, "w") as f:
                f.write("\n".join(annotations))

            created_scenes += 1

        except Exception as e:
            print(f"    ⚠️  Failed to create scene {i}: {str(e)}")
            continue

    print(f"    ✅ Created {created_scenes} realistic scenes")


def _create_multi_object_scene(
    background: np.ndarray, objects: Dict, num_objects: Tuple[int, int]
) -> Tuple[np.ndarray, List[str]]:
    """Create a scene with multiple objects placed on background."""
    scene = background.copy()
    annotations = []
    placed_boxes = []

    bg_h, bg_w = background.shape[:2]
    num_objs = random.randint(num_objects[0], num_objects[1])

    for _ in range(num_objs):
        # Choose object type (70% good, 30% defect)
        obj_type = "defect" if random.random() < 0.3 else "good"

        if not objects[obj_type]:
            continue

        # Load random object
        obj_path = random.choice(objects[obj_type])
        obj_img = cv2.imread(obj_path)

        if obj_img is None:
            continue

        # Resize object
        scale = random.uniform(0.15, 0.4)
        new_w = max(20, int(obj_img.shape[1] * scale))
        new_h = max(20, int(obj_img.shape[0] * scale))

        # Ensure object fits
        new_w = min(new_w, bg_w - 20)
        new_h = min(new_h, bg_h - 20)

        if new_w <= 0 or new_h <= 0:
            continue

        obj_resized = cv2.resize(obj_img, (new_w, new_h))

        # Find position without overlap
        max_attempts = 20
        placed = False

        for _ in range(max_attempts):
            x = random.randint(0, bg_w - new_w)
            y = random.randint(0, bg_h - new_h)

            new_box = (x, y, x + new_w, y + new_h)

            # Check overlap
            overlap = any(
                boxes_overlap(new_box, existing, 15) for existing in placed_boxes
            )

            if not overlap:
                # Place object
                try:
                    roi = scene[y : y + new_h, x : x + new_w]
                    alpha = random.uniform(0.85, 0.95)
                    blended = cv2.addWeighted(obj_resized, alpha, roi, 1 - alpha, 0)
                    scene[y : y + new_h, x : x + new_w] = blended

                    # Create annotation
                    x_center = (x + new_w / 2) / bg_w
                    y_center = (y + new_h / 2) / bg_h
                    width = new_w / bg_w
                    height = new_h / bg_h

                    class_id = 1 if obj_type == "defect" else 0
                    annotations.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )

                    placed_boxes.append(new_box)
                    placed = True
                    break

                except Exception:
                    continue

    return scene, annotations


def _create_yolo_config(output_dir: str) -> str:
    """Create YOLO configuration file."""
    yaml_content = f"""# Enhanced YOLOv8 dataset configuration
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

    return yaml_path


def _print_yolo_statistics(output_dir: str):
    """Print YOLO dataset statistics."""
    print(f"\n  📊 YOLO Dataset Statistics:")

    total_images = 0
    total_annotations = 0

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(output_dir, "images", split)
        label_dir = os.path.join(output_dir, "labels", split)

        if os.path.exists(img_dir):
            img_files = [
                f
                for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
            img_count = len(img_files)
            total_images += img_count

            # Count annotations
            ann_count = 0
            if os.path.exists(label_dir):
                for label_file in os.listdir(label_dir):
                    if label_file.endswith(".txt"):
                        label_path = os.path.join(label_dir, label_file)
                        try:
                            with open(label_path, "r") as f:
                                ann_count += len([line for line in f if line.strip()])
                        except:
                            continue

            total_annotations += ann_count
            print(f"    {split:>5}: {img_count:>4} images, {ann_count:>4} annotations")

    print(
        f"    {'Total':>5}: {total_images:>4} images, {total_annotations:>4} annotations"
    )
