import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import json
import yaml


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
        image_files = []
        for ext in ["*.JPG", "*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(list(Path(class_dir).glob(ext)))

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

            dest_path = os.path.join(dest_dir, file_path.name)
            shutil.copy(file_path, dest_path)

    print(f"Dataset organized into {output_dir}")


def augment_with_backgrounds(
    object_images_dir: str,
    output_dir: str,
    background_images: List[str] = None,
    objects_per_image: Tuple[int, int] = (1, 4),
    images_per_object: int = 3,
    multi_object_scenes: int = 100,
) -> str:
    """Create training images with objects placed on realistic backgrounds.

    This addresses the domain gap between cropped training images and
    real-world test images with backgrounds and multiple objects.
    """

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

    # Create single-object training images (similar to your cropped data but with backgrounds)
    print("  Creating single-object training images...")
    for category, files in [("good", good_files), ("defect", defect_files)]:
        category_dir = os.path.join(object_images_dir, category)

        for img_file in files:
            obj_path = os.path.join(category_dir, img_file)
            obj_img = cv2.imread(obj_path)

            if obj_img is None:
                continue

            obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)

            # Create multiple variations with different backgrounds
            for var in range(images_per_object):
                # Create random background
                bg_img = create_random_background((640, 640))

                # Place object on background
                placed_img, annotations = place_object_on_background(
                    obj_img, bg_img, category == "defect"
                )

                # Save image and annotation
                img_name = (
                    f"{category}_{os.path.splitext(img_file)[0]}_var{var:02d}.jpg"
                )
                img_path = os.path.join(output_dir, "images", img_name)
                cv2.imwrite(img_path, cv2.cvtColor(placed_img, cv2.COLOR_RGB2BGR))

                # Save YOLO annotation
                ann_name = (
                    f"{category}_{os.path.splitext(img_file)[0]}_var{var:02d}.txt"
                )
                ann_path = os.path.join(output_dir, "labels", ann_name)
                with open(ann_path, "w") as f:
                    f.write("\n".join(annotations))

                image_count += 1

    # Create multi-object scenes for advanced training
    print("  Creating multi-object training scenes...")
    for i in range(multi_object_scenes):
        bg_img = create_random_background((640, 640))
        num_objects = random.randint(objects_per_image[0], objects_per_image[1])

        final_img, annotations = place_multiple_objects(
            bg_img, good_files, defect_files, object_images_dir, num_objects
        )

        img_name = f"multi_{i:04d}.jpg"
        img_path = os.path.join(output_dir, "images", img_name)
        cv2.imwrite(img_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

        ann_name = f"multi_{i:04d}.txt"
        ann_path = os.path.join(output_dir, "labels", ann_name)
        with open(ann_path, "w") as f:
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

    # Create gradient direction (horizontal, vertical, or diagonal)
    direction = random.choice(["horizontal", "vertical", "diagonal"])

    if direction == "horizontal":
        for i in range(w):
            color_val = int(180 + 75 * (i / w))  # Light to darker
            background[:, i] = [color_val, color_val, color_val]
    elif direction == "vertical":
        for i in range(h):
            color_val = int(180 + 75 * (i / h))
            background[i, :] = [color_val, color_val, color_val]
    else:  # diagonal
        for i in range(h):
            for j in range(w):
                color_val = int(180 + 75 * ((i + j) / (h + w)))
                background[i, j] = [color_val, color_val, color_val]

    # Add subtle noise
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return background


def create_texture_background(size: Tuple[int, int]) -> np.ndarray:
    """Create textured background simulating various surfaces."""
    h, w = size[1], size[0]
    base_color = random.randint(160, 240)
    background = np.full((h, w, 3), base_color, dtype=np.uint8)

    # Add different texture patterns
    texture_type = random.choice(["fabric", "metal", "concrete"])

    if texture_type == "fabric":
        # Create fabric-like texture
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                intensity = random.randint(-15, 15)
                background[i : i + 4, j : j + 4] = np.clip(
                    background[i : i + 4, j : j + 4].astype(np.int16) + intensity,
                    0,
                    255,
                ).astype(np.uint8)

    elif texture_type == "metal":
        # Create metallic texture with horizontal lines
        for i in range(0, h, 2):
            intensity = random.randint(-10, 10)
            background[i : i + 1, :] = np.clip(
                background[i : i + 1, :].astype(np.int16) + intensity, 0, 255
            ).astype(np.uint8)

    else:  # concrete
        # Random speckled texture
        noise = np.random.normal(0, 12, (h, w)).astype(np.int16)
        for c in range(3):
            background[:, :, c] = np.clip(
                background[:, :, c].astype(np.int16) + noise, 0, 255
            ).astype(np.uint8)

    return background


def create_noisy_background(size: Tuple[int, int]) -> np.ndarray:
    """Create noisy background similar to real industrial environments."""
    h, w = size[1], size[0]

    # Random base color in realistic range
    base_colors = [
        [200, 200, 200],  # Light gray
        [180, 180, 180],  # Medium gray
        [220, 220, 220],  # Very light gray
        [190, 195, 200],  # Slightly blue-gray
    ]

    base_color = random.choice(base_colors)
    background = np.full((h, w, 3), base_color, dtype=np.uint8)

    # Add realistic noise
    noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return background


def create_industrial_background(size: Tuple[int, int]) -> np.ndarray:
    """Create background that mimics industrial conveyor belts or surfaces."""
    h, w = size[1], size[0]

    # Industrial colors (blues, grays, whites)
    industrial_colors = [
        [200, 200, 210],  # Light blue-gray
        [190, 190, 190],  # Light gray
        [210, 210, 220],  # Very light blue-gray
        [180, 185, 200],  # Blue-gray
    ]

    base_color = random.choice(industrial_colors)
    background = np.full((h, w, 3), base_color, dtype=np.uint8)

    # Add subtle patterns that might appear on industrial surfaces
    pattern_type = random.choice(["lines", "dots", "clean"])

    if pattern_type == "lines":
        # Subtle horizontal or vertical lines
        line_direction = random.choice(["horizontal", "vertical"])
        spacing = random.randint(20, 50)

        for i in range(0, h if line_direction == "horizontal" else w, spacing):
            intensity = random.randint(-8, 8)
            if line_direction == "horizontal":
                background[i : i + 1, :] = np.clip(
                    background[i : i + 1, :].astype(np.int16) + intensity, 0, 255
                ).astype(np.uint8)
            else:
                background[:, i : i + 1] = np.clip(
                    background[:, i : i + 1].astype(np.int16) + intensity, 0, 255
                ).astype(np.uint8)

    elif pattern_type == "dots":
        # Small dots or speckles
        num_dots = random.randint(20, 80)
        for _ in range(num_dots):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            intensity = random.randint(-20, 20)
            background[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = np.clip(
                background[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2].astype(
                    np.int16
                )
                + intensity,
                0,
                255,
            ).astype(np.uint8)

    # Add final noise
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

    # Scale object to reasonable size
    scale_factor = random.uniform(scale_range[0], scale_range[1])
    new_w = max(10, int(obj_w * scale_factor))  # Ensure minimum size
    new_h = max(10, int(obj_h * scale_factor))

    # Ensure the object fits in the background
    new_w = min(new_w, bg_w - 10)  # Leave some margin
    new_h = min(new_h, bg_h - 10)

    if new_w <= 0 or new_h <= 0:
        # If object is too small or background too small, return background unchanged
        return background.copy(), []

    # Resize object
    obj_resized = cv2.resize(obj_img, (new_w, new_h))

    # Ensure object has same number of channels as background
    if len(obj_resized.shape) != len(background.shape):
        if len(obj_resized.shape) == 2:  # Grayscale object
            obj_resized = cv2.cvtColor(obj_resized, cv2.COLOR_GRAY2RGB)
        elif len(background.shape) == 2:  # Grayscale background
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

    # Random position (ensure object fits)
    max_x = max(0, bg_w - new_w)
    max_y = max(0, bg_h - new_h)
    x = random.randint(0, max_x) if max_x > 0 else 0
    y = random.randint(0, max_y) if max_y > 0 else 0

    # Place object on background
    result_img = background.copy()

    try:
        # Extract ROI from background
        roi = result_img[y : y + new_h, x : x + new_w]

        # Ensure shapes match exactly
        if roi.shape != obj_resized.shape:
            # Crop or pad to match
            min_h = min(roi.shape[0], obj_resized.shape[0])
            min_w = min(roi.shape[1], obj_resized.shape[1])

            roi = roi[:min_h, :min_w]
            obj_resized = obj_resized[:min_h, :min_w]

            # Update actual dimensions used
            new_h, new_w = min_h, min_w

        # Add realistic blending
        alpha = random.uniform(0.85, 1.0)  # Slight transparency

        # Blend images
        blended = cv2.addWeighted(obj_resized, alpha, roi, 1 - alpha, 0)
        result_img[y : y + new_h, x : x + new_w] = blended

    except Exception as e:
        # Fallback: direct placement without blending
        print(f"Warning: Blending failed ({str(e)}), using direct placement")
        try:
            result_img[y : y + new_h, x : x + new_w] = obj_resized
        except Exception as e2:
            print(f"Warning: Direct placement also failed ({str(e2)}), skipping object")
            return background.copy(), []

    # Create YOLO annotation (normalized coordinates)
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
    annotations = []  # Track placed objects to avoid overlap
    placed_boxes = []

    bg_h, bg_w = background.shape[:2]

    for obj_idx in range(num_objects):
        # Choose object type (70% good, 30% defect for realistic distribution)
        is_defect = random.random() < 0.3
        files = defect_files if is_defect else good_files
        category = "defect" if is_defect else "good"

        if not files:
            continue

        # Load random object
        obj_file = random.choice(files)
        obj_path = os.path.join(source_dir, category, obj_file)
        obj_img = cv2.imread(obj_path)

        if obj_img is None:
            continue

        obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)

        # Try to place object without overlap
        placed = False
        attempts = 0

        while not placed and attempts < max_attempts:
            # Scale object (smaller for multi-object scenes)
            scale_factor = random.uniform(0.15, 0.35)
            new_w = max(20, int(obj_img.shape[1] * scale_factor))
            new_h = max(20, int(obj_img.shape[0] * scale_factor))

            # Ensure object fits in background
            new_w = min(new_w, bg_w - 20)
            new_h = min(new_h, bg_h - 20)

            if new_w <= 0 or new_h <= 0:
                break

            max_x = max(0, bg_w - new_w)
            max_y = max(0, bg_h - new_h)

            if max_x <= 0 or max_y <= 0:
                break

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            # Check overlap with existing objects
            new_box = (x, y, x + new_w, y + new_h)
            overlap = any(
                boxes_overlap(new_box, existing, min_distance=10)
                for existing in placed_boxes
            )

            if not overlap:
                try:
                    # Place object
                    obj_resized = cv2.resize(obj_img, (new_w, new_h))

                    # Extract ROI and ensure shapes match
                    roi = result_img[y : y + new_h, x : x + new_w]

                    if roi.shape != obj_resized.shape:
                        # Adjust to match shapes
                        min_h = min(roi.shape[0], obj_resized.shape[0])
                        min_w = min(roi.shape[1], obj_resized.shape[1])

                        roi = roi[:min_h, :min_w]
                        obj_resized = obj_resized[:min_h, :min_w]

                        # Update actual dimensions
                        new_h, new_w = min_h, min_w

                    # Add realistic blending
                    alpha = random.uniform(0.8, 0.95)
                    blended = cv2.addWeighted(obj_resized, alpha, roi, 1 - alpha, 0)
                    result_img[y : y + new_h, x : x + new_w] = blended

                    # Create annotation
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

                except Exception as e:
                    # If blending fails, try direct placement
                    try:
                        obj_resized = cv2.resize(obj_img, (new_w, new_h))
                        result_img[y : y + new_h, x : x + new_w] = obj_resized

                        # Create annotation
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

                    except Exception as e2:
                        print(f"Warning: Failed to place object {obj_file}: {str(e2)}")

            attempts += 1

    return result_img, annotations


def boxes_overlap(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
    min_distance: int = 5,
) -> bool:
    """Check if two boxes overlap or are too close."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Expand boxes by min_distance to ensure spacing
    x1_1 -= min_distance
    y1_1 -= min_distance
    x2_1 += min_distance
    y2_1 += min_distance

    return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)


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
                if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
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
                if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
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

    # Create YOLO directory structure
    for split in ["train", "val", "test"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    # Get all images and labels
    images_dir = os.path.join(realistic_data_dir, "images")
    labels_dir = os.path.join(realistic_data_dir, "labels")

    image_files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Shuffle and split
    random.shuffle(image_files)

    total_images = len(image_files)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    # Split files
    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:],
    }

    # Copy files to appropriate splits
    for split, files in splits.items():
        for img_file in files:
            # Copy image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_dir, "images", split, img_file)
            shutil.copy2(src_img, dst_img)

            # Copy corresponding label
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(output_dir, "labels", split, label_file)

            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                # Create empty label file if none exists
                with open(dst_label, "w") as f:
                    pass

    # Create dataset.yaml
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
        defect_types = ["scratches", "spots", "noise", "missing_parts", "discoloration"]

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

        base_name = os.path.splitext(img_file)[0]

        for i in range(num_defects_per_image):
            defect_img = img.copy()
            defect_type = random.choice(defect_types)

            if defect_type == "scratches":
                # Create realistic scratches
                num_scratches = random.randint(1, 3)
                for _ in range(num_scratches):
                    x1, y1 = random.randint(0, img.shape[1]), random.randint(
                        0, img.shape[0]
                    )
                    length = random.randint(20, 100)
                    angle = random.uniform(0, 2 * np.pi)
                    x2 = int(x1 + length * np.cos(angle))
                    y2 = int(y1 + length * np.sin(angle))

                    # Ensure coordinates are within image bounds
                    x2 = max(0, min(img.shape[1] - 1, x2))
                    y2 = max(0, min(img.shape[0] - 1, y2))

                    thickness = random.randint(1, 3)
                    color = (
                        random.randint(0, 100),
                        random.randint(0, 100),
                        random.randint(0, 100),
                    )
                    cv2.line(defect_img, (x1, y1), (x2, y2), color, thickness)

            elif defect_type == "spots":
                # Create spots/stains
                num_spots = random.randint(1, 4)
                for _ in range(num_spots):
                    cx = random.randint(0, img.shape[1] - 1)
                    cy = random.randint(0, img.shape[0] - 1)
                    radius = random.randint(3, 15)
                    color = (
                        random.randint(0, 150),
                        random.randint(0, 150),
                        random.randint(0, 150),
                    )
                    cv2.circle(defect_img, (cx, cy), radius, color, -1)

            elif defect_type == "noise":
                # Add localized noise
                x = random.randint(0, max(0, img.shape[1] - 50))
                y = random.randint(0, max(0, img.shape[0] - 50))
                w = random.randint(30, min(50, img.shape[1] - x))
                h = random.randint(30, min(50, img.shape[0] - y))

                roi = defect_img[y : y + h, x : x + w]
                noise = np.random.randint(0, 50, roi.shape, dtype=np.uint8)
                defect_img[y : y + h, x : x + w] = cv2.add(roi, noise)

            elif defect_type == "missing_parts":
                # Create missing parts (holes/dark areas)
                num_holes = random.randint(1, 2)
                for _ in range(num_holes):
                    x = random.randint(0, max(0, img.shape[1] - 40))
                    y = random.randint(0, max(0, img.shape[0] - 40))
                    w = random.randint(10, min(40, img.shape[1] - x))
                    h = random.randint(10, min(40, img.shape[0] - y))

                    color = (
                        random.randint(0, 50),
                        random.randint(0, 50),
                        random.randint(0, 50),
                    )
                    cv2.rectangle(defect_img, (x, y), (x + w, y + h), color, -1)

            elif defect_type == "discoloration":
                # Create discoloration areas
                x = random.randint(0, max(0, img.shape[1] - 60))
                y = random.randint(0, max(0, img.shape[0] - 60))
                w = random.randint(40, min(80, img.shape[1] - x))
                h = random.randint(40, min(80, img.shape[0] - y))

                # Create a subtle color shift
                roi = defect_img[y : y + h, x : x + w].astype(np.float32)
                color_shift = np.random.uniform(-30, 30, 3)
                roi += color_shift
                defect_img[y : y + h, x : x + w] = np.clip(roi, 0, 255).astype(np.uint8)

            output_path = os.path.join(
                output_dir, f"{base_name}_defect_{defect_type}_{i+1}.jpg"
            )
            cv2.imwrite(output_path, defect_img)

    total_generated = len(normal_images) * num_defects_per_image
    print(
        f"✅ Generated {total_generated} synthetic defects with {len(defect_types)} defect types"
    )


def validate_dataset(dataset_dir: str) -> dict:
    """Validate dataset structure and return statistics."""
    stats = {
        "images": {"train": 0, "val": 0, "test": 0},
        "labels": {"train": 0, "val": 0, "test": 0},
        "classes": {"normal": 0, "defect": 0},
        "errors": [],
    }

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dataset_dir, "images", split)
        label_dir = os.path.join(dataset_dir, "labels", split)

        if os.path.exists(img_dir):
            img_files = [
                f
                for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            stats["images"][split] = len(img_files)

            # Check corresponding labels
            for img_file in img_files:
                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(label_dir, label_file)

                if os.path.exists(label_path):
                    stats["labels"][split] += 1

                    # Count classes
                    with open(label_path, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.strip():
                                class_id = int(line.split()[0])
                                if class_id == 0:
                                    stats["classes"]["normal"] += 1
                                elif class_id == 1:
                                    stats["classes"]["defect"] += 1
                else:
                    stats["errors"].append(f"Missing label for {img_file} in {split}")

    return stats


def print_dataset_statistics(stats: dict) -> None:
    """Print formatted dataset statistics."""
    print("\n📊 Dataset Statistics:")
    print("=" * 40)

    print(f"Images per split:")
    for split in ["train", "val", "test"]:
        img_count = stats["images"][split]
        label_count = stats["labels"][split]
        print(f"  {split:>5}: {img_count:>4} images, {label_count:>4} labels")

    print(f"\nClass distribution:")
    total_objects = sum(stats["classes"].values())
    for class_name, count in stats["classes"].items():
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"  {class_name:>7}: {count:>4} objects ({percentage:>5.1f}%)")

    if stats["errors"]:
        print(f"\n⚠️  Errors found:")
        for error in stats["errors"][:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(stats["errors"]) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more errors")
    else:
        print(f"\n✅ No errors found in dataset structure")
