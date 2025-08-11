"""Visualization utilities for defect detection."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Visualizer:
    """Visualization utilities for defect detection results."""

    @staticmethod
    def visualize_augmentations(image_path=None, dataset_dir=None, n_samples=5):
        """Visualize data augmentation results."""
        # Get sample image
        if image_path is None and dataset_dir:
            normal_dir = f"{dataset_dir}/train/normal"
            if os.path.exists(normal_dir):
                image_files = [
                    f
                    for f in os.listdir(normal_dir)
                    if f.lower().endswith((".bmp", ".jpg", ".jpeg", ".png"))
                ]
                if image_files:
                    image_path = os.path.join(normal_dir, image_files[0])

        if not image_path or not os.path.exists(image_path):
            print("No valid image path provided")
            return None

        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))

        # Create data generator
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
        )

        # Generate augmented images
        aug_images = []
        img_array = np.expand_dims(img_resized, axis=0)
        aug_iter = datagen.flow(img_array, batch_size=1)

        for i in range(n_samples):
            aug_images.append(next(aug_iter)[0].astype("uint8"))

        # Plot results
        plt.figure(figsize=(15, 4))

        # Original
        plt.subplot(1, n_samples + 1, 1)
        plt.imshow(img_resized)
        plt.title("Original")
        plt.axis("off")

        # Augmented
        for i in range(n_samples):
            plt.subplot(1, n_samples + 1, i + 2)
            plt.imshow(aug_images[i])
            plt.title(f"Aug {i+1}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig("augmentation_visualization.png", dpi=150, bbox_inches="tight")
        plt.show()

        return aug_images

    @staticmethod
    def visualize_model_predictions(model, test_dir, num_images=10, threshold=0.5):
        """Visualize model predictions on test images."""
        # Get test images
        normal_dir = os.path.join(test_dir, "normal")
        defect_dir = os.path.join(test_dir, "defect")

        normal_images = [
            os.path.join(normal_dir, f)
            for f in os.listdir(normal_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        defect_images = [
            os.path.join(defect_dir, f)
            for f in os.listdir(defect_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        # Select random images
        selected_normals = random.sample(
            normal_images, min(num_images // 2, len(normal_images))
        )
        selected_defects = random.sample(
            defect_images, min(num_images // 2, len(defect_images))
        )

        selected_images = selected_normals + selected_defects
        random.shuffle(selected_images)

        # Create visualization
        plt.figure(figsize=(15, num_images * 2))

        for i, img_path in enumerate(selected_images[:num_images]):
            # Load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get true class
            true_class = "Normal" if "/normal/" in img_path else "Defect"

            # Get predictions
            detections = model.predict(img)

            # Original image
            plt.subplot(num_images, 2, i * 2 + 1)
            plt.imshow(img)
            plt.title(f"Original ({true_class})")
            plt.axis("off")

            # Prediction visualization
            plt.subplot(num_images, 2, i * 2 + 2)
            result_img = model.visualize_detections(img, detections, threshold)
            plt.imshow(result_img)
            plt.title("Prediction")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig("prediction_visualization.png", dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_training_curves(history, save_path="training_curves.png"):
        """Plot training history curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(history.history["accuracy"])
        axes[0, 0].plot(history.history["val_accuracy"])
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend(["Train", "Validation"])
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(history.history["loss"])
        axes[0, 1].plot(history.history["val_loss"])
        axes[0, 1].set_title("Model Loss")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend(["Train", "Validation"])
        axes[0, 1].grid(True)

        # Precision (if available)
        if "precision" in history.history:
            axes[1, 0].plot(history.history["precision"])
            axes[1, 0].plot(history.history["val_precision"])
            axes[1, 0].set_title("Model Precision")
            axes[1, 0].set_ylabel("Precision")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].legend(["Train", "Validation"])
            axes[1, 0].grid(True)

        # Recall (if available)
        if "recall" in history.history:
            axes[1, 1].plot(history.history["recall"])
            axes[1, 1].plot(history.history["val_recall"])
            axes[1, 1].set_title("Model Recall")
            axes[1, 1].set_ylabel("Recall")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].legend(["Train", "Validation"])
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
