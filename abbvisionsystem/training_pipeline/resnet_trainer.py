"""ResNet-based classification model for single object defect detection."""

import os
import json
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from typing import Tuple, Dict, Optional


class DefectClassificationModel:
    """ResNet50V2-based binary classification model for defect detection."""

    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """Initialize the classification model."""
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.class_names = {0: "normal", 1: "defect"}

    def build_model(self, use_pretrained: bool = True) -> tf.keras.Model:
        """Build ResNet50V2-based model architecture."""
        # Base model
        base_model = ResNet50V2(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet" if use_pretrained else None,
        )

        # Complete model
        self.model = Sequential(
            [
                base_model,
                GlobalAveragePooling2D(),
                BatchNormalization(),
                Dense(256, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid"),  # Binary classification
            ]
        )

        # Fine-tuning: freeze early layers, unfreeze later layers
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        return self.model

    def prepare_data_generators(
        self,
        train_dir: str,
        val_dir: str,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """Prepare data generators with augmentation."""
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
            channel_shift_range=10.0,
        )

        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode="binary",
            shuffle=True,
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False,
        )

        return train_generator, val_generator

    def train(
        self,
        train_generator: ImageDataGenerator,
        val_generator: ImageDataGenerator,
        epochs: int = 50,
        learning_rate: float = 0.0001,
        model_name: str = "defect_classification_model",
    ) -> tf.keras.callbacks.History:
        """Train the classification model."""
        if self.model is None:
            self.build_model()

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            "balanced",
            classes=np.unique(train_generator.classes),
            y=train_generator.classes,
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6, verbose=1),
            ModelCheckpoint(
                f"trained_models/{model_name}_checkpoint.h5",
                save_best_only=True,
                monitor="val_loss",
            ),
        ]

        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
        )

        return self.history

    def evaluate(
        self, test_generator: ImageDataGenerator, save_plots: bool = True
    ) -> Dict:
        """Evaluate model performance on test set."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Evaluate
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            test_generator
        )

        # Get predictions
        test_generator.reset()
        y_pred = self.model.predict(test_generator)
        y_pred_classes = (y_pred > 0.5).astype(int)
        y_true = test_generator.classes

        # Classification report
        report = classification_report(
            y_true, y_pred_classes, target_names=["Normal", "Defect"], output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)

        results = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
            "precision_curve": precision,
            "recall_curve": recall,
        }

        if save_plots:
            self._plot_results(results)

        return results

    def _plot_results(self, results: Dict) -> None:
        """Plot training and evaluation results."""
        # Training history
        if self.history:
            plt.figure(figsize=(15, 5))

            # Accuracy
            plt.subplot(1, 3, 1)
            plt.plot(self.history.history["accuracy"])
            plt.plot(self.history.history["val_accuracy"])
            plt.title("Model Accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Validation"])

            # Loss
            plt.subplot(1, 3, 2)
            plt.plot(self.history.history["loss"])
            plt.plot(self.history.history["val_loss"])
            plt.title("Model Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Validation"])

            # Precision
            plt.subplot(1, 3, 3)
            if "precision" in self.history.history:
                plt.plot(self.history.history["precision"])
                plt.plot(self.history.history["val_precision"])
                plt.title("Model Precision")
                plt.ylabel("Precision")
                plt.xlabel("Epoch")
                plt.legend(["Train", "Validation"])

            plt.tight_layout()
            plt.savefig("classification_training_history.png")
            plt.show()

        # ROC and PR curves
        plt.figure(figsize=(12, 5))

        # ROC Curve
        plt.subplot(1, 2, 1)
        plt.plot(
            results["fpr"],
            results["tpr"],
            color="darkorange",
            lw=2,
            label=f'ROC curve (area = {results["roc_auc"]:.2f})',
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(True)

        # Precision-Recall curve
        plt.subplot(1, 2, 2)
        plt.step(
            results["recall_curve"],
            results["precision_curve"],
            color="b",
            alpha=0.2,
            where="post",
        )
        plt.fill_between(
            results["recall_curve"],
            results["precision_curve"],
            step="post",
            alpha=0.2,
            color="b",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title("Precision-Recall Curve")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("classification_performance_curves.png")
        plt.show()

    def save_model(self, model_name: str = "defect_classification_model") -> None:
        """Save model in multiple formats for compatibility."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        os.makedirs("trained_models", exist_ok=True)

        # Save in both H5 and Keras formats
        self.model.save(f"trained_models/{model_name}.h5")
        self.model.save(f"trained_models/{model_name}.keras")

        # Save as TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open(f"trained_models/{model_name}.tflite", "wb") as f:
            f.write(tflite_model)

        # Save class mapping
        with open("trained_models/class_mapping.json", "w") as f:
            json.dump(self.class_names, f)

        # Save model info
        model_info = {
            "model_name": model_name,
            "input_shape": list(self.input_shape),
            "preprocessing": "normalize_to_0_1",
            "class_names": list(self.class_names.values()),
            "date_trained": str(datetime.datetime.now()),
            "model_type": "binary_classification",
            "keras_format_path": f"trained_models/{model_name}.keras",
        }

        with open(f"trained_models/{model_name}_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"Model saved in multiple formats:")
        print(f"- H5: trained_models/{model_name}.h5")
        print(f"- Keras: trained_models/{model_name}.keras")
        print(f"- TFLite: trained_models/{model_name}.tflite")
