"""Model training utilities for defect detection."""

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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
import logging

logger = logging.getLogger(__name__)


class DefectClassificationTrainer:
    """Trainer for defect classification models."""

    def __init__(self, data_dir, model_name="defect_classifier", image_size=(224, 224)):
        self.data_dir = data_dir
        self.model_name = model_name
        self.image_size = image_size
        self.model = None
        self.history = None

    def create_model(self, use_pretrained=True):
        """Create the classification model."""
        model = Sequential(
            [
                ResNet50V2(
                    input_shape=(*self.image_size, 3),
                    include_top=False,
                    weights="imagenet" if use_pretrained else None,
                ),
                GlobalAveragePooling2D(),
                BatchNormalization(),
                Dense(256, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid"),
            ]
        )

        # Fine-tuning strategy
        base_model = model.layers[0]
        base_model.trainable = True

        # Freeze early layers, unfreeze later layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        self.model = model
        return model

    def prepare_data_generators(self, batch_size=32):
        """Prepare data generators for training."""
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

        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            f"{self.data_dir}/train",
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode="binary",
            shuffle=True,
        )

        validation_generator = validation_datagen.flow_from_directory(
            f"{self.data_dir}/validation",
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False,
        )

        test_generator = test_datagen.flow_from_directory(
            f"{self.data_dir}/test",
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False,
        )

        return train_generator, validation_generator, test_generator

    def train(self, epochs=50, batch_size=32):
        """Train the model."""
        if self.model is None:
            self.create_model()

        train_gen, val_gen, test_gen = self.prepare_data_generators(batch_size)

        # Calculate class weights
        try:
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(train_gen.classes), y=train_gen.classes
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            logger.info(f"Class weights: {class_weight_dict}")
        except Exception as e:
            logger.warning(f"Could not calculate class weights: {e}")
            class_weight_dict = None

        # Define callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6, verbose=1),
            ModelCheckpoint(
                f"trained_models/{self.model_name}_checkpoint.h5",
                save_best_only=True,
                monitor="val_loss",
            ),
        ]

        # Train the model
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
        )

        return self.history

    def evaluate(self):
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        _, _, test_gen = self.prepare_data_generators()

        # Evaluate model
        test_results = self.model.evaluate(test_gen)
        print(f"Test accuracy: {test_results[1]:.4f}")
        print(f"Test loss: {test_results[0]:.4f}")

        # Get predictions for detailed analysis
        test_gen.reset()
        y_pred = self.model.predict(test_gen)
        y_pred_classes = (y_pred > 0.5).astype(int)
        y_true = test_gen.classes

        # Classification report
        print("\nClassification Report:")
        target_names = ["Normal", "Defect"]
        print(classification_report(y_true, y_pred_classes, target_names=target_names))

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred_classes)
        print(cm)

        return {
            "test_loss": test_results[0],
            "test_accuracy": test_results[1],
            "predictions": y_pred,
            "true_labels": y_true,
            "confusion_matrix": cm,
        }

    def save_model(self):
        """Save the trained model in multiple formats."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        os.makedirs("trained_models", exist_ok=True)

        # Save in multiple formats
        self.model.save(f"trained_models/{self.model_name}.h5")
        self.model.save(f"trained_models/{self.model_name}.keras")

        # Save as TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open(f"trained_models/{self.model_name}.tflite", "wb") as f:
            f.write(tflite_model)

        # Save class mapping
        class_mapping = {"0": "normal", "1": "defect"}
        with open("trained_models/class_mapping.json", "w") as f:
            json.dump(class_mapping, f)

        # Save model info
        model_info = {
            "model_name": self.model_name,
            "input_shape": [*self.image_size, 3],
            "preprocessing": "normalize_to_0_1",
            "class_names": ["normal", "defect"],
            "date_trained": str(datetime.datetime.now()),
            "model_type": "binary_classification",
            "keras_format_path": f"trained_models/{self.model_name}.keras",
        }

        with open(f"trained_models/{self.model_name}_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Model saved in multiple formats")

    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Accuracy
        axes[0].plot(self.history.history["accuracy"])
        axes[0].plot(self.history.history["val_accuracy"])
        axes[0].set_title("Model Accuracy")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend(["Train", "Validation"], loc="upper left")

        # Loss
        axes[1].plot(self.history.history["loss"])
        axes[1].plot(self.history.history["val_loss"])
        axes[1].set_title("Model Loss")
        axes[1].set_ylabel("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(["Train", "Validation"], loc="upper left")

        # Precision
        if "precision" in self.history.history:
            axes[2].plot(self.history.history["precision"])
            axes[2].plot(self.history.history["val_precision"])
            axes[2].set_title("Model Precision")
            axes[2].set_ylabel("Precision")
            axes[2].set_xlabel("Epoch")
            axes[2].legend(["Train", "Validation"], loc="upper left")

        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.show()


class AnomalyDetectionTrainer:
    """Trainer for anomaly detection using autoencoders."""

    def __init__(self, normal_data_dir, model_name="anomaly_detector"):
        self.normal_data_dir = normal_data_dir
        self.model_name = model_name
        self.autoencoder = None
        self.threshold = None

    def create_autoencoder(self, input_shape=(224, 224, 3)):
        """Create autoencoder model."""
        from tensorflow.keras.layers import (
            Input,
            Conv2D,
            MaxPooling2D,
            UpSampling2D,
            BatchNormalization,
            LeakyReLU,
        )
        from tensorflow.keras.models import Model

        # Input
        input_img = Input(shape=input_shape)

        # Encoder
        x = Conv2D(32, (3, 3), padding="same")(input_img)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2), padding="same")(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2), padding="same")(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        encoded = MaxPooling2D((2, 2), padding="same")(x)

        # Decoder
        x = Conv2D(128, (3, 3), padding="same")(encoded)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)

        # Output
        decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer="adam", loss="mse")

        self.autoencoder = autoencoder
        return autoencoder

    def train(self, epochs=50, batch_size=32):
        """Train the anomaly detection model."""
        if self.autoencoder is None:
            self.create_autoencoder()

        # Load normal images
        normal_images = self._load_normal_images()

        # Train autoencoder
        callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]

        history = self.autoencoder.fit(
            normal_images,
            normal_images,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            shuffle=True,
        )

        # Calculate threshold
        reconstructions = self.autoencoder.predict(normal_images)
        errors = np.mean(np.square(normal_images - reconstructions), axis=(1, 2, 3))
        self.threshold = np.percentile(errors, 95)

        logger.info(f"Anomaly threshold set to: {self.threshold:.6f}")

        return history

    def _load_normal_images(self):
        """Load normal images for training."""
        import cv2

        normal_images = []
        for img_file in os.listdir(self.normal_data_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_path = os.path.join(self.normal_data_dir, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    normal_images.append(img)

        normal_images = np.array(normal_images).astype("float32") / 255.0
        logger.info(f"Loaded {len(normal_images)} normal images for training")
        return normal_images

    def save_model(self):
        """Save the anomaly detection model and threshold."""
        if self.autoencoder is None or self.threshold is None:
            raise ValueError("Model not trained yet. Call train() first.")

        os.makedirs("trained_models", exist_ok=True)

        # Save model
        self.autoencoder.save(f"trained_models/{self.model_name}.keras")

        # Save threshold
        with open(f"trained_models/{self.model_name}_threshold.json", "w") as f:
            json.dump({"threshold": float(self.threshold)}, f)

        logger.info(f"Anomaly detection model saved")
