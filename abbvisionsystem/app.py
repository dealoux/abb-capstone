import sys
import os
import cv2
import numpy as np
from typing import Optional, List
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import json
from datetime import datetime

from abbvisionsystem.camera.camera import CognexCamera, BaslerCamera, WebcamCamera
from abbvisionsystem.camera.calibration import CameraCalibrator
from abbvisionsystem.preprocessing.preprocessing import (
    prepare_for_detection,
    apply_image_enhancement,
)
from abbvisionsystem.models.taco_model import TACOModel
from abbvisionsystem.models.defect_detection_model import DefectDetectionModel
from abbvisionsystem.vision_tools.vision_widget import VisionToolsWidget
from abbvisionsystem.camera.calibration_widget import CalibrationWidget


class ABBVisionMainWindow(QMainWindow):
    """Main application window for ABB Vision System."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ABB Vision System")
        self.setGeometry(100, 100, 1600, 1000)

        # Application state
        self.current_image = None
        self.detections = None
        self.camera = None
        self.model = None

        # Setup UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()

        # Load settings
        self.load_settings()

    def setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Add tabs
        self.add_detection_tab()
        self.add_vision_tools_tab()
        self.add_calibration_tab()
        self.add_training_tab()

    def setup_menu_bar(self):
        """Setup application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Image", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        save_action = QAction("Save Results", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Camera menu
        camera_menu = menubar.addMenu("Camera")

        connect_basler = QAction("Connect Basler", self)
        connect_basler.triggered.connect(self.connect_basler_camera)
        camera_menu.addAction(connect_basler)

        connect_cognex = QAction("Connect Cognex", self)
        connect_cognex.triggered.connect(self.connect_cognex_camera)
        camera_menu.addAction(connect_cognex)

        connect_webcam = QAction("Connect Webcam", self)
        connect_webcam.triggered.connect(self.connect_webcam)
        camera_menu.addAction(connect_webcam)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_status_bar(self):
        """Setup status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        # Add permanent widgets to status bar
        self.camera_status = QLabel("Camera: Disconnected")
        self.model_status = QLabel("Model: None")
        self.calibration_status = QLabel("Calibration: None")

        self.status_bar.addPermanentWidget(self.camera_status)
        self.status_bar.addPermanentWidget(self.model_status)
        self.status_bar.addPermanentWidget(self.calibration_status)

    def add_detection_tab(self):
        """Add detection system tab."""
        self.detection_widget = DetectionWidget(self)
        self.tab_widget.addTab(self.detection_widget, "🏠 Detection System")

    def add_vision_tools_tab(self):
        """Add vision tools tab."""
        self.vision_tools_widget = VisionToolsWidget(self)
        self.tab_widget.addTab(self.vision_tools_widget, "🔍 Vision Tools")

    def add_calibration_tab(self):
        """Add camera calibration tab."""
        self.calibration_widget = CalibrationWidget(self)
        self.tab_widget.addTab(self.calibration_widget, "📷 Camera Calibration")

    def add_training_tab(self):
        """Add training center tab."""
        self.training_widget = TrainingWidget(self)
        self.tab_widget.addTab(self.training_widget, "📊 Training Center")

    # Camera connection methods
    def connect_basler_camera(self):
        """Connect to Basler camera."""
        dialog = BaslerConnectionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            device_index = dialog.get_device_index()

            self.camera = BaslerCamera(device_index=device_index)
            if self.camera.connect():
                self.camera_status.setText("Camera: Basler Connected")
                self.status_bar.showMessage("Basler camera connected successfully")

                # Update detection widget
                self.detection_widget.camera_connected(self.camera)
            else:
                QMessageBox.warning(
                    self, "Connection Failed", "Failed to connect to Basler camera"
                )

    def connect_cognex_camera(self):
        """Connect to Cognex camera."""
        dialog = CognexConnectionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()

            self.camera = CognexCamera(**config)
            if self.camera.connect():
                self.camera_status.setText("Camera: Cognex Connected")
                self.status_bar.showMessage("Cognex camera connected successfully")

                # Update detection widget
                self.detection_widget.camera_connected(self.camera)
            else:
                QMessageBox.warning(
                    self, "Connection Failed", "Failed to connect to Cognex camera"
                )

    def connect_webcam(self):
        """Connect to webcam."""
        camera_id, ok = QInputDialog.getInt(
            self, "Webcam Connection", "Camera ID:", 0, 0, 10, 1
        )
        if ok:
            self.camera = WebcamCamera(camera_id=camera_id)
            if self.camera.connect():
                self.camera_status.setText("Camera: Webcam Connected")
                self.status_bar.showMessage("Webcam connected successfully")

                # Update detection widget
                self.detection_widget.camera_connected(self.camera)
            else:
                QMessageBox.warning(
                    self, "Connection Failed", "Failed to connect to webcam"
                )

    def open_image(self):
        """Open image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.current_image = image
                self.detection_widget.set_image(image)
                self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "Load Error", "Failed to load image")

    def save_results(self):
        """Save detection results."""
        if self.current_image is not None and self.detections is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Results", "", "Image Files (*.png *.jpg *.jpeg)"
            )

            if file_path:
                # Save logic here
                self.status_bar.showMessage(f"Results saved to {file_path}")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About ABB Vision System",
            "ABB Vision System v1.0\n\n"
            "Industrial computer vision system for defect detection\n"
            "and quality inspection.",
        )

    def load_settings(self):
        """Load application settings."""
        settings = QSettings("ABB", "VisionSystem")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def closeEvent(self, event):
        """Handle application close."""
        # Save settings
        settings = QSettings("ABB", "VisionSystem")
        settings.setValue("geometry", self.saveGeometry())

        # Disconnect camera
        if self.camera and self.camera.connected:
            self.camera.disconnect()

        event.accept()


class DetectionWidget(QWidget):
    """Widget for detection system functionality."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setup_ui()
        self.load_models()

    def setup_ui(self):
        """Setup detection UI."""
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Left panel - controls
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel, 1)

        # Right panel - image display and results
        display_panel = self.create_display_panel()
        layout.addWidget(display_panel, 2)

    def create_control_panel(self):
        """Create control panel."""
        panel = QGroupBox("Configuration")
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Defect Detection", "TACO Waste Sorting"])
        self.model_combo.currentTextChanged.connect(self.model_changed)
        model_layout.addWidget(QLabel("Select Model:"))
        model_layout.addWidget(self.model_combo)

        layout.addWidget(model_group)

        # Input source
        input_group = QGroupBox("Input Source")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)

        self.input_radio_upload = QRadioButton("Upload Image")
        self.input_radio_camera = QRadioButton("Camera")
        self.input_radio_upload.setChecked(True)

        input_layout.addWidget(self.input_radio_upload)
        input_layout.addWidget(self.input_radio_camera)

        layout.addWidget(input_group)

        # Image enhancement
        enhancement_group = QGroupBox("Image Enhancement")
        enhancement_layout = QFormLayout()
        enhancement_group.setLayout(enhancement_layout)

        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_label = QLabel("0")
        self.brightness_slider.valueChanged.connect(
            lambda v: self.brightness_label.setText(str(v))
        )

        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(self.brightness_slider)
        brightness_layout.addWidget(self.brightness_label)

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_label = QLabel("0")
        self.contrast_slider.valueChanged.connect(
            lambda v: self.contrast_label.setText(str(v))
        )

        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(self.contrast_slider)
        contrast_layout.addWidget(self.contrast_label)

        enhancement_layout.addRow("Brightness:", brightness_layout)
        enhancement_layout.addRow("Contrast:", contrast_layout)

        layout.addWidget(enhancement_group)

        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout()
        detection_group.setLayout(detection_layout)

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(10, 100)
        self.confidence_slider.setValue(50)
        self.confidence_label = QLabel("0.50")
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_label.setText(f"{v/100:.2f}")
        )

        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_label)

        detection_layout.addRow("Confidence:", confidence_layout)

        layout.addWidget(detection_group)

        # Action buttons
        button_group = QGroupBox("Actions")
        button_layout = QVBoxLayout()
        button_group.setLayout(button_layout)

        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_image_btn)

        self.capture_btn = QPushButton("Capture from Camera")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        button_layout.addWidget(self.capture_btn)

        self.detect_btn = QPushButton("Run Detection")
        self.detect_btn.clicked.connect(self.run_detection)
        self.detect_btn.setEnabled(False)
        button_layout.addWidget(self.detect_btn)

        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)

        layout.addWidget(button_group)

        # Camera status
        self.camera_status_group = QGroupBox("Camera Status")
        camera_status_layout = QVBoxLayout()
        self.camera_status_group.setLayout(camera_status_layout)

        self.camera_status_label = QLabel("No camera connected")
        camera_status_layout.addWidget(self.camera_status_label)

        layout.addWidget(self.camera_status_group)

        layout.addStretch()
        return panel

    def create_display_panel(self):
        """Create display panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Create tab widget for different views
        self.display_tabs = QTabWidget()
        layout.addWidget(self.display_tabs)

        # Original image tab
        self.original_tab = ImageDisplayWidget()
        self.display_tabs.addTab(self.original_tab, "Original Image")

        # Results tab
        self.results_tab = ResultsDisplayWidget()
        self.display_tabs.addTab(self.results_tab, "Detection Results")

        # Enhanced image tab
        self.enhanced_tab = ImageDisplayWidget()
        self.display_tabs.addTab(self.enhanced_tab, "Enhanced Image")

        return panel

    def load_models(self):
        """Load available models."""
        try:
            # This would be your model loading logic
            self.parent.status_bar.showMessage("Models loaded successfully")
        except Exception as e:
            QMessageBox.warning(
                self, "Model Loading Error", f"Failed to load models: {str(e)}"
            )

    def model_changed(self, model_name):
        """Handle model selection change."""
        model_map = {"Defect Detection": "defect", "TACO Waste Sorting": "taco"}

        try:
            # Load the selected model (implement your model loading logic here)
            self.parent.status_bar.showMessage(f"Loaded model: {model_name}")
            self.parent.model_status.setText(f"Model: {model_name}")
        except Exception as e:
            QMessageBox.warning(self, "Model Error", f"Failed to load model: {str(e)}")

    def camera_connected(self, camera):
        """Handle camera connection."""
        self.camera = camera
        self.capture_btn.setEnabled(True)
        self.camera_status_label.setText(f"Connected: {type(camera).__name__}")

    def set_image(self, image):
        """Set current image."""
        self.parent.current_image = image
        self.original_tab.set_image(image)
        self.detect_btn.setEnabled(True)

        # Apply enhancements and show in enhanced tab
        enhanced = self.apply_enhancements(image)
        self.enhanced_tab.set_image(enhanced)

    def load_image(self):
        """Load image from file."""
        self.parent.open_image()

    def capture_image(self):
        """Capture image from camera."""
        if self.camera and self.camera.connected:
            image = self.camera.capture_image()
            if image is not None:
                self.set_image(image)
                self.parent.status_bar.showMessage("Image captured successfully")
            else:
                QMessageBox.warning(self, "Capture Error", "Failed to capture image")

    def apply_enhancements(self, image):
        """Apply image enhancements."""
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()

        # Apply your enhancement logic here
        enhanced = apply_image_enhancement(image, brightness, contrast)
        return enhanced

    def run_detection(self):
        """Run object detection."""
        if self.parent.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        # Show progress dialog
        progress = QProgressDialog("Running detection...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        try:
            # Apply enhancements
            enhanced = self.apply_enhancements(self.parent.current_image)

            # Prepare for detection
            detection_image = prepare_for_detection(enhanced)

            # Run detection (implement your detection logic here)
            # detections = self.parent.model.predict(detection_image)

            # For now, simulate detection
            detections = {"num_detections": 0, "scores": [], "classes": []}

            self.parent.detections = detections
            self.results_tab.set_results(self.parent.current_image, detections)
            self.save_btn.setEnabled(True)

            self.parent.status_bar.showMessage("Detection completed")

        except Exception as e:
            QMessageBox.critical(self, "Detection Error", f"Detection failed: {str(e)}")
        finally:
            progress.close()

    def save_results(self):
        """Save detection results."""
        self.parent.save_results()


class ImageDisplayWidget(QWidget):
    """Widget for displaying images."""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.image = None

    def setup_ui(self):
        """Setup UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Image label
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid gray")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("No image loaded")
        self.image_label.setScaledContents(True)

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)

        layout.addWidget(scroll_area)

        # Image info
        self.info_label = QLabel("Image info will appear here")
        layout.addWidget(self.info_label)

    def set_image(self, cv_image):
        """Set image to display."""
        self.image = cv_image

        # Convert CV image to Qt format
        if len(cv_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
        else:
            h, w = cv_image.shape
            qt_image = QImage(cv_image.data, w, h, QImage.Format.Format_Grayscale8)

        # Create pixmap and set to label
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

        # Update info
        self.info_label.setText(f"Size: {w}×{h}, Channels: {len(cv_image.shape)}")


class ResultsDisplayWidget(QWidget):
    """Widget for displaying detection results."""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Results image
        self.image_widget = ImageDisplayWidget()
        layout.addWidget(self.image_widget)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(
            ["Object Type", "Confidence", "Location"]
        )
        self.results_table.setMaximumHeight(200)
        layout.addWidget(self.results_table)

        # Summary
        self.summary_label = QLabel("No detections")
        layout.addWidget(self.summary_label)

    def set_results(self, image, detections):
        """Set detection results."""
        # Display image with detection boxes (implement visualization logic)
        self.image_widget.set_image(image)

        # Update table
        num_detections = detections.get("num_detections", 0)
        self.results_table.setRowCount(num_detections)

        # Update summary
        self.summary_label.setText(f"Total detections: {num_detections}")


class TrainingWidget(QWidget):
    """Widget for training functionality."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup training UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Training type selection
        type_group = QGroupBox("Training Type")
        type_layout = QVBoxLayout()
        type_group.setLayout(type_layout)

        self.training_combo = QComboBox()
        self.training_combo.addItems(
            [
                "Pattern Templates",
                "Defect Classification",
                "Blob Detection",
                "Custom Models",
            ]
        )
        type_layout.addWidget(self.training_combo)

        layout.addWidget(type_group)

        # Training content (implement specific training interfaces)
        self.training_stack = QStackedWidget()
        layout.addWidget(self.training_stack)

        # Add different training interfaces
        self.add_pattern_training()
        self.add_defect_training()
        self.add_blob_training()
        self.add_custom_training()

        # Connect combo box to stack
        self.training_combo.currentIndexChanged.connect(
            self.training_stack.setCurrentIndex
        )

    def add_pattern_training(self):
        """Add pattern training interface."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        layout.addWidget(QLabel("Pattern Template Training"))
        layout.addWidget(
            QLabel("Upload training images and configure pattern templates")
        )

        # Add your pattern training UI here

        self.training_stack.addWidget(widget)

    def add_defect_training(self):
        """Add defect training interface."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        layout.addWidget(QLabel("Defect Detection Training"))
        layout.addWidget(QLabel("Train models to detect defects"))

        # Add your defect training UI here

        self.training_stack.addWidget(widget)

    def add_blob_training(self):
        """Add blob training interface."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        layout.addWidget(QLabel("Blob Detection Configuration"))
        layout.addWidget(QLabel("Configure blob detection parameters"))

        # Add your blob training UI here

        self.training_stack.addWidget(widget)

    def add_custom_training(self):
        """Add custom training interface."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        layout.addWidget(QLabel("Custom Model Training"))
        layout.addWidget(QLabel("Train custom computer vision models"))

        # Add your custom training UI here

        self.training_stack.addWidget(widget)


# Dialog classes for camera connections
class BaslerConnectionDialog(QDialog):
    """Dialog for Basler camera connection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect Basler Camera")
        self.setup_ui()

    def setup_ui(self):
        """Setup dialog UI."""
        layout = QFormLayout()
        self.setLayout(layout)

        self.device_index = QSpinBox()
        self.device_index.setRange(0, 10)
        layout.addRow("Device Index:", self.device_index)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def get_device_index(self):
        """Get selected device index."""
        return self.device_index.value()


class CognexConnectionDialog(QDialog):
    """Dialog for Cognex camera connection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect Cognex Camera")
        self.setup_ui()

    def setup_ui(self):
        """Setup dialog UI."""
        layout = QFormLayout()
        self.setLayout(layout)

        self.ip_address = QLineEdit("192.168.1.100")
        self.port = QLineEdit("80")
        self.username = QLineEdit()
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)

        layout.addRow("IP Address:", self.ip_address)
        layout.addRow("Port:", self.port)
        layout.addRow("Username:", self.username)
        layout.addRow("Password:", self.password)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def get_config(self):
        """Get connection configuration."""
        return {
            "ip_address": self.ip_address.text(),
            "port": self.port.text(),
            "username": self.username.text() or None,
            "password": self.password.text() or None,
        }


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("ABB Vision System")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("ABB")

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = ABBVisionMainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
