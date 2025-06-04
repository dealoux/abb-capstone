from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import cv2
import numpy as np
import os
from datetime import datetime

from abbvisionsystem.camera.calibration import CameraCalibrator


class CalibrationWidget(QWidget):
    """Widget for camera calibration functionality."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.calibrator = CameraCalibrator()
        self.setup_ui()

    def setup_ui(self):
        """Setup calibration UI."""
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Left panel - controls
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel, 1)

        # Right panel - image display
        display_panel = self.create_display_panel()
        layout.addWidget(display_panel, 2)

    def create_control_panel(self):
        """Create control panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Pattern settings
        pattern_group = QGroupBox("Chessboard Pattern")
        pattern_layout = QFormLayout()
        pattern_group.setLayout(pattern_layout)

        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(4, 50)
        self.cols_spin.setValue(39)
        pattern_layout.addRow("Columns:", self.cols_spin)

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(4, 50)
        self.rows_spin.setValue(27)
        pattern_layout.addRow("Rows:", self.rows_spin)

        self.square_size = QDoubleSpinBox()
        self.square_size.setRange(1.0, 100.0)
        self.square_size.setValue(10.0)
        self.square_size.setSuffix(" mm")
        pattern_layout.addRow("Square Size:", self.square_size)

        update_pattern_btn = QPushButton("Update Pattern")
        update_pattern_btn.clicked.connect(self.update_pattern)
        pattern_layout.addRow(update_pattern_btn)

        layout.addWidget(pattern_group)

        # Visualization options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QVBoxLayout()
        viz_group.setLayout(viz_layout)

        self.show_axes_cb = QCheckBox("Show Coordinate Axes")
        self.show_axes_cb.setChecked(True)
        viz_layout.addWidget(self.show_axes_cb)

        self.show_cube_cb = QCheckBox("Show 3D Cube")
        viz_layout.addWidget(self.show_cube_cb)

        self.show_camera_frame_cb = QCheckBox("Show Camera Frame")
        viz_layout.addWidget(self.show_camera_frame_cb)

        layout.addWidget(viz_group)

        # Image collection
        collection_group = QGroupBox("Image Collection")
        collection_layout = QVBoxLayout()
        collection_group.setLayout(collection_layout)

        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        collection_layout.addWidget(self.capture_btn)

        self.load_images_btn = QPushButton("Load Images")
        self.load_images_btn.clicked.connect(self.load_images)
        collection_layout.addWidget(self.load_images_btn)

        self.add_image_btn = QPushButton("Add Current Image")
        self.add_image_btn.clicked.connect(self.add_current_image)
        self.add_image_btn.setEnabled(False)
        collection_layout.addWidget(self.add_image_btn)

        self.clear_images_btn = QPushButton("Clear All Images")
        self.clear_images_btn.clicked.connect(self.clear_images)
        collection_layout.addWidget(self.clear_images_btn)

        # Image count
        self.image_count_label = QLabel("Images: 0")
        collection_layout.addWidget(self.image_count_label)

        layout.addWidget(collection_group)

        # Calibration
        calibration_group = QGroupBox("Calibration")
        calibration_layout = QVBoxLayout()
        calibration_group.setLayout(calibration_layout)

        self.calibrate_btn = QPushButton("Calibrate Camera")
        self.calibrate_btn.clicked.connect(self.calibrate_camera)
        calibration_layout.addWidget(self.calibrate_btn)

        # Results display
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setReadOnly(True)
        calibration_layout.addWidget(self.results_text)

        layout.addWidget(calibration_group)

        # Save/Load
        save_load_group = QGroupBox("Save/Load")
        save_load_layout = QVBoxLayout()
        save_load_group.setLayout(save_load_layout)

        self.save_btn = QPushButton("Save Calibration")
        self.save_btn.clicked.connect(self.save_calibration)
        self.save_btn.setEnabled(False)
        save_load_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Calibration")
        self.load_btn.clicked.connect(self.load_calibration)
        save_load_layout.addWidget(self.load_btn)

        layout.addWidget(save_load_group)

        layout.addStretch()
        return panel

    def create_display_panel(self):
        """Create display panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Tab widget for different views
        self.display_tabs = QTabWidget()
        layout.addWidget(self.display_tabs)

        # Original image tab
        self.original_widget = ImageDisplayWidget()
        self.display_tabs.addTab(self.original_widget, "Original")

        # Pattern detection tab
        self.pattern_widget = ImageDisplayWidget()
        self.display_tabs.addTab(self.pattern_widget, "Pattern Detection")

        # Enhanced visualization tab
        self.enhanced_widget = ImageDisplayWidget()
        self.display_tabs.addTab(self.enhanced_widget, "Enhanced Visualization")

        return panel

    def update_pattern(self):
        """Update chessboard pattern settings."""
        self.calibrator.set_chessboard_pattern(
            self.rows_spin.value(), self.cols_spin.value(), self.square_size.value()
        )
        QMessageBox.information(
            self, "Pattern Updated", "Chessboard pattern updated successfully"
        )

    def capture_image(self):
        """Capture image from camera."""
        if self.parent.camera and self.parent.camera.connected:
            image = self.parent.camera.capture_image()
            if image is not None:
                self.parent.current_image = image
                self.display_image(image)
                self.add_image_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Capture Error", "Failed to capture image")

    def load_images(self):
        """Load calibration images from files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Calibration Images",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )

        if file_paths:
            successful_adds = 0
            for file_path in file_paths:
                image = cv2.imread(file_path)
                if image is not None:
                    if self.calibrator.add_calibration_image(image):
                        successful_adds += 1

            self.update_image_count()
            QMessageBox.information(
                self,
                "Images Loaded",
                f"Successfully added {successful_adds} out of {len(file_paths)} images",
            )

    def display_image(self, image):
        """Display image in all tabs."""
        self.original_widget.set_image(image)

        # Create pattern detection visualization
        pattern_vis = self.calibrator.visualize_calibration(
            image, draw_axes=False, draw_cube=False
        )
        self.pattern_widget.set_image(pattern_vis)

        # Create enhanced visualization
        enhanced_vis = self.calibrator.draw_chessboard_pattern(
            image,
            draw_axes=self.show_axes_cb.isChecked(),
            draw_cube=self.show_cube_cb.isChecked(),
            draw_camera_frame=self.show_camera_frame_cb.isChecked(),
        )
        self.enhanced_widget.set_image(enhanced_vis)

    def add_current_image(self):
        """Add current image to calibration set."""
        if self.parent.current_image is not None:
            if self.calibrator.add_calibration_image(self.parent.current_image):
                self.update_image_count()
                QMessageBox.information(
                    self, "Image Added", "Image added to calibration set"
                )
            else:
                QMessageBox.warning(
                    self, "Add Failed", "Could not detect chessboard in image"
                )

    def clear_images(self):
        """Clear all calibration images."""
        reply = QMessageBox.question(
            self,
            "Clear Images",
            "Are you sure you want to clear all calibration images?",
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.calibrator.calibration_images.clear()
            self.update_image_count()

    def update_image_count(self):
        """Update image count display."""
        count = len(self.calibrator.calibration_images)
        self.image_count_label.setText(f"Images: {count}")

        # Enable calibration if enough images
        self.calibrate_btn.setEnabled(count >= 5)

    def calibrate_camera(self):
        """Perform camera calibration."""
        if len(self.calibrator.calibration_images) < 5:
            QMessageBox.warning(
                self, "Insufficient Images", "Need at least 5 calibration images"
            )
            return

        # Show progress dialog
        progress = QProgressDialog("Performing calibration...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        try:
            result = self.calibrator.calibrate_camera()

            if result:
                # Display results
                results_text = f"""
Calibration Successful!

Reprojection Error: {result.reprojection_error:.4f} pixels
Image Size: {result.image_size[0]}×{result.image_size[1]}

Camera Matrix:
fx = {result.camera_matrix[0,0]:.2f}
fy = {result.camera_matrix[1,1]:.2f}
cx = {result.camera_matrix[0,2]:.2f}
cy = {result.camera_matrix[1,2]:.2f}

Scale: {result.pixels_per_mm:.2f} pixels/mm
                """

                self.results_text.setText(results_text)
                self.save_btn.setEnabled(True)

                # Update parent calibration status
                self.parent.calibration_status.setText("Calibration: Complete")

                QMessageBox.information(
                    self,
                    "Calibration Complete",
                    "Camera calibration completed successfully!",
                )

                # Update visualization if current image exists
                if self.parent.current_image is not None:
                    self.display_image(self.parent.current_image)

            else:
                QMessageBox.critical(
                    self, "Calibration Failed", "Camera calibration failed"
                )

        except Exception as e:
            QMessageBox.critical(
                self, "Calibration Error", f"Calibration error: {str(e)}"
            )
        finally:
            progress.close()

    def save_calibration(self):
        """Save calibration to file."""
        if self.calibrator.calibration_result is None:
            QMessageBox.warning(self, "No Calibration", "No calibration data to save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration",
            f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)",
        )

        if file_path:
            if self.calibrator.save_calibration(file_path):
                QMessageBox.information(
                    self, "Saved", f"Calibration saved to {file_path}"
                )
            else:
                QMessageBox.critical(self, "Save Failed", "Failed to save calibration")

    def load_calibration(self):
        """Load calibration from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", "", "JSON Files (*.json)"
        )

        if file_path:
            if self.calibrator.load_calibration(file_path):
                QMessageBox.information(
                    self, "Loaded", f"Calibration loaded from {file_path}"
                )

                # Update UI
                self.save_btn.setEnabled(True)
                self.parent.calibration_status.setText("Calibration: Loaded")

                # Display results
                result = self.calibrator.calibration_result
                if result:
                    results_text = f"""
Loaded Calibration:

Reprojection Error: {result.reprojection_error:.4f} pixels
Scale: {result.pixels_per_mm:.2f} pixels/mm
                    """
                    self.results_text.setText(results_text)

            else:
                QMessageBox.critical(self, "Load Failed", "Failed to load calibration")

    def camera_connected(self, camera):
        """Handle camera connection."""
        self.capture_btn.setEnabled(True)


# Import the ImageDisplayWidget from the main file or create it here
class ImageDisplayWidget(QWidget):
    """Widget for displaying images (duplicate from main file for completeness)."""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid gray")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("No image loaded")
        self.image_label.setScaledContents(True)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)

        layout.addWidget(scroll_area)

    def set_image(self, cv_image):
        """Set image to display."""
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

        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
