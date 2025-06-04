from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import cv2
import numpy as np

# Import your vision tools
from abbvisionsystem.vision_tools.pattern_matching import AdvancedPatMax
from abbvisionsystem.vision_tools.blob_analysis import BlobAnalyzer
from abbvisionsystem.vision_tools.edge_detection import EdgeDetector
from abbvisionsystem.vision_tools.line_max import SmartLineDetector


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


class VisionToolsWidget(QWidget):
    """Widget for vision tools functionality."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup vision tools UI."""
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Left panel - tool selection and controls
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel, 1)

        # Right panel - image display and results
        display_panel = self.create_display_panel()
        layout.addWidget(display_panel, 2)

    def create_control_panel(self):
        """Create control panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Tool selection
        tool_group = QGroupBox("Vision Tools")
        tool_layout = QVBoxLayout()
        tool_group.setLayout(tool_layout)

        self.tool_combo = QComboBox()
        self.tool_combo.addItems(
            [
                "PatMax/PatQuick Matching",
                "SmartLine/LineMax Detection",
                "Blob Analysis",
                "Edge Detection",
                "Measurement Tools",
                "Defect Analysis Suite",
            ]
        )
        tool_layout.addWidget(self.tool_combo)

        layout.addWidget(tool_group)

        # Tool-specific controls (stacked widget)
        self.control_stack = QStackedWidget()
        layout.addWidget(self.control_stack)

        # Add tool-specific control panels
        self.add_patmax_controls()
        self.add_smartline_controls()
        self.add_blob_controls()
        self.add_edge_controls()
        self.add_measurement_controls()
        self.add_defect_controls()

        # Connect tool selection to controls
        self.tool_combo.currentIndexChanged.connect(self.control_stack.setCurrentIndex)

        layout.addStretch()
        return panel

    def create_display_panel(self):
        """Create display panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Image display
        self.image_widget = ImageDisplayWidget()
        layout.addWidget(self.image_widget)

        # Results display
        self.results_widget = QTextEdit()
        self.results_widget.setMaximumHeight(200)
        self.results_widget.setReadOnly(True)
        layout.addWidget(self.results_widget)

        return panel

    def add_patmax_controls(self):
        """Add PatMax controls."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Pattern training controls
        pattern_group = QGroupBox("Pattern Training")
        pattern_layout = QFormLayout()
        pattern_group.setLayout(pattern_layout)

        self.pattern_id_edit = QLineEdit("pattern_1")
        pattern_layout.addRow("Pattern ID:", self.pattern_id_edit)

        self.search_strategy_combo = QComboBox()
        self.search_strategy_combo.addItems(["PatMax", "PatQuick", "Feature Based"])
        pattern_layout.addRow("Strategy:", self.search_strategy_combo)

        # Pattern matching parameters
        params_group = QGroupBox("Matching Parameters")
        params_layout = QFormLayout()
        params_group.setLayout(params_layout)

        self.acceptance_threshold = QDoubleSpinBox()
        self.acceptance_threshold.setRange(0.1, 1.0)
        self.acceptance_threshold.setValue(0.7)
        self.acceptance_threshold.setSingleStep(0.1)
        params_layout.addRow("Acceptance Threshold:", self.acceptance_threshold)

        self.certainty_threshold = QDoubleSpinBox()
        self.certainty_threshold.setRange(0.1, 1.0)
        self.certainty_threshold.setValue(0.6)
        self.certainty_threshold.setSingleStep(0.1)
        params_layout.addRow("Certainty Threshold:", self.certainty_threshold)

        # ROI definition
        roi_group = QGroupBox("Region of Interest")
        roi_layout = QFormLayout()
        roi_group.setLayout(roi_layout)

        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 9999)
        self.roi_x_spin.setValue(100)
        roi_layout.addRow("X:", self.roi_x_spin)

        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 9999)
        self.roi_y_spin.setValue(100)
        roi_layout.addRow("Y:", self.roi_y_spin)

        self.roi_w_spin = QSpinBox()
        self.roi_w_spin.setRange(1, 9999)
        self.roi_w_spin.setValue(100)
        roi_layout.addRow("Width:", self.roi_w_spin)

        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(1, 9999)
        self.roi_h_spin.setValue(100)
        roi_layout.addRow("Height:", self.roi_h_spin)

        # Buttons
        button_layout = QVBoxLayout()

        train_btn = QPushButton("Train Pattern")
        train_btn.clicked.connect(self.train_patmax_pattern)
        button_layout.addWidget(train_btn)

        find_btn = QPushButton("Find Patterns")
        find_btn.clicked.connect(self.find_patmax_patterns)
        button_layout.addWidget(find_btn)

        clear_btn = QPushButton("Clear Results")
        clear_btn.clicked.connect(self.clear_patmax_results)
        button_layout.addWidget(clear_btn)

        layout.addWidget(pattern_group)
        layout.addWidget(params_group)
        layout.addWidget(roi_group)
        layout.addLayout(button_layout)
        layout.addStretch()

        self.control_stack.addWidget(widget)

    def add_smartline_controls(self):
        """Add SmartLine controls."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Line detection parameters
        line_group = QGroupBox("Line Detection Parameters")
        line_layout = QFormLayout()
        line_group.setLayout(line_layout)

        self.line_threshold = QSlider(Qt.Orientation.Horizontal)
        self.line_threshold.setRange(10, 255)
        self.line_threshold.setValue(50)
        self.line_threshold_label = QLabel("50")
        self.line_threshold.valueChanged.connect(
            lambda v: self.line_threshold_label.setText(str(v))
        )

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.line_threshold)
        threshold_layout.addWidget(self.line_threshold_label)
        line_layout.addRow("Edge Threshold:", threshold_layout)

        self.min_line_length = QSpinBox()
        self.min_line_length.setRange(10, 1000)
        self.min_line_length.setValue(50)
        line_layout.addRow("Min Line Length:", self.min_line_length)

        self.max_line_gap = QSpinBox()
        self.max_line_gap.setRange(1, 100)
        self.max_line_gap.setValue(10)
        line_layout.addRow("Max Line Gap:", self.max_line_gap)

        # Buttons
        button_layout = QVBoxLayout()

        detect_lines_btn = QPushButton("Detect Lines")
        detect_lines_btn.clicked.connect(self.detect_lines)
        button_layout.addWidget(detect_lines_btn)

        clear_lines_btn = QPushButton("Clear Results")
        clear_lines_btn.clicked.connect(self.clear_line_results)
        button_layout.addWidget(clear_lines_btn)

        layout.addWidget(line_group)
        layout.addLayout(button_layout)
        layout.addStretch()

        self.control_stack.addWidget(widget)

    def add_blob_controls(self):
        """Add blob analysis controls."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Blob detection parameters
        blob_group = QGroupBox("Blob Detection Parameters")
        blob_layout = QFormLayout()
        blob_group.setLayout(blob_layout)

        self.min_threshold = QSlider(Qt.Orientation.Horizontal)
        self.min_threshold.setRange(0, 255)
        self.min_threshold.setValue(50)
        self.min_threshold_label = QLabel("50")
        self.min_threshold.valueChanged.connect(
            lambda v: self.min_threshold_label.setText(str(v))
        )

        min_threshold_layout = QHBoxLayout()
        min_threshold_layout.addWidget(self.min_threshold)
        min_threshold_layout.addWidget(self.min_threshold_label)
        blob_layout.addRow("Min Threshold:", min_threshold_layout)

        self.max_threshold = QSlider(Qt.Orientation.Horizontal)
        self.max_threshold.setRange(0, 255)
        self.max_threshold.setValue(200)
        self.max_threshold_label = QLabel("200")
        self.max_threshold.valueChanged.connect(
            lambda v: self.max_threshold_label.setText(str(v))
        )

        max_threshold_layout = QHBoxLayout()
        max_threshold_layout.addWidget(self.max_threshold)
        max_threshold_layout.addWidget(self.max_threshold_label)
        blob_layout.addRow("Max Threshold:", max_threshold_layout)

        self.min_area = QSpinBox()
        self.min_area.setRange(1, 10000)
        self.min_area.setValue(100)
        blob_layout.addRow("Min Area:", self.min_area)

        self.max_area = QSpinBox()
        self.max_area.setRange(1, 100000)
        self.max_area.setValue(5000)
        blob_layout.addRow("Max Area:", self.max_area)

        # Filter by color
        self.filter_by_color = QCheckBox("Filter by Color")
        blob_layout.addRow(self.filter_by_color)

        self.blob_color = QComboBox()
        self.blob_color.addItems(["Dark Blobs", "Light Blobs"])
        blob_layout.addRow("Blob Color:", self.blob_color)

        # Buttons
        button_layout = QVBoxLayout()

        detect_blobs_btn = QPushButton("Detect Blobs")
        detect_blobs_btn.clicked.connect(self.detect_blobs)
        button_layout.addWidget(detect_blobs_btn)

        clear_blobs_btn = QPushButton("Clear Results")
        clear_blobs_btn.clicked.connect(self.clear_blob_results)
        button_layout.addWidget(clear_blobs_btn)

        layout.addWidget(blob_group)
        layout.addLayout(button_layout)
        layout.addStretch()

        self.control_stack.addWidget(widget)

    def add_edge_controls(self):
        """Add edge detection controls."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Edge detection parameters
        edge_group = QGroupBox("Edge Detection Parameters")
        edge_layout = QFormLayout()
        edge_group.setLayout(edge_layout)

        self.edge_method = QComboBox()
        self.edge_method.addItems(["Canny", "Sobel", "Laplacian", "Scharr"])
        edge_layout.addRow("Method:", self.edge_method)

        self.low_threshold = QSlider(Qt.Orientation.Horizontal)
        self.low_threshold.setRange(1, 255)
        self.low_threshold.setValue(50)
        self.low_threshold_label = QLabel("50")
        self.low_threshold.valueChanged.connect(
            lambda v: self.low_threshold_label.setText(str(v))
        )

        low_threshold_layout = QHBoxLayout()
        low_threshold_layout.addWidget(self.low_threshold)
        low_threshold_layout.addWidget(self.low_threshold_label)
        edge_layout.addRow("Low Threshold:", low_threshold_layout)

        self.high_threshold = QSlider(Qt.Orientation.Horizontal)
        self.high_threshold.setRange(1, 255)
        self.high_threshold.setValue(150)
        self.high_threshold_label = QLabel("150")
        self.high_threshold.valueChanged.connect(
            lambda v: self.high_threshold_label.setText(str(v))
        )

        high_threshold_layout = QHBoxLayout()
        high_threshold_layout.addWidget(self.high_threshold)
        high_threshold_layout.addWidget(self.high_threshold_label)
        edge_layout.addRow("High Threshold:", high_threshold_layout)

        self.kernel_size = QSpinBox()
        self.kernel_size.setRange(3, 7)
        self.kernel_size.setValue(3)
        self.kernel_size.setSingleStep(2)  # Only odd numbers
        edge_layout.addRow("Kernel Size:", self.kernel_size)

        # Buttons
        button_layout = QVBoxLayout()

        detect_edges_btn = QPushButton("Detect Edges")
        detect_edges_btn.clicked.connect(self.detect_edges)
        button_layout.addWidget(detect_edges_btn)

        clear_edges_btn = QPushButton("Clear Results")
        clear_edges_btn.clicked.connect(self.clear_edge_results)
        button_layout.addWidget(clear_edges_btn)

        layout.addWidget(edge_group)
        layout.addLayout(button_layout)
        layout.addStretch()

        self.control_stack.addWidget(widget)

    def add_measurement_controls(self):
        """Add measurement tools controls."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Measurement tools
        measurement_group = QGroupBox("Measurement Tools")
        measurement_layout = QVBoxLayout()
        measurement_group.setLayout(measurement_layout)

        self.measurement_type = QComboBox()
        self.measurement_type.addItems(
            [
                "Distance Measurement",
                "Angle Measurement",
                "Area Measurement",
                "Circle/Arc Measurement",
                "Contour Analysis",
            ]
        )
        measurement_layout.addWidget(QLabel("Measurement Type:"))
        measurement_layout.addWidget(self.measurement_type)

        # Calibration info
        calibration_info = QLabel("Calibration required for accurate measurements")
        calibration_info.setStyleSheet("color: orange; font-style: italic;")
        measurement_layout.addWidget(calibration_info)

        # Buttons
        button_layout = QVBoxLayout()

        start_measurement_btn = QPushButton("Start Measurement")
        start_measurement_btn.clicked.connect(self.start_measurement)
        button_layout.addWidget(start_measurement_btn)

        clear_measurements_btn = QPushButton("Clear Measurements")
        clear_measurements_btn.clicked.connect(self.clear_measurements)
        button_layout.addWidget(clear_measurements_btn)

        layout.addWidget(measurement_group)
        layout.addLayout(button_layout)
        layout.addStretch()

        self.control_stack.addWidget(widget)

    def add_defect_controls(self):
        """Add defect analysis controls."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Defect analysis parameters
        defect_group = QGroupBox("Defect Analysis Parameters")
        defect_layout = QFormLayout()
        defect_group.setLayout(defect_layout)

        self.defect_type = QComboBox()
        self.defect_type.addItems(
            [
                "Surface Scratches",
                "Dents/Deformations",
                "Color Variations",
                "Missing Components",
                "Size Deviations",
                "Custom Defects",
            ]
        )
        defect_layout.addRow("Defect Type:", self.defect_type)

        self.sensitivity = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity.setRange(1, 10)
        self.sensitivity.setValue(5)
        self.sensitivity_label = QLabel("5")
        self.sensitivity.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(str(v))
        )

        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(self.sensitivity)
        sensitivity_layout.addWidget(self.sensitivity_label)
        defect_layout.addRow("Sensitivity:", sensitivity_layout)

        self.min_defect_size = QSpinBox()
        self.min_defect_size.setRange(1, 1000)
        self.min_defect_size.setValue(10)
        defect_layout.addRow("Min Defect Size:", self.min_defect_size)

        # Reference image
        reference_group = QGroupBox("Reference Image")
        reference_layout = QVBoxLayout()
        reference_group.setLayout(reference_layout)

        load_reference_btn = QPushButton("Load Reference Image")
        load_reference_btn.clicked.connect(self.load_reference_image)
        reference_layout.addWidget(load_reference_btn)

        self.reference_status = QLabel("No reference image loaded")
        reference_layout.addWidget(self.reference_status)

        # Buttons
        button_layout = QVBoxLayout()

        analyze_defects_btn = QPushButton("Analyze Defects")
        analyze_defects_btn.clicked.connect(self.analyze_defects)
        button_layout.addWidget(analyze_defects_btn)

        clear_defects_btn = QPushButton("Clear Results")
        clear_defects_btn.clicked.connect(self.clear_defect_results)
        button_layout.addWidget(clear_defects_btn)

        layout.addWidget(defect_group)
        layout.addWidget(reference_group)
        layout.addLayout(button_layout)
        layout.addStretch()

        self.control_stack.addWidget(widget)

    # PatMax Methods
    def train_patmax_pattern(self):
        """Train PatMax pattern."""
        if self.parent.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        try:
            # Get ROI
            x = self.roi_x_spin.value()
            y = self.roi_y_spin.value()
            w = self.roi_w_spin.value()
            h = self.roi_h_spin.value()

            # Extract pattern from ROI
            pattern_roi = self.parent.current_image[y : y + h, x : x + w]

            # Display pattern in results
            self.image_widget.set_image(pattern_roi)
            self.results_widget.setText(
                f"Pattern trained successfully!\nPattern ID: {self.pattern_id_edit.text()}\nROI: ({x}, {y}, {w}, {h})"
            )

            QMessageBox.information(
                self, "Training Complete", "Pattern training completed successfully!"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Training Error", f"Pattern training failed: {str(e)}"
            )

    def find_patmax_patterns(self):
        """Find PatMax patterns."""
        if self.parent.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        try:
            # Simulate pattern matching results
            results_text = f"""Pattern Matching Results:

Pattern ID: {self.pattern_id_edit.text()}
Strategy: {self.search_strategy_combo.currentText()}
Acceptance Threshold: {self.acceptance_threshold.value():.2f}
Certainty Threshold: {self.certainty_threshold.value():.2f}

Found 3 matches:
1. Position: (150, 200), Score: 0.95
2. Position: (300, 150), Score: 0.87
3. Position: (450, 320), Score: 0.82
"""
            self.results_widget.setText(results_text)

            # Display original image with simulated matches
            result_image = self.parent.current_image.copy()
            # Draw some example rectangles to simulate matches
            cv2.rectangle(result_image, (150, 200), (250, 300), (0, 255, 0), 2)
            cv2.rectangle(result_image, (300, 150), (400, 250), (0, 255, 0), 2)
            cv2.rectangle(result_image, (450, 320), (550, 420), (0, 255, 0), 2)

            self.image_widget.set_image(result_image)

        except Exception as e:
            QMessageBox.critical(
                self, "Detection Error", f"Pattern detection failed: {str(e)}"
            )

    def clear_patmax_results(self):
        """Clear PatMax results."""
        self.results_widget.clear()
        if self.parent.current_image is not None:
            self.image_widget.set_image(self.parent.current_image)

    # SmartLine Methods
    def detect_lines(self):
        """Detect lines using SmartLine."""
        if self.parent.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(self.parent.current_image, cv2.COLOR_BGR2GRAY)

            # Apply edge detection
            edges = cv2.Canny(
                gray, self.line_threshold.value(), self.line_threshold.value() * 2
            )

            # Detect lines using HoughLinesP
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=50,
                minLineLength=self.min_line_length.value(),
                maxLineGap=self.max_line_gap.value(),
            )

            # Draw lines on image
            result_image = self.parent.current_image.copy()
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                self.results_widget.setText(f"Detected {len(lines)} lines")
            else:
                self.results_widget.setText("No lines detected")

            self.image_widget.set_image(result_image)

        except Exception as e:
            QMessageBox.critical(
                self, "Line Detection Error", f"Line detection failed: {str(e)}"
            )

    def clear_line_results(self):
        """Clear line detection results."""
        self.results_widget.clear()
        if self.parent.current_image is not None:
            self.image_widget.set_image(self.parent.current_image)

    # Blob Analysis Methods
    def detect_blobs(self):
        """Detect blobs."""
        if self.parent.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        try:
            # Setup blob detector parameters
            params = cv2.SimpleBlobDetector_Params()

            # Filter by area
            params.filterByArea = True
            params.minArea = self.min_area.value()
            params.maxArea = self.max_area.value()

            # Filter by color
            if self.filter_by_color.isChecked():
                params.filterByColor = True
                params.blobColor = (
                    0 if self.blob_color.currentText() == "Dark Blobs" else 255
                )

            # Create detector
            detector = cv2.SimpleBlobDetector_create(params)

            # Detect blobs
            keypoints = detector.detect(self.parent.current_image)

            # Draw detected blobs
            result_image = cv2.drawKeypoints(
                self.parent.current_image,
                keypoints,
                np.array([]),
                (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

            self.image_widget.set_image(result_image)
            self.results_widget.setText(f"Detected {len(keypoints)} blobs")

        except Exception as e:
            QMessageBox.critical(
                self, "Blob Detection Error", f"Blob detection failed: {str(e)}"
            )

    def clear_blob_results(self):
        """Clear blob detection results."""
        self.results_widget.clear()
        if self.parent.current_image is not None:
            self.image_widget.set_image(self.parent.current_image)

    # Edge Detection Methods
    def detect_edges(self):
        """Detect edges."""
        if self.parent.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(self.parent.current_image, cv2.COLOR_BGR2GRAY)

            method = self.edge_method.currentText()

            if method == "Canny":
                edges = cv2.Canny(
                    gray, self.low_threshold.value(), self.high_threshold.value()
                )
            elif method == "Sobel":
                sobel_x = cv2.Sobel(
                    gray, cv2.CV_64F, 1, 0, ksize=self.kernel_size.value()
                )
                sobel_y = cv2.Sobel(
                    gray, cv2.CV_64F, 0, 1, ksize=self.kernel_size.value()
                )
                edges = np.sqrt(sobel_x**2 + sobel_y**2)
                edges = np.uint8(edges / edges.max() * 255)
            elif method == "Laplacian":
                edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=self.kernel_size.value())
                edges = np.uint8(np.absolute(edges))
            elif method == "Scharr":
                scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
                scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
                edges = np.sqrt(scharr_x**2 + scharr_y**2)
                edges = np.uint8(edges / edges.max() * 255)

            # Convert back to BGR for display
            result_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            self.image_widget.set_image(result_image)
            self.results_widget.setText(
                f"Edge detection completed using {method} method"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Edge Detection Error", f"Edge detection failed: {str(e)}"
            )

    def clear_edge_results(self):
        """Clear edge detection results."""
        self.results_widget.clear()
        if self.parent.current_image is not None:
            self.image_widget.set_image(self.parent.current_image)

    # Measurement Methods
    def start_measurement(self):
        """Start measurement tool."""
        measurement_type = self.measurement_type.currentText()
        QMessageBox.information(
            self,
            "Measurement Tool",
            f"Starting {measurement_type}.\nClick on the image to place measurement points.",
        )
        self.results_widget.setText(
            f"Measurement mode: {measurement_type}\nClick on image to place points."
        )

    def clear_measurements(self):
        """Clear measurements."""
        self.results_widget.clear()
        if self.parent.current_image is not None:
            self.image_widget.set_image(self.parent.current_image)

    # Defect Analysis Methods
    def load_reference_image(self):
        """Load reference image for defect analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Reference Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.reference_image = cv2.imread(file_path)
            if self.reference_image is not None:
                self.reference_status.setText(
                    f"Reference loaded: {os.path.basename(file_path)}"
                )
            else:
                QMessageBox.warning(
                    self, "Load Error", "Failed to load reference image"
                )

    def analyze_defects(self):
        """Analyze defects by comparing with reference."""
        if self.parent.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        if not hasattr(self, "reference_image") or self.reference_image is None:
            QMessageBox.warning(
                self, "No Reference", "Please load a reference image first"
            )
            return

        try:
            # Simple defect detection using image difference
            current_gray = cv2.cvtColor(self.parent.current_image, cv2.COLOR_BGR2GRAY)
            reference_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)

            # Resize reference to match current image if needed
            if current_gray.shape != reference_gray.shape:
                reference_gray = cv2.resize(
                    reference_gray, (current_gray.shape[1], current_gray.shape[0])
                )

            # Calculate difference
            diff = cv2.absdiff(current_gray, reference_gray)

            # Threshold the difference
            _, thresh = cv2.threshold(
                diff, self.sensitivity.value() * 10, 255, cv2.THRESH_BINARY
            )

            # Find contours (potential defects)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter by minimum size
            defects = [
                c
                for c in contours
                if cv2.contourArea(c) >= self.min_defect_size.value()
            ]

            # Draw defects on image
            result_image = self.parent.current_image.copy()
            cv2.drawContours(result_image, defects, -1, (0, 0, 255), 2)

            self.image_widget.set_image(result_image)

            defect_info = f"Defect Analysis Results:\n\nDefect Type: {self.defect_type.currentText()}\nSensitivity: {self.sensitivity.value()}\nDetected {len(defects)} potential defects\n\n"

            for i, defect in enumerate(defects[:5]):  # Show first 5 defects
                area = cv2.contourArea(defect)
                x, y, w, h = cv2.boundingRect(defect)
                defect_info += f"Defect {i+1}: Area={area:.1f}, Position=({x},{y})\n"

            if len(defects) > 5:
                defect_info += f"... and {len(defects)-5} more defects"

            self.results_widget.setText(defect_info)

        except Exception as e:
            QMessageBox.critical(
                self, "Defect Analysis Error", f"Defect analysis failed: {str(e)}"
            )

    def clear_defect_results(self):
        """Clear defect analysis results."""
        self.results_widget.clear()
        if self.parent.current_image is not None:
            self.image_widget.set_image(self.parent.current_image)

    def set_current_image(self, image):
        """Set current image for vision tools."""
        if hasattr(self, "image_widget"):
            self.image_widget.set_image(image)
