"""Video processing module for real-time defect detection with camera integration."""

import cv2
import numpy as np
import time
import threading
import queue
from typing import Optional, Dict, Any, Callable, List, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class DetectionFrame:
    """Container for a frame with detection results."""

    frame: np.ndarray
    detections: Dict[str, Any]
    timestamp: float
    frame_number: int
    processing_time: float


class VideoProcessor:
    """Real-time video processor with defect detection capabilities."""

    def __init__(
        self,
        model,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_fps: int = 30,
        buffer_size: int = 10,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_fps = max_fps
        self.frame_interval = 1.0 / max_fps

        # Threading and buffering
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue(maxsize=buffer_size)
        self.processing_thread = None
        self.is_processing = False

        # Statistics
        self.stats = {
            "frames_processed": 0,
            "detections_made": 0,
            "average_processing_time": 0.0,
            "current_fps": 0.0,
            "defects_detected": 0,
            "last_detection_time": None,
        }

        # Video recording
        self.video_writer = None
        self.recording = False
        self.output_path = None

        # Performance optimization
        self.skip_frames = 1  # Process every nth frame
        self.detection_interval = 1  # Run detection every nth processed frame
        self.last_detection_time = 0

    def start_processing(self):
        """Start the video processing thread."""
        if not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(
                target=self._processing_loop, daemon=True
            )
            self.processing_thread.start()
            logger.info("Video processing started")

    def stop_processing(self):
        """Stop the video processing thread."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        self._cleanup_queues()
        logger.info("Video processing stopped")

    def add_frame(self, frame: np.ndarray) -> bool:
        """Add a frame to the processing queue."""
        if not self.is_processing:
            return False

        try:
            # Non-blocking put with immediate return if queue is full
            self.frame_queue.put_nowait(
                {
                    "frame": frame.copy(),
                    "timestamp": time.time(),
                    "frame_number": self.stats["frames_processed"],
                }
            )
            return True
        except queue.Full:
            # Drop frame if queue is full (maintain real-time performance)
            return False

    def get_result(self, timeout: float = 0.1) -> Optional[DetectionFrame]:
        """Get the latest detection result."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        frame_count = 0
        processing_times = []

        while self.is_processing:
            try:
                # Get frame from queue
                frame_data = self.frame_queue.get(timeout=0.1)
                frame = frame_data["frame"]
                timestamp = frame_data["timestamp"]
                frame_number = frame_data["frame_number"]

                start_time = time.time()

                # Skip frames for performance if needed
                frame_count += 1
                if frame_count % self.skip_frames != 0:
                    continue

                # Run detection based on interval
                should_detect = (
                    frame_count % (self.skip_frames * self.detection_interval) == 0
                    or time.time() - self.last_detection_time
                    > 1.0  # At least every second
                )

                if should_detect:
                    detections = self._run_detection(frame)
                    self.last_detection_time = time.time()
                else:
                    # Use previous detection or empty result
                    detections = self._get_empty_detection()

                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # Keep only last 30 processing times for moving average
                if len(processing_times) > 30:
                    processing_times.pop(0)

                # Update statistics
                self.stats["frames_processed"] += 1
                self.stats["average_processing_time"] = np.mean(processing_times)
                self.stats["current_fps"] = (
                    1.0 / processing_time if processing_time > 0 else 0
                )

                if detections and detections.get("num_detections", 0) > 0:
                    self.stats["detections_made"] += 1
                    self.stats["last_detection_time"] = datetime.now()

                    # Count defects
                    defect_count = sum(
                        1
                        for cls in detections.get("classes", [])
                        if cls == 1  # Assuming class 1 is defect
                    )
                    self.stats["defects_detected"] += defect_count

                # Create result frame
                result_frame = DetectionFrame(
                    frame=frame,
                    detections=detections,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    processing_time=processing_time,
                )

                # Add to result queue (non-blocking)
                try:
                    self.result_queue.put_nowait(result_frame)
                except queue.Full:
                    # Remove oldest result if queue is full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result_frame)
                    except queue.Empty:
                        pass

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                continue

    def _run_detection(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run detection on a frame."""
        try:
            # Determine if this is a YOLO model or classification model
            if hasattr(self.model, "predict") and "YOLO" in str(type(self.model)):
                # YOLO multi-object detection
                detections = self.model.predict(
                    frame,
                    conf_threshold=self.confidence_threshold,
                    iou_threshold=self.iou_threshold,
                )
            else:
                # Classification model
                detections = self.model.predict(frame)

            return detections or self._get_empty_detection()

        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return self._get_empty_detection()

    def _get_empty_detection(self) -> Dict[str, Any]:
        """Return empty detection result."""
        return {
            "boxes": np.array([]),
            "scores": np.array([]),
            "classes": np.array([]),
            "labels": [],
            "num_detections": 0,
        }

    def _cleanup_queues(self):
        """Clear all queues."""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

    def start_recording(
        self, output_path: str, fps: int = 30, frame_size: Tuple[int, int] = None
    ):
        """Start recording video with detection overlays."""
        try:
            self.output_path = output_path
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            if frame_size is None:
                frame_size = (640, 480)  # Default size

            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            self.recording = True
            logger.info(f"Started recording to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to start recording: {str(e)}")
            return False

    def stop_recording(self):
        """Stop video recording."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        logger.info(f"Stopped recording: {self.output_path}")

    def record_frame(self, frame_with_detections: np.ndarray):
        """Record a frame with detection overlays."""
        if self.recording and self.video_writer:
            # Ensure frame is in BGR format for video writer
            if len(frame_with_detections.shape) == 3:
                # Convert RGB to BGR if needed
                if frame_with_detections.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_with_detections
            else:
                frame_bgr = frame_with_detections

            self.video_writer.write(frame_bgr)

    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.stats.copy()

    def update_detection_params(
        self,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        skip_frames: int = None,
        detection_interval: int = None,
    ):
        """Update detection parameters dynamically."""
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        if skip_frames is not None:
            self.skip_frames = max(1, skip_frames)
        if detection_interval is not None:
            self.detection_interval = max(1, detection_interval)


class CalibratedVideoProcessor(VideoProcessor):
    """Video processor with calibration support for real-world coordinates."""

    def __init__(self, model, camera=None, **kwargs):
        super().__init__(model, **kwargs)
        self.camera = camera
        self.calibrator = (
            camera.calibrator if camera and hasattr(camera, "calibrator") else None
        )

    def _run_detection(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run detection with calibration support."""
        detections = super()._run_detection(frame)

        # Add real-world coordinates if calibrated
        if (
            self.calibrator
            and self.calibrator.calibration_result
            and detections.get("num_detections", 0) > 0
        ):

            detections["world_coordinates"] = self._convert_to_world_coordinates(
                detections.get("boxes", [])
            )

        return detections

    def _convert_to_world_coordinates(
        self, boxes: np.ndarray
    ) -> List[Dict[str, float]]:
        """Convert pixel coordinates to real-world coordinates."""
        if not self.calibrator or not self.calibrator.calibration_result:
            return []

        world_coords = []
        scale = self.calibrator.calibration_result.pixels_per_mm

        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Convert to mm (or your calibration unit)
            world_x = center_x / scale
            world_y = center_y / scale
            width_mm = (x2 - x1) / scale
            height_mm = (y2 - y1) / scale

            world_coords.append(
                {
                    "center_x_mm": world_x,
                    "center_y_mm": world_y,
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                }
            )

        return world_coords


class VideoVisualizationHelper:
    """Helper class for video visualization and overlay rendering."""

    @staticmethod
    def draw_detections_on_frame(
        frame: np.ndarray,
        detections: Dict[str, Any],
        model,
        confidence_threshold: float = 0.25,
        show_fps: bool = True,
        fps: float = 0.0,
    ) -> np.ndarray:
        """Draw detection results on frame with performance optimizations."""

        # Use the model's visualization method if available
        if hasattr(model, "visualize_detections"):
            frame_with_detections = model.visualize_detections(
                frame, detections, threshold=confidence_threshold
            )
        else:
            frame_with_detections = VideoVisualizationHelper._draw_basic_detections(
                frame, detections, confidence_threshold
            )

        # Add FPS counter
        if show_fps:
            cv2.putText(
                frame_with_detections,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Add detection count
        num_detections = detections.get("num_detections", 0)
        if num_detections > 0:
            cv2.putText(
                frame_with_detections,
                f"Objects: {num_detections}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        return frame_with_detections

    @staticmethod
    def _draw_basic_detections(
        frame: np.ndarray, detections: Dict[str, Any], confidence_threshold: float
    ) -> np.ndarray:
        """Basic detection drawing for models without visualization method."""
        result_frame = frame.copy()

        if detections.get("num_detections", 0) == 0:
            return result_frame

        boxes = detections.get("boxes", [])
        scores = detections.get("scores", [])
        classes = detections.get("classes", [])

        for i in range(len(boxes)):
            if scores[i] < confidence_threshold:
                continue

            box = boxes[i]
            score = scores[i]
            class_id = classes[i] if i < len(classes) else 0

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            color = (
                (0, 0, 255) if class_id == 1 else (0, 255, 0)
            )  # Red for defect, green for normal

            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"Class {class_id}: {score:.2f}"
            cv2.putText(
                result_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return result_frame

    @staticmethod
    def create_status_overlay(
        frame_shape: Tuple[int, int], stats: Dict[str, Any], detections: Dict[str, Any]
    ) -> np.ndarray:
        """Create a status overlay with system information."""
        overlay = np.zeros((frame_shape[0], 300, 3), dtype=np.uint8)

        # Title
        cv2.putText(
            overlay,
            "Detection Status",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Statistics
        y_pos = 70
        status_items = [
            f"Frames Processed: {stats.get('frames_processed', 0)}",
            f"Total Detections: {stats.get('detections_made', 0)}",
            f"Defects Found: {stats.get('defects_detected', 0)}",
            f"Current FPS: {stats.get('current_fps', 0):.1f}",
            f"Avg Process Time: {stats.get('average_processing_time', 0):.3f}s",
        ]

        for item in status_items:
            cv2.putText(
                overlay,
                item,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            y_pos += 25

        # Current detection info
        if detections.get("num_detections", 0) > 0:
            y_pos += 20
            cv2.putText(
                overlay,
                "Current Frame:",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            y_pos += 25

            for i, score in enumerate(detections.get("scores", [])):
                class_id = (
                    detections.get("classes", [i])[i]
                    if i < len(detections.get("classes", []))
                    else 0
                )
                class_name = "Defect" if class_id == 1 else "Normal"
                cv2.putText(
                    overlay,
                    f"  {class_name}: {score:.3f}",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                y_pos += 20

        return overlay


# Export the main classes
__all__ = [
    "VideoProcessor",
    "CalibratedVideoProcessor",
    "DetectionFrame",
    "VideoVisualizationHelper",
]
