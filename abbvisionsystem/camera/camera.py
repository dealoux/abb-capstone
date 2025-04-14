"""Camera interface classes for different camera types."""

import logging
import cv2
import requests
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseCamera(ABC):
    """Base class for all camera implementations."""

    def __init__(self):
        self.connected = False

    @abstractmethod
    def connect(self):
        """Connect to the camera.

        Returns:
            bool: Success status
        """
        pass

    @abstractmethod
    def capture_image(self):
        """Capture an image from the camera.

        Returns:
            numpy.ndarray: Captured image or None on failure
        """
        pass

    def disconnect(self):
        """Disconnect from the camera."""
        self.connected = False
        return True


class CognexCamera(BaseCamera):
    """Cognex camera implementation."""

    def __init__(self, ip_address, port="80", username=None, password=None):
        super().__init__()
        self.ip_address = ip_address
        self.port = port
        self.username = username
        self.password = password
        self.url = f"http://{ip_address}:{port}"
        self.auth = (username, password) if username and password else None

    def connect(self):
        """Connect to Cognex camera via REST API."""
        try:
            # Test connection by requesting camera info
            response = requests.get(f"{self.url}/info", auth=self.auth, timeout=5)

            if response.status_code == 200:
                self.connected = True
                logger.info(f"Connected to Cognex camera at {self.ip_address}")
                return True
            else:
                logger.error(f"Failed to connect to camera: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Camera connection error: {str(e)}")
            return False

    def capture_image(self):
        """Capture image from Cognex camera."""
        if not self.connected:
            logger.warning("Camera not connected. Call connect() first.")
            return None

        try:
            # Request image from camera
            response = requests.get(f"{self.url}/image", auth=self.auth, timeout=5)

            if response.status_code == 200:
                # Convert response content to image
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return image
            else:
                logger.error(f"Failed to capture image: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Image capture error: {str(e)}")
            return None


class WebcamCamera(BaseCamera):
    """Webcam camera implementation (fallback option)."""

    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.capture = None

    def connect(self):
        """Connect to webcam."""
        try:
            self.capture = cv2.VideoCapture(self.camera_id)

            if self.capture.isOpened():
                self.connected = True
                logger.info(f"Connected to webcam with ID {self.camera_id}")
                return True
            else:
                logger.error(f"Failed to connect to webcam with ID {self.camera_id}")
                return False

        except Exception as e:
            logger.error(f"Webcam connection error: {str(e)}")
            return False

    def capture_image(self):
        """Capture image from webcam."""
        if not self.connected or self.capture is None:
            logger.warning("Webcam not connected. Call connect() first.")
            return None

        try:
            # Capture frame
            ret, frame = self.capture.read()

            if ret:
                return frame
            else:
                logger.error("Failed to capture frame from webcam")
                return None

        except Exception as e:
            logger.error(f"Webcam capture error: {str(e)}")
            return None

    def disconnect(self):
        """Release webcam resources."""
        if self.capture is not None:
            self.capture.release()

        self.connected = False
        return True
