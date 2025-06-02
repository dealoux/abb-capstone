"""Enhanced camera interface with calibration support."""

import logging
import cv2
import requests
import numpy as np
from abc import ABC, abstractmethod
from .calibration import CameraCalibrator, CalibrationResult

try:
    from pypylon import pylon

    PYLON_AVAILABLE = True
except ImportError:
    PYLON_AVAILABLE = False
    logging.warning("Pypylon not available. Basler cameras will not be supported.")

logger = logging.getLogger(__name__)


class BaseCamera(ABC):
    """Base class for all camera implementations."""

    def __init__(self):
        self.connected = False
        self.calibrator = CameraCalibrator()
        self.auto_undistort = True

    @abstractmethod
    def connect(self):
        """Connect to the camera."""
        pass

    @abstractmethod
    def capture_image(self):
        """Capture an image from the camera."""
        pass

    def disconnect(self):
        """Disconnect from the camera."""
        self.connected = False
        return True

    def capture_corrected_image(self):
        """Capture and optionally undistort image."""
        image = self.capture_image()

        if (
            image is not None
            and self.auto_undistort
            and self.calibrator.calibration_result
        ):
            image = self.calibrator.undistort_image(image)

        return image


class BaslerCamera(BaseCamera):
    """Enhanced Basler camera implementation with calibration support."""

    def __init__(self, device_index=0):
        super().__init__()
        self.device_index = device_index
        self.camera = None
        self.exposure_time = 10000  # microseconds
        self.gain = 1.0

        if not PYLON_AVAILABLE:
            logger.error("Pypylon not installed. Cannot use Basler cameras.")

    def connect(self):
        """Connect to Basler camera."""
        if not PYLON_AVAILABLE:
            logger.error("Pypylon not installed. Cannot connect to Basler camera.")
            return False

        try:
            # Get available devices
            available_devices = pylon.TlFactory.GetInstance().EnumerateDevices()

            if not available_devices:
                logger.error("No Basler cameras found")
                return False

            if self.device_index >= len(available_devices):
                logger.error(
                    f"Camera index {self.device_index} out of range. Only {len(available_devices)} cameras available."
                )
                return False

            # Create camera instance
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateDevice(
                    available_devices[self.device_index]
                )
            )

            # Open the camera
            self.camera.Open()

            # Configure camera parameters for top-down inspection
            self._configure_camera_parameters()

            self.connected = True
            logger.info(
                f"Connected to Basler camera {self.camera.GetDeviceInfo().GetModelName()}"
            )
            return True

        except Exception as e:
            logger.error(f"Basler camera connection error: {str(e)}")
            return False

    def _configure_camera_parameters(self):
        """Configure camera parameters for optimal top-down inspection."""
        if not self.camera or not self.camera.IsOpen():
            return

        try:
            # Set exposure time
            if self.camera.ExposureTime.IsWritable():
                self.camera.ExposureTime.SetValue(self.exposure_time)
                logger.info(f"Set exposure time to {self.exposure_time} µs")

            # Set gain
            if self.camera.Gain.IsWritable():
                self.camera.Gain.SetValue(self.gain)
                logger.info(f"Set gain to {self.gain}")

            # Set acquisition mode to continuous
            if self.camera.AcquisitionMode.IsWritable():
                self.camera.AcquisitionMode.SetValue("Continuous")

            # Set pixel format to optimize for vision processing
            if self.camera.PixelFormat.IsWritable():
                self.camera.PixelFormat.SetValue("BGR8")  # or "Mono8" for grayscale

            # Enable auto white balance for color consistency
            if (
                hasattr(self.camera, "BalanceWhiteAuto")
                and self.camera.BalanceWhiteAuto.IsWritable()
            ):
                self.camera.BalanceWhiteAuto.SetValue("Continuous")

        except Exception as e:
            logger.warning(f"Could not configure all camera parameters: {e}")

    def set_exposure_time(self, exposure_us: int):
        """Set camera exposure time in microseconds."""
        self.exposure_time = exposure_us

        if (
            self.camera
            and self.camera.IsOpen()
            and self.camera.ExposureTime.IsWritable()
        ):
            try:
                self.camera.ExposureTime.SetValue(exposure_us)
                logger.info(f"Updated exposure time to {exposure_us} µs")
            except Exception as e:
                logger.error(f"Failed to set exposure time: {e}")

    def set_gain(self, gain: float):
        """Set camera gain."""
        self.gain = gain

        if self.camera and self.camera.IsOpen() and self.camera.Gain.IsWritable():
            try:
                self.camera.Gain.SetValue(gain)
                logger.info(f"Updated gain to {gain}")
            except Exception as e:
                logger.error(f"Failed to set gain: {e}")

    def capture_image(self):
        """Capture image from Basler camera."""
        if not self.connected or self.camera is None:
            logger.warning("Basler camera not connected. Call connect() first.")
            return None

        try:
            # Start grabbing a single image
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

            if self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(
                    5000, pylon.TimeoutHandling_ThrowException
                )

                if grab_result.GrabSucceeded():
                    # Convert to OpenCV format
                    image = grab_result.Array

                    # Convert to BGR color space if needed
                    if len(image.shape) == 2:  # If grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                    grab_result.Release()
                    return image
                else:
                    logger.error(
                        f"Failed to grab image: {grab_result.ErrorDescription}"
                    )
                    grab_result.Release()
                    return None
            else:
                logger.error("Camera is not grabbing images")
                return None

        except Exception as e:
            logger.error(f"Basler capture error: {str(e)}")
            return None
        finally:
            # Stop grabbing
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()

    def disconnect(self):
        """Disconnect from Basler camera."""
        if self.camera is not None:
            try:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                if self.camera.IsOpen():
                    self.camera.Close()
                self.camera = None
            except Exception as e:
                logger.error(f"Error disconnecting from camera: {str(e)}")

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
