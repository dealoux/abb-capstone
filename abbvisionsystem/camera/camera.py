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
        self.origin_point = None  # Store origin point like in Test1.py

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

    def add_calibration_image(self, image):
        """Add calibration image and set origin from first valid image - copied from Test1.py logic."""
        # Get current number of valid images before adding
        valid_images_before = len(self.calibrator.calibration_images)
        
        # Add image for calibration first
        if self.calibrator.add_calibration_image(image):
            valid_images_after = len(self.calibrator.calibration_images)
            
            # Get the detected corners from the latest calibration image
            latest_calib_data = self.calibrator.calibration_images[-1]
            corners = latest_calib_data["corners"]
            
            if corners is not None and len(corners) > 0:
                # Set origin to the first corner (top-left of chessboard pattern)
                origin_point = tuple(corners[0].ravel().astype(int))
                
                # Store origin from first valid image - copied from Test1.py
                if valid_images_after == 1:  # First valid image
                    self.origin_point = origin_point
                    logger.info(f"Origin set from first calibration image: {self.origin_point}")
                
                return True, origin_point
                
        return False, None

    def calibrate_camera_with_origin(self, image_list):
        """Perform calibration with origin setting - copied from Test1.py approach."""
        valid_images = 0
        
        for image in image_list:
            success, origin_point = self.add_calibration_image(image)
            if success:
                valid_images += 1
                logger.info(f"Added calibration image {valid_images}")
        
        # Perform calibration if we have enough images
        if valid_images >= 3:
            result = self.calibrator.calibrate_camera()
            if result:
                logger.info(f"Calibration successful! Found {valid_images} valid patterns.")
                if self.origin_point:
                    logger.info(f"Origin point: {self.origin_point}")
                return True
            else:
                logger.error("Calibration failed")
                return False
        else:
            logger.error(f"Need at least 3 valid images, got {valid_images}")
            return False

    def get_origin_point(self):
        """Get the stored origin point - copied from Test1.py."""
        return self.origin_point

    def get_calibrated_origin(self):
        """Get origin from calibration data if not stored directly - copied from Test1.py fallback logic."""
        if self.origin_point:
            return self.origin_point
        elif self.calibrator.calibration_result and self.calibrator.calibration_images:
            try:
                # Get origin from first calibration image
                first_calib = self.calibrator.calibration_images[0]
                if first_calib["corners"] is not None:
                    origin_point = tuple(first_calib["corners"][0].ravel().astype(int))
                    self.origin_point = origin_point  # Cache it
                    return origin_point
            except Exception as e:
                logger.warning(f"Could not get calibrated origin: {e}")
        return None

    def get_pixels_per_mm(self):
        """Get calibration scale factor."""
        if self.calibrator.calibration_result:
            return self.calibrator.calibration_result.pixels_per_mm
        return None

    def convert_pixel_to_mm_from_origin(self, pixel_coords):
        """Convert pixel coordinates to mm relative to origin - copied from Test1.py logic."""
        origin = self.get_calibrated_origin()
        pixels_per_mm = self.get_pixels_per_mm()
        
        if origin is None or pixels_per_mm is None:
            logger.warning("Origin not set or scale not calculated")
            return None
            
        # Calculate relative position from origin in pixels
        rel_x_pixels = pixel_coords[0] - origin[0]
        rel_y_pixels = pixel_coords[1] - origin[1]
        
        # Convert to mm
        x_mm = rel_x_pixels / pixels_per_mm
        y_mm = rel_y_pixels / pixels_per_mm
        
        return (x_mm, y_mm)

    def draw_coordinate_system_on_image(self, image):
        """Draw coordinate system on image - copied from Test1.py visualization."""
        result_image = image.copy()
        
        # Get origin and scale
        origin = self.get_calibrated_origin()
        pixels_per_mm = self.get_pixels_per_mm()
        
        if origin:
            ox, oy = origin
            
            # Draw origin point (Yellow dot) - larger like in Test1.py
            cv2.circle(result_image, (ox, oy), 8, (255, 255, 0), -1)
            
            # Coordinate axes - match Test1.py
            axis_length = 100
            # X-axis (Red) - along chessboard row
            cv2.arrowedLine(result_image, (ox, oy), (ox + axis_length, oy), 
                           (0, 0, 255), 3, tipLength=0.1)
            # Y-axis (Green) - along chessboard column  
            cv2.arrowedLine(result_image, (ox, oy), (ox, oy + axis_length), 
                           (0, 255, 0), 3, tipLength=0.1)
            
            # Labels - match Test1.py style
            cv2.putText(result_image, "O(0,0)", (ox-30, oy-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(result_image, "X", (ox + axis_length + 10, oy + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(result_image, "Y", (ox - 15, oy + axis_length + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add scale info if available
            if pixels_per_mm:
                cv2.putText(result_image, f"Scale: {pixels_per_mm:.2f} px/mm", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image

    def set_calibration_pattern(self, rows: int, cols: int, square_size_mm: float):
        """Set calibration pattern parameters - match Test1.py interface."""
        self.calibrator.set_chessboard_pattern(rows, cols, square_size_mm)

    def get_calibration_settings(self):
        """Get current calibration settings."""
        return {
            'rows': self.calibrator.chessboard_size[1],  # Convert back to rows
            'cols': self.calibrator.chessboard_size[0],  # Convert back to cols
            'square_size_mm': self.calibrator.square_size_mm
        }

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

            # Check current pixel format and available formats
            current_format = self.camera.PixelFormat.GetValue()
            logger.info(f"Current pixel format: {current_format}")
            
            # List all available pixel formats
            available_formats = [str(entry) for entry in self.camera.PixelFormat.Symbolics]
            logger.info(f"Available pixel formats: {available_formats}")
            
            # If camera supports color formats, try to set one
            if self.camera.PixelFormat.IsWritable():
                # Priority order for color formats
                preferred_formats = ["RGB8", "BGR8", "BayerRG8", "BayerBG8", "BayerGR8", "BayerGB8"]
                
                for fmt in preferred_formats:
                    if fmt in available_formats:
                        try:
                            self.camera.PixelFormat.SetValue(fmt)
                            logger.info(f"Successfully set pixel format to {fmt}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to set format {fmt}: {e}")
                            continue
                else:
                    logger.warning(f"No color formats available. Staying with {current_format}")

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
                    image = grab_result.Array
                    pixel_format = self.camera.PixelFormat.GetValue()
                    
                    logger.info(f"Captured image with format: {pixel_format}, shape: {image.shape}")
                    
                    # Handle different pixel formats
                    if pixel_format.startswith("Bayer"):
                        # Convert Bayer pattern to BGR
                        if pixel_format == "BayerRG8":
                            image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2BGR)
                        elif pixel_format == "BayerBG8":
                            image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
                        elif pixel_format == "BayerGR8":
                            image = cv2.cvtColor(image, cv2.COLOR_BAYER_GR2BGR)
                        elif pixel_format == "BayerGB8":
                            image = cv2.cvtColor(image, cv2.COLOR_BAYER_GB2BGR)
                        logger.info(f"Converted {pixel_format} to BGR")
                        
                    elif pixel_format == "RGB8":
                        # Convert RGB to BGR for OpenCV
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        logger.info("Converted RGB8 to BGR")
                        
                    elif pixel_format == "BGR8":
                        # Already in correct format
                        logger.info("Using BGR8 format as-is")
                        
                    elif len(image.shape) == 2 or pixel_format.startswith("Mono"):
                        # Grayscale/Mono format - convert to BGR
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        logger.info("Converted mono/grayscale to BGR")

                    grab_result.Release()
                    return image
                else:
                    logger.error(f"Failed to grab image: {grab_result.ErrorDescription}")
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

# In your detection page code
def detection_page():
    # Get camera instance (assuming you have access to it)
    camera = get_camera_instance()  # Your camera access method
    
    # Use calibrated origin - NOT hardcoded values
    origin_point = camera.get_calibrated_origin()
    
    if origin_point:
        ox, oy = origin_point
        st.success(f"Using calibrated origin: ({ox}, {oy})")
    else:
        # Fallback only if no calibration exists
        ox, oy = 100, 100  # Default fallback
        st.warning("No calibrated origin found, using default (100, 100)")
    
    # Use ox, oy for coordinate conversions
    if detections:
        for detection in detections:
            # Convert detection coordinates relative to calibrated origin
            mm_coords = camera.convert_pixel_to_mm_from_origin(detection.center)
            if mm_coords:
                x_mm, y_mm = mm_coords
                st.write(f"Detection at: ({x_mm:.2f}, {y_mm:.2f}) mm from origin")
