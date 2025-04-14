"""Image preprocessing functions for vision system."""

import cv2
import numpy as np


def resize_image(image, target_size=(640, 480)):
    """Resize image while maintaining aspect ratio.

    Args:
        image: Input image (numpy array)
        target_size: Target size as (width, height) tuple

    Returns:
        numpy.ndarray: Resized image
    """
    height, width = image.shape[:2]

    # Calculate scaling factor
    scale = min(target_size[0] / width, target_size[1] / height)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize the image
    resized = cv2.resize(image, (new_width, new_height))

    return resized


def prepare_for_detection(image):
    """Prepare image for object detection model.

    Args:
        image: Input image (numpy array)

    Returns:
        numpy.ndarray: Processed image ready for model input
    """
    # Convert BGR to RGB (if image is from OpenCV)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def apply_image_enhancement(image, brightness=0, contrast=0):
    """Apply basic image enhancement.

    Args:
        image: Input image (numpy array)
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)

    Returns:
        numpy.ndarray: Enhanced image
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return image
