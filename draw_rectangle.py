import numpy as np
import cv2

def draw_rectangle(img, top, bottom, left, right):
    """
    Draws a rectangle on the input image.

    Args:
    - img: numpy.ndarray - The input image.
    - top: int - The y-coordinate of the top edge of the rectangle.
    - bottom: int - The y-coordinate of the bottom edge of the rectangle.
    - left: int - The x-coordinate of the left edge of the rectangle.
    - right: int - The x-coordinate of the right edge of the rectangle.

    Returns:
    - numpy.ndarray - The input image with the rectangle drawn on it.
    """
    # Determine if the image is grayscale or color
    if len(img.shape) == 2 or img.shape[2] == 1:  # Grayscale
        color = 255  # White color for grayscale
    else:  # Color
        color = (255, 255, 255)  # White color for RGB
    
    thickness = 2
    cv2.rectangle(img, (left, top), (right, bottom), color, thickness)
    return img