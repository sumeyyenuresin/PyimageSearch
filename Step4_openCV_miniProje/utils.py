import cv2
import numpy as np
import imutils
from typing import Tuple, List


def resize_image(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Desired width (optional)
        height: Desired height (optional)
    
    Returns:
        Resized image
    """
    if width is None and height is None:
        return image
    
    (h, w) = image.shape[:2]
    
    if width is None:
        ratio = height / float(h)
        new_width = int(w * ratio)
        return cv2.resize(image, (new_width, height))
    else:
        ratio = width / float(w)
        new_height = int(h * ratio)
        return cv2.resize(image, (width, new_height))


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in the following order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: Array of 4 points
    
    Returns:
        Ordered points array
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum of coordinates: top-left will have smallest sum, bottom-right will have largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Difference of coordinates: top-right will have smallest diff, bottom-left will have largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective transformation to get a bird's eye view of the image.
    
    Args:
        image: Input image
        pts: Four corner points
    
    Returns:
        Transformed image
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width and height of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Define destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Apply perspective transformation
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Automatically determine Canny edge detection thresholds.
    
    Args:
        image: Input grayscale image
        sigma: Sigma value for threshold calculation
    
    Returns:
        Edge detected image
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def show_image(title: str, image: np.ndarray, wait_key: bool = True) -> None:
    """
    Display an image in a window.
    
    Args:
        title: Window title
        image: Image to display
        wait_key: Whether to wait for key press
    """
    cv2.imshow(title, image)
    if wait_key:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save_image(filename: str, image: np.ndarray) -> bool:
    """
    Save an image to file.
    
    Args:
        filename: Output filename
        image: Image to save
    
    Returns:
        True if successful, False otherwise
    """
    return cv2.imwrite(filename, image)