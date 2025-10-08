"""
Utility functions for OpenCV projects
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_image(image_path, color_mode=cv2.IMREAD_COLOR):
    """
    Load an image from file path
    
    Args:
        image_path (str): Path to the image file
        color_mode (int): OpenCV color mode flag
        
    Returns:
        np.ndarray: Loaded image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path, color_mode)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return image

def save_image(image, output_path):
    """
    Save an image to file
    
    Args:
        image (np.ndarray): Image to save
        output_path (str): Output file path
    """
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    success = cv2.imwrite(output_path, image)
    if not success:
        raise ValueError(f"Could not save image to: {output_path}")

def load_video(video_path):
    """
    Load a video file
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        cv2.VideoCapture: Video capture object
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    return cap

def create_video_writer(output_path, fps, frame_size, fourcc='XVID'):
    """
    Create a video writer object
    
    Args:
        output_path (str): Output video path
        fps (float): Frames per second
        frame_size (tuple): Frame size (width, height)
        fourcc (str): Video codec fourcc code
        
    Returns:
        cv2.VideoWriter: Video writer object
    """
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(output_path, fourcc_code, fps, frame_size)
    
    return writer

def display_image(image, title="Image", figsize=(10, 8)):
    """
    Display an image using matplotlib
    
    Args:
        image (np.ndarray): Image to display
        title (str): Window title
        figsize (tuple): Figure size
    """
    if len(image.shape) == 3:
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    plt.figure(figsize=figsize)
    plt.imshow(image_rgb, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()

def resize_image(image, width=None, height=None, maintain_aspect=True):
    """
    Resize an image
    
    Args:
        image (np.ndarray): Input image
        width (int): Target width
        height (int): Target height
        maintain_aspect (bool): Whether to maintain aspect ratio
        
    Returns:
        np.ndarray: Resized image
    """
    if width is None and height is None:
        return image
    
    h, w = image.shape[:2]
    
    if maintain_aspect:
        if width is None:
            # Calculate width based on height
            ratio = height / h
            width = int(w * ratio)
        elif height is None:
            # Calculate height based on width  
            ratio = width / w
            height = int(h * ratio)
        else:
            # Use the smaller ratio to maintain aspect ratio
            ratio = min(width / w, height / h)
            width = int(w * ratio)
            height = int(h * ratio)
    
    return cv2.resize(image, (width, height))

def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2, label=None):
    """
    Draw a bounding box on an image
    
    Args:
        image (np.ndarray): Input image
        bbox (tuple): Bounding box (x, y, w, h) or (x1, y1, x2, y2)
        color (tuple): Box color in BGR
        thickness (int): Line thickness
        label (str): Optional label text
        
    Returns:
        np.ndarray: Image with bounding box
    """
    img_copy = image.copy()
    
    if len(bbox) == 4:
        if all(bbox[i] >= 0 for i in range(4)):
            # Assume (x, y, w, h) format
            x, y, w, h = bbox
            x2, y2 = x + w, y + h
        else:
            # Assume (x1, y1, x2, y2) format
            x, y, x2, y2 = bbox
    
    cv2.rectangle(img_copy, (int(x), int(y)), (int(x2), int(y2)), color, thickness)
    
    if label:
        # Put label above the bounding box
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img_copy, (int(x), int(y) - label_size[1] - 10), 
                     (int(x) + label_size[0], int(y)), color, -1)
        cv2.putText(img_copy, label, (int(x), int(y) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img_copy

def create_trackbar_window(window_name, trackbars):
    """
    Create a window with trackbars
    
    Args:
        window_name (str): Name of the window
        trackbars (dict): Dictionary of trackbar configs
                         {name: {'min': min_val, 'max': max_val, 'default': default_val}}
    """
    cv2.namedWindow(window_name)
    
    def nothing(x):
        pass
    
    for name, config in trackbars.items():
        cv2.createTrackbar(name, window_name, 
                          config['default'], config['max'], nothing)

def get_trackbar_values(window_name, trackbar_names):
    """
    Get current values of trackbars
    
    Args:
        window_name (str): Name of the window
        trackbar_names (list): List of trackbar names
        
    Returns:
        dict: Dictionary of trackbar values
    """
    values = {}
    for name in trackbar_names:
        values[name] = cv2.getTrackbarPos(name, window_name)
    return values

def calculate_fps(start_time, frame_count):
    """
    Calculate FPS
    
    Args:
        start_time (float): Start time
        frame_count (int): Number of frames processed
        
    Returns:
        float: FPS value
    """
    import time
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma_x=0):
    """
    Apply Gaussian blur to an image
    
    Args:
        image (np.ndarray): Input image
        kernel_size (tuple): Blur kernel size
        sigma_x (float): Standard deviation in X direction
        
    Returns:
        np.ndarray: Blurred image
    """
    return cv2.GaussianBlur(image, kernel_size, sigma_x)

def convert_to_grayscale(image):
    """
    Convert image to grayscale
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def morphological_operations(image, operation, kernel_size=(3, 3), iterations=1):
    """
    Apply morphological operations
    
    Args:
        image (np.ndarray): Input binary image
        operation (str): Operation type ('opening', 'closing', 'gradient', 'tophat', 'blackhat')
        kernel_size (tuple): Kernel size
        iterations (int): Number of iterations
        
    Returns:
        np.ndarray: Processed image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    operations = {
        'opening': cv2.MORPH_OPEN,
        'closing': cv2.MORPH_CLOSE,
        'gradient': cv2.MORPH_GRADIENT,
        'tophat': cv2.MORPH_TOPHAT,
        'blackhat': cv2.MORPH_BLACKHAT
    }
    
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")
    
    return cv2.morphologyEx(image, operations[operation], kernel, iterations=iterations)