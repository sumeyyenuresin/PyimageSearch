"""
Basic OpenCV Operations
This module demonstrates basic image operations including loading, displaying,
resizing, and basic manipulations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_display_image(image_path):
    """
    Load and display an image using OpenCV and matplotlib
    """
    # Load image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display image
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    
    return img

def resize_image(img, width=None, height=None):
    """
    Resize image while maintaining aspect ratio
    """
    if img is None:
        return None
        
    (h, w) = img.shape[:2]
    
    if width is None and height is None:
        return img
    
    if width is None:
        ratio = height / float(h)
        new_width = int(w * ratio)
        resized = cv2.resize(img, (new_width, height))
    else:
        ratio = width / float(w)
        new_height = int(h * ratio)
        resized = cv2.resize(img, (width, new_height))
    
    return resized

def convert_colorspace(img, conversion_type='GRAY'):
    """
    Convert image to different color spaces
    """
    if img is None:
        return None
    
    if conversion_type == 'GRAY':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif conversion_type == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif conversion_type == 'LAB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    else:
        return img

def crop_image(img, x, y, w, h):
    """
    Crop image to specified region
    """
    if img is None:
        return None
    
    return img[y:y+h, x:x+w]

def image_properties(img):
    """
    Display basic image properties
    """
    if img is None:
        return
    
    print(f"Image shape: {img.shape}")
    print(f"Data type: {img.dtype}")
    print(f"Min pixel value: {img.min()}")
    print(f"Max pixel value: {img.max()}")

def main():
    """
    Main function to demonstrate basic operations
    """
    # Example usage
    image_path = "../data/jp.png"
    
    print("=== Basic OpenCV Operations Demo ===")
    
    # Load and display image
    img = load_and_display_image(image_path)
    
    if img is not None:
        # Show image properties
        print("\nImage Properties:")
        image_properties(img)
        
        # Resize image
        resized_img = resize_image(img, width=300)
        
        # Convert to grayscale
        gray_img = convert_colorspace(img, 'GRAY')
        
        # Display results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        plt.title('Resized')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(gray_img, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()