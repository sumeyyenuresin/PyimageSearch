"""
Geometric Operations
This module demonstrates geometric transformations including rotation,
translation, scaling, and perspective transformations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate_image(img, angle, center=None, scale=1.0):
    """
    Rotate image by specified angle
    """
    if img is None:
        return None
    
    (h, w) = img.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply rotation
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    
    return rotated

def translate_image(img, tx, ty):
    """
    Translate (move) image by specified offset
    """
    if img is None:
        return None
    
    (h, w) = img.shape[:2]
    
    # Translation matrix
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply translation
    translated = cv2.warpAffine(img, translation_matrix, (w, h))
    
    return translated

def scale_image(img, fx, fy):
    """
    Scale image by specified factors
    """
    if img is None:
        return None
    
    scaled = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    
    return scaled

def flip_image(img, flip_code):
    """
    Flip image
    flip_code: 0 = vertical, 1 = horizontal, -1 = both
    """
    if img is None:
        return None
    
    return cv2.flip(img, flip_code)

def perspective_transform(img, src_points, dst_points):
    """
    Apply perspective transformation
    """
    if img is None:
        return None
    
    (h, w) = img.shape[:2]
    
    # Get perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply perspective transformation
    transformed = cv2.warpPerspective(img, matrix, (w, h))
    
    return transformed

def affine_transform(img, src_points, dst_points):
    """
    Apply affine transformation
    """
    if img is None:
        return None
    
    (h, w) = img.shape[:2]
    
    # Get affine transformation matrix
    matrix = cv2.getAffineTransform(src_points, dst_points)
    
    # Apply affine transformation
    transformed = cv2.warpAffine(img, matrix, (w, h))
    
    return transformed

def main():
    """
    Main function to demonstrate geometric operations
    """
    # Load image
    image_path = "../data/jp.png"
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print("=== Geometric Operations Demo ===")
    
    # Apply transformations
    rotated_45 = rotate_image(img, 45)
    rotated_90 = rotate_image(img, 90)
    translated = translate_image(img, 50, 30)
    scaled = scale_image(img, 0.7, 0.7)
    flipped_h = flip_image(img, 1)  # Horizontal flip
    flipped_v = flip_image(img, 0)  # Vertical flip
    
    # Display results
    plt.figure(figsize=(18, 12))
    
    # Original
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    # Rotated 45 degrees
    plt.subplot(3, 3, 2)
    plt.imshow(cv2.cvtColor(rotated_45, cv2.COLOR_BGR2RGB))
    plt.title('Rotated 45°')
    plt.axis('off')
    
    # Rotated 90 degrees
    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(rotated_90, cv2.COLOR_BGR2RGB))
    plt.title('Rotated 90°')
    plt.axis('off')
    
    # Translated
    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(translated, cv2.COLOR_BGR2RGB))
    plt.title('Translated (50, 30)')
    plt.axis('off')
    
    # Scaled
    plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
    plt.title('Scaled (0.7x)')
    plt.axis('off')
    
    # Horizontal flip
    plt.subplot(3, 3, 6)
    plt.imshow(cv2.cvtColor(flipped_h, cv2.COLOR_BGR2RGB))
    plt.title('Horizontal Flip')
    plt.axis('off')
    
    # Vertical flip
    plt.subplot(3, 3, 7)
    plt.imshow(cv2.cvtColor(flipped_v, cv2.COLOR_BGR2RGB))
    plt.title('Vertical Flip')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate perspective transformation
    demo_perspective_transform(img)

def demo_perspective_transform(img):
    """
    Demonstrate perspective transformation with example points
    """
    (h, w) = img.shape[:2]
    
    # Define source points (corners of a rectangle)
    src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    
    # Define destination points (trapezoid effect)
    dst_points = np.float32([[w*0.1, h*0.1], [w*0.9, h*0.2], 
                            [w*0.2, h*0.9], [w*0.8, h*0.8]])
    
    # Apply perspective transformation
    perspective_img = perspective_transform(img, src_points, dst_points)
    
    # Display comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(perspective_img, cv2.COLOR_BGR2RGB))
    plt.title('Perspective Transform')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()