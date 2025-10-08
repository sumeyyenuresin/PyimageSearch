"""
Thresholding and Contours
This module demonstrates various thresholding techniques and contour detection
for image segmentation and object detection.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def simple_threshold(img, threshold_value=127, max_value=255, threshold_type=cv2.THRESH_BINARY):
    """
    Apply simple thresholding
    """
    if img is None:
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    _, thresh = cv2.threshold(gray, threshold_value, max_value, threshold_type)
    
    return thresh

def adaptive_threshold(img, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C, 
                      threshold_type=cv2.THRESH_BINARY, block_size=11, c=2):
    """
    Apply adaptive thresholding
    """
    if img is None:
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    thresh = cv2.adaptiveThreshold(gray, max_value, adaptive_method, 
                                  threshold_type, block_size, c)
    
    return thresh

def otsu_threshold(img):
    """
    Apply Otsu's thresholding
    """
    if img is None:
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def find_contours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    """
    Find contours in binary image
    """
    if img is None:
        return None, None
    
    # Ensure binary image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = img
    
    contours, hierarchy = cv2.findContours(binary, mode, method)
    
    return contours, hierarchy

def draw_contours(img, contours, contour_idx=-1, color=(0, 255, 0), thickness=2):
    """
    Draw contours on image
    """
    if img is None or contours is None:
        return None
    
    result = img.copy()
    cv2.drawContours(result, contours, contour_idx, color, thickness)
    
    return result

def contour_properties(contour):
    """
    Calculate various properties of a contour
    """
    properties = {}
    
    # Area
    properties['area'] = cv2.contourArea(contour)
    
    # Perimeter
    properties['perimeter'] = cv2.arcLength(contour, True)
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    properties['bounding_rect'] = (x, y, w, h)
    
    # Centroid
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        properties['centroid'] = (cx, cy)
    else:
        properties['centroid'] = (0, 0)
    
    # Aspect ratio
    properties['aspect_ratio'] = float(w) / h
    
    # Extent (ratio of contour area to bounding rectangle area)
    properties['extent'] = properties['area'] / (w * h)
    
    # Solidity (ratio of contour area to convex hull area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area != 0:
        properties['solidity'] = properties['area'] / hull_area
    else:
        properties['solidity'] = 0
    
    return properties

def filter_contours_by_area(contours, min_area=100, max_area=10000):
    """
    Filter contours by area
    """
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered_contours.append(contour)
    
    return filtered_contours

def approximate_contour(contour, epsilon_factor=0.02):
    """
    Approximate contour to reduce number of points
    """
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    return approx

def main():
    """
    Main function to demonstrate thresholding and contour detection
    """
    # Load image
    image_path = "../data/jp.png"
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print("=== Thresholding and Contours Demo ===")
    
    # Apply different thresholding techniques
    simple_thresh = simple_threshold(img, 127)
    adaptive_thresh_mean = adaptive_threshold(img, adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C)
    adaptive_thresh_gaussian = adaptive_threshold(img, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    otsu_thresh = otsu_threshold(img)
    
    # Display thresholding results
    plt.figure(figsize=(16, 12))
    
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(simple_thresh, cmap='gray')
    plt.title('Simple Threshold')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(adaptive_thresh_mean, cmap='gray')
    plt.title('Adaptive Threshold (Mean)')
    plt.axis('off')
    
    plt.subplot(3, 3, 4)
    plt.imshow(adaptive_thresh_gaussian, cmap='gray')
    plt.title('Adaptive Threshold (Gaussian)')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(otsu_thresh, cmap='gray')
    plt.title("Otsu's Threshold")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate contour detection
    demo_contour_detection(img, otsu_thresh)

def demo_contour_detection(original_img, binary_img):
    """
    Demonstrate contour detection and analysis
    """
    print("\n=== Contour Detection Demo ===")
    
    # Find contours
    contours, hierarchy = find_contours(binary_img)
    
    print(f"Found {len(contours)} contours")
    
    # Filter contours by area
    filtered_contours = filter_contours_by_area(contours, min_area=50)
    print(f"After filtering by area: {len(filtered_contours)} contours")
    
    # Draw all contours
    contour_img = draw_contours(original_img, contours, color=(0, 255, 0))
    
    # Draw filtered contours
    filtered_contour_img = draw_contours(original_img, filtered_contours, color=(255, 0, 0))
    
    # Analyze largest contour
    if filtered_contours:
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        properties = contour_properties(largest_contour)
        
        print(f"\nLargest contour properties:")
        print(f"  Area: {properties['area']:.2f}")
        print(f"  Perimeter: {properties['perimeter']:.2f}")
        print(f"  Centroid: {properties['centroid']}")
        print(f"  Aspect ratio: {properties['aspect_ratio']:.2f}")
        print(f"  Extent: {properties['extent']:.2f}")
        print(f"  Solidity: {properties['solidity']:.2f}")
        
        # Draw bounding rectangle and centroid
        analysis_img = original_img.copy()
        x, y, w, h = properties['bounding_rect']
        cv2.rectangle(analysis_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.circle(analysis_img, properties['centroid'], 5, (0, 0, 255), -1)
        cv2.drawContours(analysis_img, [largest_contour], -1, (0, 255, 0), 2)
        
        # Approximate contour
        approx_contour = approximate_contour(largest_contour)
        approx_img = original_img.copy()
        cv2.drawContours(approx_img, [approx_contour], -1, (255, 0, 255), 3)
        
        # Display results
        plt.figure(figsize=(16, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(binary_img, cmap='gray')
        plt.title('Binary Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.title(f'All Contours ({len(contours)})')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(filtered_contour_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Filtered Contours ({len(filtered_contours)})')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(analysis_img, cv2.COLOR_BGR2RGB))
        plt.title('Largest Contour Analysis')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(approx_img, cv2.COLOR_BGR2RGB))
        plt.title('Approximated Contour')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()