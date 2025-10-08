"""
Object Counting
This module demonstrates object detection and counting techniques using
contour analysis, template matching, and blob detection.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def count_objects_by_contours(img, min_area=100, max_area=10000):
    """
    Count objects using contour detection
    """
    if img is None:
        return 0, None, None
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            valid_contours.append(contour)
    
    # Draw contours on original image
    result_img = img.copy()
    cv2.drawContours(result_img, valid_contours, -1, (0, 255, 0), 2)
    
    # Add numbers to each object
    for i, contour in enumerate(valid_contours):
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(result_img, str(i+1), (cx-10, cy+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return len(valid_contours), result_img, binary

def template_matching(img, template, method=cv2.TM_CCOEFF_NORMED, threshold=0.8):
    """
    Count objects using template matching
    """
    if img is None or template is None:
        return 0, None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
    
    # Perform template matching
    result = cv2.matchTemplate(gray, template_gray, method)
    
    # Find locations where template matches
    locations = np.where(result >= threshold)
    matches = list(zip(*locations[::-1]))  # Switch x,y coordinates
    
    # Draw rectangles around matches
    result_img = img.copy()
    h, w = template_gray.shape
    
    for match in matches:
        top_left = match
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
    
    return len(matches), result_img

def blob_detection(img):
    """
    Detect and count blobs using SimpleBlobDetector
    """
    if img is None:
        return 0, None
    
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by Area
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 5000
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    
    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Detect blobs
    keypoints = detector.detect(gray)
    
    # Draw detected blobs
    result_img = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return len(keypoints), result_img

def watershed_segmentation(img):
    """
    Segment objects using watershed algorithm
    """
    if img is None:
        return 0, None
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Find sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]  # Mark boundaries in red
    
    # Count unique markers (excluding background and boundaries)
    unique_markers = np.unique(markers)
    object_count = len(unique_markers) - 2  # Exclude background (1) and boundaries (-1)
    
    return object_count, img

def analyze_shape_properties(contours):
    """
    Analyze and classify shapes based on contour properties
    """
    shape_analysis = []
    
    for i, contour in enumerate(contours):
        analysis = {}
        analysis['contour_id'] = i
        
        # Basic properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area > 0 and perimeter > 0:
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Extent
            extent = area / (w * h)
            
            # Solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Approximate polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
            
            # Classify shape
            shape_type = "Unknown"
            if vertices == 3:
                shape_type = "Triangle"
            elif vertices == 4:
                if 0.95 <= aspect_ratio <= 1.05:
                    shape_type = "Square"
                else:
                    shape_type = "Rectangle"
            elif vertices > 4:
                if circularity > 0.7:
                    shape_type = "Circle"
                else:
                    shape_type = "Polygon"
            
            analysis.update({
                'area': area,
                'perimeter': perimeter,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'solidity': solidity,
                'circularity': circularity,
                'vertices': vertices,
                'shape_type': shape_type
            })
            
            shape_analysis.append(analysis)
    
    return shape_analysis

def main():
    """
    Main function to demonstrate object counting techniques
    """
    # Load image
    image_path = "../data/tetris_blocks.png"
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        # Create a sample image with simple shapes for demo
        img = create_sample_image()
    
    print("=== Object Counting Demo ===")
    
    # Method 1: Contour-based counting
    count_contours, contour_result, binary_img = count_objects_by_contours(img)
    print(f"Objects found using contours: {count_contours}")
    
    # Method 2: Blob detection
    count_blobs, blob_result = blob_detection(img)
    print(f"Objects found using blob detection: {count_blobs}")
    
    # Method 3: Watershed segmentation
    count_watershed, watershed_result = watershed_segmentation(img)
    print(f"Objects found using watershed: {count_watershed}")
    
    # Display results
    plt.figure(figsize=(16, 12))
    
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(contour_result, cv2.COLOR_BGR2RGB))
    plt.title(f'Contour Detection ({count_contours} objects)')
    plt.axis('off')
    
    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(blob_result, cv2.COLOR_BGR2RGB))
    plt.title(f'Blob Detection ({count_blobs} blobs)')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(watershed_result, cv2.COLOR_BGR2RGB))
    plt.title(f'Watershed ({count_watershed} objects)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze shapes
    if contour_result is not None:
        demo_shape_analysis(img)

def create_sample_image():
    """
    Create a sample image with simple shapes for demonstration
    """
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img.fill(255)  # White background
    
    # Draw some shapes
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)  # Red square
    cv2.circle(img, (250, 100), 50, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(img, (350, 75), (500, 125), (255, 0, 0), -1)  # Blue rectangle
    
    # Triangle
    pts = np.array([[100, 250], [50, 350], [150, 350]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 0))  # Cyan triangle
    
    # Another circle
    cv2.circle(img, (400, 300), 40, (255, 0, 255), -1)  # Magenta circle
    
    return img

def demo_shape_analysis(img):
    """
    Demonstrate shape analysis and classification
    """
    print("\n=== Shape Analysis Demo ===")
    
    # Get contours
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    # Analyze shapes
    shape_analysis = analyze_shape_properties(filtered_contours)
    
    # Draw analysis results
    analysis_img = img.copy()
    
    for i, analysis in enumerate(shape_analysis):
        contour = filtered_contours[i]
        
        # Draw contour
        cv2.drawContours(analysis_img, [contour], -1, (0, 255, 0), 2)
        
        # Get centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Add shape label
            cv2.putText(analysis_img, analysis['shape_type'], 
                       (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Print analysis
        print(f"\nShape {i+1}: {analysis['shape_type']}")
        print(f"  Area: {analysis['area']:.2f}")
        print(f"  Circularity: {analysis['circularity']:.3f}")
        print(f"  Aspect Ratio: {analysis['aspect_ratio']:.2f}")
        print(f"  Vertices: {analysis['vertices']}")
    
    # Display shape analysis
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(analysis_img, cv2.COLOR_BGR2RGB))
    plt.title('Shape Analysis')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()