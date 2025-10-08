import cv2
import numpy as np
import argparse
from utils import show_image, save_image, resize_image


class ShapeDetector:
    def __init__(self):
        pass
    
    def detect(self, contour):
        """
        Detect the shape of a contour.
        
        Args:
            contour: Input contour
        
        Returns:
            String describing the detected shape
        """
        # Initialize the shape name
        shape = "unidentified"
        
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        # If the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        
        # If the shape has 4 vertices, it is either a square or rectangle
        elif len(approx) == 4:
            # Compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            
            # A square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        
        # If the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        
        # Otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        
        return shape


def preprocess_image(image):
    """
    Preprocess image for shape detection.
    
    Args:
        image: Input image
    
    Returns:
        Processed grayscale image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    
    return thresh


def detect_shapes(image, min_area=500):
    """
    Detect shapes in an image.
    
    Args:
        image: Input image
        min_area: Minimum contour area to consider
    
    Returns:
        Tuple of (detected shapes list, annotated image)
    """
    # Preprocess the image
    thresh = preprocess_image(image)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize the shape detector
    sd = ShapeDetector()
    
    # Create a copy of the image for drawing
    output = image.copy()
    
    detected_shapes = []
    
    # Loop over the contours
    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) < min_area:
            continue
        
        # Detect the shape
        shape = sd.detect(contour)
        
        # Calculate contour center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        # Draw the contour and center of the shape on the image
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
        cv2.circle(output, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(output, shape, (cX - 20, cY - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Store the detected shape info
        detected_shapes.append({
            'shape': shape,
            'center': (cX, cY),
            'area': cv2.contourArea(contour),
            'contour': contour
        })
    
    return detected_shapes, output


def count_shapes(detected_shapes):
    """
    Count the number of each shape type.
    
    Args:
        detected_shapes: List of detected shapes
    
    Returns:
        Dictionary with shape counts
    """
    shape_counts = {}
    for shape_info in detected_shapes:
        shape = shape_info['shape']
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    
    return shape_counts


def filter_shapes_by_area(detected_shapes, min_area=None, max_area=None):
    """
    Filter shapes by area.
    
    Args:
        detected_shapes: List of detected shapes
        min_area: Minimum area threshold
        max_area: Maximum area threshold
    
    Returns:
        Filtered list of shapes
    """
    filtered = []
    for shape_info in detected_shapes:
        area = shape_info['area']
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        filtered.append(shape_info)
    
    return filtered


def main():
    parser = argparse.ArgumentParser(description='Detect geometric shapes in an image')
    parser.add_argument('-i', '--image', required=True, help='Path to input image')
    parser.add_argument('-o', '--output', help='Path to output image')
    parser.add_argument('--min-area', type=int, default=500, help='Minimum contour area')
    parser.add_argument('--max-area', type=int, help='Maximum contour area')
    parser.add_argument('--show', action='store_true', help='Show the detection results')
    parser.add_argument('--count', action='store_true', help='Display shape counts')
    parser.add_argument('--resize', type=int, help='Resize image width for processing')
    
    args = parser.parse_args()
    
    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    # Resize image if requested
    if args.resize:
        image = resize_image(image, width=args.resize)
        print(f"Resized image to width: {args.resize}")
    
    print("Detecting shapes in the image...")
    
    # Detect shapes
    detected_shapes, result_image = detect_shapes(image, args.min_area)
    
    # Filter by area if specified
    if args.max_area:
        detected_shapes = filter_shapes_by_area(detected_shapes, args.min_area, args.max_area)
    
    print(f"Detected {len(detected_shapes)} shapes")
    
    # Count shapes if requested
    if args.count:
        shape_counts = count_shapes(detected_shapes)
        print("\nShape counts:")
        for shape, count in shape_counts.items():
            print(f"  {shape.capitalize()}: {count}")
    
    # Show detailed information about each shape
    print("\nDetected shapes:")
    for i, shape_info in enumerate(detected_shapes, 1):
        print(f"  {i}. {shape_info['shape'].capitalize()} - "
              f"Center: {shape_info['center']}, Area: {shape_info['area']:.0f}")
    
    # Show results if requested
    if args.show:
        # Show original and preprocessed images
        thresh = preprocess_image(image)
        show_image("Original", image, wait_key=False)
        show_image("Preprocessed", thresh, wait_key=False)
        show_image("Shape Detection Results", result_image)
    
    # Save result
    if args.output:
        if save_image(args.output, result_image):
            print(f"Result saved to {args.output}")
        else:
            print(f"Error: Could not save image to {args.output}")
    
    print("Shape detection completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        print("\nExample usage:")
        print("python shape_detector.py -i shapes.jpg --show --count")
        print("python shape_detector.py -i shapes.jpg --min-area 1000 --max-area 5000 -o detected.jpg")
        print("python shape_detector.py -i shapes.jpg --resize 800 --show")