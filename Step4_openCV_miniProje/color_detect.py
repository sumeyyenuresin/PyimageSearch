import cv2
import numpy as np
import argparse
from utils import show_image, save_image


def create_color_mask(image, lower_bound, upper_bound):
    """
    Create a mask for a specific color range.
    
    Args:
        image: Input image in BGR format
        lower_bound: Lower HSV bound for color
        upper_bound: Upper HSV bound for color
    
    Returns:
        Binary mask
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    return mask


def detect_color(image, color_name="blue"):
    """
    Detect and isolate a specific color in an image.
    
    Args:
        image: Input image
        color_name: Name of color to detect
    
    Returns:
        Tuple of (mask, result_image)
    """
    # Define color ranges in HSV
    color_ranges = {
        "red1": (np.array([0, 50, 50]), np.array([10, 255, 255])),
        "red2": (np.array([170, 50, 50]), np.array([180, 255, 255])),
        "blue": (np.array([100, 50, 50]), np.array([130, 255, 255])),
        "green": (np.array([40, 50, 50]), np.array([80, 255, 255])),
        "yellow": (np.array([20, 50, 50]), np.array([30, 255, 255])),
        "orange": (np.array([10, 50, 50]), np.array([20, 255, 255])),
        "purple": (np.array([130, 50, 50]), np.array([170, 255, 255])),
        "cyan": (np.array([80, 50, 50]), np.array([100, 255, 255]))
    }
    
    if color_name == "red":
        # Red color wraps around in HSV, so we need two ranges
        mask1 = create_color_mask(image, *color_ranges["red1"])
        mask2 = create_color_mask(image, *color_ranges["red2"])
        mask = cv2.bitwise_or(mask1, mask2)
    elif color_name in color_ranges:
        lower, upper = color_ranges[color_name]
        mask = create_color_mask(image, lower, upper)
    else:
        raise ValueError(f"Color '{color_name}' not supported. Available colors: {list(color_ranges.keys()) + ['red']}")
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return mask, result


def find_color_contours(image, color_name="blue", min_area=500):
    """
    Find contours of objects with a specific color.
    
    Args:
        image: Input image
        color_name: Color to detect
        min_area: Minimum contour area to consider
    
    Returns:
        List of contours and annotated image
    """
    mask, _ = detect_color(image, color_name)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Draw contours on the original image
    result = image.copy()
    for i, contour in enumerate(filtered_contours):
        # Draw contour
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
        
        # Add label
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(result, f"{color_name.capitalize()} {i+1}", 
                       (cX - 30, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return filtered_contours, result


def main():
    parser = argparse.ArgumentParser(description='Detect and isolate colors in an image')
    parser.add_argument('-i', '--image', required=True, help='Path to input image')
    parser.add_argument('-c', '--color', default='blue', 
                       help='Color to detect (red, blue, green, yellow, orange, purple, cyan)')
    parser.add_argument('-o', '--output', help='Path to output image')
    parser.add_argument('--show', action='store_true', help='Show the detection results')
    parser.add_argument('--contours', action='store_true', help='Find and draw contours')
    parser.add_argument('--min-area', type=int, default=500, help='Minimum contour area')
    
    args = parser.parse_args()
    
    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    print(f"Detecting {args.color} color in the image...")
    
    try:
        if args.contours:
            # Find contours of the specified color
            contours, result = find_color_contours(image, args.color, args.min_area)
            print(f"Found {len(contours)} {args.color} objects")
        else:
            # Simple color detection
            mask, result = detect_color(image, args.color)
        
        # Show results if requested
        if args.show:
            show_image("Original", image, wait_key=False)
            if not args.contours:
                show_image(f"Mask for {args.color}", mask, wait_key=False)
            show_image(f"{args.color.capitalize()} Detection Result", result)
        
        # Save result
        if args.output:
            if save_image(args.output, result):
                print(f"Result saved to {args.output}")
            else:
                print(f"Error: Could not save image to {args.output}")
        
        print("Color detection completed successfully!")
        
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        print("\nExample usage:")
        print("python color_detect.py -i input.jpg -c blue --show")
        print("python color_detect.py -i input.jpg -c red --contours --min-area 1000 -o output.jpg")
        print("\nAvailable colors: red, blue, green, yellow, orange, purple, cyan")