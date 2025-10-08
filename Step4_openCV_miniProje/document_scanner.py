import cv2
import numpy as np
import argparse
from utils import show_image, save_image, resize_image, order_points, four_point_transform, auto_canny


def preprocess_image(image):
    """
    Preprocess image for document detection.
    
    Args:
        image: Input image
    
    Returns:
        Tuple of (grayscale, blurred, edged) images
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection
    edged = auto_canny(blurred)
    
    return gray, blurred, edged


def find_document_contour(edged):
    """
    Find the contour of the document in the edge-detected image.
    
    Args:
        edged: Edge-detected image
    
    Returns:
        Document contour or None if not found
    """
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Loop over the contours
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # If our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            return approx
    
    return None


def enhance_document(image):
    """
    Enhance the scanned document image.
    
    Args:
        image: Input document image
    
    Returns:
        Enhanced image
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive threshold to get a binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned


def scan_document(image, debug=False):
    """
    Complete document scanning pipeline.
    
    Args:
        image: Input image
        debug: Whether to return debug images
    
    Returns:
        Scanned document or None if scanning failed
    """
    original = image.copy()
    
    # Step 1: Preprocess the image
    gray, blurred, edged = preprocess_image(image)
    
    # Step 2: Find document contour
    doc_contour = find_document_contour(edged)
    
    if doc_contour is None:
        print("Could not find document contour")
        return None
    
    # Step 3: Apply perspective transformation
    warped = four_point_transform(original, doc_contour.reshape(4, 2))
    
    # Step 4: Enhance the document
    enhanced = enhance_document(warped)
    
    if debug:
        return {
            'original': original,
            'gray': gray,
            'blurred': blurred,
            'edged': edged,
            'contour': doc_contour,
            'warped': warped,
            'enhanced': enhanced
        }
    
    return enhanced


def draw_contour_on_image(image, contour):
    """
    Draw the detected document contour on the image.
    
    Args:
        image: Input image
        contour: Document contour
    
    Returns:
        Image with contour drawn
    """
    result = image.copy()
    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
    return result


def auto_scan_document(image_path, output_path=None, show_steps=False):
    """
    Automatically scan a document from an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save scanned document
        show_steps: Whether to show intermediate steps
    
    Returns:
        True if successful, False otherwise
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False
    
    # Resize image for better processing
    original_height = image.shape[0]
    if original_height > 1000:
        image = resize_image(image, height=1000)
        print(f"Resized image for processing")
    
    # Scan the document
    result = scan_document(image, debug=show_steps)
    
    if result is None:
        return False
    
    if show_steps and isinstance(result, dict):
        # Show all intermediate steps
        show_image("1. Original", result['original'], wait_key=False)
        show_image("2. Grayscale", result['gray'], wait_key=False)
        show_image("3. Blurred", result['blurred'], wait_key=False)
        show_image("4. Edges", result['edged'], wait_key=False)
        
        # Show contour detection
        contour_image = draw_contour_on_image(result['original'], result['contour'])
        show_image("5. Document Contour", contour_image, wait_key=False)
        
        show_image("6. Perspective Corrected", result['warped'], wait_key=False)
        show_image("7. Final Scanned Document", result['enhanced'])
        
        scanned = result['enhanced']
    else:
        scanned = result
    
    # Save result
    if output_path:
        if save_image(output_path, scanned):
            print(f"Scanned document saved to {output_path}")
            return True
        else:
            print(f"Error: Could not save image to {output_path}")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Scan and transform documents from images')
    parser.add_argument('-i', '--image', required=True, help='Path to input image')
    parser.add_argument('-o', '--output', default='scanned_document.jpg', help='Output filename')
    parser.add_argument('--show-steps', action='store_true', help='Show intermediate processing steps')
    parser.add_argument('--show', action='store_true', help='Show the final result')
    parser.add_argument('--enhance-only', action='store_true', help='Only enhance without perspective correction')
    
    args = parser.parse_args()
    
    if args.enhance_only:
        # Load and enhance image without perspective correction
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image from {args.image}")
            return
        
        enhanced = enhance_document(image)
        
        if args.show:
            show_image("Original", image, wait_key=False)
            show_image("Enhanced Document", enhanced)
        
        if save_image(args.output, enhanced):
            print(f"Enhanced document saved to {args.output}")
        else:
            print(f"Error: Could not save image to {args.output}")
    else:
        # Full document scanning pipeline
        success = auto_scan_document(args.image, args.output, args.show_steps)
        
        if success and args.show and not args.show_steps:
            # Show just the original and result
            original = cv2.imread(args.image)
            scanned = cv2.imread(args.output)
            
            if original is not None and scanned is not None:
                show_image("Original", original, wait_key=False)
                show_image("Scanned Document", scanned)
        
        if success:
            print("Document scanning completed successfully!")
        else:
            print("Document scanning failed!")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        print("\nExample usage:")
        print("python document_scanner.py -i document.jpg --show-steps")
        print("python document_scanner.py -i document.jpg -o scanned.jpg --show")
        print("python document_scanner.py -i document.jpg --enhance-only")