import cv2
import numpy as np
import argparse
from utils import show_image, save_image


def rotate_image(image, angle):
    """
    Rotate an image by a given angle.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
    
    Returns:
        Rotated image
    """
    # Get image dimensions
    (h, w) = image.shape[:2]
    
    # Calculate the center of the image
    center = (w // 2, h // 2)
    
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to avoid cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                            flags=cv2.INTER_CUBIC, 
                            borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def rotate_bound(image, angle):
    """
    Rotate an image while keeping all pixels within bounds.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
    
    Returns:
        Rotated image with all pixels visible
    """
    # Get image dimensions
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # Compute new bounding dimensions
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    # Perform rotation
    return cv2.warpAffine(image, M, (nW, nH))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Rotate an image')
    parser.add_argument('-i', '--image', required=True, help='Path to input image')
    parser.add_argument('-a', '--angle', type=float, default=45, help='Rotation angle in degrees')
    parser.add_argument('-o', '--output', help='Path to output image')
    parser.add_argument('--show', action='store_true', help='Show the rotated image')
    
    args = parser.parse_args()
    
    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    print(f"Original image shape: {image.shape}")
    
    # Rotate the image
    rotated = rotate_bound(image, args.angle)
    print(f"Rotated image shape: {rotated.shape}")
    
    # Show the images if requested
    if args.show:
        show_image("Original", image, wait_key=False)
        show_image(f"Rotated by {args.angle} degrees", rotated)
    
    # Save the rotated image
    if args.output:
        if save_image(args.output, rotated):
            print(f"Rotated image saved to {args.output}")
        else:
            print(f"Error: Could not save image to {args.output}")
    
    print("Image rotation completed successfully!")


if __name__ == "__main__":
    # Example usage without command line arguments
    try:
        main()
    except SystemExit:
        # If no arguments provided, show example
        print("\nExample usage:")
        print("python rotate_image.py -i input.jpg -a 45 -o rotated.jpg --show")
        print("\nThis will rotate input.jpg by 45 degrees, save as rotated.jpg, and display the result.")