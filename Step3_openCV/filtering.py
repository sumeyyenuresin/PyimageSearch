"""
Image Filtering
This module demonstrates various image filtering techniques including
blur, noise reduction, edge detection, and morphological operations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_blur(img, kernel_size=(15, 15), sigma_x=0):
    """
    Apply Gaussian blur filter
    """
    if img is None:
        return None
    
    return cv2.GaussianBlur(img, kernel_size, sigma_x)

def median_blur(img, kernel_size=5):
    """
    Apply median blur filter (good for removing salt-and-pepper noise)
    """
    if img is None:
        return None
    
    return cv2.medianBlur(img, kernel_size)

def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter (preserves edges while reducing noise)
    """
    if img is None:
        return None
    
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def sobel_edge_detection(img):
    """
    Apply Sobel edge detection
    """
    if img is None:
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply Sobel operators
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine gradients
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    return sobel_combined.astype(np.uint8)

def canny_edge_detection(img, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection
    """
    if img is None:
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    return cv2.Canny(gray, low_threshold, high_threshold)

def laplacian_edge_detection(img):
    """
    Apply Laplacian edge detection
    """
    if img is None:
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    return np.absolute(laplacian).astype(np.uint8)

def morphological_operations(img, operation='opening', kernel_size=(5, 5)):
    """
    Apply morphological operations
    operations: 'erosion', 'dilation', 'opening', 'closing'
    """
    if img is None:
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    kernel = np.ones(kernel_size, np.uint8)
    
    if operation == 'erosion':
        return cv2.erode(gray, kernel, iterations=1)
    elif operation == 'dilation':
        return cv2.dilate(gray, kernel, iterations=1)
    elif operation == 'opening':
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    else:
        return gray

def sharpen_image(img):
    """
    Sharpen image using kernel convolution
    """
    if img is None:
        return None
    
    # Sharpening kernel
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    
    return cv2.filter2D(img, -1, kernel)

def add_noise(img, noise_type='gaussian'):
    """
    Add noise to image for demonstration purposes
    """
    if img is None:
        return None
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        noisy_img = cv2.add(img, noise)
    elif noise_type == 'salt_pepper':
        noisy_img = img.copy()
        # Salt noise
        coords = [np.random.randint(0, i-1, int(0.05 * img.size)) 
                 for i in img.shape]
        noisy_img[coords[0], coords[1]] = 255
        # Pepper noise
        coords = [np.random.randint(0, i-1, int(0.05 * img.size)) 
                 for i in img.shape]
        noisy_img[coords[0], coords[1]] = 0
    else:
        noisy_img = img
    
    return noisy_img

def main():
    """
    Main function to demonstrate filtering operations
    """
    # Load image
    image_path = "../data/jp.png"
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print("=== Image Filtering Demo ===")
    
    # Add noise for demonstration
    noisy_img = add_noise(img, 'gaussian')
    
    # Apply various filters
    gaussian_filtered = gaussian_blur(noisy_img)
    median_filtered = median_blur(noisy_img)
    bilateral_filtered = bilateral_filter(noisy_img)
    sharpened = sharpen_image(img)
    
    # Edge detection
    sobel_edges = sobel_edge_detection(img)
    canny_edges = canny_edge_detection(img)
    laplacian_edges = laplacian_edge_detection(img)
    
    # Display blur and noise reduction results
    plt.figure(figsize=(16, 12))
    
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
    plt.title('Noisy Image')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Blur')
    plt.axis('off')
    
    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))
    plt.title('Median Filter')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2RGB))
    plt.title('Bilateral Filter')
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.title('Sharpened')
    plt.axis('off')
    
    plt.subplot(3, 3, 7)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title('Sobel Edges')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.imshow(laplacian_edges, cmap='gray')
    plt.title('Laplacian Edges')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate morphological operations
    demo_morphological_operations(img)

def demo_morphological_operations(img):
    """
    Demonstrate morphological operations
    """
    # Convert to binary image for better morphological demonstration
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    erosion = morphological_operations(binary, 'erosion')
    dilation = morphological_operations(binary, 'dilation')
    opening = morphological_operations(binary, 'opening')
    closing = morphological_operations(binary, 'closing')
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(erosion, cmap='gray')
    plt.title('Erosion')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(dilation, cmap='gray')
    plt.title('Dilation')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(opening, cmap='gray')
    plt.title('Opening')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(closing, cmap='gray')
    plt.title('Closing')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()