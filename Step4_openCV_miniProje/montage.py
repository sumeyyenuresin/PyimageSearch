import cv2
import numpy as np
import argparse
import os
import glob
from utils import resize_image, show_image, save_image


def create_montage(images, grid_size=(3, 3), image_size=(200, 200), spacing=10):
    """
    Create a montage from a list of images.
    
    Args:
        images: List of images
        grid_size: Tuple of (rows, cols) for the grid
        image_size: Size to resize each image to
        spacing: Spacing between images
    
    Returns:
        Montage image
    """
    rows, cols = grid_size
    img_height, img_width = image_size
    
    # Calculate montage dimensions
    montage_width = cols * img_width + (cols - 1) * spacing
    montage_height = rows * img_height + (rows - 1) * spacing
    
    # Create blank montage
    montage = np.ones((montage_height, montage_width, 3), dtype=np.uint8) * 255
    
    # Place images in the montage
    for i, image in enumerate(images[:rows * cols]):
        if image is None:
            continue
            
        # Calculate position
        row = i // cols
        col = i % cols
        
        x = col * (img_width + spacing)
        y = row * (img_height + spacing)
        
        # Resize image
        if len(image.shape) == 3:
            resized = cv2.resize(image, (img_width, img_height))
        else:
            # Convert grayscale to BGR
            gray_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            resized = cv2.resize(gray_bgr, (img_width, img_height))
        
        # Place in montage
        montage[y:y+img_height, x:x+img_width] = resized
    
    return montage


def load_images_from_folder(folder_path, max_images=None, extensions=('*.jpg', '*.jpeg', '*.png', '*.bmp')):
    """
    Load images from a folder.
    
    Args:
        folder_path: Path to the folder containing images
        max_images: Maximum number of images to load
        extensions: Tuple of file extensions to look for
    
    Returns:
        List of loaded images
    """
    images = []
    image_files = []
    
    # Collect all image files
    for extension in extensions:
        pattern = os.path.join(folder_path, extension)
        image_files.extend(glob.glob(pattern, recursive=False))
    
    # Sort files for consistent ordering
    image_files.sort()
    
    # Limit number of images if specified
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} image files")
    
    # Load images
    for i, file_path in enumerate(image_files):
        print(f"Loading image {i+1}/{len(image_files)}: {os.path.basename(file_path)}")
        image = cv2.imread(file_path)
        if image is not None:
            images.append(image)
        else:
            print(f"Warning: Could not load {file_path}")
    
    return images


def create_adaptive_montage(images, max_cols=5, image_size=(200, 200), spacing=10):
    """
    Create a montage with adaptive grid size based on number of images.
    
    Args:
        images: List of images
        max_cols: Maximum number of columns
        image_size: Size to resize each image to
        spacing: Spacing between images
    
    Returns:
        Montage image
    """
    num_images = len(images)
    
    if num_images == 0:
        return None
    
    # Calculate optimal grid size
    cols = min(num_images, max_cols)
    rows = int(np.ceil(num_images / cols))
    
    print(f"Creating montage with {rows} rows and {cols} columns for {num_images} images")
    
    return create_montage(images, grid_size=(rows, cols), image_size=image_size, spacing=spacing)


def add_labels_to_montage(montage, labels, grid_size, image_size, spacing=10):
    """
    Add labels to a montage.
    
    Args:
        montage: Montage image
        labels: List of labels
        grid_size: Grid size used for montage
        image_size: Size of each image
        spacing: Spacing between images
    
    Returns:
        Montage with labels
    """
    rows, cols = grid_size
    img_height, img_width = image_size
    
    labeled_montage = montage.copy()
    
    for i, label in enumerate(labels[:rows * cols]):
        row = i // cols
        col = i % cols
        
        x = col * (img_width + spacing)
        y = row * (img_height + spacing)
        
        # Add label at the bottom of each image
        cv2.putText(labeled_montage, str(label), 
                   (x + 5, y + img_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return labeled_montage


def main():
    parser = argparse.ArgumentParser(description='Create an image montage')
    parser.add_argument('-f', '--folder', required=True, help='Path to folder containing images')
    parser.add_argument('-o', '--output', default='montage.jpg', help='Output montage filename')
    parser.add_argument('-s', '--size', nargs=2, type=int, default=[200, 200], 
                       help='Size of each image in montage (width height)')
    parser.add_argument('-g', '--grid', nargs=2, type=int, 
                       help='Grid size (rows cols). If not specified, will be calculated automatically')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to include')
    parser.add_argument('--spacing', type=int, default=10, help='Spacing between images')
    parser.add_argument('--show', action='store_true', help='Show the montage')
    parser.add_argument('--labels', action='store_true', help='Add filename labels to images')
    
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.isdir(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist")
        return
    
    # Load images
    images = load_images_from_folder(args.folder, args.max_images)
    
    if not images:
        print("No images found in the specified folder")
        return
    
    # Create montage
    image_size = tuple(args.size)
    
    if args.grid:
        montage = create_montage(images, tuple(args.grid), image_size, args.spacing)
    else:
        montage = create_adaptive_montage(images, image_size=image_size, spacing=args.spacing)
    
    if montage is None:
        print("Failed to create montage")
        return
    
    # Add labels if requested
    if args.labels:
        # Get filenames as labels
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            pattern = os.path.join(args.folder, ext)
            image_files.extend(glob.glob(pattern))
        
        image_files.sort()
        if args.max_images:
            image_files = image_files[:args.max_images]
        
        labels = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
        
        if args.grid:
            grid_size = tuple(args.grid)
        else:
            cols = min(len(images), 5)
            rows = int(np.ceil(len(images) / cols))
            grid_size = (rows, cols)
        
        montage = add_labels_to_montage(montage, labels, grid_size, image_size, args.spacing)
    
    # Show montage if requested
    if args.show:
        show_image("Montage", montage)
    
    # Save montage
    if save_image(args.output, montage):
        print(f"Montage saved to {args.output}")
        print(f"Montage size: {montage.shape[1]}x{montage.shape[0]} pixels")
    else:
        print(f"Error: Could not save montage to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        print("\nExample usage:")
        print("python montage.py -f ./images --show")
        print("python montage.py -f ./images -s 150 150 -g 2 3 --labels -o my_montage.jpg")
        print("python montage.py -f ./images --max-images 12 --spacing 15")