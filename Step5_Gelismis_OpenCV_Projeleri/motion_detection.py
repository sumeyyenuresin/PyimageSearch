"""
Motion Detection Module

This module implements various motion detection algorithms using OpenCV.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from .utils import load_video, create_video_writer, save_image

class MotionDetector:
    """
    A class for detecting motion in video streams using background subtraction
    """
    
    def __init__(self, method='MOG2'):
        """
        Initialize motion detector
        
        Args:
            method (str): Background subtraction method ('MOG2', 'KNN', 'GMG')
        """
        self.method = method
        self.background_subtractor = self._create_background_subtractor()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def _create_background_subtractor(self):
        """Create background subtractor based on method"""
        if self.method == 'MOG2':
            return cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        elif self.method == 'KNN':
            return cv2.createBackgroundSubtractorKNN(detectShadows=True)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def detect_motion(self, frame):
        """
        Detect motion in a frame
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (motion_mask, motion_areas)
        """
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove shadows (they appear as gray in MOG2)
        fg_mask[fg_mask == 127] = 0
        
        # Morphological operations to clean up the mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 500  # Minimum area threshold
        motion_areas = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, w, h))
        
        return fg_mask, motion_areas
    
    def process_video(self, video_path, output_path=None, show_live=False):
        """
        Process a video file for motion detection
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video (optional)
            show_live (bool): Whether to show live preview
            
        Returns:
            list: List of motion detection results per frame
        """
        cap = load_video(video_path)
        results = []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer if output path is provided
        writer = None
        if output_path:
            writer = create_video_writer(output_path, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect motion
            motion_mask, motion_areas = self.detect_motion(frame)
            
            # Draw bounding boxes around motion areas
            result_frame = frame.copy()
            motion_detected = len(motion_areas) > 0
            
            for (x, y, w, h) in motion_areas:
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_frame, 'Motion', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add status text
            status = "Motion Detected" if motion_detected else "No Motion"
            color = (0, 255, 0) if motion_detected else (0, 0, 255)
            cv2.putText(result_frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time
                print(f"Processed {frame_count}/{total_frames} frames (FPS: {current_fps:.2f})")
            
            # Save frame if writer is available
            if writer:
                writer.write(result_frame)
            
            # Show live preview if requested
            if show_live:
                # Resize for display if too large
                display_frame = result_frame
                if width > 1280 or height > 720:
                    scale = min(1280/width, 720/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_frame = cv2.resize(result_frame, (new_width, new_height))
                
                cv2.imshow('Motion Detection', display_frame)
                cv2.imshow('Motion Mask', motion_mask)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"results/motion/frame_{frame_count:06d}.jpg"
                    save_image(result_frame, save_path)
                    print(f"Saved frame to {save_path}")
            
            # Store results
            results.append({
                'frame_number': frame_count,
                'motion_detected': motion_detected,
                'motion_areas': motion_areas,
                'motion_count': len(motion_areas)
            })
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_live:
            cv2.destroyAllWindows()
        
        print(f"Processing complete! Total frames: {frame_count}")
        return results

class FrameDifferenceDetector:
    """
    Motion detection using frame difference method
    """
    
    def __init__(self, threshold=25, min_area=500):
        """
        Initialize frame difference detector
        
        Args:
            threshold (int): Difference threshold
            min_area (int): Minimum contour area
        """
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None
        
    def detect_motion(self, frame):
        """
        Detect motion using frame difference
        
        Args:
            frame (np.ndarray): Current frame
            
        Returns:
            tuple: (motion_mask, motion_areas)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray), []
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        motion_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, w, h))
        
        # Update previous frame
        self.prev_frame = gray
        
        return thresh, motion_areas

def main():
    """
    Main function for testing motion detection
    """
    # Example usage
    detector = MotionDetector(method='MOG2')
    
    # Test with webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit, 's' to save current frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect motion
        motion_mask, motion_areas = detector.detect_motion(frame)
        
        # Draw results
        result_frame = frame.copy()
        for (x, y, w, h) in motion_areas:
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Motion Detection', result_frame)
        cv2.imshow('Motion Mask', motion_mask)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            save_image(result_frame, f"results/motion/webcam_{timestamp}.jpg")
            print(f"Saved frame: webcam_{timestamp}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()