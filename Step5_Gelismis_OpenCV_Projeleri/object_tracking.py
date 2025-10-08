"""
Object Tracking Module

This module implements various object tracking algorithms using OpenCV.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from .utils import load_video, create_video_writer, save_image

class ObjectTracker:
    """
    A class for tracking objects in video streams using various tracking algorithms
    """
    
    def __init__(self, tracker_type='CSRT'):
        """
        Initialize object tracker
        
        Args:
            tracker_type (str): Type of tracker ('CSRT', 'KCF', 'MOSSE', 'MIL', 'BOOSTING', 'TLD', 'MEDIANFLOW')
        """
        self.tracker_type = tracker_type
        self.tracker = None
        self.initialized = False
        
    def _create_tracker(self):
        """Create tracker based on type"""
        trackers = {
            'CSRT': cv2.TrackerCSRT_create,
            'KCF': cv2.TrackerKCF_create,
            'MOSSE': cv2.TrackerMOSSE_create,
            'MIL': cv2.TrackerMIL_create,
            'BOOSTING': cv2.TrackerBoosting_create,
            'TLD': cv2.TrackerTLD_create,
            'MEDIANFLOW': cv2.TrackerMedianFlow_create
        }
        
        if self.tracker_type not in trackers:
            raise ValueError(f"Unknown tracker type: {self.tracker_type}")
        
        return trackers[self.tracker_type]()
    
    def initialize(self, frame, bbox):
        """
        Initialize tracker with first frame and bounding box
        
        Args:
            frame (np.ndarray): First frame
            bbox (tuple): Initial bounding box (x, y, w, h)
            
        Returns:
            bool: True if initialization successful
        """
        self.tracker = self._create_tracker()
        self.initialized = self.tracker.init(frame, bbox)
        return self.initialized
    
    def update(self, frame):
        """
        Update tracker with new frame
        
        Args:
            frame (np.ndarray): New frame
            
        Returns:
            tuple: (success, bbox) where bbox is (x, y, w, h)
        """
        if not self.initialized:
            return False, None
        
        success, bbox = self.tracker.update(frame)
        return success, bbox
    
    def track_video(self, video_path, initial_bbox=None, output_path=None, show_live=True):
        """
        Track object in video
        
        Args:
            video_path (str): Path to input video
            initial_bbox (tuple): Initial bounding box (x, y, w, h). If None, user will select
            output_path (str): Path to save output video (optional)
            show_live (bool): Whether to show live preview
            
        Returns:
            list: List of tracking results per frame
        """
        cap = load_video(video_path)
        results = []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return results
        
        # Get initial bounding box
        if initial_bbox is None:
            print("Select object to track. Press ENTER after selection.")
            bbox = cv2.selectROI("Select Object", frame, False)
            cv2.destroyWindow("Select Object")
            
            if bbox[2] == 0 or bbox[3] == 0:
                print("No object selected")
                return results
        else:
            bbox = initial_bbox
        
        # Initialize tracker
        if not self.initialize(frame, bbox):
            print("Failed to initialize tracker")
            return results
        
        print(f"Tracking initialized with {self.tracker_type} tracker")
        
        # Create video writer if output path is provided
        writer = None
        if output_path:
            writer = create_video_writer(output_path, fps, (width, height))
        
        frame_count = 1
        tracking_success_count = 0
        start_time = time.time()
        
        # Process first frame
        result_frame = self._draw_tracking_result(frame, True, bbox, frame_count)
        if writer:
            writer.write(result_frame)
        
        results.append({
            'frame_number': frame_count,
            'success': True,
            'bbox': bbox,
            'center': (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
        })
        
        tracking_success_count += 1
        
        if show_live:
            cv2.imshow('Object Tracking', result_frame)
        
        print(f"Processing {total_frames} frames...")
        
        # Process remaining frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update tracker
            success, bbox = self.update(frame)
            
            # Draw result
            result_frame = self._draw_tracking_result(frame, success, bbox, frame_count)
            
            if success:
                tracking_success_count += 1
                center = (int(bbox[0] + bbox[2]//2), int(bbox[1] + bbox[3]//2))
            else:
                center = None
            
            # Save frame if writer is available
            if writer:
                writer.write(result_frame)
            
            # Show live preview if requested
            if show_live:
                cv2.imshow('Object Tracking', result_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"results/tracking/frame_{frame_count:06d}.jpg"
                    save_image(result_frame, save_path)
                    print(f"Saved frame to {save_path}")
                elif key == ord('r'):
                    # Reinitialize tracker
                    print("Select new object to track")
                    new_bbox = cv2.selectROI("Select Object", frame, False)
                    if new_bbox[2] > 0 and new_bbox[3] > 0:
                        if self.initialize(frame, new_bbox):
                            print("Tracker reinitialized")
                            bbox = new_bbox
                            success = True
            
            # Store results
            results.append({
                'frame_number': frame_count,
                'success': success,
                'bbox': bbox if success else None,
                'center': center
            })
            
            # Progress update
            if frame_count % 50 == 0:
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time
                success_rate = tracking_success_count / frame_count * 100
                print(f"Processed {frame_count}/{total_frames} frames "
                      f"(FPS: {current_fps:.2f}, Success: {success_rate:.1f}%)")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_live:
            cv2.destroyAllWindows()
        
        final_success_rate = tracking_success_count / frame_count * 100
        print(f"Tracking complete! Success rate: {final_success_rate:.1f}%")
        
        return results
    
    def _draw_tracking_result(self, frame, success, bbox, frame_number):
        """
        Draw tracking result on frame
        
        Args:
            frame (np.ndarray): Input frame
            success (bool): Whether tracking was successful
            bbox (tuple): Bounding box (x, y, w, h)
            frame_number (int): Current frame number
            
        Returns:
            np.ndarray: Frame with tracking visualization
        """
        result_frame = frame.copy()
        
        if success and bbox is not None:
            # Draw bounding box
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point
            center_x, center_y = x + w//2, y + h//2
            cv2.circle(result_frame, (center_x, center_y), 3, (0, 255, 0), -1)
            
            # Add tracker info
            info_text = f"{self.tracker_type} - TRACKING"
            cv2.putText(result_frame, info_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add coordinates
            coord_text = f"({center_x}, {center_y})"
            cv2.putText(result_frame, coord_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Tracking failed
            cv2.putText(result_frame, f"{self.tracker_type} - LOST", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add frame number
        cv2.putText(result_frame, f"Frame: {frame_number}", (10, result_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame

class MultiObjectTracker:
    """
    Track multiple objects simultaneously
    """
    
    def __init__(self, tracker_type='CSRT'):
        """
        Initialize multi-object tracker
        
        Args:
            tracker_type (str): Type of tracker to use for each object
        """
        self.tracker_type = tracker_type
        self.trackers = []
        self.colors = []
        
    def add_object(self, frame, bbox):
        """
        Add new object to track
        
        Args:
            frame (np.ndarray): Current frame
            bbox (tuple): Bounding box (x, y, w, h)
            
        Returns:
            int: Object ID
        """
        tracker = ObjectTracker(self.tracker_type)
        if tracker.initialize(frame, bbox):
            self.trackers.append(tracker)
            # Generate random color for this object
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            self.colors.append(color)
            return len(self.trackers) - 1
        return -1
    
    def update_all(self, frame):
        """
        Update all trackers
        
        Args:
            frame (np.ndarray): Current frame
            
        Returns:
            list: List of (success, bbox) for each tracker
        """
        results = []
        for tracker in self.trackers:
            success, bbox = tracker.update(frame)
            results.append((success, bbox))
        return results
    
    def draw_all(self, frame, results):
        """
        Draw all tracking results
        
        Args:
            frame (np.ndarray): Input frame
            results (list): List of (success, bbox) for each tracker
            
        Returns:
            np.ndarray: Frame with all tracking visualizations
        """
        result_frame = frame.copy()
        
        for i, (success, bbox) in enumerate(results):
            color = self.colors[i]
            
            if success and bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
                
                # Add object ID
                cv2.putText(result_frame, f"ID: {i}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_frame

def main():
    """
    Main function for testing object tracking
    """
    # Example usage with webcam
    tracker = ObjectTracker('CSRT')
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 's' to select object, 'q' to quit")
    
    initialized = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if initialized:
            # Update tracker
            success, bbox = tracker.update(frame)
            result_frame = tracker._draw_tracking_result(frame, success, bbox, 0)
        else:
            result_frame = frame.copy()
            cv2.putText(result_frame, "Press 's' to select object", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not initialized:
            # Select object
            bbox = cv2.selectROI("Select Object", frame, False)
            if bbox[2] > 0 and bbox[3] > 0:
                if tracker.initialize(frame, bbox):
                    initialized = True
                    print("Tracking initialized")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()