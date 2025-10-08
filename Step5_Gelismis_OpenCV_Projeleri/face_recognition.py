"""
Face Recognition Module

This module implements face detection and recognition using various algorithms.
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path
import face_recognition as fr
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .utils import load_image, save_image

class FaceDetector:
    """
    Face detection using OpenCV's Haar Cascades and HOG
    """
    
    def __init__(self, method='haar', cascade_path=None):
        """
        Initialize face detector
        
        Args:
            method (str): Detection method ('haar', 'dnn', 'hog')
            cascade_path (str): Path to Haar cascade file
        """
        self.method = method
        
        if method == 'haar':
            if cascade_path is None:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
        elif method == 'dnn':
            # Load DNN model for face detection
            self.net = cv2.dnn.readNetFromTensorflow(
                'models/opencv_face_detector_uint8.pb',
                'models/opencv_face_detector.pbtxt'
            )
    
    def detect_faces(self, image, min_confidence=0.5):
        """
        Detect faces in an image
        
        Args:
            image (np.ndarray): Input image
            min_confidence (float): Minimum confidence threshold for DNN
            
        Returns:
            list: List of face bounding boxes (x, y, w, h)
        """
        if self.method == 'haar':
            return self._detect_haar(image)
        elif self.method == 'dnn':
            return self._detect_dnn(image, min_confidence)
        elif self.method == 'hog':
            return self._detect_hog(image)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def _detect_haar(self, image):
        """Detect faces using Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces.tolist()
    
    def _detect_dnn(self, image, min_confidence):
        """Detect faces using DNN"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append([x1, y1, x2 - x1, y2 - y1])
        
        return faces
    
    def _detect_hog(self, image):
        """Detect faces using HOG (face_recognition library)"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = fr.face_locations(rgb_image, model='hog')
        
        faces = []
        for (top, right, bottom, left) in face_locations:
            faces.append([left, top, right - left, bottom - top])
        
        return faces

class FaceRecognizer:
    """
    Face recognition using face_recognition library and SVM classifier
    """
    
    def __init__(self):
        """Initialize face recognizer"""
        self.known_encodings = []
        self.known_names = []
        self.classifier = None
        self.detector = FaceDetector(method='hog')
    
    def add_person(self, image_path, person_name):
        """
        Add a person to the recognition database
        
        Args:
            image_path (str): Path to person's image
            person_name (str): Name of the person
            
        Returns:
            bool: True if person added successfully
        """
        try:
            image = load_image(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get face encodings
            encodings = fr.face_encodings(rgb_image)
            
            if len(encodings) == 0:
                print(f"No face found in {image_path}")
                return False
            
            # Use the first face found
            encoding = encodings[0]
            self.known_encodings.append(encoding)
            self.known_names.append(person_name)
            
            print(f"Added {person_name} to database")
            return True
            
        except Exception as e:
            print(f"Error adding person {person_name}: {e}")
            return False
    
    def load_dataset(self, dataset_path):
        """
        Load dataset from directory structure
        Expected structure:
        dataset_path/
        ├── person1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── person2/
            ├── image1.jpg
            └── image2.jpg
        
        Args:
            dataset_path (str): Path to dataset directory
            
        Returns:
            int: Number of images loaded
        """
        dataset_path = Path(dataset_path)
        loaded_count = 0
        
        if not dataset_path.exists():
            print(f"Dataset path does not exist: {dataset_path}")
            return 0
        
        print(f"Loading dataset from {dataset_path}")
        
        for person_dir in dataset_path.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            print(f"Processing {person_name}...")
            
            for image_file in person_dir.glob('*.jpg'):
                if self.add_person(str(image_file), person_name):
                    loaded_count += 1
        
        print(f"Loaded {loaded_count} images for {len(set(self.known_names))} people")
        return loaded_count
    
    def train_classifier(self):
        """
        Train SVM classifier on known encodings
        
        Returns:
            float: Training accuracy
        """
        if len(self.known_encodings) == 0:
            print("No training data available")
            return 0.0
        
        X = np.array(self.known_encodings)
        y = np.array(self.known_names)
        
        # Split data for training and testing
        if len(X) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test = X, X
            y_train, y_test = y, y
        
        # Train SVM classifier
        self.classifier = SVC(kernel='linear', probability=True)
        self.classifier.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Classifier trained with accuracy: {accuracy:.2f}")
        return accuracy
    
    def recognize_faces(self, image, confidence_threshold=0.6):
        """
        Recognize faces in an image
        
        Args:
            image (np.ndarray): Input image
            confidence_threshold (float): Minimum confidence for recognition
            
        Returns:
            list: List of (bbox, name, confidence) tuples
        """
        if self.classifier is None:
            print("Classifier not trained. Call train_classifier() first.")
            return []
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = fr.face_locations(rgb_image, model='hog')
        face_encodings = fr.face_encodings(rgb_image, face_locations)
        
        results = []
        
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Predict with classifier
            probabilities = self.classifier.predict_proba([encoding])[0]
            max_prob_index = np.argmax(probabilities)
            confidence = probabilities[max_prob_index]
            
            if confidence >= confidence_threshold:
                name = self.classifier.classes_[max_prob_index]
            else:
                name = "Unknown"
            
            bbox = (left, top, right - left, bottom - top)
            results.append((bbox, name, confidence))
        
        return results
    
    def save_model(self, model_path):
        """
        Save trained model to file
        
        Args:
            model_path (str): Path to save model
        """
        model_data = {
            'encodings': self.known_encodings,
            'names': self.known_names,
            'classifier': self.classifier
        }
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load trained model from file
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.known_encodings = model_data['encodings']
            self.known_names = model_data['names']
            self.classifier = model_data['classifier']
            
            print(f"Model loaded from {model_path}")
            print(f"Loaded {len(self.known_encodings)} encodings for {len(set(self.known_names))} people")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def process_video(self, video_path, output_path=None, show_live=True):
        """
        Process video for face recognition
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video (optional)
            show_live (bool): Whether to show live preview
            
        Returns:
            list: List of recognition results per frame
        """
        from .utils import load_video, create_video_writer
        
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
        process_every_n_frames = 3  # Process every 3rd frame for speed
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every nth frame for speed
            if frame_count % process_every_n_frames == 0:
                recognition_results = self.recognize_faces(frame)
            else:
                recognition_results = []
            
            # Draw results
            result_frame = self._draw_recognition_results(frame, recognition_results)
            
            # Save frame if writer is available
            if writer:
                writer.write(result_frame)
            
            # Show live preview if requested
            if show_live:
                cv2.imshow('Face Recognition', result_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"results/faces/frame_{frame_count:06d}.jpg"
                    save_image(result_frame, save_path)
                    print(f"Saved frame to {save_path}")
            
            # Store results
            results.append({
                'frame_number': frame_count,
                'faces': recognition_results
            })
            
            # Progress update
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_live:
            cv2.destroyAllWindows()
        
        print(f"Processing complete! Total frames: {frame_count}")
        return results
    
    def _draw_recognition_results(self, frame, results):
        """
        Draw recognition results on frame
        
        Args:
            frame (np.ndarray): Input frame
            results (list): List of (bbox, name, confidence) tuples
            
        Returns:
            np.ndarray: Frame with recognition visualization
        """
        result_frame = frame.copy()
        
        for bbox, name, confidence in results:
            x, y, w, h = bbox
            
            # Choose color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw name and confidence
            label = f"{name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(result_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Text
            cv2.putText(result_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_frame

def main():
    """
    Main function for testing face recognition
    """
    recognizer = FaceRecognizer()
    
    # Example: Load dataset and train
    # recognizer.load_dataset('data/faces')
    # recognizer.train_classifier()
    # recognizer.save_model('models/face_recognition_model.pkl')
    
    # Example: Load existing model
    # recognizer.load_model('models/face_recognition_model.pkl')
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Recognize faces
        results = recognizer.recognize_faces(frame)
        result_frame = recognizer._draw_recognition_results(frame, results)
        
        cv2.imshow('Face Recognition', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()