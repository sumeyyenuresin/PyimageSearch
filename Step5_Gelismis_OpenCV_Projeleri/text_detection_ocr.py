"""
Text Detection and OCR Module

This module implements text detection and optical character recognition using various methods.
"""

import cv2
import numpy as np
import pytesseract
import easyocr
from pathlib import Path
import re
from .utils import load_image, save_image

class TextDetector:
    """
    Text detection using OpenCV's EAST detector and other methods
    """
    
    def __init__(self, method='east', east_model_path=None):
        """
        Initialize text detector
        
        Args:
            method (str): Detection method ('east', 'contour', 'mser')
            east_model_path (str): Path to EAST model file
        """
        self.method = method
        
        if method == 'east' and east_model_path:
            self.net = cv2.dnn.readNet(east_model_path)
    
    def detect_text(self, image, min_confidence=0.5):
        """
        Detect text regions in an image
        
        Args:
            image (np.ndarray): Input image
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            list: List of text bounding boxes
        """
        if self.method == 'east':
            return self._detect_east(image, min_confidence)
        elif self.method == 'contour':
            return self._detect_contour(image)
        elif self.method == 'mser':
            return self._detect_mser(image)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def _detect_east(self, image, min_confidence):
        """Detect text using EAST detector"""
        if not hasattr(self, 'net'):
            raise ValueError("EAST model not loaded. Provide east_model_path.")
        
        orig = image.copy()
        (H, W) = image.shape[:2]
        
        # Set the new width and height and determine the ratio
        (newW, newH) = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)
        
        # Resize the image and grab dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]
        
        # Define output layer names for EAST detector
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        
        # Construct blob and perform forward pass
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(layerNames)
        
        # Decode predictions
        boxes = self._decode_predictions(scores, geometry, min_confidence)
        
        # Scale bounding boxes back to original image size
        scaled_boxes = []
        for (x, y, w, h) in boxes:
            x = int(x * rW)
            y = int(y * rH)
            w = int(w * rW)
            h = int(h * rH)
            scaled_boxes.append((x, y, w, h))
        
        return scaled_boxes
    
    def _decode_predictions(self, scores, geometry, min_confidence):
        """Decode EAST predictions"""
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        
        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            
            for x in range(0, numCols):
                if scoresData[x] < min_confidence:
                    continue
                
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                
                rects.append((startX, startY, endX - startX, endY - startY))
                confidences.append(scoresData[x])
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(rects, confidences, min_confidence, 0.4)
        
        boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                boxes.append(rects[i])
        
        return boxes
    
    def _detect_contour(self, image):
        """Detect text using contour analysis"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 10:  # Filter by aspect ratio
                    text_boxes.append((x, y, w, h))
        
        return text_boxes
    
    def _detect_mser(self, image):
        """Detect text using MSER (Maximally Stable Extremal Regions)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create MSER object
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        text_boxes = []
        for region in regions:
            if len(region) > 10:  # Filter small regions
                x, y, w, h = cv2.boundingRect(region)
                aspect_ratio = w / h
                if 0.1 < aspect_ratio < 10 and w > 10 and h > 10:
                    text_boxes.append((x, y, w, h))
        
        return text_boxes

class OCREngine:
    """
    Optical Character Recognition using Tesseract and EasyOCR
    """
    
    def __init__(self, engine='tesseract', languages=['en']):
        """
        Initialize OCR engine
        
        Args:
            engine (str): OCR engine ('tesseract', 'easyocr')
            languages (list): List of languages to recognize
        """
        self.engine = engine
        self.languages = languages
        
        if engine == 'easyocr':
            self.reader = easyocr.Reader(languages)
    
    def extract_text(self, image, bbox=None, preprocess=True):
        """
        Extract text from image or image region
        
        Args:
            image (np.ndarray): Input image
            bbox (tuple): Bounding box (x, y, w, h) for region extraction
            preprocess (bool): Whether to preprocess image
            
        Returns:
            str: Extracted text
        """
        # Extract region if bbox is provided
        if bbox is not None:
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
        
        # Preprocess image
        if preprocess:
            roi = self._preprocess_for_ocr(roi)
        
        # Extract text based on engine
        if self.engine == 'tesseract':
            return self._extract_tesseract(roi)
        elif self.engine == 'easyocr':
            return self._extract_easyocr(roi)
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine}")
    
    def _preprocess_for_ocr(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize if too small
        height, width = gray.shape
        if height < 30 or width < 30:
            scale = max(30/height, 30/width, 2.0)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh
    
    def _extract_tesseract(self, image):
        """Extract text using Tesseract"""
        try:
            # Configure Tesseract
            config = r'--oem 3 --psm 6'  # Use LSTM OCR Engine Mode with uniform text block
            text = pytesseract.image_to_string(image, config=config, lang='+'.join(self.languages))
            return text.strip()
        except Exception as e:
            print(f"Tesseract error: {e}")
            return ""
    
    def _extract_easyocr(self, image):
        """Extract text using EasyOCR"""
        try:
            results = self.reader.readtext(image)
            text_parts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter by confidence
                    text_parts.append(text)
            return ' '.join(text_parts)
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return ""
    
    def get_detailed_results(self, image, bbox=None):
        """
        Get detailed OCR results with bounding boxes and confidence
        
        Args:
            image (np.ndarray): Input image
            bbox (tuple): Bounding box for region extraction
            
        Returns:
            list: List of (text, bbox, confidence) tuples
        """
        # Extract region if bbox is provided
        if bbox is not None:
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]
            offset = (x, y)
        else:
            roi = image
            offset = (0, 0)
        
        results = []
        
        if self.engine == 'tesseract':
            # Get detailed Tesseract results
            data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and int(data['conf'][i]) > 30:  # Filter by confidence
                    x = data['left'][i] + offset[0]
                    y = data['top'][i] + offset[1]
                    w = data['width'][i]
                    h = data['height'][i]
                    confidence = int(data['conf'][i]) / 100.0
                    results.append((text, (x, y, w, h), confidence))
        
        elif self.engine == 'easyocr':
            # Get EasyOCR results
            easyocr_results = self.reader.readtext(roi)
            for (bbox_coords, text, confidence) in easyocr_results:
                if confidence > 0.5:
                    # Convert EasyOCR bbox format to (x, y, w, h)
                    x_coords = [point[0] for point in bbox_coords]
                    y_coords = [point[1] for point in bbox_coords]
                    x = int(min(x_coords)) + offset[0]
                    y = int(min(y_coords)) + offset[1]
                    w = int(max(x_coords) - min(x_coords))
                    h = int(max(y_coords) - min(y_coords))
                    results.append((text, (x, y, w, h), confidence))
        
        return results

class TextRecognitionPipeline:
    """
    Complete text detection and recognition pipeline
    """
    
    def __init__(self, detector_method='contour', ocr_engine='tesseract', languages=['en']):
        """
        Initialize text recognition pipeline
        
        Args:
            detector_method (str): Text detection method
            ocr_engine (str): OCR engine to use
            languages (list): Languages for OCR
        """
        self.detector = TextDetector(method=detector_method)
        self.ocr = OCREngine(engine=ocr_engine, languages=languages)
    
    def process_image(self, image_path, output_path=None):
        """
        Process image for text detection and recognition
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save result image
            
        Returns:
            dict: Processing results
        """
        # Load image
        image = load_image(image_path)
        
        # Detect text regions
        text_boxes = self.detector.detect_text(image)
        
        # Extract text from each region
        results = []
        result_image = image.copy()
        
        for i, bbox in enumerate(text_boxes):
            # Extract text
            text = self.ocr.extract_text(image, bbox)
            
            if text.strip():  # Only process non-empty text
                x, y, w, h = bbox
                
                # Draw bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add text label
                label = f"{i+1}: {text[:20]}..."  # Show first 20 characters
                cv2.putText(result_image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                results.append({
                    'region_id': i + 1,
                    'bbox': bbox,
                    'text': text,
                    'cleaned_text': self._clean_text(text)
                })
        
        # Save result image if path provided
        if output_path:
            save_image(result_image, output_path)
        
        return {
            'image_path': image_path,
            'text_regions': len(results),
            'results': results,
            'full_text': ' '.join([r['cleaned_text'] for r in results])
        }
    
    def process_video(self, video_path, output_path=None, sample_rate=30):
        """
        Process video for text recognition
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video
            sample_rate (int): Process every nth frame
            
        Returns:
            list: List of frame results
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
        
        print(f"Processing {total_frames} frames (sampling every {sample_rate} frames)...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every nth frame
            if frame_count % sample_rate == 0:
                # Detect and recognize text
                text_boxes = self.detector.detect_text(frame)
                frame_results = []
                
                for bbox in text_boxes:
                    text = self.ocr.extract_text(frame, bbox)
                    if text.strip():
                        frame_results.append({
                            'bbox': bbox,
                            'text': text,
                            'cleaned_text': self._clean_text(text)
                        })
                
                # Draw results on frame
                result_frame = self._draw_text_results(frame, frame_results)
                
                results.append({
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'text_regions': len(frame_results),
                    'results': frame_results
                })
            else:
                result_frame = frame
            
            # Save frame if writer is available
            if writer:
                writer.write(result_frame)
            
            # Progress update
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        print(f"Processing complete! Processed {len(results)} frames with text")
        return results
    
    def _clean_text(self, text):
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        
        return text
    
    def _draw_text_results(self, frame, results):
        """Draw text recognition results on frame"""
        result_frame = frame.copy()
        
        for i, result in enumerate(results):
            bbox = result['bbox']
            text = result['text']
            
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add text label
            label = f"{i+1}: {text[:15]}..."
            cv2.putText(result_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result_frame

def main():
    """
    Main function for testing text detection and OCR
    """
    # Example usage
    pipeline = TextRecognitionPipeline(
        detector_method='contour',
        ocr_engine='tesseract',
        languages=['en']
    )
    
    # Test with single image
    # result = pipeline.process_image('data/input/test_image.jpg', 'results/text/result.jpg')
    # print(f"Detected {result['text_regions']} text regions")
    # print(f"Full text: {result['full_text']}")
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit, 's' to save current frame")
    detector = TextDetector(method='contour')
    ocr = OCREngine(engine='tesseract')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect text regions
        text_boxes = detector.detect_text(frame)
        
        # Draw bounding boxes
        result_frame = frame.copy()
        for i, bbox in enumerate(text_boxes):
            x, y, w, h = bbox
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_frame, f"Text {i+1}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Text Detection', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save and process current frame
            import time
            timestamp = int(time.time())
            frame_path = f"results/text/frame_{timestamp}.jpg"
            save_image(result_frame, frame_path)
            
            # Extract text from detected regions
            for i, bbox in enumerate(text_boxes):
                text = ocr.extract_text(frame, bbox)
                if text.strip():
                    print(f"Region {i+1}: {text}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()