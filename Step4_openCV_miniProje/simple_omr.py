import cv2
import numpy as np
import argparse
from utils import show_image, save_image, resize_image, four_point_transform, order_points


class SimpleOMR:
    def __init__(self, choices_per_question=5):
        self.choices_per_question = choices_per_question
        self.questions = []
        self.answers = []
    
    def preprocess_image(self, image):
        """
        Preprocess the OMR sheet image.
        
        Args:
            image: Input image
        
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get binary image
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        return thresh
    
    def find_answer_sheet_contour(self, thresh):
        """
        Find the contour of the answer sheet.
        
        Args:
            thresh: Thresholded image
        
        Returns:
            Answer sheet contour or None
        """
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Look for the largest rectangular contour
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If we have 4 points, we likely found our answer sheet
            if len(approx) == 4:
                return approx
        
        return None
    
    def extract_bubbles(self, thresh, num_questions, choices_per_question=5):
        """
        Extract bubble regions from the answer sheet.
        
        Args:
            thresh: Thresholded image
            num_questions: Number of questions
            choices_per_question: Number of choices per question
        
        Returns:
            List of bubble contours organized by question
        """
        # Find all contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (bubbles should be similar in size)
        bubble_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 <= area <= 400:  # Adjust these values based on your bubble size
                bubble_contours.append(contour)
        
        # Sort bubbles by position (top to bottom, left to right)
        bubble_contours = sorted(bubble_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
        
        # Group bubbles by questions
        questions_bubbles = []
        for i in range(0, len(bubble_contours), choices_per_question):
            question_bubbles = bubble_contours[i:i + choices_per_question]
            if len(question_bubbles) == choices_per_question:
                questions_bubbles.append(question_bubbles)
        
        return questions_bubbles[:num_questions]
    
    def check_bubble_filled(self, thresh, contour, fill_threshold=0.6):
        """
        Check if a bubble is filled.
        
        Args:
            thresh: Thresholded image
            contour: Bubble contour
            fill_threshold: Threshold for considering a bubble filled
        
        Returns:
            True if bubble is filled, False otherwise
        """
        # Create mask for the bubble
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Apply mask and count non-zero pixels
        masked = cv2.bitwise_and(thresh, thresh, mask=mask)
        total_pixels = cv2.countNonZero(mask)
        filled_pixels = cv2.countNonZero(masked)
        
        # Calculate fill percentage
        if total_pixels > 0:
            fill_percentage = filled_pixels / total_pixels
            return fill_percentage >= fill_threshold
        
        return False
    
    def grade_omr_sheet(self, image, answer_key, num_questions=None, show_results=False):
        """
        Grade an OMR answer sheet.
        
        Args:
            image: Input OMR sheet image
            answer_key: List of correct answers (0-indexed)
            num_questions: Number of questions (auto-detected if None)
            show_results: Whether to show visual results
        
        Returns:
            Dictionary with grading results
        """
        original = image.copy()
        
        # Preprocess image
        thresh = self.preprocess_image(image)
        
        # Find answer sheet contour and apply perspective correction
        sheet_contour = self.find_answer_sheet_contour(thresh)
        if sheet_contour is not None:
            # Apply perspective transformation
            warped = four_point_transform(original, sheet_contour.reshape(4, 2))
            warped_thresh = self.preprocess_image(warped)
        else:
            warped = original
            warped_thresh = thresh
        
        # Auto-detect number of questions if not provided
        if num_questions is None:
            num_questions = len(answer_key)
        
        # Extract bubbles
        questions_bubbles = self.extract_bubbles(warped_thresh, num_questions, self.choices_per_question)
        
        if len(questions_bubbles) == 0:
            print("No bubbles detected!")
            return None
        
        # Process each question
        student_answers = []
        correct_answers = 0
        
        for q_idx, question_bubbles in enumerate(questions_bubbles):
            # Check which bubbles are filled
            filled_bubbles = []
            for choice_idx, bubble in enumerate(question_bubbles):
                if self.check_bubble_filled(warped_thresh, bubble):
                    filled_bubbles.append(choice_idx)
            
            # Determine the answer
            if len(filled_bubbles) == 1:
                student_answer = filled_bubbles[0]
                student_answers.append(student_answer)
                
                # Check if correct
                if q_idx < len(answer_key) and student_answer == answer_key[q_idx]:
                    correct_answers += 1
            elif len(filled_bubbles) == 0:
                student_answers.append(None)  # No answer
            else:
                student_answers.append(-1)  # Multiple answers (invalid)
        
        # Calculate score
        total_questions = len(answer_key)
        score_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        results = {
            'student_answers': student_answers,
            'correct_answers': correct_answers,
            'total_questions': total_questions,
            'score_percentage': score_percentage,
            'questions_detected': len(questions_bubbles)
        }
        
        # Show results if requested
        if show_results:
            result_image = self.draw_results(warped, questions_bubbles, student_answers, answer_key)
            show_image("OMR Grading Results", result_image)
        
        return results
    
    def draw_results(self, image, questions_bubbles, student_answers, answer_key):
        """
        Draw grading results on the image.
        
        Args:
            image: Input image
            questions_bubbles: List of bubble contours for each question
            student_answers: Student's answers
            answer_key: Correct answers
        
        Returns:
            Annotated image
        """
        result = image.copy()
        
        for q_idx, (question_bubbles, student_answer) in enumerate(zip(questions_bubbles, student_answers)):
            correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None
            
            for choice_idx, bubble in enumerate(question_bubbles):
                # Get bubble center
                M = cv2.moments(bubble)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    continue
                
                # Color coding
                if choice_idx == student_answer:
                    if choice_idx == correct_answer:
                        # Correct answer - green
                        cv2.circle(result, (cX, cY), 20, (0, 255, 0), 3)
                    else:
                        # Wrong answer - red
                        cv2.circle(result, (cX, cY), 20, (0, 0, 255), 3)
                elif choice_idx == correct_answer and student_answer != correct_answer:
                    # Show correct answer - blue
                    cv2.circle(result, (cX, cY), 15, (255, 0, 0), 2)
        
        return result


def create_sample_answer_key(num_questions, choices_per_question=5):
    """
    Create a sample answer key for testing.
    
    Args:
        num_questions: Number of questions
        choices_per_question: Number of choices per question
    
    Returns:
        List of random answers
    """
    return [np.random.randint(0, choices_per_question) for _ in range(num_questions)]


def main():
    parser = argparse.ArgumentParser(description='Simple OMR (Optical Mark Recognition) System')
    parser.add_argument('-i', '--image', required=True, help='Path to OMR sheet image')
    parser.add_argument('-k', '--answer-key', nargs='+', type=int, 
                       help='Answer key (space-separated integers, 0-indexed)')
    parser.add_argument('-q', '--questions', type=int, help='Number of questions')
    parser.add_argument('-c', '--choices', type=int, default=5, help='Number of choices per question')
    parser.add_argument('--show', action='store_true', help='Show grading results')
    parser.add_argument('--demo', action='store_true', help='Run with sample answer key')
    parser.add_argument('-o', '--output', help='Save result image to file')
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    # Resize image if too large
    if image.shape[0] > 1000:
        image = resize_image(image, height=1000)
        print("Resized image for processing")
    
    # Create OMR instance
    omr = SimpleOMR(args.choices)
    
    # Prepare answer key
    if args.demo:
        # Create sample answer key
        num_questions = args.questions or 10
        answer_key = create_sample_answer_key(num_questions, args.choices)
        print(f"Using sample answer key: {answer_key}")
    elif args.answer_key:
        answer_key = args.answer_key
        print(f"Using provided answer key: {answer_key}")
    else:
        print("Error: Please provide an answer key using --answer-key or use --demo mode")
        return
    
    # Grade the OMR sheet
    print("Processing OMR sheet...")
    results = omr.grade_omr_sheet(image, answer_key, args.questions, args.show)
    
    if results is None:
        print("Failed to process OMR sheet")
        return
    
    # Display results
    print(f"\n=== OMR GRADING RESULTS ===")
    print(f"Questions detected: {results['questions_detected']}")
    print(f"Total questions: {results['total_questions']}")
    print(f"Correct answers: {results['correct_answers']}")
    print(f"Score: {results['score_percentage']:.1f}%")
    
    print(f"\nDetailed results:")
    for i, answer in enumerate(results['student_answers']):
        correct = answer_key[i] if i < len(answer_key) else "N/A"
        if answer is None:
            status = "No answer"
        elif answer == -1:
            status = "Multiple answers (invalid)"
        elif answer == correct:
            status = "Correct ✓"
        else:
            status = "Wrong ✗"
        
        print(f"  Question {i+1}: Answer={answer}, Correct={correct}, Status={status}")
    
    # Save results if requested
    if args.output and args.show:
        # Re-process to get the result image
        thresh = omr.preprocess_image(image)
        sheet_contour = omr.find_answer_sheet_contour(thresh)
        if sheet_contour is not None:
            warped = four_point_transform(image, sheet_contour.reshape(4, 2))
        else:
            warped = image
        
        questions_bubbles = omr.extract_bubbles(omr.preprocess_image(warped), 
                                               len(answer_key), args.choices)
        result_image = omr.draw_results(warped, questions_bubbles, 
                                       results['student_answers'], answer_key)
        
        if save_image(args.output, result_image):
            print(f"Result image saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        print("\nExample usage:")
        print("python simple_omr.py -i answer_sheet.jpg --answer-key 0 1 2 3 4 --show")
        print("python simple_omr.py -i answer_sheet.jpg --demo -q 10 --show")
        print("python simple_omr.py -i answer_sheet.jpg -k 0 1 2 1 0 -c 4 -o result.jpg")