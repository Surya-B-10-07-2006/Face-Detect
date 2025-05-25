import cv2
import numpy as np
from datetime import datetime
import os
import webbrowser
import time

class FaceDetector:
    def __init__(self):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Load face detection cascade
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_eye.xml'))
        self.smile_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_smile.xml'))
        
        # Define colors for emotions
        self.emotion_colors = {
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'surprise': (0, 165, 255), # Orange
            'neutral': (255, 255, 255) # White
        }

        # YouTube song links for each emotion
        self.emotion_songs = {
            'happy': 'https://www.youtube.com/watch?v=oLgzs8nut3A',
            'sad': 'https://www.youtube.com/watch?v=QNdtanXFNFw',
            'angry': 'https://www.youtube.com/watch?v=AAUv3HEaHq8',
            'surprise': 'https://www.youtube.com/watch?v=VzppuKWR-5U',
            'neutral': 'https://www.youtube.com/watch?v=TKeU1bLlAcc'
        }
        
        # Track last song played time
        self.last_song_time = datetime.now()
        self.song_cooldown = 30  # seconds between song changes
        self.last_emotion = None

        # Button properties
        self.button_rect = None
        self.button_text = "Capture & Play"
        self.button_color = (0, 255, 0)  # Green
        self.button_hover_color = (0, 200, 0)  # Darker green
        self.button_active = False

    def detect_emotion(self, face_roi):
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Detect smile/mouth
        smile = self.smile_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Initialize emotions with neutral values
        emotions = {
            'happy': 0.1,
            'sad': 0.1,
            'angry': 0.1,
            'surprise': 0.1,
            'neutral': 0.6  # Initialize neutral higher as a baseline
        }
        
        # Process mouth/smile detection
        if len(smile) > 0:
            largest_smile = max(smile, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_smile
            mouth_ratio = w / h if h > 0 else 0
            
            # Surprise detection - wide open mouth (O shape)
            if mouth_ratio > 2.5:  # Increased threshold for more pronounced O shape
                emotions['surprise'] = 0.95
                emotions['happy'] = 0.05
                emotions['neutral'] = 0.1
            # Happy detection - upturned corners
            elif mouth_ratio > 1.8:  # Wider smile
                emotions['happy'] = 0.95
                emotions['neutral'] = 0.1
            # Sad detection - downturned corners
            elif mouth_ratio < 1.2:  # Narrower mouth
                emotions['sad'] = 0.8
                emotions['neutral'] = 0.2
            # Angry detection - tight lips
            elif mouth_ratio < 1.0:  # Very narrow mouth
                emotions['angry'] = 0.8
                emotions['neutral'] = 0.2
        
        # Process eye detection
        if len(eyes) > 0:
            eye_areas = [w * h for (x, y, w, h) in eyes]
            eye_positions = [(x + w/2, y + h/2) for (x, y, w, h) in eyes]
            
            if len(eyes) == 2:  # Both eyes detected
                # Calculate eye metrics
                eye_distance = abs(eye_positions[0][0] - eye_positions[1][0])
                eye_height_diff = abs(eye_positions[0][1] - eye_positions[1][1])
                eye_area_ratio = max(eye_areas) / min(eye_areas) if min(eye_areas) > 0 else 1
                avg_eye_area = sum(eye_areas) / len(eye_areas)
                
                # Surprise detection - wide open eyes
                if avg_eye_area > 800:  # Larger eyes
                    emotions['surprise'] = max(emotions['surprise'], 0.9)
                    emotions['neutral'] = 0.1
                
                # Angry detection - narrowed eyes and furrowed brows
                elif (avg_eye_area < 500 and  # Smaller eyes
                      eye_area_ratio > 1.3 and  # Uneven eye sizes
                      eye_height_diff > 5):  # Eyes at different heights
                    emotions['angry'] = max(emotions['angry'], 0.9)
                    emotions['neutral'] = 0.1
                
                # Sad detection - droopy eyes and raised inner eyebrows
                elif (avg_eye_area < 450 and  # Smaller eyes
                      eye_area_ratio < 1.2 and  # Similar eye sizes
                      eye_height_diff > 8):  # Eyes at different heights
                    emotions['sad'] = max(emotions['sad'], 0.9)
                    emotions['neutral'] = 0.1
                
                # Happy detection - slight squint with crow's feet
                elif (450 < avg_eye_area < 700 and  # Medium-sized eyes
                      eye_area_ratio < 1.2):  # Similar eye sizes
                    emotions['happy'] = max(emotions['happy'], 0.8)
                    emotions['neutral'] = 0.2
                
                # Neutral detection - balanced eye features
                elif (500 < avg_eye_area < 800 and  # Normal-sized eyes
                      eye_area_ratio < 1.1 and  # Very similar eye sizes
                      eye_height_diff < 5):  # Eyes at similar heights
                    emotions['neutral'] = max(emotions['neutral'], 0.9)
            
            # Single eye detection (partial face)
            elif len(eyes) == 1:
                eye_area = eye_areas[0]
                if eye_area > 700:  # Large eye
                    emotions['surprise'] = max(emotions['surprise'], 0.8)
                elif eye_area < 400:  # Small eye
                    emotions['sad'] = max(emotions['sad'], 0.7)
        
        # Re-normalize emotions
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions

    def draw_button(self, frame):
        """Draw the capture button in the corner"""
        # Button dimensions
        button_width = 150
        button_height = 40
        margin = 20
        
        # Position in top-right corner
        x = frame.shape[1] - button_width - margin
        y = margin
        
        # Store button rectangle for click detection
        self.button_rect = (x, y, button_width, button_height)
        
        # Draw button background
        color = self.button_hover_color if self.button_active else self.button_color
        cv2.rectangle(frame, (x, y), (x + button_width, y + button_height), color, -1)
        
        # Draw button border
        cv2.rectangle(frame, (x, y), (x + button_width, y + button_height), (255, 255, 255), 2)
        
        # Calculate text position to center it
        text_size = cv2.getTextSize(self.button_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x + (button_width - text_size[0]) // 2
        text_y = y + (button_height + text_size[1]) // 2
        
        # Draw button text
        cv2.putText(frame, self.button_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def check_button_click(self, x, y):
        """Check if the button was clicked"""
        if self.button_rect is None:
            return False
            
        button_x, button_y, button_width, button_height = self.button_rect
        return (button_x <= x <= button_x + button_width and
                button_y <= y <= button_y + button_height)

    def analyze_and_play(self, frame):
        """Analyze the current frame and play the corresponding song"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Get the first face
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect emotions
            emotions = self.detect_emotion(face_roi)
            
            # Get the top emotion
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion, confidence = top_emotion
            
            if confidence > 0.5:
                # Get the song URL for this emotion
                song_url = self.emotion_songs.get(emotion)
                if song_url:
                    print(f"\nDetected {emotion} emotion! Opening song...")
                    webbrowser.open(song_url)
                    return True
        
        return False

    def start_detection(self):
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Face detection started. Press 'q' to quit.")
        print("\nEmotion Songs:")
        for emotion, song in self.emotion_songs.items():
            print(f"- {emotion.capitalize()}: {song}")
        
        while True:
            # Read frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # Process each face
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract face ROI for emotion detection
                face_roi = frame[y:y+h, x:x+w]
                
                # Detect emotions
                emotions = self.detect_emotion(face_roi)
                
                # Sort emotions by probability
                sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
                
                # Display top emotion
                top_emotion, confidence = sorted_emotions[0]
                emotion_text = f"{top_emotion.capitalize()}: {int(confidence*100)}%"
                
                # Use emotion color
                color = self.emotion_colors.get(top_emotion, (255, 255, 255))
                
                # Draw emotion text above face
                cv2.putText(frame, emotion_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw the capture button
            self.draw_button(frame)

            # Display the frame
            cv2.imshow('Face Detection', frame)

            # Handle mouse events
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if self.check_button_click(x, y):
                        print("\nAnalyzing current frame...")
                        if self.analyze_and_play(frame):
                            print("Song opened successfully!")
                        else:
                            print("No face detected or confidence too low!")
                elif event == cv2.EVENT_MOUSEMOVE:
                    self.button_active = self.check_button_click(x, y)

            cv2.setMouseCallback('Face Detection', mouse_callback)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceDetector()
    detector.start_detection() 