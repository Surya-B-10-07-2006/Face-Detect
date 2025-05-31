import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from face_detection import FaceDetector

class FaceDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Emotion Detection")
        
        # Set fixed window size
        self.root.geometry("1200x800")
        self.root.resizable(False, False)
        
        # Initialize face detector
        self.detector = FaceDetector()
        self.is_running = False
        self.current_emotion = None
        self.face_detection_confidence = 0.0
        self.emotion_history = []
        self.confidence_threshold = 0.5
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure("Modern.TFrame", background="#1E1E1E")
        self.style.configure("Modern.TButton",
                           background="#2C3E50",
                           foreground="white",
                           font=("Arial", 12, "bold"),
                           padding=10)
        
        # Create main container
        self.main_container = ttk.Frame(self.root, style="Modern.TFrame")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel (Camera Feed)
        self.left_panel = ttk.Frame(self.main_container, style="Modern.TFrame")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Camera feed frame
        self.camera_frame = ttk.Frame(self.left_panel, style="Modern.TFrame")
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera feed label
        self.video_label = ttk.Label(self.camera_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Create right panel (Controls and Emotion Levels)
        self.right_panel = ttk.Frame(self.main_container, style="Modern.TFrame", width=360)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # Control buttons frame
        self.control_frame = ttk.Frame(self.right_panel, style="Modern.TFrame")
        self.control_frame.pack(fill=tk.X, pady=10)
        
        # Add control buttons
        self.start_button = ttk.Button(self.control_frame, text="▶ Start",
                                     command=self.toggle_detection,
                                     style="Modern.TButton",
                                     width=25)
        self.start_button.pack(pady=5)
        
        self.stop_button = ttk.Button(self.control_frame, text="⏹ Stop",
                                    command=self.stop_detection,
                                    style="Modern.TButton",
                                    width=25,
                                    state=tk.DISABLED)
        self.stop_button.pack(pady=5)
        
        self.capture_button = ttk.Button(self.control_frame, text="📸 Capture & Play",
                                       command=self.capture_and_play,
                                       style="Modern.TButton",
                                       width=25,
                                       state=tk.DISABLED)
        self.capture_button.pack(pady=5)
        
        # Emotion levels frame
        self.emotion_levels_frame = ttk.Frame(self.right_panel, style="Modern.TFrame")
        self.emotion_levels_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create emotion level indicators
        self.emotion_levels = {}
        emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral']
        
        for emotion in emotions:
            container = ttk.Frame(self.emotion_levels_frame, style="Modern.TFrame")
            container.pack(fill=tk.X, pady=5)
            
            # Emotion label
            label = ttk.Label(container, text=emotion.capitalize(),
                            foreground="white",
                            font=("Arial", 12, "bold"))
            label.pack(side=tk.LEFT, padx=5)
            
            # Level indicator
            level = ttk.Label(container, width=30, background="#2C3E50")
            level.pack(side=tk.RIGHT, padx=5)
            
            self.emotion_levels[emotion] = level
        
        # Set up video update
        self.update_frame()

    def toggle_detection(self):
        if not self.is_running:
            if not self.detector.cap.isOpened():
                return
            self.is_running = True
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.capture_button.configure(state=tk.NORMAL)
        else:
            self.stop_detection()

    def stop_detection(self):
        self.is_running = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.capture_button.configure(state=tk.DISABLED)
        self.current_emotion = None

    def process_emotion(self, emotions):
        top_emotion = max(emotions.items(), key=lambda x: x[1])
        emotion, confidence = top_emotion
        
        self.emotion_history.append((emotion, confidence))
        if len(self.emotion_history) > 5:
            self.emotion_history.pop(0)
        
        avg_confidence = sum(c for e, c in self.emotion_history if e == emotion) / len([e for e, c in self.emotion_history if e == emotion])
        
        if avg_confidence >= self.confidence_threshold:
            self.current_emotion = emotion
            self.face_detection_confidence = avg_confidence
            return True
        return False

    def update_emotion_levels(self, emotions):
        for emotion, confidence in emotions.items():
            level = int(confidence * 100)
            color = f"#{int(39 * (1 - confidence)):02x}{int(174 * confidence):02x}{int(96 * confidence):02x}"
            self.emotion_levels[emotion].configure(background=color)

    def update_frame(self):
        if self.is_running:
            try:
                ret, frame = self.detector.cap.read()
                if not ret:
                    self.stop_detection()
                    return

                # Resize frame for better performance
                frame = cv2.resize(frame, (640, 480))
                
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces and emotions
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                if len(faces) > 0:
                    face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = face
                    
                    # Add padding to face region
                    padding = int(min(w, h) * 0.1)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Extract face ROI for emotion detection
                    face_roi = frame[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        face_roi = cv2.resize(face_roi, (48, 48))
                        emotions = self.detector.detect_emotion(face_roi)
                        self.update_emotion_levels(emotions)
                        
                        if self.process_emotion(emotions):
                            self.current_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                            # Add emotion text above the face
                            emotion_text = f"Emotion: {self.current_emotion.capitalize()}"
                            confidence_text = f"Confidence: {self.face_detection_confidence:.2f}"
                            
                            # Calculate text position
                            text_y = max(y1 - 10, 30)
                            
                            # Add background rectangle for better visibility
                            text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame_rgb, 
                                        (x1, text_y - text_size[1] - 10),
                                        (x1 + text_size[0], text_y + 5),
                                        (0, 0, 0),
                                        -1)
                            
                            # Add emotion text
                            cv2.putText(frame_rgb,
                                      emotion_text,
                                      (x1, text_y),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.7,
                                      (0, 255, 0),
                                      2)
                            
                            # Add confidence text
                            cv2.putText(frame_rgb,
                                      confidence_text,
                                      (x1, text_y + 25),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.7,
                                      (0, 255, 0),
                                      2)

                # Convert frame to PhotoImage
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                
                # Update video label
                self.video_label.configure(image=photo)
                self.video_label.image = photo  # Keep a reference

            except Exception as e:
                print(f"Error: {str(e)}")
                self.stop_detection()
        
        # Schedule next update
        self.root.after(30, self.update_frame)

    def capture_and_play(self):
        if self.current_emotion and self.face_detection_confidence >= self.confidence_threshold:
            youtube_url = self.get_youtube_url_for_emotion(self.current_emotion)
            if youtube_url:
                # Open YouTube URL in default web browser
                import webbrowser
                webbrowser.open(youtube_url)
            else:
                self.current_emotion = None
        else:
            self.current_emotion = None

    def get_youtube_url_for_emotion(self, emotion):
        emotion_urls = {
            'happy': 'https://www.youtube.com/watch?v=uQnfRdmSXl0',
            'sad': 'https://www.youtube.com/watch?v=V8Yv8F6KiG0',
            'angry': 'https://www.youtube.com/watch?v=AAUv3HEaHq8',
            'surprise': 'https://www.youtube.com/watch?v=VzppuKWR-5U',
            'neutral': 'https://www.youtube.com/watch?v=TKeU1bLlAcc'
        }
        return emotion_urls.get(emotion)

def main():
    root = tk.Tk()
    app = FaceDetectionUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 