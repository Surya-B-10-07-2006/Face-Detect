import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
import time
import vlc
import glob
import threading
from PIL import Image, ImageTk

class FaceDetector:
    def __init__(self):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Initialize VLC instance
        self.vlc_instance = vlc.Instance()
        self.player = self.vlc_instance.media_player_new()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Face Emotion Detection")
        self.root.geometry("1280x720")  # 16:9 aspect ratio
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for webcam
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create webcam canvas with fixed size
        self.webcam_canvas = tk.Canvas(self.left_panel, width=640, height=480)
        self.webcam_canvas.pack(pady=10)
        
        # Create control buttons
        self.create_control_buttons()
        
        # Create right panel for VLC player
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create VLC player frame
        self.player_frame = ttk.Frame(self.right_panel)
        self.player_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create VLC player canvas with same size as webcam
        self.player_canvas = tk.Canvas(self.player_frame, width=640, height=480, bg='black')
        self.player_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create media control buttons frame
        self.media_control_frame = ttk.Frame(self.right_panel)
        self.media_control_frame.pack(pady=10)
        
        # Create media control buttons
        self.create_media_control_buttons()
        
        # Set VLC player window
        if os.name == 'nt':  # Windows
            self.player.set_hwnd(self.player_canvas.winfo_id())
        else:  # Linux/Mac
            self.player.set_xwindow(self.player_canvas.winfo_id())
        
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

        # Video paths for each emotion
        self.emotion_videos = {
            'happy': self.get_video_list('happy'),
            'sad': self.get_video_list('sad'),
            'neutral': self.get_video_list('neutral'),
            'angry': self.get_video_list('angry'),
            'surprise': self.get_video_list('surprise')
        }
        
        # Video player state
        self.current_video_index = 0
        self.is_playing = False
        self.current_emotion_videos = []
        self.current_video = None
        self.is_detecting = False
        self.current_emotion = None
        self.emotion_confidence = 0.0

    def create_control_buttons(self):
        # Create button frame
        button_frame = ttk.Frame(self.left_panel)
        button_frame.pack(pady=10)
        
        # Create buttons with consistent width
        button_width = 15  # Width in characters
        
        self.start_button = ttk.Button(button_frame, text="Start Detection", width=button_width, command=self.start_detection)
        self.start_button.pack(pady=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", width=button_width, command=self.stop_detection)
        self.stop_button.pack(pady=5)
        
        self.capture_button = ttk.Button(button_frame, text="Capture & Play", width=button_width, command=self.capture_and_play)
        self.capture_button.pack(pady=5)
        
        self.next_button = ttk.Button(button_frame, text="Next Video", width=button_width, command=self.next_video)
        self.next_button.pack(pady=5)
        
        self.prev_button = ttk.Button(button_frame, text="Previous Video", width=button_width, command=self.prev_video)
        self.prev_button.pack(pady=5)

    def create_media_control_buttons(self):
        """Create media control buttons for VLC player"""
        # Create button frame
        button_frame = ttk.Frame(self.media_control_frame)
        button_frame.pack(pady=5)
        
        # Create buttons with consistent width
        button_width = 8  # Width in characters
        
        # Play/Pause button
        self.play_pause_button = ttk.Button(button_frame, text="⏯️ Play", width=button_width, command=self.toggle_play_pause)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)
        
        # Volume control buttons
        self.volume_down_button = ttk.Button(button_frame, text="🔈 -", width=button_width, command=self.decrease_volume)
        self.volume_down_button.pack(side=tk.LEFT, padx=5)
        
        self.volume_up_button = ttk.Button(button_frame, text="🔊 +", width=button_width, command=self.increase_volume)
        self.volume_up_button.pack(side=tk.LEFT, padx=5)
        
        # Mute button
        self.mute_button = ttk.Button(button_frame, text="🔇 Mute", width=button_width, command=self.toggle_mute)
        self.mute_button.pack(side=tk.LEFT, padx=5)
        
        # Stop button
        self.stop_button = ttk.Button(button_frame, text="⏹️ Stop", width=button_width, command=self.stop_video)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def get_video_list(self, emotion):
        """Get list of videos for an emotion"""
        video_dir = os.path.join('songs', emotion)
        if os.path.exists(video_dir):
            videos = glob.glob(os.path.join(video_dir, '*.mp4'))
            return videos
        return []

    def start_detection(self):
        self.is_detecting = True
        self.update_webcam()

    def stop_detection(self):
        self.is_detecting = False

    def capture_and_play(self):
        if self.is_detecting and self.current_emotion:
            self.play_emotion_video(self.current_emotion)

    def play_emotion_video(self, emotion):
        """Play the video associated with the detected emotion"""
        self.emotion_videos[emotion] = self.get_video_list(emotion)
        if emotion in self.emotion_videos and self.emotion_videos[emotion]:
            self.current_emotion_videos = self.emotion_videos[emotion]
            self.current_video_index = 0
            return self.play_current_video()
        return False

    def play_current_video(self):
        """Play the current video"""
        if self.current_emotion_videos:
            try:
                video_path = self.current_emotion_videos[self.current_video_index]
                media = self.vlc_instance.media_new(video_path)
                self.player.set_media(media)
                self.player.play()
                self.is_playing = True
                self.current_video = os.path.basename(video_path)
                self.play_pause_button.configure(text="⏯️ Pause")
                return True
            except Exception as e:
                print(f"Error playing video: {str(e)}")
                self.is_playing = False
                self.current_video = None
        return False

    def next_video(self):
        """Play next video in the list"""
        if self.current_emotion_videos:
            self.player.stop()
            self.current_video_index = (self.current_video_index + 1) % len(self.current_emotion_videos)
            return self.play_current_video()
        return False

    def prev_video(self):
        """Play previous video in the list"""
        if self.current_emotion_videos:
            self.player.stop()
            self.current_video_index = (self.current_video_index - 1) % len(self.current_emotion_videos)
            return self.play_current_video()
        return False

    def update_webcam(self):
        """Update webcam feed"""
        if self.is_detecting:
            ret, frame = self.cap.read()
            if ret:
                # Resize frame to match canvas size
                frame = cv2.resize(frame, (640, 480))
                
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces and emotions
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_roi = frame[y:y+h, x:x+w]
                    emotions = self.detect_emotion(face_roi)
                    sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
                    self.current_emotion, self.emotion_confidence = sorted_emotions[0]
                    
                    # Draw rectangle and emotion text
                    cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    emotion_text = f"{self.current_emotion.capitalize()}: {int(self.emotion_confidence*100)}%"
                    cv2.putText(frame_rgb, emotion_text, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert to PhotoImage
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                
                # Update canvas
                self.webcam_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.webcam_canvas.photo = photo
            
            # Schedule next update
            self.root.after(10, self.update_webcam)

    def detect_emotion(self, face_roi):
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Detect smile/mouth
        smile = self.smile_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Initialize emotions with equal baseline
        emotions = {
            'happy': 0.2,
            'sad': 0.2,
            'angry': 0.2,
            'surprise': 0.2,
            'neutral': 0.2
        }
        
        # Process mouth/smile detection
        if len(smile) > 0:
            largest_smile = max(smile, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_smile
            mouth_ratio = w / h if h > 0 else 0
            
            if mouth_ratio > 2.0:
                emotions['surprise'] = 0.9
                emotions['neutral'] = 0.1
            elif mouth_ratio > 1.5:
                emotions['happy'] = 0.9
                emotions['neutral'] = 0.1
            elif mouth_ratio < 1.3:
                emotions['sad'] = 0.9
                emotions['neutral'] = 0.1
            elif mouth_ratio < 1.1:
                emotions['angry'] = 0.9
                emotions['neutral'] = 0.1
        
        # Process eye detection
        if len(eyes) > 0:
            eye_areas = [w * h for (x, y, w, h) in eyes]
            eye_positions = [(x + w/2, y + h/2) for (x, y, w, h) in eyes]
            
            if len(eyes) == 2:
                eye_distance = abs(eye_positions[0][0] - eye_positions[1][0])
                eye_height_diff = abs(eye_positions[0][1] - eye_positions[1][1])
                eye_area_ratio = max(eye_areas) / min(eye_areas) if min(eye_areas) > 0 else 1
                avg_eye_area = sum(eye_areas) / len(eye_areas)
                
                if avg_eye_area > 600:
                    emotions['surprise'] = max(emotions['surprise'], 0.9)
                    emotions['neutral'] = 0.1
                elif avg_eye_area < 400 and eye_area_ratio > 1.2:
                    emotions['angry'] = max(emotions['angry'], 0.9)
                    emotions['neutral'] = 0.1
                elif avg_eye_area < 350 and eye_height_diff > 6:
                    emotions['sad'] = max(emotions['sad'], 0.9)
                    emotions['neutral'] = 0.1
                elif 350 < avg_eye_area < 600:
                    emotions['happy'] = max(emotions['happy'], 0.9)
                    emotions['neutral'] = 0.1
        
        # Re-normalize emotions
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions

    def toggle_play_pause(self):
        """Toggle between play and pause"""
        if self.player.is_playing():
            self.player.pause()
            self.play_pause_button.configure(text="⏯️ Play")
        else:
            self.player.play()
            self.play_pause_button.configure(text="⏯️ Pause")

    def increase_volume(self):
        """Increase volume by 10%"""
        current_volume = self.player.audio_get_volume()
        new_volume = min(current_volume + 10, 100)
        self.player.audio_set_volume(new_volume)

    def decrease_volume(self):
        """Decrease volume by 10%"""
        current_volume = self.player.audio_get_volume()
        new_volume = max(current_volume - 10, 0)
        self.player.audio_set_volume(new_volume)

    def toggle_mute(self):
        """Toggle mute state"""
        is_muted = self.player.audio_get_mute()
        self.player.audio_set_mute(not is_muted)
        self.mute_button.configure(text="🔇 Unmute" if not is_muted else "🔇 Mute")

    def stop_video(self):
        """Stop video playback"""
        self.player.stop()
        self.play_pause_button.configure(text="⏯️ Play")

    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    detector = FaceDetector()
    detector.run() 