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
import sys
import json
from datetime import datetime
import random
import platform

class FaceEmotionDetection:
    def __init__(self):
        # Initialize VLC instance with verbose logging
        self.instance = vlc.Instance('--verbose 2')
        self.player = self.instance.media_player_new()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Face Emotion Detection")
        self.root.geometry("1280x720")
        self.root.configure(bg='#1a1a1a')
        
        # Load CSS styles
        self.load_styles()
        
        # Create main frame
        self.main_frame = tk.Frame(self.root, bg='#2d2d2d')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create left and right panels
        self.left_panel = tk.Frame(self.main_frame, bg='#2d2d2d')
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.right_panel = tk.Frame(self.main_frame, bg='#2d2d2d')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Create webcam canvas
        self.webcam_canvas = tk.Canvas(self.left_panel, width=640, height=480, bg='black')
        self.webcam_canvas.pack(pady=10)
        
        # Create VLC player canvas
        self.player_canvas = tk.Canvas(self.right_panel, width=640, height=480, bg='black')
        self.player_canvas.pack(pady=10)
        
        # Create control buttons frame
        self.control_frame = tk.Frame(self.left_panel, bg='#2d2d2d')
        self.control_frame.pack(pady=10)
        
        # Create media control buttons frame
        self.media_control_frame = tk.Frame(self.right_panel, bg='#2d2d2d')
        self.media_control_frame.pack(pady=10)
        
        # Create buttons with tooltips
        self.create_control_buttons()
        self.create_media_buttons()
        
        # Create status labels
        self.status_label = tk.Label(self.left_panel, text="Status: Ready", bg='#2d2d2d', fg='white')
        self.status_label.pack(pady=5)
        
        self.emotion_label = tk.Label(self.left_panel, text="Emotion: None", bg='#2d2d2d', fg='white')
        self.emotion_label.pack(pady=5)
        
        # Initialize variables
        self.is_detecting = False
        self.current_emotion = None
        self.video_list = {}
        self.current_video_index = 0
        self.is_playing = False
        self.is_muted = False
        self.volume = 100
        
        # Emotion detection variables
        self.emotion_buffer = []  # Buffer to store recent emotions
        self.buffer_size = 10     # Number of emotions to store
        self.last_emotion_change = time.time()  # Time of last emotion change
        self.emotion_change_delay = 2.0  # Minimum seconds between emotion changes
        self.emotion_confidence = {}  # Store confidence for each emotion
        
        # Load emotion videos
        self.load_emotion_videos()
        
        # Set up mouse callback
        self.webcam_canvas.bind("<Button-1>", self.mouse_callback)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Error: Could not open webcam")
            return
        
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Start webcam update
        self.update_webcam()
        
    def load_styles(self):
        """Load CSS styles from file"""
        try:
            with open('styles.css', 'r') as f:
                self.styles = f.read()
        except FileNotFoundError:
            self.show_error("Error: styles.css file not found")
            self.styles = ""
    
    def create_control_buttons(self):
        """Create control buttons with tooltips"""
        # Start Detection button
        self.start_button = tk.Button(
            self.control_frame,
            text="Start Detection",
            command=self.start_detection,
            bg='#2ecc71',
            fg='white',
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.start_button, "Start face detection")
        
        # Stop Detection button
        self.stop_button = tk.Button(
            self.control_frame,
            text="Stop Detection",
            command=self.stop_detection,
            bg='#e74c3c',
            fg='white',
            relief=tk.FLAT,
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.stop_button, "Stop face detection")
        
        # Capture & Play button
        self.capture_button = tk.Button(
            self.control_frame,
            text="Capture & Play",
            command=self.capture_and_play,
            bg='#f1c40f',
            fg='white',
            relief=tk.FLAT,
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.capture_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.capture_button, "Capture current emotion and play video")
    
    def create_media_buttons(self):
        """Create media control buttons with tooltips"""
        # Play/Pause button
        self.play_button = tk.Button(
            self.media_control_frame,
            text="⏯️",
            command=self.toggle_play_pause,
            bg='#3498db',
            fg='white',
            relief=tk.FLAT,
            width=3,
            height=1
        )
        self.play_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.play_button, "Play/Pause")
        
        # Previous button
        self.prev_button = tk.Button(
            self.media_control_frame,
            text="⏮️",
            command=self.prev_video,
            bg='#3498db',
            fg='white',
            relief=tk.FLAT,
            width=3,
            height=1
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.prev_button, "Previous Video")
        
        # Next button
        self.next_button = tk.Button(
            self.media_control_frame,
            text="⏭️",
            command=self.next_video,
            bg='#3498db',
            fg='white',
            relief=tk.FLAT,
            width=3,
            height=1
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.next_button, "Next Video")
        
        # Volume up button
        self.volume_up_button = tk.Button(
            self.media_control_frame,
            text="🔊",
            command=self.volume_up,
            bg='#9b59b6',
            fg='white',
            relief=tk.FLAT,
            width=3,
            height=1
        )
        self.volume_up_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.volume_up_button, "Increase Volume")
        
        # Volume down button
        self.volume_down_button = tk.Button(
            self.media_control_frame,
            text="🔉",
            command=self.volume_down,
            bg='#9b59b6',
            fg='white',
            relief=tk.FLAT,
            width=3,
            height=1
        )
        self.volume_down_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.volume_down_button, "Decrease Volume")
        
        # Mute button
        self.mute_button = tk.Button(
            self.media_control_frame,
            text="🔇",
            command=self.toggle_mute,
            bg='#e67e22',
            fg='white',
            relief=tk.FLAT,
            width=3,
            height=1
        )
        self.mute_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.mute_button, "Mute/Unmute")
        
        # Stop button
        self.stop_media_button = tk.Button(
            self.media_control_frame,
            text="⏹️",
            command=self.stop_video,
            bg='#e74c3c',
            fg='white',
            relief=tk.FLAT,
            width=3,
            height=1
        )
        self.stop_media_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.stop_media_button, "Stop Video")
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = tk.Label(
                tooltip,
                text=text,
                bg='#333',
                fg='white',
                relief=tk.SOLID,
                borderwidth=1,
                padx=5,
                pady=2
            )
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
        
        widget.bind('<Enter>', show_tooltip)
    
    def show_error(self, message):
        """Show error message"""
        error_label = tk.Label(
            self.root,
            text=message,
            fg='#e74c3c',
            bg='rgba(231, 76, 60, 0.1)',
            padx=10,
            pady=5,
            relief=tk.SOLID,
            borderwidth=1
        )
        error_label.pack(pady=10)
        self.root.after(3000, error_label.destroy)
    
    def show_success(self, message):
        """Show success message"""
        success_label = tk.Label(
            self.root,
            text=message,
            fg='#2ecc71',
            bg='rgba(46, 204, 113, 0.1)',
            padx=10,
            pady=5,
            relief=tk.SOLID,
            borderwidth=1
        )
        success_label.pack(pady=10)
        self.root.after(3000, success_label.destroy)
    
    def update_webcam(self):
        """Update webcam feed"""
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.is_detecting:
                    # If detecting, let detect_emotion handle the frame display
                    self.detect_emotion(frame)
                else:
                    # If not detecting, just display the frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize frame to fit canvas
                    canvas_width = self.webcam_canvas.winfo_width()
                    canvas_height = self.webcam_canvas.winfo_height()
                    if canvas_width > 1 and canvas_height > 1:
                        frame_rgb = cv2.resize(frame_rgb, (canvas_width, canvas_height))
                    
                    # Convert to PhotoImage
                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                    
                    # Update canvas
                    self.webcam_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Schedule next update
        self.root.after(10, self.update_webcam)
    
    def detect_emotion(self, frame):
        """Detect emotion in the frame with smoothing and stability"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Get face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Get current emotion with confidence
            current_time = time.time()
            new_emotion = self.get_emotion_with_confidence(face_roi)
            
            # Add to buffer
            self.emotion_buffer.append(new_emotion)
            if len(self.emotion_buffer) > self.buffer_size:
                self.emotion_buffer.pop(0)
            
            # Only update emotion if enough time has passed
            if current_time - self.last_emotion_change >= self.emotion_change_delay:
                # Get most common emotion from buffer
                if self.emotion_buffer:
                    emotion_counts = {}
                    for emotion, _ in self.emotion_buffer:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
                    
                    # Only update if emotion has changed
                    if most_common_emotion != self.current_emotion:
                        self.current_emotion = most_common_emotion
                        self.last_emotion_change = current_time
                        self.emotion_label.config(text=f"Emotion: {self.current_emotion}")
                        self.capture_button.config(state=tk.NORMAL)
            
            # Add emotion text above the face box
            emotion_text = f"Emotion: {self.current_emotion if self.current_emotion else 'Detecting...'}"
            cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert frame to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to fit canvas
            canvas_width = self.webcam_canvas.winfo_width()
            canvas_height = self.webcam_canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                frame_rgb = cv2.resize(frame_rgb, (canvas_width, canvas_height))
            
            # Convert to PhotoImage and update canvas
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.webcam_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    
    def get_emotion_with_confidence(self, face_roi):
        """Get emotion with confidence score using facial features"""
        # Initialize emotion scores
        emotions = {
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'neutral': 0.0
        }
        
        # Analyze facial features (simplified version)
        # In a real application, you would use a proper emotion detection model
        
        # Get face dimensions
        height, width = face_roi.shape
        
        # Analyze mouth region (bottom third of face)
        mouth_region = face_roi[int(height*0.6):height, :]
        mouth_edges = cv2.Canny(mouth_region, 100, 200)
        mouth_curvature = np.sum(mouth_edges) / (width * height * 0.4)
        
        # Analyze eye regions (top third of face)
        eye_region = face_roi[:int(height*0.3), :]
        eye_edges = cv2.Canny(eye_region, 100, 200)
        eye_intensity = np.sum(eye_edges) / (width * height * 0.3)
        
        # Update emotion scores based on features
        if mouth_curvature > 0.5:
            emotions['happy'] = 0.8
            emotions['neutral'] = 0.2
        elif mouth_curvature < 0.2:
            emotions['sad'] = 0.8
            emotions['neutral'] = 0.2
        elif eye_intensity > 0.6:
            emotions['angry'] = 0.8
            emotions['neutral'] = 0.2
        else:
            emotions['neutral'] = 0.9
            emotions['happy'] = 0.1
        
        # Get emotion with highest confidence
        max_emotion = max(emotions.items(), key=lambda x: x[1])
        return max_emotion
    
    def start_detection(self):
        """Start face detection"""
        self.is_detecting = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Detecting")
        self.show_success("Face detection started")
    
    def stop_detection(self):
        """Stop face detection"""
        self.is_detecting = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.capture_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        self.emotion_label.config(text="Emotion: None")
        self.show_success("Face detection stopped")
    
    def capture_and_play(self):
        """Capture current emotion and play video"""
        if self.current_emotion:
            self.play_emotion_video(self.current_emotion)
            self.show_success(f"Playing {self.current_emotion} video")
    
    def load_emotion_videos(self):
        """Load videos for each emotion"""
        emotions = ['happy', 'sad', 'angry', 'neutral']
        for emotion in emotions:
            self.video_list[emotion] = self.get_video_list(emotion)
    
    def get_video_list(self, emotion):
        """Get list of videos for an emotion"""
        video_dir = os.path.join('songs', emotion)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            return []
        
        videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
        return [os.path.join(video_dir, v) for v in videos]
    
    def play_emotion_video(self, emotion):
        """Play a video for the given emotion"""
        if emotion in self.video_list and self.video_list[emotion]:
            self.current_video_index = 0
            self.play_current_video()
        else:
            self.show_error(f"No videos found for {emotion} emotion")
    
    def play_current_video(self):
        """Play the current video"""
        if not self.video_list[self.current_emotion]:
            return
        
        video_path = self.video_list[self.current_emotion][self.current_video_index]
        
        # Create media
        media = self.instance.media_new(video_path)
        self.player.set_media(media)
        
        # Set window ID based on platform
        if platform.system() == "Windows":
            self.player.set_hwnd(self.player_canvas.winfo_id())
        else:
            self.player.set_xwindow(self.player_canvas.winfo_id())
        
        # Play video
        self.player.play()
        self.is_playing = True
        self.show_success(f"Playing: {os.path.basename(video_path)}")
    
    def toggle_play_pause(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.player.pause()
            self.is_playing = False
        else:
            self.player.play()
            self.is_playing = True
    
    def volume_up(self):
        """Increase volume"""
        self.volume = min(100, self.volume + 10)
        self.player.audio_set_volume(self.volume)
    
    def volume_down(self):
        """Decrease volume"""
        self.volume = max(0, self.volume - 10)
        self.player.audio_set_volume(self.volume)
    
    def toggle_mute(self):
        """Toggle mute"""
        self.is_muted = not self.is_muted
        self.player.audio_set_mute(self.is_muted)
    
    def stop_video(self):
        """Stop video playback"""
        self.player.stop()
        self.is_playing = False
    
    def mouse_callback(self, event):
        """Handle mouse clicks"""
        # Check if click is within capture button bounds
        button_x = self.capture_button.winfo_x()
        button_y = self.capture_button.winfo_y()
        button_width = self.capture_button.winfo_width()
        button_height = self.capture_button.winfo_height()
        
        if (button_x <= event.x <= button_x + button_width and
            button_y <= event.y <= button_y + button_height):
            self.capture_and_play()
    
    def next_video(self):
        """Play next video in the list"""
        if not self.current_emotion or not self.video_list[self.current_emotion]:
            self.show_error("No videos available")
            return
            
        # Stop current video
        self.player.stop()
        
        # Move to next video
        self.current_video_index = (self.current_video_index + 1) % len(self.video_list[self.current_emotion])
        
        # Play new video
        self.play_current_video()
        self.show_success("Playing next video")

    def prev_video(self):
        """Play previous video in the list"""
        if not self.current_emotion or not self.video_list[self.current_emotion]:
            self.show_error("No videos available")
            return
            
        # Stop current video
        self.player.stop()
        
        # Move to previous video
        self.current_video_index = (self.current_video_index - 1) % len(self.video_list[self.current_emotion])
        
        # Play new video
        self.play_current_video()
        self.show_success("Playing previous video")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'player'):
            self.player.stop()

if __name__ == "__main__":
    app = FaceEmotionDetection()
    app.run() 