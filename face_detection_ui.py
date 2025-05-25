import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
from face_detection import FaceDetector
import webbrowser

class ModernButton(ttk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(style='Modern.TButton')
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.original_style = self.cget('style')

    def on_enter(self, e):
        self.configure(background=self.master.master.hover_color)

    def on_leave(self, e):
        self.configure(background=self.master.master.button_color)

class ModernLabelFrame(ttk.LabelFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(style='Modern.TLabelframe')

class FaceDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Emotion Detection")
        self.root.geometry("1200x800")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use clam theme as base
        
        # Configure colors
        self.bg_color = "#1E1E1E"  # Dark background
        self.accent_color = "#00BFFF"  # Deep Sky Blue
        self.text_color = "#FFFFFF"  # White
        self.button_color = "#2C3E50"  # Dark blue-gray
        self.hover_color = "#3498DB"  # Bright blue
        self.success_color = "#2ECC71"  # Emerald
        self.warning_color = "#E74C3C"  # Red
        
        # Configure styles
        self.style.configure('Modern.TButton',
            background=self.button_color,
            foreground=self.text_color,
            font=('Segoe UI', 12, 'bold'),
            padding=15,
            borderwidth=0,
            relief="flat")
            
        self.style.configure('Modern.TLabelframe',
            background=self.bg_color,
            foreground=self.text_color,
            font=('Segoe UI', 12, 'bold'),
            borderwidth=2,
            relief="solid")
        
        self.style.configure('Modern.TLabelframe.Label',
            background=self.bg_color,
            foreground=self.text_color,
            font=('Segoe UI', 12, 'bold'))
        
        self.style.configure('Modern.TLabel',
            background=self.bg_color,
            foreground=self.text_color,
            font=('Segoe UI', 12))
            
        self.style.configure('Modern.TFrame',
            background=self.bg_color)
        
        # Configure root window
        self.root.configure(bg=self.bg_color)
        
        # Initialize face detector
        self.detector = FaceDetector()
        self.is_running = False
        self.current_emotion = None
        self.emotion_colors = {
            'happy': '#FFD700',    # Gold
            'sad': '#4169E1',      # Royal Blue
            'angry': '#FF4500',    # Orange Red
            'surprise': '#FFA500', # Orange
            'neutral': '#FFFFFF'   # White
        }

        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, style='Modern.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

        # Create left panel for video feed
        self.video_frame = ModernLabelFrame(self.main_frame, text="Video Feed")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.video_label = ttk.Label(self.video_frame, style='Modern.TLabel')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Create right panel for controls and info
        self.control_frame = ModernLabelFrame(self.main_frame, text="Controls")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=15, pady=15)

        # Add start/stop button with icon
        self.start_button = ModernButton(
            self.control_frame,
            text="▶ Start Detection",
            command=self.toggle_detection
        )
        self.start_button.pack(pady=15, padx=20, fill=tk.X)

        # Add stop and play button with icon
        self.stop_play_button = ModernButton(
            self.control_frame,
            text="⏹ Stop & Play Last Emotion",
            command=self.stop_and_play,
            state=tk.DISABLED
        )
        self.stop_play_button.pack(pady=15, padx=20, fill=tk.X)

        # Add capture button with icon
        self.capture_button = ModernButton(
            self.control_frame,
            text="📸 Capture & Play Song",
            command=self.capture_and_play,
            state=tk.DISABLED
        )
        self.capture_button.pack(pady=15, padx=20, fill=tk.X)

        # Add emotion display with enhanced styling
        self.emotion_frame = ModernLabelFrame(self.control_frame, text="Current Emotion")
        self.emotion_frame.pack(fill=tk.X, padx=20, pady=15)
        
        self.emotion_label = ttk.Label(
            self.emotion_frame,
            text="No emotion detected",
            font=("Segoe UI", 28, "bold"),
            style='Modern.TLabel'
        )
        self.emotion_label.pack(pady=20)

        # Add emotion colors legend with enhanced styling
        self.legend_frame = ModernLabelFrame(self.control_frame, text="Emotion Colors")
        self.legend_frame.pack(fill=tk.X, padx=20, pady=15)
        
        emotion_emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'surprise': '😲',
            'neutral': '😐'
        }
        
        for emotion, color in self.emotion_colors.items():
            frame = ttk.Frame(self.legend_frame, style='Modern.TFrame')
            frame.pack(fill=tk.X, pady=8)
            
            emoji_label = ttk.Label(
                frame,
                text=emotion_emojis[emotion],
                font=("Segoe UI", 24),
                style='Modern.TLabel'
            )
            emoji_label.pack(side=tk.LEFT, padx=15)
            
            color_label = ttk.Label(
                frame,
                text="●",
                foreground=color,
                font=("Segoe UI", 24),
                style='Modern.TLabel'
            )
            color_label.pack(side=tk.LEFT, padx=5)
            
            emotion_label = ttk.Label(
                frame,
                text=emotion.capitalize(),
                font=("Segoe UI", 14),
                style='Modern.TLabel'
            )
            emotion_label.pack(side=tk.LEFT)

        # Add status bar with modern style
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            style='Modern.TLabel',
            padding=(10, 5)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def toggle_detection(self):
        if not self.is_running:
            # Check if camera is opened
            if not self.detector.cap.isOpened():
                self.status_var.set("Error: Could not open webcam. Is it in use by another app?")
                return
            self.is_running = True
            self.start_button.configure(text="Stop Detection")
            self.stop_play_button.configure(state=tk.NORMAL)
            self.capture_button.configure(state=tk.NORMAL)
            self.status_var.set("Detection running...")
            self.start_detection()
        else:
            self.is_running = False
            self.start_button.configure(text="Start Detection")
            self.stop_play_button.configure(state=tk.DISABLED)
            self.capture_button.configure(state=tk.DISABLED)
            self.status_var.set("Detection stopped")

    def start_detection(self):
        def update_frame():
            if not self.is_running:
                return

            ret, frame = self.detector.cap.read()
            if ret:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces and emotions
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Extract face ROI for emotion detection
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Detect emotions
                    emotions = self.detector.detect_emotion(face_roi)
                    
                    # Get top emotion
                    top_emotion = max(emotions.items(), key=lambda x: x[1])
                    self.current_emotion = top_emotion[0]
                    
                    # Update emotion label
                    self.emotion_label.configure(
                        text=f"{self.current_emotion.capitalize()}\n{int(top_emotion[1]*100)}%",
                        foreground=self.emotion_colors.get(self.current_emotion, 'black')
                    )

                # Convert frame to PhotoImage
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                
                # Update video label
                self.video_label.configure(image=photo)
                self.video_label.image = photo

            # Schedule next update
            self.root.after(10, update_frame)

        # Start detection in a separate thread
        threading.Thread(target=update_frame, daemon=True).start()

    def capture_and_play(self):
        if self.current_emotion:
            song_url = self.detector.emotion_songs.get(self.current_emotion)
            if song_url:
                self.status_var.set(f"Opening song for {self.current_emotion} emotion...")
                webbrowser.open(song_url)
            else:
                self.status_var.set("No song found for this emotion")
        else:
            self.status_var.set("No emotion detected")

    def stop_and_play(self):
        if self.current_emotion:
            self.is_running = False
            self.start_button.configure(text="Start Detection")
            self.stop_play_button.configure(state=tk.DISABLED)
            self.capture_button.configure(state=tk.DISABLED)
            self.status_var.set(f"Playing song for last detected emotion: {self.current_emotion}")
            
            # Play the song for the last detected emotion
            song_url = self.detector.emotion_songs.get(self.current_emotion)
            if song_url:
                webbrowser.open(song_url)
            else:
                self.status_var.set("No song found for this emotion")
        else:
            self.status_var.set("No emotion was detected before stopping")

def main():
    root = tk.Tk()
    app = FaceDetectionUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 