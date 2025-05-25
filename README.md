# Face Emotion Detection with Music Player

A Python application that detects facial emotions in real-time using your webcam and plays corresponding songs based on the detected emotion.

## Features

- Real-time face detection
- Emotion recognition (Happy, Sad, Angry, Surprised, Neutral)
- Modern GUI interface with Tkinter
- Automatic song playback based on detected emotions
- Visual feedback with emotion colors and emojis

## Requirements

- Python 3.8 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-deduct.git
cd face-deduct
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python face_detection_ui.py
```

2. Click "Start Detection" to begin face detection
3. The application will detect your face and emotions in real-time
4. Use the buttons to:
   - Start/Stop detection
   - Capture and play a song for the current emotion
   - Stop detection and play the last detected emotion's song

## Project Structure

- `face_detection_ui.py` - Main GUI application
- `face_detection.py` - Core face detection and emotion recognition logic
- `requirements.txt` - List of required Python packages
- `known_faces/` - Directory for storing face recognition data

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for face detection
- MediaPipe for facial landmarks
- Tkinter for the GUI 