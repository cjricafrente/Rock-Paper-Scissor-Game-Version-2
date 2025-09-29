## About

rpsv2 is an interactive Rock, Paper, Scissors game that uses YOLO (for object detection / annotation) combined with MediaPipe (for hand tracking / gesture detection) to recognize your hand gestures in real time and pit them against the computer.

## Features

Real-time gesture detection (rock / paper / scissors) using MediaPipe hands.

YOLO-based annotation / object detection pipeline integration.

Visual overlay of detection / classification results.

Simple game logic: computer makes a random choice, your gesture is matched, and result is displayed (win / lose / draw).

Modular design (you can tweak models, detection thresholds, etc.).

## Requirements

Python 3.7+

OpenCV

MediaPipe

YOLO / object detection dependencies (depending on which YOLO version you use: KD, YOLOv5, YOLOv8, etc.)

(Possibly) PyTorch or other deep learning libraries (depending on how YOLO is integrated)

## Installation

Clone the repository:

git clone https://github.com/cjricafrente/rpsv2.git
cd rpsv2


Install required Python packages (example using requirements.txt if present):

pip install -r requirements.txt


If there's no requirements.txt, you might need to manually install:

pip install opencv-python mediapipe  # plus any YOLO / DL libraries


(Optional) Download any model weights or YOLO configuration files the project uses, and place them in the appropriate folder (e.g. model/).

## Usage

Run the main script (or whichever entry point the repository uses). For example:

python main.py


or

python play.py


Then:

Allow camera access (if asked).

The application will capture video frames and detect your hand gesture.

The computer picks a gesture at random.

The result is shown overlayed on the video (you win / draw / lose).

Use a keyboard key (e.g. Esc) to quit.

(Adjust the above instructions depending on what the repository’s actual script names and behavior are.)

## Project Structure

Here’s a typical layout (adjust according to actual repo):

rpsv2/
├── model/                 # YOLO / ML model weights / configs
├── src/                   # source code (detection, game logic)
│   ├── yolo_detector.py
│   ├── mediapipe_gesture.py
│   ├── game_logic.py
│   └── main.py
├── requirements.txt
├── README.md
└── (maybe) assets/         # images, sample data, etc.


You can update this section to show the real folder/file setup.

## How It Works

Capture: The camera feed is captured using OpenCV.

YOLO Detection (optional or combined): YOLO model detects objects / regions of interest in the frame.

MediaPipe Hand Tracking / Landmarks: For detected region(s) or full frame, MediaPipe extracts hand landmarks.

Gesture Classification: Based on landmark positions (e.g. finger angles, tip positions), the code classifies the gesture as Rock / Paper / Scissors (or Unknown).

Game Logic: The computer randomly selects its move. The player’s detected gesture is matched, and result is determined (win / draw / lose).

Display / Overlay: The result, gesture icons, annotation boxes, etc. are overlayed onto the video frame and shown in real time.

Feel free to expand this “How It Works” section with formulas, diagrams, or pseudocode.

## Future Improvements

Improve gesture recognition accuracy (e.g. more robust toward lighting, background)

Add scorekeeping (best of N, cumulative score)

Add multiplayer support (local / network)

Support more gestures (like “Lizard”, “Spock”)

Add UI elements (buttons, menus)

Optimize performance (fps, detection speed)

Train a custom YOLO model to detect hand region more robustly
