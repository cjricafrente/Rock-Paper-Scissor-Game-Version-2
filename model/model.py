import cv2
import random
import time
import numpy as np
from ultralytics import YOLO

class RockPaperScissorsGame:
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)
        self.class_labels = ['rock', 'paper', 'scissors']
        self.gesture_images = self._load_gesture_icons()
        self.start_game = False
        self.game_in_progress = False
        self.countdown_start = 0

        self.COLORS = {
            'background': (30, 30, 30),
            'rectangle': (100, 100, 100),
            'text_primary': (255, 255, 255),
            'text_highlight': (0, 255, 255),
            'win': (0, 255, 0),
            'lose': (0, 0, 255)
        }

        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    def _load_gesture_icons(self):
        """Load gesture icons."""
        gesture_images = {}
        for label in self.class_labels:
            img = cv2.imread(f'icons/{label}.png', cv2.IMREAD_UNCHANGED)
            if img is not None:
                img = cv2.resize(img, (280, 200))  # Adjust size for better visibility
                gesture_images[label] = img
        return gesture_images

    def determine_winner(self, player_choice, ai_choice):
        if player_choice == ai_choice:
            return "Draw", self.COLORS['text_highlight']
        elif (
            (player_choice == 'rock' and ai_choice == 'scissors') or
            (player_choice == 'paper' and ai_choice == 'rock') or
            (player_choice == 'scissors' and ai_choice == 'paper')
        ):
            return "You Win!", self.COLORS['win']
        else:
            return "AI Wins!", self.COLORS['lose']

    def overlay_transparent_image(self, background, overlay, x_offset, y_offset):
        if overlay.shape[2] == 4:
            b, g, r, a = cv2.split(overlay)
            overlay = cv2.merge((b, g, r))
            mask = a / 255.0
            for c in range(3):
                background[y_offset:y_offset+overlay.shape[0], 
                          x_offset:x_offset+overlay.shape[1], c] = \
                    background[y_offset:y_offset+overlay.shape[0], 
                               x_offset:x_offset+overlay.shape[1], c] * (1 - mask) + \
                    overlay[:, :, c] * mask
        else:
            background[y_offset:y_offset+overlay.shape[0], 
                       x_offset:x_offset+overlay.shape[1]] = overlay
        return background

    def draw_initial_ui(self, frame, camera_frame):
        """Draw initial UI with camera frame in the Player section."""
        frame[:] = self.COLORS['background']
        camera_frame = cv2.resize(camera_frame, (280, 300))
        cv2.rectangle(frame, (30, 100), (310, 400), self.COLORS['rectangle'], 2)
        cv2.rectangle(frame, (330, 100), (610, 400), self.COLORS['rectangle'], 2)
        cv2.putText(frame, "Player", (100, 90), self.FONT, 0.8, self.COLORS['text_primary'], 2)
        cv2.putText(frame, "AI", (470, 90), self.FONT, 0.8, self.COLORS['text_primary'], 2)
        frame[100:400, 30:310] = camera_frame
        cv2.putText(frame, "Press SPACE to start", (150, 450), self.FONT, 0.8, self.COLORS['text_highlight'], 2)
        return frame

    def run_game(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, camera_frame = cap.read()
            if not ret:
                print("Error reading frame.")
                break

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            if not self.game_in_progress:
                frame = self.draw_initial_ui(frame, camera_frame)
            else:
                frame = self.run_game_round(frame, camera_frame)

            cv2.imshow("Rock Paper Scissors AI", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                self.start_game = True
                self.game_in_progress = True
                self.countdown_start = time.time()
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_game_round(self, frame, camera_frame):
        """Execute a single game round."""
        # Resize and place the camera feed in the "Player" section
        camera_frame = cv2.resize(camera_frame, (280, 300))
        frame[100:400, 30:310] = camera_frame

        # Countdown logic
        countdown = int(3 - (time.time() - self.countdown_start))
        if countdown > 0:
        # Display the countdown on the screen
            cv2.putText(frame, f"Get ready! {countdown}", (200, 50),
                    self.FONT, 1, self.COLORS['win'], 2)
            return frame

        # Perform object detection to identify the player's gesture
        results = self.model.predict(source=camera_frame, show=False, conf=0.8)
        detected_gestures = self._get_detected_gestures(results)

        # Determine player and AI choices
        player_choice = detected_gestures[0][0] if detected_gestures else "None"
        ai_choice = random.choice(self.class_labels)

        # Determine the winner
        result, result_color = self.determine_winner(player_choice, ai_choice)

        # Resize the AI's gesture image to fit its square
        if ai_choice in self.gesture_images:
            ai_image = cv2.resize(self.gesture_images[ai_choice], (280, 300))
            frame = self.overlay_transparent_image(
                frame,
                ai_image,
                330, 100
            )

        # Draw the results horizontally at the bottom
        result_text = f"You: {player_choice}   AI: {ai_choice}   Result: {result}"
        cv2.putText(frame, result_text, (50, 450),
            self.FONT, 0.7, result_color, 2)

        # Pause for 3 seconds to display results
        cv2.imshow("Rock Paper Scissors AI", frame)
        cv2.waitKey(3000)  # Pause for 3 seconds

        # Reset game state for the next round
        self.game_in_progress = False
        return frame
    
    def _get_detected_gestures(self, results):
        """Extract and sort detected gestures from YOLO results."""
        detected_gestures = []
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result
            cls = int(cls)
            if conf > 0.8:  # Only consider gestures with high confidence
                center_y = (y1 + y2) / 2  # Calculate center for sorting
                detected_gestures.append((self.class_labels[cls], conf, center_y))

        # Sort by y-coordinate to get the closest gesture
        return sorted(detected_gestures, key=lambda x: x[2]) if detected_gestures else []




def main():
    game = RockPaperScissorsGame()
    game.run_game()

if __name__ == "__main__":
    main()
