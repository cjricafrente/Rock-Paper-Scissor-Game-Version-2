import cv2
import random
from ultralytics import YOLO
import time

# Load the YOLO model
model = YOLO('best.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Define class labels (rock, paper, scissors)
class_labels = ['rock', 'paper', 'scissors']

# Load gesture images
gesture_images = {}
for label in class_labels:
    gesture_images[label] = cv2.imread(f'icons/{label}.png')

# Function to determine the winner
def determine_winner(player_choice, ai_choice):
    if player_choice == ai_choice:
        return "Draw"
    elif (player_choice == 'rock' and ai_choice == 'scissors') or \
         (player_choice == 'paper' and ai_choice == 'rock') or \
         (player_choice == 'scissors' and ai_choice == 'paper'):
        return "You Win!"
    else:
        return "AI Wins!"

start_game = False
game_in_progress = False
countdown_start = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Resize the frame to improve display
    resized_frame = cv2.resize(frame, (640, 480))

    if not game_in_progress:
        # Draw player and AI squares, shifted downward to avoid overlap with start text
        cv2.rectangle(resized_frame, (30, 100), (310, 400), (255, 255, 255), 2)
        cv2.rectangle(resized_frame, (330, 100), (610, 400), (255, 255, 255), 2)
        cv2.putText(resized_frame, "Player", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(resized_frame, "AI", (470, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(resized_frame, "Press SPACE to start the game", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        # Display the countdown
        countdown = int(3 - (time.time() - countdown_start))
        if countdown > 0:
            cv2.putText(resized_frame, f"Get ready! {countdown}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Perform object detection
            results = model.predict(source=resized_frame, show=False, conf=0.8)

            # Initialize a list to store detected gestures and their confidences along with their y-coordinate
            detected_gestures = []

            # Check detections and collect gestures with their confidences and y-coordinate
            for result in results[0].boxes.data:
                _, _, _, _, conf, cls = result
                cls = int(cls)
                if conf > 0.8:  # Ensure confidence threshold
                    # Calculate the center of the bounding box (x, y)
                    x1, y1, x2, y2 = result[:4]
                    center_y = (y1 + y2) / 2  # Calculate the vertical center of the bounding box
                    
                    detected_gestures.append((class_labels[cls], conf, center_y))

            # Sort by y-coordinate (ascending) to get the gesture that is closest to the camera
            detected_gestures.sort(key=lambda x: x[2])

            # Choose the closest gesture with the highest confidence
            if detected_gestures:
                player_choice = detected_gestures[0][0]  # Choose the gesture with the closest center (topmost)
            else:
                player_choice = "None"

            # AI makes a random choice
            ai_choice = random.choice(class_labels)

            # Determine the winner
            winner = determine_winner(player_choice, ai_choice)

            # Display the result in the UI
            ai_image = gesture_images.get(ai_choice)

            if ai_image is not None:
                ai_image = cv2.resize(ai_image, (280, 280))
                
                # Overlay the AI image onto the frame
                resized_frame[50:330, 330:610] = ai_image

            # Display the text result
            cv2.putText(resized_frame, f"You: {player_choice}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(resized_frame, f"AI: {ai_choice}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(resized_frame, winner, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Reset the game state after displaying the result
            cv2.imshow("Rock Paper Scissors Game", resized_frame)
            cv2.waitKey(3000)
            game_in_progress = False

    # Display the frame
    cv2.imshow("Rock Paper Scissors Game", resized_frame)

    # Key press handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Press 'Space' to start the game
        start_game = True
        game_in_progress = True
        countdown_start = time.time()
    if key == ord('q'):  # Press 'q' to exit
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
