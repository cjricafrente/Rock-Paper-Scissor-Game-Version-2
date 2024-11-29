import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)

# Function to convert coordinates to YOLO format
def convert_to_yolo_format(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    return x_center, y_center, width, height

# Path to the validation image dataset
valid_folder = r'D:\\PythonProjects\\rps-yolov2\\annotation\\datasetsv2\\final\\valid'
output_folder = r'D:\\PythonProjects\\rps-yolov2\\annotation\\datasetsv2\\final\\valid'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Mapping of subfolder names to class IDs
class_mapping = {
    'c0': 0,  # rock
    'c1': 1,  # paper
    'c2': 2,  # scissors
}

# Iterate over the subfolders for the valid folder
for subfolder, class_id in class_mapping.items():
    subfolder_path = os.path.join(valid_folder, subfolder)
    output_subfolder_path = os.path.join(output_folder, subfolder)
    
    # Ensure the output subfolder exists
    os.makedirs(output_subfolder_path, exist_ok=True)
    
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith(('.jpg', '.png')):  # Check for image files
                image_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Error loading image: {image_path}")
                    continue

                print(f"Processing image: {image_path}")  # Debug print

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    print("Hands detected.")  # Debug print
                    with open(os.path.join(output_subfolder_path, filename.split('.')[0] + '.txt'), 'w') as file:
                        for hand_landmarks in results.multi_hand_landmarks:
                            x_min, y_min = img.shape[1], img.shape[0]
                            x_max, y_max = 0, 0

                            # Find bounding box around the hand
                            for landmark in hand_landmarks.landmark:
                                x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                                x_min, y_min = min(x, x_min), min(y, y_min)
                                x_max, y_max = max(x, x_max), max(y, y_max)

                            # Convert to YOLO format
                            x_center, y_center, width, height = convert_to_yolo_format(x_min, y_min, x_max - x_min, y_max - y_min, img.shape[1], img.shape[0])
                            file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                else:
                    print(f"No hands detected in {image_path}")  # Debug print

print("Annotations for the validation set created successfully.")
