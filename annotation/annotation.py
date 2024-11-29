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

# Path to your image dataset
image_folder = r'D:\\PythonProjects\\rps-yolov2\\annotation\\datasetsv2\\final\\train'
output_folder = r'D:\\PythonProjects\\rps-yolov2\\annotation\\annotaited-datasets'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Mapping of subfolder names to class IDs
class_mapping = {
    'c0': 0,  # rock
    'c1': 1,  # paper (if exists)
    'c2': 2,  # scissors (if exists)
}

# Iterate through the subfolders and generate annotations
for subfolder, class_id in class_mapping.items():
    subfolder_path = os.path.join(image_folder, subfolder)
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith(('.jpg', '.png')):
                image_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Error loading image: {image_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    with open(os.path.join(output_folder, filename.split('.')[0] + '.txt'), 'w') as file:
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

print("Annotations created successfully.")
