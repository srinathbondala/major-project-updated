# import cv2
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import torchvision.transforms as transforms
# from collections import defaultdict, deque
# import os
# import json

# LABELS_FILE = "kinetics_labels.json"

# def download_kinetics_labels():
#     labels_url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
    
#     # Fetch and save the labels if the file does not exist
#     if not os.path.exists(LABELS_FILE):
#         import requests
#         response = requests.get(labels_url)
#         labels = {i: label.strip() for i, label in enumerate(response.text.split("\n")) if label}
        
#         with open(LABELS_FILE, "w") as f:
#             json.dump(labels, f)

# # Ensure labels are downloaded
# download_kinetics_labels()

# # Load the labels from the JSON file
# with open(LABELS_FILE, "r") as f:
#     kinetics_labels = json.load(f)

# def get_action_label(action_index):
#     return kinetics_labels.get(str(action_index), "Unknown")



# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load YOLOv8 model for person detection
# yolo_model = YOLO("yolov8n.pt")  # Change to 'yolov8s.pt' for better accuracy
# yolo_model.to(device)

# # Initialize DeepSORT Tracker
# deep_sort = DeepSort(max_age=15)

# # Path to X3D-M model checkpoint
# MODEL_PATH = "x3d_m.pth"

# # Load X3D-M Model
# def load_x3d_model():
#     model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
#     torch.save(model.state_dict(), MODEL_PATH)  # Cache locally
#     return model

# # Try loading the model or download if missing
# if os.path.exists(MODEL_PATH):
#     print("Loading existing X3D-M model...")
#     x3d_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=False)
#     x3d_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# else:
#     print("Downloading X3D-M model...")
#     x3d_model = load_x3d_model()

# x3d_model.to(device)
# x3d_model.eval()

# # Open video file or webcam
# video_path = 0  # Use 0 for webcam, or provide video file path
# cap = cv2.VideoCapture('example1.mp4')

# # Store frames for activity recognition
# track_frame_queues = defaultdict(lambda: deque(maxlen=16))

# # Preprocessing for X3D Model
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  # Resize for X3D
#     transforms.ToTensor()
# ])

# plt.ion()  # Enable interactive mode for real-time updating

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLO detection
#     results = yolo_model(frame, verbose=False)  
#     detections = []

#     for result in results:
#         for box in result.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = box
#             if int(class_id) == 0 and score > 0.3:  # Class 0 = 'person'
#                 detections.append(([x1, y1, x2, y2], score, None))

#     # Update DeepSORT tracker
#     tracks = deep_sort.update_tracks(detections, frame=frame)

#     for track in tracks:
#         if not track.is_confirmed():
#             continue
#         track_id = track.track_id
#         x1, y1, x2, y2 = map(int, track.to_ltrb())

#         # Draw bounding box & ID
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Store frames for action recognition
#         person_crop = frame[y1:y2, x1:x2]
#         if person_crop.size > 0:
#             track_frame_queues[track_id].append(person_crop)

#         if len(track_frame_queues[track_id]) == 16:
#             try:
#                 # Convert list of frames into a tensor: (16, 3, 182, 182)
#                 frames = torch.stack([transform(track_frame_queues[track_id][i]) for i in range(16)])

#                 # Reshape to match X3D input format: (1, 3, 16, 182, 182)
#                 frame_tensor = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)

#                 with torch.no_grad():
#                     predictions = x3d_model(frame_tensor)
#                     # action_label = torch.argmax(predictions, dim=1).item()
#                     action_label = torch.argmax(predictions, dim=1).item()
#                 action_name = get_action_label(action_label)  # Get the actual action name
#                 cv2.putText(frame, f'Action: {action_name}', (x1, y1 - 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#                 # cv2.putText(frame, f'Action: {action_label}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#             except Exception as e:
#                 print(f"Error in action recognition: {e}")

#     # Convert BGR to RGB for Matplotlib
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     plt.imshow(rgb_frame)
#     plt.axis("off")
#     plt.pause(0.001)
#     plt.clf()

# cap.release()
# plt.ioff()  # Disable interactive mode
# plt.show()  # Ensure last frame is shown
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchvision.transforms as transforms
from collections import defaultdict, deque
import os
import json

LABELS_FILE = "kinetics_labels.json"

def download_kinetics_labels():
    labels_url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
    
    if not os.path.exists(LABELS_FILE):
        import requests
        response = requests.get(labels_url)
        labels = {i: label.strip() for i, label in enumerate(response.text.split("\n")) if label}
        
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f)

download_kinetics_labels()

with open(LABELS_FILE, "r") as f:
    kinetics_labels = json.load(f)

def get_action_label(action_index):
    return kinetics_labels.get(str(action_index), "Unknown")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("yolov8n.pt")
yolo_model.to(device)

deep_sort = DeepSort(max_age=15)

MODEL_PATH = "x3d_m.pth"

def load_x3d_model():
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
    torch.save(model.state_dict(), MODEL_PATH)
    return model

if os.path.exists(MODEL_PATH):
    print("Loading existing X3D-M model...")
    x3d_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=False)
    x3d_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print("Downloading X3D-M model...")
    x3d_model = load_x3d_model()

x3d_model.to(device)
x3d_model.eval()

video_path = 0
cap = cv2.VideoCapture('example.mp4')

track_frame_queues = defaultdict(lambda: deque(maxlen=16))

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

plt.ion()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, verbose=False)  
    detections = []

    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if int(class_id) == 0 and score > 0.3:
                detections.append(([x1, y1, x2, y2], score, None))

    tracks = deep_sort.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size > 0:
            track_frame_queues[track_id].append(person_crop)

        if len(track_frame_queues[track_id]) == 16:
            try:
                frames = torch.stack([transform(track_frame_queues[track_id][i]) for i in range(16)])
                frame_tensor = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)

                with torch.no_grad():
                    predictions = x3d_model(frame_tensor)
                    action_label = torch.argmax(predictions, dim=1).item()
                action_name = get_action_label(action_label)
                cv2.putText(frame, f'Action: {action_name}', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            except Exception as e:
                print(f"Error in action recognition: {e}")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.imshow(rgb_frame)
    plt.axis("off")
    plt.pause(0.001)
    plt.clf()

cap.release()
plt.ioff()
plt.show()
