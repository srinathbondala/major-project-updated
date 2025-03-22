import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
import os
import json

# --- Download Kinetics Labels ---
LABELS_FILE = "kinetics_labels.json"

def download_kinetics_labels():
    labels_url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
    if not os.path.exists(LABELS_FILE):
        import requests
        response = requests.get(labels_url)
        # Create a dictionary mapping index -> label (ignoring empty lines)
        labels = {i: label.strip() for i, label in enumerate(response.text.split("\n")) if label.strip()}
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f)

download_kinetics_labels()

with open(LABELS_FILE, "r") as f:
    kinetics_labels = json.load(f)

def get_action_label(action_index):
    return kinetics_labels.get(str(action_index), "Unknown")

# --- Set Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load YOLO Model with no_grad ---
print("Loading YOLOv8 model...")
with torch.no_grad():
    yolo_model = YOLO("yolov8m.pt")  # Auto-downloads weights if needed.
yolo_model.to(device)

# --- Initialize Deep SORT ---
deep_sort = DeepSort(max_age=16)

# --- Load the X3D-M Action Recognition Model & Script It for Speed ---
MODEL_PATH = "x3d_m.pth"

def load_and_script_x3d():
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
    model.to(device)  # Move model to GPU
    # Save state dict (optional)
    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    # Script the model for faster inference using an input tensor on the correct device
    scripted_model = torch.jit.trace(model, torch.randn(1, 3, 16, 224, 224).to(device))
    return scripted_model

if os.path.exists(MODEL_PATH):
    print("Loading existing X3D-M model...")
    x3d_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=False)
    x3d_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    x3d_model.to(device)  # Ensure the model is on the same device as the input
    x3d_model.eval()
    x3d_model = torch.jit.trace(x3d_model, torch.randn(1, 3, 16, 224, 224).to(device))
else:
    print("Downloading and scripting X3D-M model...")
    x3d_model = load_and_script_x3d()
x3d_model.to(device)

# Optionally, if your device supports fp16, convert the model:
# x3d_model.half()

# --- Open Video File ---
video_path = "example.mp4"  # Ensure this file is in your filesystem.
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Error reading video file.")
    cap.release()
    exit()

height, width = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- Initialize Frame Queues for Each Track ---
track_frame_queues = defaultdict(lambda: deque(maxlen=16))

# --- Define Optimized Transformation Function ---
def transform_crop(crop):
    # Resize crop to (224,224) using cv2.resize
    resized = cv2.resize(crop, (224, 224))
    # Convert BGR to RGB and scale to [0,1]
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
    # Convert to tensor and rearrange dimensions to (C, H, W)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    # Normalize using ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor

frame_count = 0

print("Starting video processing...")
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- YOLO Detection ---
        results = yolo_model(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = box
                # Person class in COCO is 0; use a threshold of 0.3
                if int(class_id) == 0 and score > 0.3:
                    detections.append(([x1, y1, x2, y2], score, None))

        # --- Update Deep SORT Tracker ---
        tracks = deep_sort.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Crop and transform detected person region
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # Append raw crop to the queue for this track
            track_frame_queues[track_id].append(crop)

            # When we have 16 frames, perform action recognition in batch
            if len(track_frame_queues[track_id]) == 16:
                try:
                    # Process all 16 crops at once
                    crops = list(track_frame_queues[track_id])
                    tensors = [transform_crop(c) for c in crops]
                    frames_tensor = torch.stack(tensors, dim=1)  # Shape: (3, 16, 224, 224)
                    # Add batch dimension: (1, 3, 16, 224, 224)
                    input_tensor = frames_tensor.unsqueeze(0).to(device)
                    # Optionally use half precision: input_tensor = input_tensor.half()
                    predictions = x3d_model(input_tensor)
                    action_index = torch.argmax(predictions, dim=1).item()
                    action_name = get_action_label(action_index)
                    cv2.putText(frame, f'Action: {action_name}', (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Error in action recognition: {e}")

        out.write(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print("Processing complete. Output saved to output.mp4")
