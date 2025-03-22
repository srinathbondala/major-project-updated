# import cv2
# import torch
# import onnxruntime as ort
# import numpy as np
# import os
# import json
# import tempfile
# import streamlit as st
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from collections import defaultdict, deque

# # --- Download Kinetics Labels ---
# LABELS_FILE = "kinetics_labels.json"

# def download_kinetics_labels():
#     labels_url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
#     if not os.path.exists(LABELS_FILE):
#         import requests
#         response = requests.get(labels_url)
#         labels = {i: label.strip() for i, label in enumerate(response.text.split("\n")) if label.strip()}
#         with open(LABELS_FILE, "w") as f:
#             json.dump(labels, f)

# download_kinetics_labels()

# with open(LABELS_FILE, "r") as f:
#     kinetics_labels = json.load(f)

# def get_action_label(action_index):
#     return kinetics_labels.get(str(action_index), "Unknown")

# # --- Set Device ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- Load YOLOv8 Model ---
# st.text("Loading YOLOv8 model...")
# with torch.no_grad():
#     yolo_model = YOLO("yolov8m.pt")  # Downloads weights if needed.
# yolo_model.to(device)

# # --- Initialize Deep SORT ---
# deep_sort = DeepSort(max_age=16)

# # --- Export or Load ONNX X3D-M Model ---
# ONNX_MODEL_PATH = "x3d_m.onnx"
# if not os.path.exists(ONNX_MODEL_PATH):
#     st.text("Exporting X3D-M model to ONNX format...")
#     model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
#     model.to(device)
#     model.eval()
#     dummy_input = torch.randn(1, 3, 16, 224, 224).to(device)
#     torch.onnx.export(model, dummy_input, ONNX_MODEL_PATH, opset_version=12)
    
# providers = ['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
# onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

# # --- Define Transformation Function ---
# def transform_crop(crop):
#     # Resize crop to 224x224, convert from BGR to RGB, and normalize.
#     resized = cv2.resize(crop, (224, 224))
#     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
#     tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3, 224, 224)
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#     return (tensor - mean) / std

# # --- Initialize Frame Queues for Each Track ---
# track_frame_queues = defaultdict(lambda: deque(maxlen=16))

# # --- Streamlit Sidebar: Tracking Options ---
# st.sidebar.header("Tracking Options")
# selected_track_id = st.sidebar.text_input("Enter DeepSORT Track ID to track (leave blank for all):", "")

# # --- Main Streamlit App ---
# st.title("Real-Time Human Activity Recognition")
# st.markdown("Upload a video file or use your webcam for live processing.")

# video_source = st.radio("Select Video Source:", ["Upload Video", "Live Camera"])

# if video_source == "Upload Video":
#     uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         tfile.write(uploaded_file.read())
#         video_path = tfile.name
#         cap = cv2.VideoCapture(video_path)
#     else:
#         cap = None
# else:
#     cap = cv2.VideoCapture(0)

# start_button = st.button("Start Recognition")

# if start_button and cap is not None:
#     stframe = st.empty()
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # --- YOLO Detection ---
#         results = yolo_model(frame, verbose=False)
#         detections = []
#         for result in results:
#             # Iterate through each detected box
#             for box in result.boxes.data.tolist():
#                 x1, y1, x2, y2, score, class_id = box
#                 # Person class in COCO is 0; threshold set to 0.3
#                 if int(class_id) == 0 and score > 0.3:
#                     detections.append(([x1, y1, x2, y2], score, None))

#         # --- Update Deep SORT Tracker ---
#         tracks = deep_sort.update_tracks(detections, frame=frame)
#         i=0
#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
#             track_id = track.track_id
#             # If a specific track id is provided, only process that one
#             if selected_track_id and str(track_id) != selected_track_id:
#                 continue

#             # x1, y1, x2, y2 = map(int,detections[i][0])
#             # i=i+1
#             x1, y1, x2, y2 = map(int, track.to_ltrb())
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Crop the detected person region and update the frame queue for this track
#             crop = frame[y1:y2, x1:x2]
#             if crop.size == 0:
#                 continue
#             track_frame_queues[track_id].append(crop)

#             # When 16 frames are collected for this track, perform action recognition
#             if len(track_frame_queues[track_id]) == 16:
#                 try:
#                     crops = list(track_frame_queues[track_id])
#                     tensors = [transform_crop(c) for c in crops]
#                     # Stack tensors along a new dimension: (3, 16, 224, 224)
#                     frames_tensor = torch.stack(tensors, dim=1)
#                     # Add batch dimension: (1, 3, 16, 224, 224)
#                     input_tensor = frames_tensor.unsqueeze(0).to(device)
#                     input_np = input_tensor.detach().cpu().numpy()
#                     ort_inputs = {onnx_session.get_inputs()[0].name: input_np}
#                     predictions = onnx_session.run(None, ort_inputs)[0]
#                     action_index = int(np.argmax(predictions, axis=1)[0])
#                     action_name = get_action_label(action_index)
#                     cv2.putText(frame, f'Action: {action_name}', (x1, y1 - 30),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#                 except Exception as e:
#                     st.error(f"Error in action recognition: {e}")

#         stframe.image(frame, channels="BGR")
#         frame_count += 1

#     cap.release()
#     st.success("Processing complete.")
import cv2
import torch
import onnxruntime as ort
import numpy as np
import os
import json
import tempfile
import streamlit as st
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque

### ----- Helper Functions and Setup ----- ###

# Download Kinetics labels for activity recognition.
LABELS_FILE = "kinetics_labels.json"
def download_kinetics_labels():
    labels_url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
    if not os.path.exists(LABELS_FILE):
        import requests
        response = requests.get(labels_url)
        labels = {i: label.strip() for i, label in enumerate(response.text.split("\n")) if label.strip()}
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f)
download_kinetics_labels()

with open(LABELS_FILE, "r") as f:
    kinetics_labels = json.load(f)
def get_action_label(action_index):
    return kinetics_labels.get(str(action_index), "Unknown")

# Letterbox: Resize image to fixed shape (e.g., 640x640) with padding while preserving aspect ratio.
def letterbox(im, new_shape=(640,640), color=(114,114,114), auto=True, scaleFill=False, scaleup=True):
    shape = im.shape[:2]  # (height, width)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

# Transformation for activity recognition: resize crop to 224x224 and normalize.
def transform_crop(crop):
    resized = cv2.resize(crop, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3,224,224)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std

# Compute Intersection-over-Union (IoU) between two boxes.
def compute_iou(box1, box2):
    # Both boxes in format: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2Area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    unionArea = box1Area + box2Area - interArea
    if unionArea == 0:
        return 0
    return interArea / unionArea

### ----- Model and Tracker Initialization ----- ###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv8 model for detection.
st.text("Loading YOLOv8 model...")
with torch.no_grad():
    yolo_model = YOLO("yolov8m.pt")  # Change to a different variant if needed.
yolo_model.to(device)

# Initialize DeepSORT tracker.
deep_sort = DeepSort(max_age=16)

# Export or load ONNX X3D-M model for activity recognition.
ONNX_MODEL_PATH = "x3d_m.onnx"
if not os.path.exists(ONNX_MODEL_PATH):
    st.text("Exporting X3D-M model to ONNX format...")
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, 16, 224, 224).to(device)
    torch.onnx.export(model, dummy_input, ONNX_MODEL_PATH, opset_version=12)
providers = ['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

# Initialize frame queues for each track.
track_frame_queues = defaultdict(lambda: deque(maxlen=16))

### ----- Streamlit UI Setup ----- ###

st.sidebar.header("Tracking Options")
selected_track_id = st.sidebar.text_input("Enter DeepSORT Track ID to track (leave blank for all):", "")

st.title("Real-Time Human Activity Recognition")
st.markdown("Upload a video or use your webcam. The app uses YOLOv8 for detection, DeepSORT for tracking, "
            "and an ONNX-exported X3D-M model for activity recognition. The bounding boxes for drawing/cropping come from YOLO.")

video_source = st.radio("Select Video Source:", ["Upload Video", "Live Camera"])
if video_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        cap = cv2.VideoCapture(video_path)
    else:
        cap = None
else:
    cap = cv2.VideoCapture(0)

start_button = st.button("Start Recognition")
FRAME_SAMPLE_RATE = 2

### ----- Main Processing Loop ----- ###

if start_button and cap is not None:
    st_frame = st.empty()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SAMPLE_RATE != 0:
            continue 
        # Get original frame dimensions.
        h, w, _ = frame.shape

        # YOLO detection using letterbox preprocessing.
        img_letterbox, scale, pad = letterbox(frame, new_shape=(640,640))
        results = yolo_model(img_letterbox, verbose=False)
        raw_detections = []  # YOLO detections.
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # boxes in letterbox image coordinates.
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, score, cls in zip(boxes, scores, classes):
                if int(cls) == 0 and score > 0.3:
                    # Scale detection boxes back to original frame coordinates.
                    box[0] = (box[0] - pad[0]) / scale
                    box[1] = (box[1] - pad[1]) / scale
                    box[2] = (box[2] - pad[0]) / scale
                    box[3] = (box[3] - pad[1]) / scale
                    box[0] = np.clip(box[0], 0, w-1)
                    box[1] = np.clip(box[1], 0, h-1)
                    box[2] = np.clip(box[2], 0, w-1)
                    box[3] = np.clip(box[3], 0, h-1)
                    raw_detections.append((box.tolist(), float(score), None))
        
        # Update DeepSORT tracker.
        tracks = deep_sort.update_tracks(raw_detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            if selected_track_id and str(track_id) != selected_track_id:
                continue

            # Get DeepSORT predicted box.
            track_box = list(map(int, track.to_ltrb()))
            best_iou = 0
            best_box = None
            # Associate YOLO detection with track using IoU.
            for det in raw_detections:
                det_box = det[0]
                iou = compute_iou(track_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_box = det_box
            final_box = best_box if best_box is not None and best_iou > 0.3 else track_box
            final_box = list(map(int, final_box))
            
            # Draw the final bounding box and track ID.
            cv2.rectangle(frame, (final_box[0], final_box[1]), (final_box[2], final_box[3]), (0,255,0), 2)
            cv2.putText(frame, f'ID: {track_id}', (final_box[0], final_box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Crop the region using YOLO's detection boundaries.
            crop = frame[final_box[1]:final_box[3], final_box[0]:final_box[2]]
            if crop.size == 0:
                continue
            track_frame_queues[track_id].append(crop)

            # When 16 frames are collected, perform activity recognition.
            if len(track_frame_queues[track_id]) == 16:
                try:
                    crops = list(track_frame_queues[track_id])
                    tensors = [transform_crop(c) for c in crops]
                    frames_tensor = torch.stack(tensors, dim=1)  # shape: (3, 16, 224, 224)
                    input_tensor = frames_tensor.unsqueeze(0).to(device)  # shape: (1, 3, 16, 224, 224)
                    input_np = input_tensor.detach().cpu().numpy()
                    ort_inputs = {onnx_session.get_inputs()[0].name: input_np}
                    predictions = onnx_session.run(None, ort_inputs)[0]
                    action_index = int(np.argmax(predictions, axis=1)[0])
                    action_name = get_action_label(action_index)
                    cv2.putText(frame, f'Action: {action_name}', (final_box[0], final_box[1]-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                except Exception as e:
                    st.error(f"Error in activity recognition: {e}")

        st_frame.image(frame, channels="BGR")
        # frame_count += 1

    cap.release()
    st.success("Processing complete.")

if __name__ == "__main__":
    pass
