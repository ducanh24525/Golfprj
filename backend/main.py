import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms
from model import EventDetector, NormalizePose, ToTensor, Normalize

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="uploads"), name="static")

# Load YOLO model
# Assuming yolo11m-pose.pt is in the root directory
YOLO_MODEL_PATH = "../yolo11m-pose.pt"
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"Warning: could not load YOLO model at {YOLO_MODEL_PATH}. Error: {e}")
    yolo_model = None

# Load Golf Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EventDetector(
    pretrain=False,
    rgb_lstm_layers=1,
    rgb_lstm_hidden=256,
    pose_lstm_layers=1,
    pose_lstm_hidden=128,
    bidirectional=True,
    dropout=False
).to(device)

CHECKPOINT_PATH = "../swingnet_iter_1800.pth.tar"
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Golf EventDetector model loaded successfully!")
else:
    print(f"Warning: checkpoint {CHECKPOINT_PATH} not found.")

model.eval()

transform = transforms.Compose([
    NormalizePose(),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_poses(video_path):
    cap = cv2.VideoCapture(video_path)
    poses = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO on frame
        results = yolo_model(frame, verbose=False)
        if len(results) > 0 and results[0].keypoints is not None:
            # Get the first person's keypoints (assuming 1 person)
            # results[0].keypoints.data has shape (num_persons, 17, 3)
            kpts = results[0].keypoints.data[0].cpu().numpy() # (17, 3)
            
            # Map YOLO (17) to GolfDB (13)
            # GolfDB 13 points:
            # 0: Nose (0), 1: L-Shoulder (5), 2: R-Shoulder (6), 3: L-Elbow (7)
            # 4: R-Elbow (8), 5: L-Wrist (9), 6: R-Wrist (10), 7: L-Hip (11)
            # 8: R-Hip (12), 9: L-Knee (13), 10: R-Knee (14), 11: L-Ankle (15), 12: R-Ankle (16)
            
            mapped_kpts = np.zeros((13, 3), dtype=np.float32)
            mapping = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            for i, yolo_idx in enumerate(mapping):
                mapped_kpts[i] = kpts[yolo_idx]
            poses.append(mapped_kpts)
        else:
            poses.append(np.zeros((13, 3), dtype=np.float32))
            
    cap.release()
    return np.array(poses) # (T, 13, 3)

def extract_frames(video_path, target_size=224):
    cap = cv2.VideoCapture(video_path)
    images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size))
        images.append(frame)
    cap.release()
    return np.array(images)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    video_path = UPLOAD_DIR / file.filename
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # 1. Extract poses
    print("Extracting poses...")
    poses = extract_poses(str(video_path))
    
    # 2. Extract frames
    print("Extracting frames...")
    images = extract_frames(str(video_path))
    
    if len(images) == 0:
        return JSONResponse(status_code=400, content={"error": "Could not read video frames."})
        
    T = len(images)
    if len(poses) < T:
        # pad poses if needed
        pad = np.zeros((T - len(poses), 13, 3))
        poses = np.concatenate([poses, pad], axis=0)
    elif len(poses) > T:
        poses = poses[:T]

    sample = {'images': images, 'poses': poses}
    sample = transform(sample)
    
    images_t = sample['images'].to(device).unsqueeze(0) # (1, T, 3, H, W)
    poses_t = sample['poses'].to(device).unsqueeze(0)   # (1, T, 13, 3)
    
    seq_length = 64
    probs_all = []
    
    print(f"Running inference on {T} frames...")
    with torch.no_grad():
        batch_idx = 0
        while batch_idx * seq_length < T:
            start = batch_idx * seq_length
            end = min((batch_idx + 1) * seq_length, T)
            logits = model(images_t[:, start:end], poses_t[:, start:end])
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
            batch_idx += 1
            
    probs_all = np.concatenate(probs_all, axis=0) # (T, 9)
    
    # 3. Create annotated video
    print("Creating annotated video...")
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30
    
    out_filename = f"annotated_{os.path.splitext(file.filename)[0]}.mp4"
    out_path = UPLOAD_DIR / out_filename
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    
    event_names = ['Address', 'Toe-up', 'Mid-backswing', 'Top', 'Mid-downswing', 'Impact', 'Mid-follow-through', 'Finish']
    # Compute event time segments based on predictions
    event_segments = []
    current_event = None
    start_idx = 0
    for i in range(T):
        probs = probs_all[i]
        pred_cls = int(np.argmax(probs))
        pred_conf = float(probs[pred_cls])
        if pred_conf > 0.5 and pred_cls < len(event_names):
            event_name = event_names[pred_cls]
            if current_event != event_name:
                if current_event is not None:
                    end_idx = i - 1
                    event_segments.append({
                        "name": current_event,
                        "start_frame": start_idx,
                        "end_frame": end_idx,
                        "start_time_sec": start_idx / fps,
                        "end_time_sec": (end_idx + 1) / fps
                    })
                current_event = event_name
                start_idx = i
        else:
            if current_event is not None:
                end_idx = i - 1
                event_segments.append({
                    "name": current_event,
                    "start_frame": start_idx,
                    "end_frame": end_idx,
                    "start_time_sec": start_idx / fps,
                    "end_time_sec": (end_idx + 1) / fps
                })
                current_event = None
    if current_event is not None:
        end_idx = T - 1
        event_segments.append({
            "name": current_event,
            "start_frame": start_idx,
            "end_frame": end_idx,
            "start_time_sec": start_idx / fps,
            "end_time_sec": (end_idx + 1) / fps
        })
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx >= len(probs_all): break
        
        probs = probs_all[frame_idx]
        pred_cls = np.argmax(probs)
        pred_conf = probs[pred_cls]
        
        if pred_cls < 8 and pred_conf > 0.5:
            text_pred = f"{event_names[pred_cls]} ({pred_conf:.2f})"
            color_pred = (0, 255, 0)
        else:
            text_pred = "No Event"
            color_pred = (200, 200, 200)
            
        cv2.putText(frame, text_pred, (20, 40), font, 1, color_pred, 2, cv2.LINE_AA)
        
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print("Done processing!")
    
    # Format probabilities for the frontend
    probs_list = probs_all.tolist()
    
    return {
        "status": "success",
        "video_url": f"/static/{out_filename}",
        "probabilities": probs_list,
        "event_names": event_names,
        "event_segments": event_segments,
    }
