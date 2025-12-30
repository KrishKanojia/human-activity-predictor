import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np
import os
import requests
from collections import Counter
import tempfile
import time

# ----------------------------- Setup -----------------------------
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'logistic_model.joblib')
scaler_path = os.path.join(base_path, 'std_scaler.joblib')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

labels = {0: 'WALKING', 1: 'WALKING_UPSTAIRS', 2: 'WALKING_DOWNSTAIRS',
          3: 'SITTING', 4: 'STANDING', 5: 'LAYING'}

# Download Pose Landmarker model if missing
model_file = os.path.join(base_path, 'pose_landmarker_lite.task')
model_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'

if not os.path.exists(model_file):
    with st.spinner("Downloading Pose Landmarker model (~30MB)..."):
        response = requests.get(model_url)
        with open(model_file, 'wb') as f:
            f.write(response.content)

# MediaPipe Setup
BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_file),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
)

pose_landmarker = PoseLandmarker.create_from_options(options)

POSE_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(17,19),(19,15),(15,21),
    (12,14),(14,16),(16,18),(18,20),(20,16),(16,22),
    (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32),(32,28)
])

def draw_landmarks(image, landmarks):
    h, w, _ = image.shape
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            cv2.line(image, (int(start.x * w), int(start.y * h)),
                     (int(end.x * w), int(end.y * h)), (255, 255, 255), 3)

# ----------------------------- Streamlit UI -----------------------------
st.title("📸 🎥 Human Activity Recognition")

uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4", "mov", "avi"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_type = uploaded_file.type

    if file_type.startswith('image'):
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = pose_landmarker.detect_for_video(mp_image, timestamp_ms=0)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            features = np.zeros((1, 561))
            features[0, :len(landmark_array)] = landmark_array
            prediction = model.predict(scaler.transform(features))[0]
            st.subheader(f"🔍 Detected Activity: **{labels.get(prediction)}**")
            annotated = image.copy()
            draw_landmarks(annotated, landmarks)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    elif file_type.startswith('video'):
        # Use a context manager for the temporary file to ensure it's handled correctly
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
            tfile.write(file_bytes)
            video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use 'avc1' codec for browser compatibility
        out_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        out = cv2.VideoWriter(out_temp.name, fourcc, fps, (width, height))

        frame_count = 0
        predictions = []
        stframe = st.empty()
        progress_bar = st.progress(0)
        skip_frames = 2 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Performance: Frame Skipping
            if frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_count / fps * 1000)
            results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                features = np.zeros((1, 561))
                features[0, :len(landmark_array)] = landmark_array
                pred = model.predict(scaler.transform(features))[0]
                predictions.append(pred)
                draw_landmarks(frame, landmarks)
                cv2.putText(frame, labels.get(pred, "Unknown"), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            out.write(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            frame_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            time.sleep(0.01) # UI Refresh stability

        cap.release()
        out.release()
        time.sleep(1) # Wait for Windows to release file locks

        if predictions:
            most_common = Counter(predictions).most_common(1)[0][0]
            st.success(f"🎯 **Final Prediction: {labels.get(most_common)}**")
            st.video(out_temp.name)

        # Cleanup
        try:
            if os.path.exists(video_path): os.remove(video_path)
            # We keep out_temp until the user closes the app to allow video streaming
        except Exception: pass