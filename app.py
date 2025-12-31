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
import io
import av 



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



# MediaPipe Setup (New Tasks API)

BaseOptions = python.BaseOptions

VisionRunningMode = vision.RunningMode

PoseLandmarker = vision.PoseLandmarker

PoseLandmarkerOptions = vision.PoseLandmarkerOptions



options = PoseLandmarkerOptions(

    base_options=BaseOptions(model_asset_path=model_file),

    running_mode=VisionRunningMode.VIDEO,  # Important: VIDEO mode for better temporal consistency

    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

)



pose_landmarker = PoseLandmarker.create_from_options(options)



# Pose connections for drawing

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

st.title("📸 🎥 Human Activity Recognition (Image & Video)")



st.write("""

Upload an **image** or **video** of a person performing one of these activities:  

**Walking, Walking Upstairs/Downstairs, Sitting, Standing, Laying**



*Note: Model trained on sensor data (UCI HAR), so pose-based predictions are approximate.*

""")



uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4", "mov", "avi"])



if uploaded_file is not None:

    file_bytes = uploaded_file.read()

    file_type = uploaded_file.type



    if file_type.startswith('image'):

        # ------------------- IMAGE PROCESSING -------------------

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

            activity = labels.get(prediction, "Unknown")



            annotated = image.copy()

            draw_landmarks(annotated, landmarks)



            st.subheader(f"🔍 Detected Activity: **{activity}**")

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        else:

            st.error("No pose detected in the image.")



    elif file_type.startswith('video'):
        # ------------------- VIDEO PROCESSING -------------------
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(file_bytes)
        tfile.close()  # Close the file handle
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        
        # Safely get FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or fps > 120:  # Invalid or unrealistic FPS
            fps = 30.0  # Common fallback
        fps_int = int(round(fps))  # PyAV prefers integer

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_count = 0
        predictions = []

        stframe = st.empty()
        progress_bar = st.progress(0)
        st.info("Processing video... This may take a few moments.")

        # In-memory H.264 MP4 encoding with PyAV
        buffer = io.BytesIO()
        container = av.open(buffer, mode='w', format='mp4')
        stream = container.add_stream('h264', rate=fps_int)  # Use integer
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'  # Essential for browser compatibility
        stream.options = {'crf': '23'}  # Good quality/size trade-off

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

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

            # Encode and mux frame
            frame_av = av.VideoFrame.from_ndarray(frame, format='bgr24')
            for packet in stream.encode(frame_av):
                container.mux(packet)

            # Live preview
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            frame_count += 1
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))

        # Flush remaining packets
        for packet in stream.encode(None):
            container.mux(packet)

        container.close()
        
        # Get video bytes
        buffer.seek(0)
        video_bytes = buffer.getvalue()
        buffer.close()

        cap.release()
        os.unlink(video_path)
        progress_bar.empty()
        st.info("")  # Clear processing message

        # Final prediction (majority vote)
        if predictions:
            most_common = Counter(predictions).most_common(1)[0][0]
            count = Counter(predictions).most_common(1)[0][1]
            final_activity = labels.get(most_common, "Unknown")
            st.success(f"**Most Frequent Activity: {final_activity}** "
                       f"({count} / {len(predictions)} frames with pose detected)")
        else:
            st.warning("No pose detected in any frame.")

        # Display final annotated video
        st.video(video_bytes)