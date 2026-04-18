import streamlit as st
import cv2
import os
from ultralytics import YOLO
from supervision import ByteTrack, Detections
import numpy as np

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL (FAST INIT)
# =========================
@st.cache_resource
def load_model():
    model = YOLO("models/best.pt")

    # 🔥 SPEED OPTIMIZATIONS
    model.fuse()
    model.to("cuda") if hasattr(model, "to") else None

    return model

model = load_model()

tracker = ByteTrack()

st.title("Fast Drone Inference System")

video_file = st.file_uploader("Upload MP4", type=["mp4"])

conf = st.slider("Confidence", 0.05, 0.5, 0.15)

# =========================
# FAST DRAW FUNCTION (NO .plot())
# =========================
def draw_boxes(frame, results):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = f"{cls}:{conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    return frame

# =========================
# MAIN PIPELINE
# =========================
if video_file:

    input_path = os.path.join(UPLOAD_DIR, video_file.name)

    with open(input_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(3))
    h = int(cap.get(4))

    output_path = os.path.join(OUTPUT_DIR, "out_" + video_file.name)

    # 🔥 FASTER ENCODER SETTINGS
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    stframe = st.empty()
    progress = st.progress(0)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0

    # =========================
    # WARMUP (CRITICAL)
    # =========================
    dummy = np.zeros((640,640,3), dtype=np.uint8)
    model.predict(dummy, verbose=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        i += 1

        # =========================
        # RESIZE FOR SPEED
        # =========================
        frame_small = cv2.resize(frame, (640, 640))

        # =========================
        # YOLO INFERENCE (FAST MODE)
        # =========================
        results = model.predict(
            frame_small,
            conf=conf,
            imgsz=640,
            device="cpu",
            half=True,
            verbose=False
        )[0]

        # =========================
        # BYTE TRACK (LIGHT MODE)
        # =========================
        detections = Detections.from_ultralytics(results)
        tracks = tracker.update_with_detections(detections)

        # =========================
        # DRAW (FAST CUSTOM)
        # =========================
        annotated = draw_boxes(frame_small, results)

        # resize back only at end
        annotated = cv2.resize(annotated, (w, h))

        out.write(annotated)

        # =========================
        # UI UPDATE (LIMITED FREQUENCY)
        # =========================
        if i % 3 == 0:   # reduce Streamlit overhead
            stframe.image(annotated, channels="BGR")

        progress.progress(i / frame_count)

    cap.release()
    out.release()

    st.success("Done!")

    st.video(output_path)