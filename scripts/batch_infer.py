import os
import cv2
from ultralytics import YOLO
from supervision import ByteTrack, Detections
from tqdm import tqdm

INPUT_DIR = "input_videos"
OUTPUT_DIR = "output_videos"
RESULTS_FILE = "results/summary.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

model = YOLO("models/best.pt")
model.fuse()
tracker = ByteTrack()


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(3))
    h = int(cap.get(4))

    out_path = os.path.join(OUTPUT_DIR, os.path.basename(video_path))
    out = cv2.VideoWriter(out_path,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))

    class_counts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.2)[0]

        detections = Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        annotated = results.plot()

        # Count detections
        for cls in results.boxes.cls:
            cls = int(cls)
            class_counts[cls] = class_counts.get(cls, 0) + 1

        out.write(annotated)

    cap.release()
    out.release()

    return class_counts


def main():
    summary = {}

    for video in tqdm(os.listdir(INPUT_DIR)):
        if not video.endswith(".mp4"):
            continue

        path = os.path.join(INPUT_DIR, video)
        counts = process_video(path)

        summary[video] = counts

    # Save results
    with open(RESULTS_FILE, "w") as f:
        for vid, counts in summary.items():
            f.write(f"{vid}: {counts}\n")


if __name__ == "__main__":
    main()