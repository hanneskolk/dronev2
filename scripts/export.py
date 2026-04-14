from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

#model.export(format="onnx")   # cross-platform
model.export(format="engine") # optional (NVIDIA only)