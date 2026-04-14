from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # small model

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,   # Colab GPU
    workers=4,

    # HIGH IMPACT SETTINGS
    cache=True,
    mosaic=1.0,
    mixup=0.2,

    # PRIORITY: LOW MISSED OBJECTS
    conf=0.15,
    iou=0.5
)