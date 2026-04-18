from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # small model

model.train(
    data="dataset/dataset.yml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,   # Colab GPU
    workers=4,

    # HIGH IMPACT SETTINGS
    hsv_s=0.7,
    hsv_v=0.5,
    degrees=10,
    scale=0.7,
    cache=True,
    mosaic=1.0,
    mixup=0.2,

    optimizer='auto',
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3,
    # PRIORITY: LOW MISSED OBJECTS
    conf=0.15,
    iou=0.5
)