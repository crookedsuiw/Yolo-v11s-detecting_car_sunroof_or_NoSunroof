from ultralytics import YOLO
model = YOLO('yolo11s.pt')

model.train(
    data = 'custom_dataset.yaml',
    imgsz = 640,
    device = 'cpu',
    batch = 8,
    epochs = 100,
    workers = 0
)