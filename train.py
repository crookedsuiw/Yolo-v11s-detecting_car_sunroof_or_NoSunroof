from ultralytics import YOLO
model = YOLO('yolo11s.pt')

model.train(
    data = 'custom_dataset.yaml',
    imgsz = 640,
    device = 'cpu', #set to 0 if you have access to gpu
    batch = 8,
    epochs = 100,
    workers = 0
)