from ultralytics import YOLO

# Load a model
model = YOLO('./runs/detect/train/weights/best.pt')  # load a custom model

metrics = model.val(data="./data/VisDrone.yaml", device=0, batch=2, iou=0.7, conf=0.001)