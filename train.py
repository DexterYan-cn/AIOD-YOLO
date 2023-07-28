from ultralytics import YOLO

model = YOLO("./models/AIOD-YOLO.yaml")  # build a YOLOv8n model from scratch

model.train(data="./data/VisDrone.yaml", epochs=300, batch=16, imgsz=640)  # train the model