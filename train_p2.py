import os
from ultralytics import YOLO,RTDETR
import warnings
warnings.filterwarnings('ignore')
# Load a model
# model = YOLO(model="yolov8s.yaml")  # yolov8s
# model = YOLO(model="yolov8s-p2.yaml")  # yolov8s+head
# model = YOLO(model="yolov8s-p2-repvgg.yaml")  # yolov8s+repvgg
model = YOLO(r"C:\Users\LTJ\Desktop\YOLO-main\ultralytics\cfg\models\v8\yolov8-try(1).yaml")  # yolov8s+repvgg+sf
# model.load('yolov8s.pt')  # 论文未加载预训练权重
# Use the model
model.train(data=r"C:\Users\LTJ\Desktop\YOLO-main\VisDrone.yaml", imgsz=640, epochs=300, workers=0, batch=4, cache=True, project='runs/train/YOLOv8',device=0)

# path = model.export(format="onnx", dynamic=True)  # export the mode l to ONNX format
