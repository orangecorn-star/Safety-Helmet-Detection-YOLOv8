#coding:utf-8
from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt") 

if __name__ == '__main__':

    results = model.train(
        data='data.yaml', 
        epochs=100, 
        batch=8,
        imgsz=640,
        device=0, 
        cache=False,
        workers=0,
        name='train_yolov8_baseline'
    )