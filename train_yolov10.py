#coding:utf-8
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolov10n.pt")  
    
    results = model.train(
        data='data.yaml', 
        epochs=100, 
        batch=8,    
        imgsz=640,
        device=0, 
        workers=0,
        name='train_yolov10_baseline' 
    )