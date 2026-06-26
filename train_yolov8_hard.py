#coding:utf-8
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolov8n.pt")  
    
    results = model.train(
        data='data.yaml', 
        epochs=100, 
        batch=8,    # 它的显存占用极小，8 batch 完全没问题，甚至可以开到 16
        imgsz=640,
        device=0, 
        workers=0,
        name='train_yolov8_enhanced' # 给这次的新成果单独建一个文件夹，与之前的区分开
    )