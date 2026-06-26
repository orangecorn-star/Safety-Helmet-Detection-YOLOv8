#coding:utf-8
from ultralytics import YOLO

if __name__ == '__main__':
    # 【最完美的轻量级对比】：调用清华大学 2024 年最新提出的 YOLOv10n
    # 核心差异：无 NMS 后处理 (NMS-Free)，真正的端到端架构
    model = YOLO("yolov10n.pt")  
    
    results = model.train(
        data='data.yaml', 
        epochs=100, 
        batch=8,    # 它的显存占用极小，8 batch 完全没问题，甚至可以开到 16
        imgsz=640,
        device=0, 
        workers=0,
        name='train_yolov10_baseline' 
    )