#coding:utf-8
from ultralytics import YOLO


# 加载模型
model = YOLO("yolov8n.pt") 

if __name__ == '__main__':
    # 核心修改点说明：
    # 1. device=0：启用你的第一块独立显卡进行火力全开地训练。
    # 2. epochs=100：GPU 速度很快，可以直接跑 100 轮，这样模型收敛得更好，为你的论文提供更好看的 loss 曲线。
    # 3. batch=16：批次大小调高。如果你的显卡显存较大（比如 8G 以上），甚至可以改为 32。如果报错 RuntimeError: CUDA out of memory，就降回 8。
    # 4. workers=2：负责搬运数据的线程数。Windows 系统下开太高容易假死或报错，建议设为 2 或 4。如果训练启动时卡住不动，请改回 workers=0。
    
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