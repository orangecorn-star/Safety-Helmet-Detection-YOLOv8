#coding:utf-8
from ultralytics import YOLO
import cv2
import os

# 训练好的模型路径
model = YOLO("D:/桌面/下作业/人工智能/YOLOv8-Helmet-Detection-Enhance/runs/detect/train/weights/best.pt")

# 需要检测的图片地址（可自行更换）
img_path = "D:/桌面/下作业/人工智能/YOLOv8-Helmet-Detection-Enhance/困难场景/4.jpg"

# 保存结果的目标文件夹
save_dir = "D:/桌面/下作业/人工智能/YOLOv8-Helmet-Detection-Enhance/imageTest"

# 如果目标文件夹不存在，则创建
os.makedirs(save_dir, exist_ok=True)

# 检测图片
results = model(img_path)
res = results[0].plot()

# 显示图片（按任意键关闭窗口）
cv2.imshow("YOLOv8 Detection", res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存带检测框的图片
# 获取原文件名，构造保存路径
base_name = os.path.basename(img_path)
save_path = os.path.join(save_dir, base_name)
cv2.imwrite(save_path, res)
print(f"检测结果已保存至: {save_path}")