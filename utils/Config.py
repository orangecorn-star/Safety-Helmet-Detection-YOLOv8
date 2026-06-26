#coding:utf-8

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 训练完成后生成的最佳权重路径（YOLOv8 默认会保存在 runs/detect/...）
model_path = 'runs/detect/train/weights/best.pt'

# 类别映射修改为 2 类，去除原有的非机动车
names = {0: 'helmet', 1: "without"}
CH_names = ['头盔', '未佩戴']