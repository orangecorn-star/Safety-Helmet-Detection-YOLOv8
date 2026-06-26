#coding:utf-8
import os
import shutil
from ultralytics import YOLO

# ================= 路径配置 =================
# 原困难样本图片库
HARD_IMG_SRC = r".\datasets\Hard_Samples"
# 当前的全局验证集目录
VAL_IMG_DIR = r".\datasets\images\val"
VAL_LBL_DIR = r".\datasets\labels\val"

# 为困难验证集专门新建的孤立目录
HARD_VAL_IMG = r".\datasets\images\val_hard"
HARD_VAL_LBL = r".\datasets\labels\val_hard"
# ============================================

def isolate_hard_val_set():
    """将混入全局 val 的 40 张困难样本提取到独立目录"""
    os.makedirs(HARD_VAL_IMG, exist_ok=True)
    os.makedirs(HARD_VAL_LBL, exist_ok=True)
    
    count = 0
    # 遍历原始困难样本库里的所有图片（兼容各种后缀）
    for img_name in os.listdir(HARD_IMG_SRC):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + ".txt"
        
        val_img_path = os.path.join(VAL_IMG_DIR, img_name)
        val_lbl_path = os.path.join(VAL_LBL_DIR, txt_name)
        
        # 严格判断：如果这张困难样本正好被划分到了全局验证集中
        if os.path.exists(val_img_path) and os.path.exists(val_lbl_path):
            shutil.copy(val_img_path, os.path.join(HARD_VAL_IMG, img_name))
            shutil.copy(val_lbl_path, os.path.join(HARD_VAL_LBL, txt_name))
            count += 1
            
    print(f"✅ 成功提取了 {count} 张困难验证集图片用于专项测试！")

def create_hard_yaml():
    """生成专门用于困难样本测试的 YAML 文件"""
    yaml_content = """
path: ./datasets
train: images/train
val: images/val_hard
nc: 2
names:
  0: helmet
  1: without
"""
    with open("hard_data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print("✅ hard_data.yaml 配置文件已生成！")

def run_validation():
    print("\n🚀 [第一回合] 开始测试 Baseline 基础模型 (旧模型)...")
    # 加载第一版模型（基础版）
    model_base = YOLO(r".\runs\detect\train_yolov8_baseline\weights\best.pt")
    model_base.val(data="hard_data.yaml", imgsz=640, batch=8, workers=0, name="val_baseline_hard")

    print("\n🚀 [第二回合] 开始测试 Enhanced 增强模型 (新模型)...")
    # 加载第二版模型（注意：如果你第二次训练的文件夹名不是 train_enhanced，请自行修改下方路径）
    model_enh = YOLO(r".\runs\detect\train_yolov8_enhanced\weights\best.pt")
    model_enh.val(data="hard_data.yaml", imgsz=640, batch=8, workers=0, name="val_enhanced_hard")
    
    print("\n🎉 专项验证全部完成！请去 runs/detect 目录下查看两者的惊人差距！")

if __name__ == '__main__':
    isolate_hard_val_set()
    create_hard_yaml()
    run_validation()