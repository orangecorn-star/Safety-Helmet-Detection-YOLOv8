#coding:utf-8
import os
import shutil
from ultralytics import YOLO

# ================= Path Configuration =================
# Original hard sample image directory
HARD_IMG_SRC = r".\datasets\Hard_Samples"
# Global validation set directories
VAL_IMG_DIR = r".\datasets\images\val"
VAL_LBL_DIR = r".\datasets\labels\val"

# Isolated hard validation set directories
HARD_VAL_IMG = r".\datasets\images\val_hard"
HARD_VAL_LBL = r".\datasets\labels\val_hard"
# ====================================================

def isolate_hard_val_set():
    """Extract the 40 hard samples that were mixed into the global validation set"""
    os.makedirs(HARD_VAL_IMG, exist_ok=True)
    os.makedirs(HARD_VAL_LBL, exist_ok=True)

    count = 0
    # Iterate over all images in the hard sample source directory
    for img_name in os.listdir(HARD_IMG_SRC):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + ".txt"

        val_img_path = os.path.join(VAL_IMG_DIR, img_name)
        val_lbl_path = os.path.join(VAL_LBL_DIR, txt_name)

        # Strict check: only copy if this hard sample exists in the global validation set
        if os.path.exists(val_img_path) and os.path.exists(val_lbl_path):
            shutil.copy(val_img_path, os.path.join(HARD_VAL_IMG, img_name))
            shutil.copy(val_lbl_path, os.path.join(HARD_VAL_LBL, txt_name))
            count += 1

    print(f" Successfully extracted {count} hard validation images for specialized testing!")

def create_hard_yaml():
    """Generate a YAML file specifically for hard sample testing"""
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
    print(" hard_data.yaml configuration file generated!")

def run_validation():
    print("\n [Round 1] Testing Baseline model...")
    model_base = YOLO(r".\runs\detect\train_yolov8_baseline\weights\best.pt")
    model_base.val(data="hard_data.yaml", imgsz=640, batch=8, workers=0, name="val_baseline_hard")

    print("\n [Round 2] Testing Enhanced model...")
    model_enh = YOLO(r".\runs\detect\train_yolov8_enhanced\weights\best.pt")
    model_enh.val(data="hard_data.yaml", imgsz=640, batch=8, workers=0, name="val_enhanced_hard")

    print("\n Specialized validation complete! Check runs/detect for the results.")

if __name__ == '__main__':
    isolate_hard_val_set()
    create_hard_yaml()
    run_validation()