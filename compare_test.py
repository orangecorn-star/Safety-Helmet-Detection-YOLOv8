#coding:utf-8
from ultralytics import YOLO
import cv2
import numpy as np
import os

BASE_MODEL = r".\runs\detect\train_yolov8_baseline\weights\best.pt"  
ENH_MODEL = r".\runs\detect\train_yolov8_enhanced\weights\best.pt"    
TEST_IMG_DIR = r".\datasets\images\val_hard"
SAVE_DIR = r".\compare_results_clean"

def clean_draw(img, results, title, is_enhanced=False):
    draw_img = img.copy()
    main_color = (0, 150, 0) if is_enhanced else (0, 0, 200) 
    
    cv2.rectangle(draw_img, (0, 0), (draw_img.shape[1], 40), main_color, -1)
    cv2.putText(draw_img, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{results[0].names[cls_id]} {conf:.2f}"
        
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), main_color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(draw_img, (x1, y1 - 25), (x1 + w, y1), main_color, -1)
        cv2.putText(draw_img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
    return draw_img

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Force check if model weight files exist
    if not os.path.exists(BASE_MODEL) or not os.path.exists(ENH_MODEL):
        print(f" Model weight files not found!\nPlease check:\n{BASE_MODEL}\n{ENH_MODEL}")
        return

    model_base = YOLO(BASE_MODEL)
    model_enh = YOLO(ENH_MODEL)

    # Deep scan the folder
    all_files = os.listdir(TEST_IMG_DIR)
    print(f" Found {len(all_files)} files in {TEST_IMG_DIR}.")
    
    # Relax image format restrictions
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    test_imgs = [f for f in all_files if str(f).lower().endswith(valid_exts)]
    print(f" Found {len(test_imgs)} images with valid extensions. Generating comparison images...")
    
    if len(test_imgs) == 0:
        print(" Warning: No valid image files found. Please check the folder path and file formats!")
        return

    success_count = 0
    for img_name in test_imgs:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        
        try:
            orig_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if orig_img is None:
                print(f" Failed to read image, possibly corrupted: {img_name}")
                continue
        except Exception as e:
            print(f" Error reading image {img_name}: {e}")
            continue
        
        h, w = orig_img.shape[:2]
        new_w = 640
        orig_img = cv2.resize(orig_img, (new_w, int(h * (new_w / w))))

        res_base = model_base(orig_img, verbose=False)
        res_enh = model_enh(orig_img, verbose=False)

        img_b = clean_draw(orig_img, res_base, "Baseline Model (Conf+)")
        img_e = clean_draw(orig_img, res_enh, "Enhanced Model (Conf+)", True)

        combined_img = np.hstack((img_b, img_e))
        cv2.imwrite(os.path.join(SAVE_DIR, f"clean_conf_{img_name}"), combined_img)
        print(f" Successfully processed: {img_name}")
        success_count += 1
        
    print(f"\n Task completed! Successfully generated {success_count} comparison images, saved to {SAVE_DIR}.")

if __name__ == '__main__':
    main()