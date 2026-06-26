#coding:utf-8
import os

def analyze_bbox_area(label_dir):
    total_boxes = 0
    small_boxes = 0
    # YOLO 归一化面积，假设 640x640 图像，32x32 的小目标面积占比为 (32/640)^2 = 0.0025
    SMALL_THRESH = 0.0025 

    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            with open(os.path.join(label_dir, file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        w, h = float(parts[3]), float(parts[4])
                        area = w * h
                        total_boxes += 1
                        if area < SMALL_THRESH:
                            small_boxes += 1
                            
    small_ratio = (small_boxes / total_boxes * 100) if total_boxes > 0 else 0
    return total_boxes, small_ratio

if __name__ == '__main__':
    # 替换为你实际的 labels 路径
    global_val_dir = r".\datasets\labels\val"
    hard_val_dir = r".\datasets\labels\val_hard"

    g_total, g_ratio = analyze_bbox_area(global_val_dir)
    h_total, h_ratio = analyze_bbox_area(hard_val_dir)

    print(f"📊 全局验证集: 总框数 {g_total}, 小目标占比 {g_ratio:.1f}%")
    print(f"📊 困难专项集: 总框数 {h_total}, 小目标占比 {h_ratio:.1f}%")