import os
import xml.etree.ElementTree as ET
import random
import shutil

# ==================== 配置区域 ====================
# 1. 你的源 VOC 数据集路径（根据你的实际电脑路径修改）
VOC_ROOT = r"D:\桌面\下作业\人工智能\数据集\VOC2028\VOC2028"
XML_DIR = os.path.join(VOC_ROOT, "Annotations")
IMG_DIR = os.path.join(VOC_ROOT, "JPEGImages")

# 2. 你的新工程中的目标 YOLO 数据集保存路径
YOLO_ROOT = r"./datasets"

# 3. 类别对齐映射（核心：将你数据集里的 xml 英文标签映射到 YOLO 的 0 和 1）
# 提示：请打开你几张 xml 文件看看里面的 <name> 标签是不是 hat 和 person
CLASS_MAPPING = {
    "hat": 0,       # 0 代表安全帽（已佩戴）
    "person": 1     # 1 代表未佩戴
}

# 4. 训练集和验证集划分比例（80%用于训练，20%用于验证）
TRAIN_RATIO = 0.8
# ==================================================

def convert_box(size, box):
    """将 VOC 的 [xmin, ymin, xmax, ymax] 转换为 YOLO 的 [x_center, y_center, w, h] 并归一化"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x_center * dw, y_center * dh, w * dw, h * dh)

def setup_dirs():
    """创建 YOLO 格式所需的标准目录结构"""
    for sub in ['images', 'labels']:
        for split in ['train', 'val']:
            os.makedirs(os.path.join(YOLO_ROOT, sub, split), exist_ok=True)

def main():
    setup_dirs()
    
    # 获取所有图片（去除后缀，只拿文件名）
    all_files = [os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(42) # 固定随机种子，保证每次运行划分结果一致
    random.shuffle(all_files)
    
    # 划分训练集和验证集
    split_idx = int(len(all_files) * TRAIN_RATIO)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"数据总数: {len(all_files)} | 训练集: {len(train_files)} | 验证集: {len(val_files)}")
    
    for split_name, file_list in [('train', train_files), ('val', val_files)]:
        for file_name in file_list:
            img_ext = ".jpg" 
            src_img_path = os.path.join(IMG_DIR, file_name + img_ext)
            if not os.path.exists(src_img_path):
                src_img_path = os.path.join(IMG_DIR, file_name + ".png")
                img_ext = ".png"
                if not os.path.exists(src_img_path): continue
                
            src_xml_path = os.path.join(XML_DIR, file_name + ".xml")
            if not os.path.exists(src_xml_path): continue
                
            # 目标路径
            dst_img_path = os.path.join(YOLO_ROOT, "images", split_name, file_name + img_ext)
            dst_txt_path = os.path.join(YOLO_ROOT, "labels", split_name, file_name + ".txt")
            
            # 解析 XML
            tree = ET.parse(src_xml_path)
            root = tree.getroot()
            size = root.find('size')
            w_img = int(size.find('width').text)
            h_img = int(size.find('height').text)
            
            has_valid_object = False
            yolo_lines = []
            
            for obj in root.findall('object'):
                cls_name = obj.find('name').text.strip()
                if cls_name not in CLASS_MAPPING:
                    continue # 自动过滤和忽略不需要的非机动车等干扰类别
                    
                cls_id = CLASS_MAPPING[cls_name]
                xmlbox = obj.find('bndbox')
                box = (float(xmlbox.find('xmin').text), 
                       float(xmlbox.find('ymin').text), 
                       float(xmlbox.find('xmax').text), 
                       float(xmlbox.find('ymax').text))
                
                # 坐标格式转换
                bb = convert_box((w_img, h_img), box)
                yolo_lines.append(f"{cls_id} {' '.join([f'{x:.6f}' for x in bb])}\n")
                has_valid_object = True
                
            # 只有当图片中包含我们需要的有效目标时，才进行搬移和生成标签
            if has_valid_object:
                shutil.copy(src_img_path, dst_img_path)
                with open(dst_txt_path, 'w', encoding='utf-8') as f:
                    f.writelines(yolo_lines)

    print(">> 数据集转换与自动划分完成！标准的 YOLO 格式目录已成功在当前目录下的 ./datasets 中生成。")

if __name__ == '__main__':
    main()