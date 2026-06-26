#coding:utf-8
import os
import random
import shutil

# ==================== 1. 路径配置区域 ====================
# 困难样本的源路径 (图片和标签)
SRC_IMG_DIR = r".\datasets\Hard_Samples"
SRC_LBL_DIR = r".\datasets\Hard_Samples_labels"

# 现有 YOLO 数据集的目标路径
DST_IMG_TRAIN = r".\datasets\images\train"
DST_IMG_VAL = r".\datasets\images\val"
DST_LBL_TRAIN = r".\datasets\labels\train"
DST_LBL_VAL = r".\datasets\labels\val"

# 训练集划分比例 (80% 进入 train，20% 进入 val)
TRAIN_RATIO = 0.8
# =========================================================

def merge_samples():
    print("🚀 开始检索困难样本库...")
    
    # 1. 检查所有的图片文件
    all_imgs = [f for f in os.listdir(SRC_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    valid_pairs = []
    
    # 2. 严格核对：只有同时具备图片和对应 txt 标签的样本才会被采用
    for img_name in all_imgs:
        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + ".txt"
        
        img_path = os.path.join(SRC_IMG_DIR, img_name)
        txt_path = os.path.join(SRC_LBL_DIR, txt_name)
        
        if os.path.exists(txt_path):
            valid_pairs.append((img_name, txt_name))
            
    total_valid = len(valid_pairs)
    print(f"✅ 成功匹配到 {total_valid} 对有效的 [图片 + 标签] 样本！")
    
    if total_valid == 0:
        print("❌ 未找到有效样本，请检查路径是否正确。")
        return

    # 3. 随机打乱样本，确保训练集和验证集的数据分布均匀
    random.seed(42)  # 固定随机种子，保证可复现
    random.shuffle(valid_pairs)
    
    # 4. 计算划分节点
    split_idx = int(total_valid * TRAIN_RATIO)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    print(f"📦 准备分配: {len(train_pairs)} 个样本分配至训练集(train), {len(val_pairs)} 个样本分配至验证集(val)...")
    
    # 5. 执行复制操作
    def copy_files(pairs, dest_img_dir, dest_lbl_dir, dataset_type):
        count = 0
        for img_name, txt_name in pairs:
            # 构造完整的源路径和目标路径
            src_img = os.path.join(SRC_IMG_DIR, img_name)
            src_lbl = os.path.join(SRC_LBL_DIR, txt_name)
            
            dst_img = os.path.join(dest_img_dir, img_name)
            dst_lbl = os.path.join(dest_lbl_dir, txt_name)
            
            # 复制文件
            shutil.copy(src_img, dst_img)
            shutil.copy(src_lbl, dst_lbl)
            count += 1
        print(f"✔️ {dataset_type} 数据集融合完毕！已成功汇入 {count} 个新样本。")

    # 分别汇入 train 和 val
    copy_files(train_pairs, DST_IMG_TRAIN, DST_LBL_TRAIN, "Train (训练)")
    copy_files(val_pairs, DST_IMG_VAL, DST_LBL_VAL, "Val (验证)")
    
    print("\n🎉 全部困难样本融合大功告成！现有的数据集已全面升级为增强版。")

if __name__ == '__main__':
    merge_samples()