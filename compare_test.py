#coding:utf-8
from ultralytics import YOLO
import cv2
import numpy as np
import os

# ================= 1. 路径配置区域 =================
# 基础模型 (第一版) 权重路径
BASE_MODEL_PATH = r".\runs\detect\train\weights\best.pt"

# 增强模型 (第二版) 权重路径 
# ⚠️ 注意：请检查你 runs/detect/ 目录下，第二次训练的文件夹是叫 train2 还是 train_enhanced，并在此处对应修改！
ENH_MODEL_PATH = r".\runs\detect\train_enhanced\weights\best.pt" 

# 困难样本测试集目录 (刚才剥离出来的那个文件夹)
TEST_IMG_DIR = r"D.\datasets\images\val_hard"

# 对比图保存输出的文件夹
SAVE_DIR = r".\compare_results"
# ===================================================

def main():
    # 创建保存输出结果的文件夹
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("⏳ 正在加载两个模型，请稍候...")
    try:
        model_base = YOLO(BASE_MODEL_PATH)
        model_enh = YOLO(ENH_MODEL_PATH)
    except Exception as e:
        print(f"❌ 模型加载失败，请检查权重路径是否正确！\n错误信息: {e}")
        return

    # 获取困难测试集里的图片列表
    img_list = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_list:
        print("❌ 在 val_hard 文件夹中没有找到图片！")
        return

    # 自动挑选前 5 张图片进行对比测试
    test_imgs = img_list[:5] 

    print(f"\n🚀 开始对比推理，共测试 {len(test_imgs)} 张图片...")
    for img_name in test_imgs:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        
        # 1. 分别使用两个模型进行推理 (verbose=False 用于关闭终端刷屏输出)
        res_base = model_base(img_path, verbose=False)
        res_enh = model_enh(img_path, verbose=False)

        # 2. 获取画好检测框的图片矩阵
        img_b = res_base[0].plot()
        img_e = res_enh[0].plot()

        # 3. 统一图片缩放尺寸 (将图片宽度统一缩放至 640，保持宽高比)
        h, w = img_b.shape[:2]
        new_w = 640
        new_h = int(h * (new_w / w))
        img_b = cv2.resize(img_b, (new_w, new_h))
        img_e = cv2.resize(img_e, (new_w, new_h))

        # 4. 在图片左上角添加文本标记，方便论文排版阅读
        # Baseline 标记为红色，Enhanced 标记为绿色
        cv2.putText(img_b, "Baseline Model", (15, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(img_e, "Enhanced Model", (15, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

        # 5. 左右拼接两张图片 (hstack = horizontal stack)
        combined_img = np.hstack((img_b, img_e))

        # 6. 保存对比结果
        save_path = os.path.join(SAVE_DIR, f"compare_{img_name}")
        cv2.imwrite(save_path, combined_img)
        print(f"✔️ 对比图已生成: {save_path}")

    print(f"\n🎉 所有对比任务完成！请前往 {SAVE_DIR} 目录查看生成的拼接对比图。")

if __name__ == '__main__':
    main()