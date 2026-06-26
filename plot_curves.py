#coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= 学术排版全局设置 =================
# 强制使用新罗马字体 (Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
# ====================================================

BASE_CSV = r"./runs/detect/train_yolov8_baseline/results.csv"
ENH_CSV = r"./runs/detect/train_yolov8_enhanced/results.csv"
SAVE_PATH = r"./compare_curves.png"

def main():
    if not os.path.exists(BASE_CSV) or not os.path.exists(ENH_CSV):
        print("❌ 找不到 CSV 文件，请检查路径是否正确！")
        return

    df_base = pd.read_csv(BASE_CSV)
    df_enh = pd.read_csv(ENH_CSV)
    df_base.columns = df_base.columns.str.strip()
    df_enh.columns = df_enh.columns.str.strip()

    epoch_base, loss_base, map_base = df_base['epoch'], df_base['train/box_loss'], df_base['metrics/mAP50(B)']
    epoch_enh, loss_enh, map_enh = df_enh['epoch'], df_enh['train/box_loss'], df_enh['metrics/mAP50(B)']

    plt.figure(figsize=(14, 5), dpi=300)

    # ---------- 第 1 张图：Train Box Loss 对比 ----------
    plt.subplot(1, 2, 1)
    plt.plot(epoch_base, loss_base, color='#1f77b4', linestyle='-', linewidth=2, label='Baseline')
    plt.plot(epoch_enh, loss_enh, color='#d62728', linestyle='--', linewidth=2, label='Enhanced (Hard Samples)')
    # 【已删除 title，严格遵循学术规范】
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Box Loss', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    # 【强制图例位于右上角内部】
    plt.legend(fontsize=12, loc='upper right')

    # ---------- 第 2 张图：mAP50 对比 ----------
    plt.subplot(1, 2, 2)
    plt.plot(epoch_base, map_base, color='#1f77b4', linestyle='-', linewidth=2, label='Baseline')
    plt.plot(epoch_enh, map_enh, color='#d62728', linestyle='--', linewidth=2, label='Enhanced (Hard Samples)')
    # 【已删除 title，严格遵循学术规范】
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('mAP@0.5', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    # 【强制图例位于右下角内部】
    plt.legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"🎉 完美！带新罗马字体且无标题的对比曲线图已保存至: {SAVE_PATH}")

if __name__ == '__main__':
    main()