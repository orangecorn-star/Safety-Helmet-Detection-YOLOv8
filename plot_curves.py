#coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= Global settings for academic formatting =================
# Force Times New Roman font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
# ====================================================

BASE_CSV = r"./runs/detect/train_yolov8_baseline/results.csv"
ENH_CSV = r"./runs/detect/train_yolov8_enhanced/results.csv"
SAVE_PATH = r"./compare_curves.png"

def main():
    if not os.path.exists(BASE_CSV) or not os.path.exists(ENH_CSV):
        print("❌ CSV file not found. Please check the path!")
        return

    df_base = pd.read_csv(BASE_CSV)
    df_enh = pd.read_csv(ENH_CSV)
    df_base.columns = df_base.columns.str.strip()
    df_enh.columns = df_enh.columns.str.strip()

    epoch_base, loss_base, map_base = df_base['epoch'], df_base['train/box_loss'], df_base['metrics/mAP50(B)']
    epoch_enh, loss_enh, map_enh = df_enh['epoch'], df_enh['train/box_loss'], df_enh['metrics/mAP50(B)']

    plt.figure(figsize=(14, 5), dpi=300)

    # ---------- Figure 1: Train Box Loss Comparison ----------
    plt.subplot(1, 2, 1)
    plt.plot(epoch_base, loss_base, color='#1f77b4', linestyle='-', linewidth=2, label='Baseline')
    plt.plot(epoch_enh, loss_enh, color='#d62728', linestyle='--', linewidth=2, label='Enhanced (Hard Samples)')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Box Loss', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')

    # ---------- Figure 2: mAP50 Comparison ----------
    plt.subplot(1, 2, 2)
    plt.plot(epoch_base, map_base, color='#1f77b4', linestyle='-', linewidth=2, label='Baseline')
    plt.plot(epoch_enh, map_enh, color='#d62728', linestyle='--', linewidth=2, label='Enhanced (Hard Samples)')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('mAP@0.5', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f" Comparison curve saved to: {SAVE_PATH}")

if __name__ == '__main__':
    main()