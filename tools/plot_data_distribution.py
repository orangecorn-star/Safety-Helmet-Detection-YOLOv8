#coding:utf-8
import os
import matplotlib.pyplot as plt
import numpy as np

# ================= 学术排版全局设置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False 
# ====================================================

def get_area_distribution(label_dir):
    """统计标签目录中的目标尺寸分布"""
    small, medium, large = 0, 0, 0
    # 根据 MS COCO 标准定义相对面积阈值 (假设640分辨率)
    # 小目标: < 32x32 (面积 < 0.0025)
    # 中目标: 32x32 ~ 96x96 (面积 0.0025 ~ 0.0225)
    # 大目标: > 96x96 (面积 > 0.0225)
    
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            with open(os.path.join(label_dir, file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        w, h = float(parts[3]), float(parts[4])
                        area = w * h
                        if area < 0.0025:
                            small += 1
                        elif area < 0.0225:
                            medium += 1
                        else:
                            large += 1
                            
    total = small + medium + large
    
    # ======== 【关键修改点：在这里加 print】 ========
    folder_name = os.path.basename(label_dir)
    print(f"📁 文件夹 [{folder_name}] 统计结果：")
    print(f"   -> 总框数: {total}")
    print(f"   -> 小目标数: {small}")
    print("-" * 40)
    # ===============================================

    if total == 0:
        return 0, 0, 0
    # 返回百分比
    return (small/total*100, medium/total*100, large/total*100)

def main():
    # 替换为你实际的相对路径
    global_val_dir = r"./datasets/labels/val"
    hard_val_dir = r"./datasets/labels/val_hard"

    if not os.path.exists(global_val_dir) or not os.path.exists(hard_val_dir):
        print("❌ 找不到 labels 文件夹，请先运行你的流水线前几步！")
        return

    # 获取分布数据
    g_s, g_m, g_l = get_area_distribution(global_val_dir)
    h_s, h_m, h_l = get_area_distribution(hard_val_dir)

    print("📊 正在生成数据分布对比图...")
    
    # 开始绘图
    labels = ['Small Objects', 'Medium Objects', 'Large Objects']
    global_dist = [g_s, g_m, g_l]
    hard_dist = [h_s, h_m, h_l]

    x = np.arange(len(labels))
    width = 0.35  # 柱状图宽度

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    # 绘制分组柱状图
    rects1 = ax.bar(x - width/2, global_dist, width, label='Global Dataset', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width/2, hard_dist, width, label='Hard Samples (Ours)', color='#d62728', edgecolor='black')

    # 添加文字标签、坐标轴
    ax.set_ylabel('Proportion (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 100) # y轴限制在0-100%
    
    # 将图例放在图内右上角
    ax.legend(fontsize=12, loc='upper right')

    # 为柱状图添加具体数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移3个像素
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontname='Times New Roman')

    autolabel(rects1)
    autolabel(rects2)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = "data_distribution_comparison.png"
    plt.savefig(save_path)
    print(f"🎉 数据分布图已生成并保存至: {save_path}")

if __name__ == '__main__':
    main()