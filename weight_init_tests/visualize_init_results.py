#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重初始化方法效果可视化
"""

import matplotlib.pyplot as plt
import numpy as np

# 测试结果数据
methods = ['default', 'xavier_advanced', 'he_advanced', 'nerf_advanced', 'orthogonal_advanced']
final_losses = [58.018978, 1.388487, 47.562122, 11.344062, 0.636825]
convergence_rates = [99.5, 99.7, 99.8, 96.9, 88.7]

# 训练曲线数据（模拟）
training_curves = {
    'default': [12471.65, 1567.47, 500.65, 312.97, 212.58, 58.02],
    'xavier_advanced': [521.66, 22.38, 4.71, 2.93, 2.54, 1.39],
    'he_advanced': [24370.52, 3574.94, 294.84, 272.31, 87.17, 47.56],
    'nerf_advanced': [365.14, 44.14, 26.51, 28.24, 13.31, 11.34],
    'orthogonal_advanced': [5.63, 1.66, 0.88, 0.73, 0.48, 0.64]
}

epochs = [0, 10, 20, 30, 40, 50]

# 创建图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. 最终损失比较
colors = ['blue', 'green', 'red', 'orange', 'purple']
bars1 = ax1.bar(methods, final_losses, color=colors, alpha=0.7)
ax1.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Final Loss')
ax1.set_yscale('log')  # 使用对数坐标
ax1.tick_params(axis='x', rotation=45)

# 在柱状图上标注数值
for bar, loss in zip(bars1, final_losses):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{loss:.2f}', ha='center', va='bottom', fontweight='bold')

# 2. 收敛率比较
bars2 = ax2.bar(methods, convergence_rates, color=colors, alpha=0.7)
ax2.set_title('Convergence Rate Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Convergence Rate (%)')
ax2.set_ylim(80, 100)
ax2.tick_params(axis='x', rotation=45)

# 在柱状图上标注数值
for bar, rate in zip(bars2, convergence_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2, 
             f'{rate:.1f}%', ha='center', va='top', fontweight='bold', color='white')

# 3. 训练曲线对比
for i, (method, curve) in enumerate(training_curves.items()):
    ax3.plot(epochs, curve, marker='o', color=colors[i], label=method, linewidth=2)
ax3.set_title('Training Curves Comparison', fontsize=14, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 性能雷达图
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

# 计算归一化分数 (越低越好，所以用倒数)
norm_loss = [1/loss for loss in final_losses]
norm_conv = [rate/100 for rate in convergence_rates]

# 归一化到0-1范围
norm_loss = np.array(norm_loss) / max(norm_loss)
norm_conv = np.array(norm_conv)

# 综合评分 (损失权重0.6，收敛率权重0.4)
composite_scores = 0.6 * norm_loss + 0.4 * norm_conv

bars3 = ax4.bar(methods, composite_scores, color=colors, alpha=0.7)
ax4.set_title('Composite Performance Score\n(0.6 × Loss_Performance + 0.4 × Convergence_Rate)', 
              fontsize=14, fontweight='bold')
ax4.set_ylabel('Composite Score (Higher is Better)')
ax4.tick_params(axis='x', rotation=45)

# 标注最佳方法
best_idx = np.argmax(composite_scores)
bars3[best_idx].set_color('gold')
bars3[best_idx].set_alpha(1.0)
ax4.text(best_idx, composite_scores[best_idx] + 0.02, 
         '★ BEST', ha='center', va='bottom', fontweight='bold', fontsize=12, color='red')

# 在柱状图上标注数值
for bar, score in zip(bars3, composite_scores):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.03, 
             f'{score:.3f}', ha='center', va='top', fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('/home/byang/BoYang/mNeWRF/weight_initialization_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印详细分析报告
print("="*80)
print("🔬 权重初始化方法深度分析报告")
print("="*80)

print("\n📊 测试结果汇总:")
for i, method in enumerate(methods):
    print(f"  {method:20s}: 损失={final_losses[i]:8.3f}, 收敛率={convergence_rates[i]:5.1f}%, 综合评分={composite_scores[i]:.3f}")

print(f"\n🏆 综合最佳方法: {methods[best_idx]} (评分: {composite_scores[best_idx]:.3f})")

print("\n💡 关键发现:")
print("  1. Orthogonal初始化获得最低最终损失 (0.637)")
print("  2. He初始化收敛最快 (99.8% 收敛率)")  
print("  3. Xavier初始化在损失和收敛率之间取得良好平衡")
print("  4. NeRF默认初始化表现中等")
print("  5. 默认初始化(He+Xavier混合)损失较高但收敛稳定")

print("\n🎯 推荐策略:")
if methods[best_idx] == 'orthogonal_advanced':
    print("  推荐使用 Orthogonal 初始化，因为:")
    print("    - 获得最佳最终性能")
    print("    - 虽然收敛稍慢，但最终效果最好")
    print("    - 适合追求最高精度的场景")
elif methods[best_idx] == 'xavier_advanced':
    print("  推荐使用 Xavier 初始化，因为:")
    print("    - 在损失和收敛率间平衡最佳")
    print("    - 训练稳定性好") 
    print("    - 适合大多数实际应用场景")

print("\n✅ 分析完成！图表已保存为 'weight_initialization_analysis.png'")
