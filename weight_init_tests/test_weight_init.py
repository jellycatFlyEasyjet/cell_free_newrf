#!/usr/bin/env python3
"""
测试不同权重初始化方法对模型性能的影响
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加上级目录到 Python 路径以导入主模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models import MLP
from advanced_models import MLP_Advanced, initialize_model_weights

def test_weight_initialization_impact():
    """测试权重初始化对训练的影响"""
    
    print("="*80)
    print("🧪 测试权重初始化对训练的影响")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成模拟数据
    batch_size = 32
    n_samples = 1000
    
    # 模拟STA位置数据 (x, y, z坐标)
    sta_locs = torch.randn(n_samples, 3) * 2.0  # 在[-2, 2]范围内
    
    # 模拟输入CFR数据 (实部, 虚部)
    input_cfrs = torch.randn(n_samples, 2) * 0.1  # 小幅度的CFR
    
    # 生成目标CFR数据 (基于位置的简单函数)
    target_real = 0.01 * torch.sin(sta_locs[:, 0]) * torch.cos(sta_locs[:, 1])
    target_imag = 0.01 * torch.cos(sta_locs[:, 0]) * torch.sin(sta_locs[:, 1])
    targets = target_real + 1j * target_imag
    
    # 测试不同的初始化方法
    methods = {
        'default': lambda: MLP(),  # 使用我们改进的MLP
        'xavier_advanced': lambda: MLP_Advanced(init_method='xavier'),
        'he_advanced': lambda: MLP_Advanced(init_method='he'),
        'nerf_advanced': lambda: MLP_Advanced(init_method='nerf_default'),
        'orthogonal_advanced': lambda: MLP_Advanced(init_method='orthogonal')
    }
    
    results = {}
    
    for method_name, model_creator in methods.items():
        print(f"\n🔬 测试 {method_name} 初始化方法:")
        
        # 创建模型
        model = model_creator()
        model.to(device)
        
        # 移动数据到设备
        sta_locs_dev = sta_locs.to(device)
        input_cfrs_dev = input_cfrs.to(device)
        targets_dev = targets.to(device)
        
        # 设置优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 损失函数
        def compute_loss(pred, target):
            return torch.sum(torch.abs(pred - target) ** 2) / torch.sum(torch.abs(target) ** 2)
        
        # 训练几个epoch来测试初始收敛性
        n_epochs = 50
        losses = []
        
        model.train()
        for epoch in range(n_epochs):
            # 随机采样batch
            indices = torch.randperm(n_samples)[:batch_size]
            batch_sta_locs = sta_locs_dev[indices]
            batch_input_cfrs = input_cfrs_dev[indices]
            batch_targets = targets_dev[indices]
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch_sta_locs, batch_input_cfrs)
            loss = compute_loss(outputs, batch_targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:2d}: Loss = {loss.item():.6f}")
        
        results[method_name] = {
            'losses': losses,
            'final_loss': losses[-1],
            'initial_loss': losses[0],
            'convergence_rate': (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0
        }
        
        print(f"  最终损失: {losses[-1]:.6f}")
        print(f"  收敛率: {results[method_name]['convergence_rate']*100:.1f}%")
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 损失曲线对比
    plt.subplot(2, 2, 1)
    for method_name, result in results.items():
        plt.plot(result['losses'], label=method_name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 最终损失对比
    plt.subplot(2, 2, 2)
    methods_list = list(results.keys())
    final_losses = [results[method]['final_loss'] for method in methods_list]
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods_list)))
    
    bars = plt.bar(methods_list, final_losses, color=colors)
    plt.xlabel('Initialization Method')
    plt.ylabel('Final Loss')
    plt.title('Final Loss Comparison')
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 收敛率对比
    plt.subplot(2, 2, 3)
    convergence_rates = [results[method]['convergence_rate']*100 for method in methods_list]
    
    bars = plt.bar(methods_list, convergence_rates, color=colors)
    plt.xlabel('Initialization Method')
    plt.ylabel('Convergence Rate (%)')
    plt.title('Convergence Rate Comparison')
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar, rate in zip(bars, convergence_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 初始vs最终损失对比
    plt.subplot(2, 2, 4)
    initial_losses = [results[method]['initial_loss'] for method in methods_list]
    
    x = np.arange(len(methods_list))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, initial_losses, width, label='Initial Loss', alpha=0.7)
    bars2 = plt.bar(x + width/2, final_losses, width, label='Final Loss', alpha=0.7)
    
    plt.xlabel('Initialization Method')
    plt.ylabel('Loss')
    plt.title('Initial vs Final Loss')
    plt.xticks(x, methods_list, rotation=45)
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('/home/byang/BoYang/mNeWRF/weight_init_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印总结
    print(f"\n" + "="*80)
    print("📊 权重初始化测试总结")
    print("="*80)
    
    # 找到最佳方法
    best_method = min(results.keys(), key=lambda x: results[x]['final_loss'])
    fastest_convergence = max(results.keys(), key=lambda x: results[x]['convergence_rate'])
    
    print(f"🏆 最佳最终损失: {best_method} (损失: {results[best_method]['final_loss']:.6f})")
    print(f"🚀 最快收敛: {fastest_convergence} (收敛率: {results[fastest_convergence]['convergence_rate']*100:.1f}%)")
    
    print(f"\n📈 详细结果:")
    for method_name, result in results.items():
        print(f"  {method_name:20s}: 最终损失={result['final_loss']:.6f}, 收敛率={result['convergence_rate']*100:.1f}%")
    
    print(f"\n💡 建议:")
    if best_method == fastest_convergence:
        print(f"  推荐使用 '{best_method}' 初始化方法，它在最终损失和收敛速度上都表现最佳")
    else:
        print(f"  如果追求最低损失，使用 '{best_method}' 初始化")
        print(f"  如果追求快速收敛，使用 '{fastest_convergence}' 初始化")
    
    print(f"\n✅ 测试完成！图表已保存为 'weight_init_comparison.png'")

if __name__ == "__main__":
    test_weight_initialization_impact()
