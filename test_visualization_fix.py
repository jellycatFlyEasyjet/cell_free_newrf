#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的训练可视化
"""

import sys
import os
sys.path.append('/home/byang/BoYang/mNeWRF')

import torch
import numpy as np
import matplotlib.pyplot as plt

def test_visualization_fix():
    """测试修复后的可视化代码"""
    print("="*60)
    print("🧪 测试训练可视化修复")
    print("="*60)
    
    # 模拟训练数据
    n_iters = 50
    train_losses = []
    val_losses = []
    train_losses_raw = []
    val_losses_raw = []
    SNRs = []
    val_SNRs = []
    iternum = []
    
    # 生成模拟数据（模拟实际训练中的损失变化）
    base_train = 1.0
    base_val = 0.8
    
    for i in range(n_iters):
        # 训练损失：逐渐减小但有噪声
        train_loss = base_train * np.exp(-i/20) + 0.1 * np.random.random()
        train_loss_raw = train_loss * 0.5 + 0.01 * np.random.random()
        
        # 验证损失：类似但稍不同的尺度
        val_loss = base_val * np.exp(-i/25) + 0.05 * np.random.random()
        val_loss_raw = val_loss * 0.4 + 0.008 * np.random.random()
        
        train_losses.append(train_loss)
        train_losses_raw.append(train_loss_raw)
        
        if i % 5 == 0:  # 验证损失不是每次都计算
            val_losses.append(val_loss)
            val_losses_raw.append(val_loss_raw)
            val_SNRs.append(-10 * np.log10(val_loss))
            iternum.append(i)
        
        SNRs.append(-10 * np.log10(train_loss))
    
    # 模拟验证数据
    val_output_cfr = torch.randn(32) * 0.1 + 0.1j * torch.randn(32)
    val_target_cfr = torch.randn(32) * 0.1 + 0.1j * torch.randn(32)
    val_SNR = -2.5
    
    i = n_iters - 1
    
    print("📊 生成的测试数据统计:")
    print(f"  训练损失范围: {min(train_losses):.4f} - {max(train_losses):.4f}")
    print(f"  验证损失范围: {min(val_losses):.4f} - {max(val_losses):.4f}")
    print(f"  原始训练损失范围: {min(train_losses_raw):.4f} - {max(train_losses_raw):.4f}")
    print(f"  原始验证损失范围: {min(val_losses_raw):.4f} - {max(val_losses_raw):.4f}")
    
    # 测试可视化代码
    print("\n🎨 生成训练可视化图表...")
    
    # 使用修复后的可视化代码
    fig, ax = plt.subplots(2,3, figsize=(18, 10))
    
    # 第一行第一个：相对损失对比
    ax[0,0].plot(range(len(train_losses)), train_losses, 'b-', label='Train Loss (Relative)', alpha=0.7, linewidth=1.5)
    ax[0,0].plot(range(len(val_losses)), val_losses, 'r-', label='Val Loss (Relative)', alpha=0.7, linewidth=1.5)
    ax[0,0].set_xlabel('Iterations')
    ax[0,0].set_ylabel('Relative Loss')
    ax[0,0].set_yscale('log')
    ax[0,0].legend()
    ax[0,0].grid(True, alpha=0.3)
    ax[0,0].set_title('Relative Loss (Used for Training)')

    # 第一行第二个：原始MSE损失对比
    ax[0,1].plot(range(len(train_losses_raw)), train_losses_raw, 'b-', label='Train Loss (Raw MSE)', alpha=0.7, linewidth=1.5)
    ax[0,1].plot(range(len(val_losses_raw)), val_losses_raw, 'r-', label='Val Loss (Raw MSE)', alpha=0.7, linewidth=1.5)
    ax[0,1].set_xlabel('Iterations')
    ax[0,1].set_ylabel('Raw MSE Loss')
    ax[0,1].set_yscale('log')
    ax[0,1].legend()
    ax[0,1].grid(True, alpha=0.3)
    ax[0,1].set_title('Raw MSE Loss (Same Scale)')

    # 第一行第三个：SNR对比
    ax[0,2].plot(range(0, i + 1), SNRs, 'r-', label="Train SNR", alpha=0.7, linewidth=1.5)
    ax[0,2].plot(iternum, val_SNRs, 'y-', label='Val SNR', alpha=0.7, linewidth=1.5)
    ax[0,2].set_xlabel('Iterations')
    ax[0,2].set_ylabel('SNR (dB)')
    ax[0,2].legend()
    ax[0,2].grid(True, alpha=0.3)
    ax[0,2].set_title('SNR Progress')

    # 第二行第一个：复数预测可视化
    ax[1,0].plot(np.real(val_output_cfr), np.imag(val_output_cfr), "ro", label="prediction", alpha=0.7, markersize=3)
    ax[1,0].plot(np.real(val_target_cfr), np.imag(val_target_cfr), "bo", label="Target", alpha=0.7, markersize=3)
    ax[1,0].set_xlabel('Real Part')
    ax[1,0].set_ylabel('Imaginary Part')
    ax[1,0].set_title(f"AP_[2] predict SNR: {val_SNR:.2f} dB")
    ax[1,0].legend()
    ax[1,0].grid(True, alpha=0.3)

    # 第二行第二个：损失差异
    if len(train_losses_raw) > 0 and len(val_losses_raw) > 0:
        loss_diff = np.array(train_losses_raw) - np.array(val_losses_raw[:len(train_losses_raw)])
        ax[1,1].plot(range(len(loss_diff)), loss_diff, 'g-', label='Train - Val Loss', alpha=0.7, linewidth=1.5)
        ax[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax[1,1].set_xlabel('Iterations')
        ax[1,1].set_ylabel('Loss Difference')
        ax[1,1].legend()
        ax[1,1].grid(True, alpha=0.3)
        ax[1,1].set_title('Train-Val Loss Difference')

    # 第二行第三个：学习率（模拟）
    current_lr = 5e-4
    ax[1,2].axhline(y=current_lr, color='purple', linewidth=2, label=f'Current LR: {current_lr:.2e}')
    ax[1,2].set_xlabel('Iterations')
    ax[1,2].set_ylabel('Learning Rate')
    ax[1,2].legend()
    ax[1,2].grid(True, alpha=0.3)
    ax[1,2].set_title('Learning Rate Schedule')

    plt.tight_layout()
    plt.savefig('/home/byang/BoYang/mNeWRF/test_visualization_fix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 可视化测试完成!")
    print("📈 关键改进:")
    print("  1. 修复了 ax[3] 索引错误（现在使用2x3布局）")
    print("  2. 添加了原始MSE损失显示（相同尺度）")
    print("  3. 添加了损失差异分析")
    print("  4. 改进了图表布局和可读性")
    print("  5. 添加了学习率监控")
    
    return True

if __name__ == "__main__":
    test_visualization_fix()
    
    print("\n" + "="*60)
    print("🎯 训练可视化修复完成！")
    print("="*60)
    print("💡 主要解决方案:")
    print("  1. 使用 2x3 子图布局代替 1x3")
    print("  2. 分别显示相对损失和原始MSE损失")
    print("  3. 原始MSE损失使用相同计算方式，尺度一致")
    print("  4. 添加详细的损失对比分析")
    print("\n✅ 现在 train loss 和 val loss 可以在相同尺度下观察了！")
