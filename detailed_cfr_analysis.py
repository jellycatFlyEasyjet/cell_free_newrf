#!/usr/bin/env python3
"""
详细分析不同AP之间的CFR数据分布和统计特性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def analyze_ap_cfr_distribution(dataset_path):
    """分析每个AP的CFR数据分布"""
    
    print("="*80)
    print("📡 AP-CFR详细分析报告")
    print("="*80)
    
    # 加载数据集
    dataset = pd.read_pickle(dataset_path)
    dataset = dataset.dropna()  # 移除包含NaN的行
    
    print(f"清理后数据集大小: {len(dataset)} 条记录")
    
    # 按AP分组分析CFR数据
    ap_cfr_stats = {}
    
    for ap_id in sorted(dataset['TxID'].unique()):
        print(f"\n📊 分析 AP {int(ap_id)} 的CFR数据...")
        
        ap_data = dataset[dataset['TxID'] == ap_id]
        print(f"  AP {int(ap_id)} 记录数: {len(ap_data)}")
        
        # 提取CFR数据
        cfr_magnitudes = []
        cfr_phases = []
        cfr_real = []
        cfr_imag = []
        cfr_complex_values = []
        
        for idx, row in ap_data.iterrows():
            try:
                cfr = row['CSI']
                if isinstance(cfr, np.ndarray) and len(cfr) > 0:
                    # 确保是复数
                    if not np.iscomplexobj(cfr):
                        cfr = cfr.astype(np.complex128)
                    
                    cfr_flat = cfr.flatten()
                    cfr_complex_values.extend(cfr_flat)
                    cfr_magnitudes.extend(np.abs(cfr_flat))
                    cfr_phases.extend(np.angle(cfr_flat))
                    cfr_real.extend(np.real(cfr_flat))
                    cfr_imag.extend(np.imag(cfr_flat))
                    
            except Exception as e:
                print(f"    ⚠️ 处理第{idx}条记录时出错: {e}")
                continue
        
        # 统计信息
        if cfr_magnitudes:
            stats = {
                'count': len(cfr_magnitudes),
                'magnitude': {
                    'min': np.min(cfr_magnitudes),
                    'max': np.max(cfr_magnitudes),
                    'mean': np.mean(cfr_magnitudes),
                    'std': np.std(cfr_magnitudes),
                    'median': np.median(cfr_magnitudes)
                },
                'phase': {
                    'min': np.min(cfr_phases),
                    'max': np.max(cfr_phases),
                    'mean': np.mean(cfr_phases),
                    'std': np.std(cfr_phases),
                    'median': np.median(cfr_phases)
                },
                'real': {
                    'min': np.min(cfr_real),
                    'max': np.max(cfr_real),
                    'mean': np.mean(cfr_real),
                    'std': np.std(cfr_real),
                    'median': np.median(cfr_real)
                },
                'imag': {
                    'min': np.min(cfr_imag),
                    'max': np.max(cfr_imag),
                    'mean': np.mean(cfr_imag),
                    'std': np.std(cfr_imag),
                    'median': np.median(cfr_imag)
                },
                'complex_values': cfr_complex_values[:100]  # 保存前100个用于可视化
            }
            
            ap_cfr_stats[int(ap_id)] = stats
            
            print(f"  📈 统计结果:")
            print(f"    样本数量: {stats['count']}")
            print(f"    幅度: {stats['magnitude']['mean']:.6f} ± {stats['magnitude']['std']:.6f} (范围: {stats['magnitude']['min']:.6f} - {stats['magnitude']['max']:.6f})")
            print(f"    相位: {stats['phase']['mean']:.6f} ± {stats['phase']['std']:.6f} (范围: {stats['phase']['min']:.6f} - {stats['phase']['max']:.6f})")
            print(f"    实部: {stats['real']['mean']:.6f} ± {stats['real']['std']:.6f} (范围: {stats['real']['min']:.6f} - {stats['real']['max']:.6f})")
            print(f"    虚部: {stats['imag']['mean']:.6f} ± {stats['imag']['std']:.6f} (范围: {stats['imag']['min']:.6f} - {stats['imag']['max']:.6f})")
    
    # 创建对比图表
    print(f"\n📊 创建AP对比可视化...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('不同AP的CFR数据分布对比', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # 1. 幅度对比 - 直方图
    for i, (ap_id, stats) in enumerate(ap_cfr_stats.items()):
        # 重新提取数据用于直方图
        ap_data = dataset[dataset['TxID'] == ap_id]
        magnitudes = []
        for _, row in ap_data.iterrows():
            try:
                cfr = row['CSI']
                if isinstance(cfr, np.ndarray) and len(cfr) > 0:
                    if not np.iscomplexobj(cfr):
                        cfr = cfr.astype(np.complex128)
                    magnitudes.extend(np.abs(cfr.flatten()))
            except:
                continue
        
        if magnitudes:
            axes[0, 0].hist(magnitudes, bins=30, alpha=0.7, 
                           label=f'AP {ap_id}', color=colors[i % len(colors)])
    
    axes[0, 0].set_title('CFR幅度分布对比')
    axes[0, 0].set_xlabel('幅度')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 2. 相位对比 - 直方图
    for i, (ap_id, stats) in enumerate(ap_cfr_stats.items()):
        ap_data = dataset[dataset['TxID'] == ap_id]
        phases = []
        for _, row in ap_data.iterrows():
            try:
                cfr = row['CSI']
                if isinstance(cfr, np.ndarray) and len(cfr) > 0:
                    if not np.iscomplexobj(cfr):
                        cfr = cfr.astype(np.complex128)
                    phases.extend(np.angle(cfr.flatten()))
            except:
                continue
        
        if phases:
            axes[0, 1].hist(phases, bins=30, alpha=0.7, 
                           label=f'AP {ap_id}', color=colors[i % len(colors)])
    
    axes[0, 1].set_title('CFR相位分布对比')
    axes[0, 1].set_xlabel('相位 (弧度)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].legend()
    
    # 3. 复数平面分布对比
    for i, (ap_id, stats) in enumerate(ap_cfr_stats.items()):
        if 'complex_values' in stats and stats['complex_values']:
            complex_vals = stats['complex_values']
            real_parts = [np.real(c) for c in complex_vals]
            imag_parts = [np.imag(c) for c in complex_vals]
            
            axes[0, 2].scatter(real_parts, imag_parts, alpha=0.6, s=20,
                             label=f'AP {ap_id}', color=colors[i % len(colors)])
    
    axes[0, 2].set_title('CFR复数平面分布对比')
    axes[0, 2].set_xlabel('实部')
    axes[0, 2].set_ylabel('虚部')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 幅度统计对比 - 箱线图
    magnitude_data = []
    ap_labels = []
    
    for ap_id, stats in ap_cfr_stats.items():
        ap_data = dataset[dataset['TxID'] == ap_id]
        ap_magnitudes = []
        for _, row in ap_data.iterrows():
            try:
                cfr = row['CSI']
                if isinstance(cfr, np.ndarray) and len(cfr) > 0:
                    if not np.iscomplexobj(cfr):
                        cfr = cfr.astype(np.complex128)
                    ap_magnitudes.extend(np.abs(cfr.flatten()))
            except:
                continue
        
        if ap_magnitudes:
            magnitude_data.append(ap_magnitudes)
            ap_labels.append(f'AP {ap_id}')
    
    if magnitude_data:
        axes[0, 3].boxplot(magnitude_data, labels=ap_labels)
        axes[0, 3].set_title('CFR幅度分布箱线图')
        axes[0, 3].set_ylabel('幅度')
    
    # 5-8. 统计指标对比
    ap_ids = list(ap_cfr_stats.keys())
    
    # 5. 平均幅度对比
    mean_magnitudes = [ap_cfr_stats[ap_id]['magnitude']['mean'] for ap_id in ap_ids]
    std_magnitudes = [ap_cfr_stats[ap_id]['magnitude']['std'] for ap_id in ap_ids]
    
    axes[1, 0].bar([f'AP {ap_id}' for ap_id in ap_ids], mean_magnitudes, 
                   yerr=std_magnitudes, capsize=5, color=colors[:len(ap_ids)])
    axes[1, 0].set_title('平均CFR幅度对比')
    axes[1, 0].set_ylabel('平均幅度')
    
    # 6. 相位标准差对比
    phase_stds = [ap_cfr_stats[ap_id]['phase']['std'] for ap_id in ap_ids]
    
    axes[1, 1].bar([f'AP {ap_id}' for ap_id in ap_ids], phase_stds, 
                   color=colors[:len(ap_ids)])
    axes[1, 1].set_title('CFR相位标准差对比')
    axes[1, 1].set_ylabel('相位标准差')
    
    # 7. 实部vs虚部标准差
    real_stds = [ap_cfr_stats[ap_id]['real']['std'] for ap_id in ap_ids]
    imag_stds = [ap_cfr_stats[ap_id]['imag']['std'] for ap_id in ap_ids]
    
    x_pos = np.arange(len(ap_ids))
    width = 0.35
    
    axes[1, 2].bar(x_pos - width/2, real_stds, width, label='实部标准差', alpha=0.8)
    axes[1, 2].bar(x_pos + width/2, imag_stds, width, label='虚部标准差', alpha=0.8)
    axes[1, 2].set_title('实部vs虚部标准差对比')
    axes[1, 2].set_ylabel('标准差')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels([f'AP {ap_id}' for ap_id in ap_ids])
    axes[1, 2].legend()
    
    # 8. 样本数量对比
    sample_counts = [ap_cfr_stats[ap_id]['count'] for ap_id in ap_ids]
    
    axes[1, 3].bar([f'AP {ap_id}' for ap_id in ap_ids], sample_counts, 
                   color=colors[:len(ap_ids)])
    axes[1, 3].set_title('CFR样本数量对比')
    axes[1, 3].set_ylabel('样本数量')
    
    plt.tight_layout()
    plt.savefig('/home/byang/BoYang/mNeWRF/ap_cfr_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印总结
    print(f"\n" + "="*80)
    print("📋 AP-CFR分析总结")
    print("="*80)
    
    for ap_id, stats in ap_cfr_stats.items():
        print(f"\n📡 AP {ap_id}:")
        print(f"  样本数量: {stats['count']}")
        print(f"  幅度 (平均 ± 标准差): {stats['magnitude']['mean']:.6f} ± {stats['magnitude']['std']:.6f}")
        print(f"  相位 (平均 ± 标准差): {stats['phase']['mean']:.6f} ± {stats['phase']['std']:.6f}")
        print(f"  实部 (平均 ± 标准差): {stats['real']['mean']:.6f} ± {stats['real']['std']:.6f}")
        print(f"  虚部 (平均 ± 标准差): {stats['imag']['mean']:.6f} ± {stats['imag']['std']:.6f}")
    
    # 检查AP间差异
    print(f"\n🔍 AP间差异分析:")
    magnitude_means = [ap_cfr_stats[ap_id]['magnitude']['mean'] for ap_id in sorted(ap_cfr_stats.keys())]
    phase_means = [ap_cfr_stats[ap_id]['phase']['mean'] for ap_id in sorted(ap_cfr_stats.keys())]
    
    print(f"  幅度平均值变异系数: {np.std(magnitude_means) / np.mean(magnitude_means) * 100:.2f}%")
    print(f"  相位平均值标准差: {np.std(phase_means):.4f} 弧度")
    
    print(f"\n✅ 分析完成！图表已保存为 'ap_cfr_comparison.png'")

if __name__ == "__main__":
    dataset_path = '/home/byang/BoYang/NeWRF-main/simulator/datasets/conference_500STA_4APs.pkl'
    analyze_ap_cfr_distribution(dataset_path)
