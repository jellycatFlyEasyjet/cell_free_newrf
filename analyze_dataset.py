#!/usr/bin/env python3
"""
分析 conference_500STA_4APs.pkl 数据集中CFR数据的分布情况
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def analyze_cfr_dataset(dataset_path):
    """分析CFR数据集的详细信息"""
    
    print("="*80)
    print("📊 CFR数据集分析报告")
    print("="*80)
    
    # 加载数据集
    try:
        dataset = pd.read_pickle(dataset_path)
        print(f"✅ 成功加载数据集: {dataset_path}")
        print(f"📋 数据集形状: {dataset.shape}")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return
    
    print("\n" + "-"*50)
    print("🔍 基本信息")
    print("-"*50)
    
    # 基本信息
    print(f"总记录数: {len(dataset)}")
    print(f"列名: {list(dataset.columns)}")
    print(f"数据类型:")
    for col in dataset.columns:
        print(f"  {col}: {dataset[col].dtype}")
    
    # 检查缺失值
    missing_data = dataset.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\n⚠️ 缺失数据:")
        print(missing_data[missing_data > 0])
    else:
        print("\n✅ 无缺失数据")
    
    print("\n" + "-"*50)
    print("📡 AP和STA分布")
    print("-"*50)
    
    # AP和STA分布
    if 'TxID' in dataset.columns:
        tx_counts = Counter(dataset['TxID'])
        print(f"AP (TxID) 数量: {len(tx_counts)}")
        print(f"AP分布: {dict(tx_counts)}")
    
    if 'RxID' in dataset.columns:
        rx_counts = Counter(dataset['RxID'])
        print(f"STA (RxID) 数量: {len(rx_counts)}")
        print(f"STA ID范围: {min(rx_counts.keys())} - {max(rx_counts.keys())}")
    
    print("\n" + "-"*50)
    print("📊 CFR数据分析")
    print("-"*50)
    
    # CFR数据分析
    if 'CSI' in dataset.columns:
        print("CSI (Channel State Information) 数据:")
        
        # 随机采样一些CFR数据进行分析
        sample_size = min(100, len(dataset))
        sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        cfr_shapes = []
        cfr_magnitudes = []
        cfr_phases = []
        cfr_real_parts = []
        cfr_imag_parts = []
        
        for idx in sample_indices:
            try:
                cfr_data = dataset.iloc[idx]['CSI']
                if isinstance(cfr_data, np.ndarray) and len(cfr_data) > 0:
                    # 确保是复数数组
                    if np.iscomplexobj(cfr_data):
                        cfr_complex = cfr_data
                    else:
                        cfr_complex = cfr_data.astype(np.complex128)
                    
                    cfr_shapes.append(cfr_complex.shape)
                    cfr_magnitudes.extend(np.abs(cfr_complex.flatten()))
                    cfr_phases.extend(np.angle(cfr_complex.flatten()))
                    cfr_real_parts.extend(np.real(cfr_complex.flatten()))
                    cfr_imag_parts.extend(np.imag(cfr_complex.flatten()))
                    
            except Exception as e:
                print(f"  ⚠️ 处理第{idx}条记录时出错: {e}")
                continue
        
        if cfr_shapes:
            shape_counter = Counter([str(shape) for shape in cfr_shapes])
            print(f"  CFR数据形状分布: {dict(shape_counter)}")
            
            if cfr_magnitudes:
                print(f"  幅度统计:")
                print(f"    最小值: {np.min(cfr_magnitudes):.6f}")
                print(f"    最大值: {np.max(cfr_magnitudes):.6f}")
                print(f"    平均值: {np.mean(cfr_magnitudes):.6f}")
                print(f"    标准差: {np.std(cfr_magnitudes):.6f}")
                
                print(f"  相位统计 (弧度):")
                print(f"    最小值: {np.min(cfr_phases):.6f}")
                print(f"    最大值: {np.max(cfr_phases):.6f}")
                print(f"    平均值: {np.mean(cfr_phases):.6f}")
                print(f"    标准差: {np.std(cfr_phases):.6f}")
                
                print(f"  实部统计:")
                print(f"    最小值: {np.min(cfr_real_parts):.6f}")
                print(f"    最大值: {np.max(cfr_real_parts):.6f}")
                print(f"    平均值: {np.mean(cfr_real_parts):.6f}")
                print(f"    标准差: {np.std(cfr_real_parts):.6f}")
                
                print(f"  虚部统计:")
                print(f"    最小值: {np.min(cfr_imag_parts):.6f}")
                print(f"    最大值: {np.max(cfr_imag_parts):.6f}")
                print(f"    平均值: {np.mean(cfr_imag_parts):.6f}")
                print(f"    标准差: {np.std(cfr_imag_parts):.6f}")
    
    print("\n" + "-"*50)
    print("🗺️ 位置信息分析")
    print("-"*50)
    
    # 位置信息分析
    if 'TxPos' in dataset.columns:
        print("AP位置 (TxPos) 信息:")
        tx_positions = []
        for pos in dataset['TxPos'].dropna():
            if isinstance(pos, (list, np.ndarray)) and len(pos) >= 3:
                tx_positions.append(pos[:3])
        
        if tx_positions:
            tx_positions = np.array(tx_positions)
            print(f"  X坐标范围: {np.min(tx_positions[:, 0]):.2f} - {np.max(tx_positions[:, 0]):.2f}")
            print(f"  Y坐标范围: {np.min(tx_positions[:, 1]):.2f} - {np.max(tx_positions[:, 1]):.2f}")
            print(f"  Z坐标范围: {np.min(tx_positions[:, 2]):.2f} - {np.max(tx_positions[:, 2]):.2f}")
    
    if 'RxPos' in dataset.columns:
        print("\nSTA位置 (RxPos) 信息:")
        rx_positions = []
        for pos in dataset['RxPos'].dropna():
            if isinstance(pos, (list, np.ndarray)) and len(pos) >= 3:
                rx_positions.append(pos[:3])
        
        if rx_positions:
            rx_positions = np.array(rx_positions)
            print(f"  X坐标范围: {np.min(rx_positions[:, 0]):.2f} - {np.max(rx_positions[:, 0]):.2f}")
            print(f"  Y坐标范围: {np.min(rx_positions[:, 1]):.2f} - {np.max(rx_positions[:, 1]):.2f}")
            print(f"  Z坐标范围: {np.min(rx_positions[:, 2]):.2f} - {np.max(rx_positions[:, 2]):.2f}")
    
    print("\n" + "-"*50)
    print("📈 数据可视化")
    print("-"*50)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CFR数据集分析可视化', fontsize=16)
    
    # 1. AP-STA对分布
    if 'TxID' in dataset.columns and 'RxID' in dataset.columns:
        pair_counts = dataset.groupby(['TxID', 'RxID']).size()
        ap_sta_matrix = pair_counts.unstack(fill_value=0)
        
        sns.heatmap(ap_sta_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('AP-STA对数量分布')
        axes[0, 0].set_xlabel('STA ID (RxID)')
        axes[0, 0].set_ylabel('AP ID (TxID)')
    
    # 2. CFR幅度分布
    if cfr_magnitudes:
        axes[0, 1].hist(cfr_magnitudes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('CFR幅度分布')
        axes[0, 1].set_xlabel('幅度')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_yscale('log')
    
    # 3. CFR相位分布
    if cfr_phases:
        axes[0, 2].hist(cfr_phases, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('CFR相位分布')
        axes[0, 2].set_xlabel('相位 (弧度)')
        axes[0, 2].set_ylabel('频次')
    
    # 4. 实部vs虚部散点图
    if cfr_real_parts and cfr_imag_parts:
        # 随机采样用于可视化
        n_points = min(1000, len(cfr_real_parts))
        indices = np.random.choice(len(cfr_real_parts), n_points, replace=False)
        sample_real = [cfr_real_parts[i] for i in indices]
        sample_imag = [cfr_imag_parts[i] for i in indices]
        
        axes[1, 0].scatter(sample_real, sample_imag, alpha=0.5, s=1)
        axes[1, 0].set_title('CFR复数平面分布')
        axes[1, 0].set_xlabel('实部')
        axes[1, 0].set_ylabel('虚部')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. STA位置分布
    if rx_positions is not None and len(rx_positions) > 0:
        axes[1, 1].scatter(rx_positions[:, 0], rx_positions[:, 1], alpha=0.6, s=10)
        axes[1, 1].set_title('STA位置分布 (X-Y平面)')
        axes[1, 1].set_xlabel('X坐标 (m)')
        axes[1, 1].set_ylabel('Y坐标 (m)')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. AP位置分布
    if tx_positions is not None and len(tx_positions) > 0:
        unique_tx_pos = np.unique(tx_positions, axis=0)
        axes[1, 2].scatter(unique_tx_pos[:, 0], unique_tx_pos[:, 1], 
                          c='red', s=100, marker='^', label='AP')
        axes[1, 2].set_title('AP位置分布 (X-Y平面)')
        axes[1, 2].set_xlabel('X坐标 (m)')
        axes[1, 2].set_ylabel('Y坐标 (m)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('/home/byang/BoYang/mNeWRF/dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 可视化图表已保存为 'dataset_analysis.png'")
    
    print("\n" + "="*80)
    print("📋 分析完成")
    print("="*80)

if __name__ == "__main__":
    dataset_path = '/home/byang/BoYang/NeWRF-main/simulator/datasets/conference_500STA_4APs.pkl'
    analyze_cfr_dataset(dataset_path)
