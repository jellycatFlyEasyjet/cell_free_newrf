#!/usr/bin/env python3
"""
Detailed AP-CFR Distribution Analysis with English Labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def analyze_ap_cfr_distribution_english(dataset_path):
    """Analyze CFR data distribution for each AP with English labels"""
    
    print("="*80)
    print("📡 AP-CFR Detailed Analysis Report")
    print("="*80)
    
    # Load dataset
    dataset = pd.read_pickle(dataset_path)
    dataset = dataset.dropna()  # Remove rows with NaN
    
    print(f"Dataset size after cleaning: {len(dataset)} records")
    
    # Analyze CFR data by AP
    ap_cfr_stats = {}
    
    for ap_id in sorted(dataset['TxID'].unique()):
        print(f"\n📊 Analyzing CFR data for AP {int(ap_id)}...")
        
        ap_data = dataset[dataset['TxID'] == ap_id]
        print(f"  AP {int(ap_id)} record count: {len(ap_data)}")
        
        # Extract CFR data
        cfr_magnitudes = []
        cfr_phases = []
        cfr_real = []
        cfr_imag = []
        cfr_complex_values = []
        
        for idx, row in ap_data.iterrows():
            try:
                cfr = row['CSI']
                if isinstance(cfr, np.ndarray) and len(cfr) > 0:
                    # Ensure complex numbers
                    if not np.iscomplexobj(cfr):
                        cfr = cfr.astype(np.complex128)
                    
                    cfr_flat = cfr.flatten()
                    cfr_complex_values.extend(cfr_flat)
                    cfr_magnitudes.extend(np.abs(cfr_flat))
                    cfr_phases.extend(np.angle(cfr_flat))
                    cfr_real.extend(np.real(cfr_flat))
                    cfr_imag.extend(np.imag(cfr_flat))
                    
            except Exception as e:
                print(f"    ⚠️ Error processing record {idx}: {e}")
                continue
        
        # Statistics
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
                'complex_values': cfr_complex_values[:100]  # Save first 100 for visualization
            }
            
            ap_cfr_stats[int(ap_id)] = stats
            
            print(f"  📈 Statistics:")
            print(f"    Sample count: {stats['count']}")
            print(f"    Magnitude: {stats['magnitude']['mean']:.6f} ± {stats['magnitude']['std']:.6f} (range: {stats['magnitude']['min']:.6f} - {stats['magnitude']['max']:.6f})")
            print(f"    Phase: {stats['phase']['mean']:.6f} ± {stats['phase']['std']:.6f} (range: {stats['phase']['min']:.6f} - {stats['phase']['max']:.6f})")
            print(f"    Real: {stats['real']['mean']:.6f} ± {stats['real']['std']:.6f} (range: {stats['real']['min']:.6f} - {stats['real']['max']:.6f})")
            print(f"    Imaginary: {stats['imag']['mean']:.6f} ± {stats['imag']['std']:.6f} (range: {stats['imag']['min']:.6f} - {stats['imag']['max']:.6f})")
    
    # Create comparison charts
    print(f"\n📊 Creating AP comparison visualization...")
    
    # Set matplotlib parameters
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('CFR Data Distribution Comparison Across Different APs', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # 1. Magnitude comparison - histogram
    for i, (ap_id, stats) in enumerate(ap_cfr_stats.items()):
        # Re-extract data for histogram
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
    
    axes[0, 0].set_title('CFR Magnitude Distribution Comparison')
    axes[0, 0].set_xlabel('Magnitude')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 2. Phase comparison - histogram
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
    
    axes[0, 1].set_title('CFR Phase Distribution Comparison')
    axes[0, 1].set_xlabel('Phase (radians)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. Complex plane distribution comparison
    for i, (ap_id, stats) in enumerate(ap_cfr_stats.items()):
        if 'complex_values' in stats and stats['complex_values']:
            complex_vals = stats['complex_values']
            real_parts = [np.real(c) for c in complex_vals]
            imag_parts = [np.imag(c) for c in complex_vals]
            
            axes[0, 2].scatter(real_parts, imag_parts, alpha=0.6, s=20,
                             label=f'AP {ap_id}', color=colors[i % len(colors)])
    
    axes[0, 2].set_title('CFR Complex Plane Distribution Comparison')
    axes[0, 2].set_xlabel('Real Part')
    axes[0, 2].set_ylabel('Imaginary Part')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Magnitude statistics comparison - box plot
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
        axes[0, 3].boxplot(magnitude_data, tick_labels=ap_labels)
        axes[0, 3].set_title('CFR Magnitude Distribution Box Plot')
        axes[0, 3].set_ylabel('Magnitude')
    
    # 5-8. Statistical metrics comparison
    ap_ids = list(ap_cfr_stats.keys())
    
    # 5. Average magnitude comparison
    mean_magnitudes = [ap_cfr_stats[ap_id]['magnitude']['mean'] for ap_id in ap_ids]
    std_magnitudes = [ap_cfr_stats[ap_id]['magnitude']['std'] for ap_id in ap_ids]
    
    axes[1, 0].bar([f'AP {ap_id}' for ap_id in ap_ids], mean_magnitudes, 
                   yerr=std_magnitudes, capsize=5, color=colors[:len(ap_ids)])
    axes[1, 0].set_title('Average CFR Magnitude Comparison')
    axes[1, 0].set_ylabel('Average Magnitude')
    
    # 6. Phase standard deviation comparison
    phase_stds = [ap_cfr_stats[ap_id]['phase']['std'] for ap_id in ap_ids]
    
    axes[1, 1].bar([f'AP {ap_id}' for ap_id in ap_ids], phase_stds, 
                   color=colors[:len(ap_ids)])
    axes[1, 1].set_title('CFR Phase Standard Deviation Comparison')
    axes[1, 1].set_ylabel('Phase Standard Deviation')
    
    # 7. Real vs Imaginary standard deviation
    real_stds = [ap_cfr_stats[ap_id]['real']['std'] for ap_id in ap_ids]
    imag_stds = [ap_cfr_stats[ap_id]['imag']['std'] for ap_id in ap_ids]
    
    x_pos = np.arange(len(ap_ids))
    width = 0.35
    
    axes[1, 2].bar(x_pos - width/2, real_stds, width, label='Real Part Std', alpha=0.8)
    axes[1, 2].bar(x_pos + width/2, imag_stds, width, label='Imaginary Part Std', alpha=0.8)
    axes[1, 2].set_title('Real vs Imaginary Part Standard Deviation')
    axes[1, 2].set_ylabel('Standard Deviation')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels([f'AP {ap_id}' for ap_id in ap_ids])
    axes[1, 2].legend()
    
    # 8. Sample count comparison
    sample_counts = [ap_cfr_stats[ap_id]['count'] for ap_id in ap_ids]
    
    axes[1, 3].bar([f'AP {ap_id}' for ap_id in ap_ids], sample_counts, 
                   color=colors[:len(ap_ids)])
    axes[1, 3].set_title('CFR Sample Count Comparison')
    axes[1, 3].set_ylabel('Sample Count')
    
    plt.tight_layout()
    plt.savefig('/home/byang/BoYang/mNeWRF/ap_cfr_comparison_english.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n" + "="*80)
    print("📋 AP-CFR Analysis Summary")
    print("="*80)
    
    for ap_id, stats in ap_cfr_stats.items():
        print(f"\n📡 AP {ap_id}:")
        print(f"  Sample count: {stats['count']}")
        print(f"  Magnitude (mean ± std): {stats['magnitude']['mean']:.6f} ± {stats['magnitude']['std']:.6f}")
        print(f"  Phase (mean ± std): {stats['phase']['mean']:.6f} ± {stats['phase']['std']:.6f}")
        print(f"  Real part (mean ± std): {stats['real']['mean']:.6f} ± {stats['real']['std']:.6f}")
        print(f"  Imaginary part (mean ± std): {stats['imag']['mean']:.6f} ± {stats['imag']['std']:.6f}")
    
    # Check differences between APs
    print(f"\n🔍 Inter-AP Difference Analysis:")
    magnitude_means = [ap_cfr_stats[ap_id]['magnitude']['mean'] for ap_id in sorted(ap_cfr_stats.keys())]
    phase_means = [ap_cfr_stats[ap_id]['phase']['mean'] for ap_id in sorted(ap_cfr_stats.keys())]
    
    print(f"  Magnitude mean coefficient of variation: {np.std(magnitude_means) / np.mean(magnitude_means) * 100:.2f}%")
    print(f"  Phase mean standard deviation: {np.std(phase_means):.4f} radians")
    
    print(f"\n✅ Analysis complete! Charts saved as 'ap_cfr_comparison_english.png'")

if __name__ == "__main__":
    dataset_path = '/home/byang/BoYang/NeWRF-main/simulator/datasets/conference_500STA_4APs.pkl'
    analyze_ap_cfr_distribution_english(dataset_path)
