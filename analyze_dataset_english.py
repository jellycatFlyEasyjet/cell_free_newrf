#!/usr/bin/env python3
"""
CFR Dataset Analysis with English Labels
Fixed matplotlib display issues with Chinese characters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def analyze_cfr_dataset_english(dataset_path):
    """Analyze CFR dataset with English labels"""
    
    print("="*80)
    print("📊 CFR Dataset Analysis Report")
    print("="*80)
    
    # Load dataset
    try:
        dataset = pd.read_pickle(dataset_path)
        print(f"✅ Successfully loaded dataset: {dataset_path}")
        print(f"📋 Dataset shape: {dataset.shape}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    print("\n" + "-"*50)
    print("🔍 Basic Information")
    print("-"*50)
    
    # Basic info
    print(f"Total records: {len(dataset)}")
    print(f"Columns: {list(dataset.columns)}")
    print(f"Data types:")
    for col in dataset.columns:
        print(f"  {col}: {dataset[col].dtype}")
    
    # Check missing data
    missing_data = dataset.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\n⚠️ Missing data:")
        print(missing_data[missing_data > 0])
    else:
        print("\n✅ No missing data")
    
    print("\n" + "-"*50)
    print("📡 AP and STA Distribution")
    print("-"*50)
    
    # AP and STA distribution
    if 'TxID' in dataset.columns:
        tx_counts = Counter(dataset['TxID'])
        print(f"Number of APs (TxID): {len(tx_counts)}")
        print(f"AP distribution: {dict(tx_counts)}")
    
    if 'RxID' in dataset.columns:
        rx_counts = Counter(dataset['RxID'])
        print(f"Number of STAs (RxID): {len(rx_counts)}")
        print(f"STA ID range: {min(rx_counts.keys())} - {max(rx_counts.keys())}")
    
    print("\n" + "-"*50)
    print("📊 CFR Data Analysis")
    print("-"*50)
    
    # CFR data analysis
    if 'CSI' in dataset.columns:
        print("CSI (Channel State Information) data:")
        
        # Sample CFR data for analysis
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
                    # Ensure complex array
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
                print(f"  ⚠️ Error processing record {idx}: {e}")
                continue
        
        if cfr_shapes:
            shape_counter = Counter([str(shape) for shape in cfr_shapes])
            print(f"  CFR data shape distribution: {dict(shape_counter)}")
            
            if cfr_magnitudes:
                print(f"  Magnitude statistics:")
                print(f"    Min: {np.min(cfr_magnitudes):.6f}")
                print(f"    Max: {np.max(cfr_magnitudes):.6f}")
                print(f"    Mean: {np.mean(cfr_magnitudes):.6f}")
                print(f"    Std: {np.std(cfr_magnitudes):.6f}")
                
                print(f"  Phase statistics (radians):")
                print(f"    Min: {np.min(cfr_phases):.6f}")
                print(f"    Max: {np.max(cfr_phases):.6f}")
                print(f"    Mean: {np.mean(cfr_phases):.6f}")
                print(f"    Std: {np.std(cfr_phases):.6f}")
                
                print(f"  Real part statistics:")
                print(f"    Min: {np.min(cfr_real_parts):.6f}")
                print(f"    Max: {np.max(cfr_real_parts):.6f}")
                print(f"    Mean: {np.mean(cfr_real_parts):.6f}")
                print(f"    Std: {np.std(cfr_real_parts):.6f}")
                
                print(f"  Imaginary part statistics:")
                print(f"    Min: {np.min(cfr_imag_parts):.6f}")
                print(f"    Max: {np.max(cfr_imag_parts):.6f}")
                print(f"    Mean: {np.mean(cfr_imag_parts):.6f}")
                print(f"    Std: {np.std(cfr_imag_parts):.6f}")
    
    print("\n" + "-"*50)
    print("🗺️ Position Information Analysis")
    print("-"*50)
    
    # Position analysis
    tx_positions = None
    rx_positions = None
    
    if 'TxPos' in dataset.columns:
        print("AP Position (TxPos) information:")
        tx_positions = []
        for pos in dataset['TxPos'].dropna():
            if isinstance(pos, (list, np.ndarray)) and len(pos) >= 3:
                tx_positions.append(pos[:3])
        
        if tx_positions:
            tx_positions = np.array(tx_positions)
            print(f"  X coordinate range: {np.min(tx_positions[:, 0]):.2f} - {np.max(tx_positions[:, 0]):.2f}")
            print(f"  Y coordinate range: {np.min(tx_positions[:, 1]):.2f} - {np.max(tx_positions[:, 1]):.2f}")
            print(f"  Z coordinate range: {np.min(tx_positions[:, 2]):.2f} - {np.max(tx_positions[:, 2]):.2f}")
    
    if 'RxPos' in dataset.columns:
        print("\nSTA Position (RxPos) information:")
        rx_positions = []
        for pos in dataset['RxPos'].dropna():
            if isinstance(pos, (list, np.ndarray)) and len(pos) >= 3:
                rx_positions.append(pos[:3])
        
        if rx_positions:
            rx_positions = np.array(rx_positions)
            print(f"  X coordinate range: {np.min(rx_positions[:, 0]):.2f} - {np.max(rx_positions[:, 0]):.2f}")
            print(f"  Y coordinate range: {np.min(rx_positions[:, 1]):.2f} - {np.max(rx_positions[:, 1]):.2f}")
            print(f"  Z coordinate range: {np.min(rx_positions[:, 2]):.2f} - {np.max(rx_positions[:, 2]):.2f}")
    
    print("\n" + "-"*50)
    print("📈 Data Visualization")
    print("-"*50)
    
    # Set matplotlib parameters for better display
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CFR Dataset Analysis Visualization', fontsize=16)
    
    # 1. AP-STA pair distribution
    if 'TxID' in dataset.columns and 'RxID' in dataset.columns:
        pair_counts = dataset.groupby(['TxID', 'RxID']).size()
        ap_sta_matrix = pair_counts.unstack(fill_value=0)
        
        sns.heatmap(ap_sta_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('AP-STA Pair Count Distribution')
        axes[0, 0].set_xlabel('STA ID (RxID)')
        axes[0, 0].set_ylabel('AP ID (TxID)')
    
    # 2. CFR magnitude distribution
    if cfr_magnitudes:
        axes[0, 1].hist(cfr_magnitudes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('CFR Magnitude Distribution')
        axes[0, 1].set_xlabel('Magnitude')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
    
    # 3. CFR phase distribution
    if cfr_phases:
        axes[0, 2].hist(cfr_phases, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('CFR Phase Distribution')
        axes[0, 2].set_xlabel('Phase (radians)')
        axes[0, 2].set_ylabel('Frequency')
    
    # 4. Real vs Imaginary scatter plot
    if cfr_real_parts and cfr_imag_parts:
        # Random sampling for visualization
        n_points = min(1000, len(cfr_real_parts))
        indices = np.random.choice(len(cfr_real_parts), n_points, replace=False)
        sample_real = [cfr_real_parts[i] for i in indices]
        sample_imag = [cfr_imag_parts[i] for i in indices]
        
        axes[1, 0].scatter(sample_real, sample_imag, alpha=0.5, s=1)
        axes[1, 0].set_title('CFR Complex Plane Distribution')
        axes[1, 0].set_xlabel('Real Part')
        axes[1, 0].set_ylabel('Imaginary Part')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. STA position distribution
    if rx_positions is not None and len(rx_positions) > 0:
        axes[1, 1].scatter(rx_positions[:, 0], rx_positions[:, 1], alpha=0.6, s=10)
        axes[1, 1].set_title('STA Position Distribution (X-Y Plane)')
        axes[1, 1].set_xlabel('X Coordinate (m)')
        axes[1, 1].set_ylabel('Y Coordinate (m)')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. AP position distribution
    if tx_positions is not None and len(tx_positions) > 0:
        unique_tx_pos = np.unique(tx_positions, axis=0)
        axes[1, 2].scatter(unique_tx_pos[:, 0], unique_tx_pos[:, 1], 
                          c='red', s=100, marker='^', label='AP')
        axes[1, 2].set_title('AP Position Distribution (X-Y Plane)')
        axes[1, 2].set_xlabel('X Coordinate (m)')
        axes[1, 2].set_ylabel('Y Coordinate (m)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('/home/byang/BoYang/mNeWRF/dataset_analysis_english.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved as 'dataset_analysis_english.png'")
    
    print("\n" + "="*80)
    print("📋 Analysis Complete")
    print("="*80)

if __name__ == "__main__":
    dataset_path = '/home/byang/BoYang/NeWRF-main/simulator/datasets/conference_500STA_4APs.pkl'
    analyze_cfr_dataset_english(dataset_path)
