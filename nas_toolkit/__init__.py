"""
NAS Toolkit - Neural Architecture Search for NeWRF
==================================================

A comprehensive toolkit for neural architecture search and hyperparameter optimization
specifically designed for NeWRF (Neural Wireless Radio Field) models.

Features:
- Two-stage NAS (Architecture + Hyperparameters)
- Real-time visualization and progress tracking
- Intelligent pruning with baseline comparison
- Jupyter notebook support for interactive use
- Automatic result analysis and reporting

Quick Start:
-----------
1. Jupyter Notebook (Recommended):
   jupyter notebook Quick_NAS_Test.ipynb

2. Interactive launcher:
   python run_nas.py

3. Python API:
   from nas_toolkit import run_nas
   results = run_nas(architecture_trials=10, hyperparams_trials=15)

Author: GitHub Copilot
Date: August 2025
"""

import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import main NAS functions
try:
    from .optuna_nas import (
        run_two_stage_nas,
        evaluate_baseline_model,
        load_and_test_best_model,
        OptimizedMLP,
        create_progress_plot,
        progress_tracker
    )
    
    # Convenience function
    def run_nas(architecture_trials=30, hyperparams_trials=50, storage_url=None):
        """
        Run Neural Architecture Search with default settings
        
        Args:
            architecture_trials: Number of architecture search trials
            hyperparams_trials: Number of hyperparameter search trials
            storage_url: Optuna storage URL (optional)
            
        Returns:
            tuple: (arch_study, hp_study, final_best_params)
        """
        return run_two_stage_nas(architecture_trials, hyperparams_trials, storage_url)
    
    __all__ = [
        'run_nas',
        'run_two_stage_nas',
        'evaluate_baseline_model',
        'load_and_test_best_model',
        'OptimizedMLP',
        'create_progress_plot',
        'progress_tracker'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import NAS functions: {e}")
    print("Please ensure all dependencies are installed:")
    print("  pip install optuna matplotlib tqdm")

# Version info
__version__ = "1.0.0"
__author__ = "GitHub Copilot"

def get_info():
    """Get NAS toolkit information"""
    return {
        'version': __version__,
        'author': __author__,
        'description': 'Neural Architecture Search toolkit for NeWRF',
        'features': [
            'Two-stage NAS (Architecture + Hyperparameters)',
            'Real-time visualization',
            'Intelligent pruning',
            'Multiple testing modes',
            'Automatic result analysis'
        ]
    }

if __name__ == "__main__":
    print("🚀 NAS Toolkit for NeWRF")
    print("=" * 40)
    
    info = get_info()
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print(f"Description: {info['description']}")
    print("\nFeatures:")
    for feature in info['features']:
        print(f"  ✅ {feature}")
    
    print("\n🔧 Quick Start:")
    print("  python fast_test.py        # Fast test (3-5 min)")
    print("  python intelligent_test.py # Standard test (15 min)")  
    print("  python visualization_test.py # Complete test (8-10 min)")
    
    print("\n📁 Files:")
    files = [
        'optuna_nas.py - Main NAS implementation',
        'fast_test.py - Quick functionality test',
        'intelligent_test.py - Standard NAS test',
        'visualization_test.py - Complete visualization test',
        'NAS_使用说明.md - Usage documentation (Chinese)',
        '可视化功能说明.md - Visualization guide (Chinese)',
        '可视化实现成功.md - Implementation report (Chinese)'
    ]
    
    for file in files:
        print(f"  📄 {file}")
