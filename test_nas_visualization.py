#!/usr/bin/env python3
"""
NAS可视化测试脚本
可视化功能包括：
1. 实时进度条和状态显示
2. 损失函数变化图表
3. 架构和超参数搜索进度对比
4. 自动保存进度图表
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optuna_nas import *
import time
import matplotlib.pyplot as plt

def test_visualization_nas():
    """测试NAS可视化功能"""
    print("🚀 开始NAS可视化测试")
    print("=" * 60)
    
    # 配置matplotlib中文显示
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置小规模测试参数 - 快速验证可视化功能
    architecture_trials = 5  # 架构搜索试验数
    hyperparams_trials = 8   # 超参数搜索试验数
    
    print(f"📊 测试配置:")
    print(f"   架构搜索试验: {architecture_trials}")
    print(f"   超参数搜索试验: {hyperparams_trials}")
    print(f"   预计总时间: ~8-10分钟")
    print(f"   可视化更新: 每30秒")
    print()
    
    print("📈 可视化功能:")
    print("   ✅ 实时进度条 (tqdm)")
    print("   ✅ 后台进度监控")
    print("   ✅ 自动生成图表")
    print("   ✅ 性能对比图")
    print("   ✅ 阶段进度条")
    print()
    
    # 开始NAS搜索
    start_time = time.time()
    
    try:
        print("🎬 开始两阶段NAS搜索（带可视化）...")
        
        # 运行NAS（已包含可视化功能）
        arch_study, hp_study, final_params = run_two_stage_nas(
            architecture_trials=architecture_trials,
            hyperparams_trials=hyperparams_trials,
            storage_url="sqlite:///nas_visualization_test.db"
        )
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🏁 NAS可视化测试完成!")
        print(f"⏱️  总耗时: {total_time/60:.2f} 分钟")
        
        if final_params:
            print("📊 生成的可视化文件:")
            
            # 检查生成的图表文件
            viz_files = [
                'nas_progress.png',
                'final_nas_results.png'
            ]
            
            for file in viz_files:
                if os.path.exists(file):
                    print(f"   ✅ {file}")
                else:
                    print(f"   ❌ {file} (未生成)")
            
            print("\n📈 可视化内容说明:")
            print("   🔹 左上: 架构搜索损失变化曲线")
            print("   🔹 右上: 超参数搜索损失变化曲线")  
            print("   🔹 左下: 各阶段完成进度条")
            print("   🔹 右下: Baseline vs 最佳模型性能对比")
            
            # 创建额外的详细分析图
            create_detailed_analysis_plot(arch_study, hp_study, final_params)
            
        else:
            print("❌ 搜索失败，请检查日志")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 测试总结:")
    print("   如果看到进度条和状态更新，说明可视化功能正常")
    print("   如果生成了PNG图表文件，说明图表功能正常")
    print("   你可以在运行目录中查看生成的图表文件")
    print("=" * 60)


def create_detailed_analysis_plot(arch_study, hp_study, final_params):
    """Create detailed analysis charts"""
    try:
        if not arch_study or not hp_study:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NAS Detailed Analysis Report', fontsize=16, fontweight='bold')
        
        # 1. Architecture search trial distribution
        arch_trials = [t for t in arch_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if arch_trials:
            arch_values = [t.value for t in arch_trials]
            ax1.hist(arch_values, bins=min(10, len(arch_values)), alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(arch_study.best_value, color='red', linestyle='--', linewidth=2, label=f'Best: {arch_study.best_value:.6f}')
            ax1.set_xlabel('Validation Loss')
            ax1.set_ylabel('Number of Trials')
            ax1.set_title('Architecture Search Results Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Hyperparameter search trial distribution
        hp_trials = [t for t in hp_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if hp_trials:
            hp_values = [t.value for t in hp_trials]
            ax2.hist(hp_values, bins=min(10, len(hp_values)), alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(hp_study.best_value, color='red', linestyle='--', linewidth=2, label=f'Best: {hp_study.best_value:.6f}')
            ax2.set_xlabel('Validation Loss')
            ax2.set_ylabel('Number of Trials')
            ax2.set_title('Hyperparameter Search Results Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Trial status statistics
        arch_complete = len([t for t in arch_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        arch_pruned = len([t for t in arch_study.trials if t.state == optuna.trial.TrialState.PRUNED])
        arch_failed = len([t for t in arch_study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        hp_complete = len([t for t in hp_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        hp_pruned = len([t for t in hp_study.trials if t.state == optuna.trial.TrialState.PRUNED])
        hp_failed = len([t for t in hp_study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        categories = ['Complete', 'Pruned', 'Failed']
        arch_counts = [arch_complete, arch_pruned, arch_failed]
        hp_counts = [hp_complete, hp_pruned, hp_failed]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, arch_counts, width, label='Architecture Search', alpha=0.7, color='blue')
        bars2 = ax3.bar(x + width/2, hp_counts, width, label='Hyperparameter Search', alpha=0.7, color='green')
        
        ax3.set_xlabel('Trial Status')
        ax3.set_ylabel('Number of Trials')
        ax3.set_title('Trial Status Statistics')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        # 4. Best parameter display
        if final_params:
            param_text = "Best Configuration Parameters:\n"
            arch_params = ['n_layers', 'hidden_dim', 'dropout_rate', 'activation', 'use_batch_norm', 'init_method', 'skip_connections']
            hp_params = ['batch_size', 'learning_rate', 'weight_decay', 'scheduler_patience', 'scheduler_factor']
            
            param_text += "\nArchitecture Parameters:\n"
            for param in arch_params:
                if param in final_params:
                    param_text += f"  {param}: {final_params[param]}\n"
            
            param_text += "\nHyperparameters:\n"
            for param in hp_params:
                if param in final_params:
                    param_text += f"  {param}: {final_params[param]}\n"
            
            ax4.text(0.05, 0.95, param_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax4.set_title('Best Configuration')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('detailed_nas_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Detailed analysis chart saved: detailed_nas_analysis.png")
        
    except Exception as e:
        print(f"Failed to create detailed analysis chart: {e}")


def monitor_progress_realtime():
    """Real-time progress monitoring independent function"""
    print("📱 Starting real-time progress monitoring...")
    print("   (This function will continuously display NAS search progress)")
    print("   Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            progress_tracker.print_status()
            create_progress_plot('realtime_progress.png')
            print(f"📊 Progress chart updated: realtime_progress.png")
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Waiting 30 seconds for next update...")
            print("-" * 60)
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n⏹️  Progress monitoring stopped")
    except Exception as e:
        print(f"Progress monitoring error: {e}")


if __name__ == "__main__":
    # Check if necessary packages are installed
    try:
        import matplotlib.pyplot as plt
        import tqdm
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install matplotlib tqdm")
        exit(1)
    
    print("🧪 NAS Visualization Function Test")
    print("Choose test mode:")
    print("1. Complete NAS test (with visualization)")
    print("2. Real-time progress monitoring only")
    
    choice = input("Please enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_visualization_nas()
    elif choice == "2":
        monitor_progress_realtime()
    else:
        print("❌ Invalid choice")
        test_visualization_nas()
