"""
快速NAS测试脚本 - 带可视化功能
预计运行时间: 3-5分钟
包含完整的可视化功能测试
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from optuna_nas import run_two_stage_nas, load_and_test_best_model, evaluate_baseline_model
import logging
import os
import time
import matplotlib.pyplot as plt

# 配置matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fast_nas_test.log'),
        logging.StreamHandler()
    ]
)

def test_fast_nas():
    """快速NAS测试 - 带可视化功能"""
    try:
        print("🚀 快速NAS测试开始")
        print("=" * 50)
        logging.info("🚀 开始快速NAS测试")
        
        # 快速测试配置
        architecture_trials = 3
        hyperparams_trials = 3
        
        print(f"⚡ 快速测试配置:")
        print(f"   架构搜索: {architecture_trials} 试验")
        print(f"   超参数搜索: {hyperparams_trials} 试验") 
        print(f"   预计时间: 3-5分钟")
        print(f"   包含可视化: ✅")
        print()
        
        start_time = time.time()
        
        # 清理之前的数据库文件
        db_files = ['two_stage_nas.db', 'two_stage_nas.db-journal', 'fast_nas_test.db']
        for f in db_files:
            if os.path.exists(f):
                os.remove(f)
                logging.info(f"清理旧数据库文件: {f}")
        
        logging.info("⚡ 使用快速模式：减少迭代次数")
        
        # 运行快速NAS测试（带可视化）
        arch_study, hp_study, final_params = run_two_stage_nas(
            architecture_trials=architecture_trials,
            hyperparams_trials=hyperparams_trials,
            storage_url="sqlite:///fast_nas_test.db"
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("✅ 快速NAS测试完成!")
        print(f"⏱️  总耗时: {elapsed_time/60:.2f} 分钟")
        
        if final_params is None:
            print("❌ NAS搜索失败！")
            logging.error("❌ NAS搜索失败！")
            return
        
        print("\n📊 生成的可视化文件:")
        viz_files = ['nas_progress.png', 'final_nas_results.png']
        for file in viz_files:
            if os.path.exists(file):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file}")
        
        print(f"\n🎯 测试结果:")
        baseline_perf = evaluate_baseline_model()
        
        if arch_study and hp_study:
            best_loss = min(arch_study.best_value, hp_study.best_value) if arch_study and hp_study else float('inf')
            improvement = (baseline_perf['loss'] - best_loss) / baseline_perf['loss'] * 100 if best_loss != float('inf') else 0
            
            print(f"   Baseline: {baseline_perf['loss']:.6f}")
            print(f"   最佳结果: {best_loss:.6f}")
            print(f"   改进程度: {improvement:+.2f}%")
            
            if improvement > 0:
                print("   🎉 找到了更好的配置!")
            else:
                print("   📊 结果记录完成，可用于分析")
        
        logging.info("✅ 快速NAS测试完成！")
        
        # 尝试加载最佳模型
        try:
            model, params = load_and_test_best_model()
            logging.info("✅ 最佳模型加载成功！")
            
            # 显示配置
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(f"最佳模型参数数: {total_params:,}")
            logging.info(f"最佳配置: {params}")
                    
        except Exception as e:
            logging.error(f"❌ 模型加载失败: {e}")
        
        print("\n" + "=" * 50)
        print("📋 快速测试总结:")
        print("   - 验证了NAS搜索流程")
        print("   - 测试了可视化功能")
        print("   - 生成了进度图表")
        print("   - 适合快速验证功能")
            
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        logging.error(f"❌ 快速NAS测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fast_nas()
