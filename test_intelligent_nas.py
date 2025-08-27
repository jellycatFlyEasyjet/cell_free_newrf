"""
智能剪枝NAS测试脚本
测试两阶段搜索和与baseline的比较
"""

import sys
sys.path.append('/home/byang/BoYang/mNeWRF')

from optuna_nas import run_two_stage_nas, load_and_test_best_model, evaluate_baseline_model
import logging
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('intelligent_nas_test.log'),
        logging.StreamHandler()
    ]
)

def test_intelligent_nas():
    """测试智能剪枝的两阶段NAS"""
    try:
        logging.info("🧪 开始测试智能剪枝两阶段NAS")
        
        # 清理之前的数据库文件
        db_files = ['two_stage_nas.db', 'two_stage_nas.db-journal']
        for f in db_files:
            if os.path.exists(f):
                os.remove(f)
                logging.info(f"清理旧数据库文件: {f}")
        
        # 先评估baseline（这会缓存结果）
        baseline_perf = evaluate_baseline_model()
        
        # 运行小规模智能NAS测试
        arch_study, hp_study, final_params = run_two_stage_nas(
            architecture_trials=5,   # 小规模测试
            hyperparams_trials=8
        )
        
        if final_params is None:
            logging.error("❌ NAS搜索失败！")
            return
        
        logging.info("✅ 智能NAS测试完成！")
        
        # 尝试加载最佳模型
        try:
            model, params = load_and_test_best_model()
            logging.info("✅ 最佳模型加载成功！")
            
            # 显示详细对比
            total_params = sum(p.numel() for p in model.parameters())
            logging.info("\n" + "="*50)
            logging.info("📊 最终对比结果:")
            logging.info(f"Baseline - 参数数: {baseline_perf['model_params']:,}, Loss: {baseline_perf['loss']:.6f}, SNR: {baseline_perf['snr']:.2f} dB")
            logging.info(f"最佳模型 - 参数数: {total_params:,}, 配置如下:")
            
            # 显示架构配置
            arch_keys = ['n_layers', 'hidden_dim', 'dropout_rate', 'activation', 'use_batch_norm', 'init_method', 'skip_connections']
            logging.info("  架构配置:")
            for key in arch_keys:
                if key in params:
                    logging.info(f"    {key}: {params[key]}")
            
            # 显示超参数配置
            hp_keys = ['batch_size', 'learning_rate', 'weight_decay', 'scheduler_patience', 'scheduler_factor']
            logging.info("  超参数配置:")
            for key in hp_keys:
                if key in params:
                    logging.info(f"    {key}: {params[key]}")
                    
        except Exception as e:
            logging.error(f"❌ 模型加载失败: {e}")
            
    except Exception as e:
        logging.error(f"❌ 智能NAS测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_intelligent_nas()
