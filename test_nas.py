"""
两阶段 NAS 测试脚本
"""

import sys
sys.path.append('/home/byang/BoYang/mNeWRF')

from optuna_nas import run_two_stage_nas, load_and_test_best_model
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nas_test.log'),
        logging.StreamHandler()
    ]
)

def test_nas():
    """测试两阶段NAS"""
    try:
        logging.info("🧪 开始测试两阶段NAS")
        
        # 运行小规模测试
        arch_study, hp_study, final_params = run_two_stage_nas(
            architecture_trials=3,  # 小规模测试
            hyperparams_trials=5
        )
        
        logging.info("✅ NAS测试完成！")
        
        # 尝试加载最佳模型
        try:
            model, params = load_and_test_best_model()
            logging.info("✅ 最佳模型加载成功！")
            
            # 显示参数数量
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(f"模型参数总数: {total_params:,}")
            
        except Exception as e:
            logging.error(f"❌ 模型加载失败: {e}")
            
    except Exception as e:
        logging.error(f"❌ NAS测试失败: {e}")
        raise

if __name__ == "__main__":
    test_nas()
