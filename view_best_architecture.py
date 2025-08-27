#!/usr/bin/env python3
"""
查看NAS搜索得到的最佳模型架构 - 完整版本
展示 get_architecture_info 函数的实际用途
"""

import pickle
import torch
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optuna_nas import OptimizedMLP

class FakeTrial:
    """用于重建模型的虚拟trial类"""
    def __init__(self, params):
        self.params = params
    
    def suggest_int(self, name, low, high):
        return self.params.get(name, (low + high) // 2)
    
    def suggest_float(self, name, low, high):
        return self.params.get(name, (low + high) / 2)
    
    def suggest_loguniform(self, name, low, high):
        return self.params.get(name, (low * high) ** 0.5)
    
    def suggest_categorical(self, name, choices):
        return self.params.get(name, choices[0])

def load_and_view_best_model(params_file='final_best_params.pkl'):
    """加载并查看最佳模型架构"""
    try:
        # 加载最佳参数
        with open(params_file, 'rb') as f:
            best_params = pickle.load(f)
        
        print("📋 最佳模型参数:")
        print("=" * 60)
        
        # 分类显示参数
        architecture_params = {}
        hyperparams = {}
        
        for key, value in best_params.items():
            if any(x in key for x in ['layer_', 'n_layers', 'use_batch_norm', 'init_method', 'skip_connections']):
                architecture_params[key] = value
            else:
                hyperparams[key] = value
        
        # 显示架构参数
        print("🏗️ 架构参数:")
        for key, value in sorted(architecture_params.items()):
            print(f"   {key}: {value}")
        
        print("\n⚙️ 超参数:")
        for key, value in sorted(hyperparams.items()):
            print(f"   {key}: {value}")
        
        # 创建模型并显示详细架构
        print("\n" + "=" * 60)
        fake_trial = FakeTrial(best_params)
        model = OptimizedMLP(fake_trial)
        
        # 显示详细架构信息
        model.print_architecture()
        
        # 显示额外统计信息
        print(f"\n📊 模型统计:")
        arch_info = model.get_architecture_info()
        print(f"   总参数量: {arch_info['total_params']:,}")
        print(f"   可训练参数: {arch_info['trainable_params']:,}")
        
        # 计算每层参数量
        print(f"\n📈 每层参数详情:")
        prev_size = 12  # 输入维度
        total_layer_params = 0
        
        for i, config in enumerate(arch_info['layer_configs']):
            current_size = config['hidden_dim']
            layer_params = prev_size * current_size + current_size  # 权重 + 偏置
            total_layer_params += layer_params
            print(f"   Layer {i}: {prev_size} → {current_size} = {layer_params:,} 参数")
            prev_size = current_size
        
        # 输出层参数
        output_params = prev_size * 2 + 2  # 输出到复数值 (real + imag)
        total_layer_params += output_params
        print(f"   Output Layer: {prev_size} → 2 = {output_params:,} 参数")
        print(f"   层参数总计: {total_layer_params:,}")
        
        return model, best_params
        
    except FileNotFoundError:
        print(f"❌ 未找到参数文件 '{params_file}'")
        print("请先运行完整的NAS搜索: python optuna_nas.py")
        return None, None
    except Exception as e:
        print(f"❌ 加载模型时出错: {e}")
        return None, None

def view_study_results(study_file='architecture_study.db'):
    """查看Optuna研究结果"""
    try:
        import optuna
        
        print("\n📊 Optuna 研究结果:")
        print("=" * 60)
        
        # 加载架构搜索研究
        if os.path.exists('architecture_study.db'):
            study = optuna.load_study(
                study_name='architecture_optimization',
                storage='sqlite:///architecture_study.db'
            )
            
            print(f"🔍 架构搜索:")
            print(f"   总试验数: {len(study.trials)}")
            print(f"   最佳试验: #{study.best_trial.number}")
            print(f"   最佳损失: {study.best_value:.6f}")
            
            # 显示最重要的参数
            print(f"\n🏆 最佳架构参数:")
            for key, value in study.best_params.items():
                if 'layer_' in key or key in ['n_layers', 'use_batch_norm', 'init_method']:
                    print(f"   {key}: {value}")
        
        # 加载超参数搜索研究
        if os.path.exists('hyperparams_study.db'):
            hp_study = optuna.load_study(
                study_name='hyperparams_optimization', 
                storage='sqlite:///hyperparams_study.db'
            )
            
            print(f"\n⚙️ 超参数搜索:")
            print(f"   总试验数: {len(hp_study.trials)}")
            print(f"   最佳试验: #{hp_study.best_trial.number}")
            print(f"   最佳损失: {hp_study.best_value:.6f}")
            
    except ImportError:
        print("❌ 需要安装optuna: pip install optuna")
    except Exception as e:
        print(f"❌ 查看研究结果时出错: {e}")

if __name__ == "__main__":
    print("🔍 查看最佳神经网络架构")
    print("=" * 60)
    
    # 查看最佳模型
    model, params = load_and_view_best_model()
    
    if model is not None:
        print(f"\n✅ 成功加载最佳模型!")
        
        # 可选：查看Optuna研究结果
        print("\n" + "="*60)
        view_study_results()
    
    print("\n🎯 架构查看完成!")
