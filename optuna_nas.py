
#%%
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from tqdm import trange, tqdm
import logging
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loaders import DatasetLoader
from config_function import *
from models import MLP  # 导入baseline模型
import random
import matplotlib.pyplot as plt
from datetime import datetime
import threading

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量存储baseline性能
BASELINE_PERFORMANCE = None

# 全局变量用于进度可视化
class ProgressTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.baseline_progress = 0
        self.arch_trials_completed = 0
        self.arch_trials_total = 0
        self.arch_best_loss = float('inf')
        self.hp_trials_completed = 0
        self.hp_trials_total = 0
        self.hp_best_loss = float('inf')
        self.current_stage = "Preparing"
        self.stage_progress = 0
        self.start_time = time.time()
        self.arch_losses = []
        self.hp_losses = []
        self.baseline_loss = None
    
    def update_baseline(self, progress):
        self.baseline_progress = progress
        self.current_stage = f"Baseline Evaluation ({progress}%)"
    
    def set_baseline_complete(self, loss):
        self.baseline_loss = loss
        self.current_stage = "Baseline Complete, Preparing Architecture Search"
    
    def set_arch_stage(self, total_trials):
        self.arch_trials_total = total_trials
        self.current_stage = "Architecture Search in Progress"
    
    def update_arch_trial(self, completed, best_loss):
        self.arch_trials_completed = completed
        if best_loss < self.arch_best_loss:
            self.arch_best_loss = best_loss
        self.arch_losses.append(best_loss)
        progress = int((completed / self.arch_trials_total) * 100) if self.arch_trials_total > 0 else 0
        self.current_stage = f"Architecture Search ({completed}/{self.arch_trials_total}) - {progress}%"
    
    def set_hp_stage(self, total_trials):
        self.hp_trials_total = total_trials
        self.current_stage = "Hyperparameter Search in Progress"
    
    def update_hp_trial(self, completed, best_loss):
        self.hp_trials_completed = completed
        if best_loss < self.hp_best_loss:
            self.hp_best_loss = best_loss
        self.hp_losses.append(best_loss)
        progress = int((completed / self.hp_trials_total) * 100) if self.hp_trials_total > 0 else 0
        self.current_stage = f"Hyperparameter Search ({completed}/{self.hp_trials_total}) - {progress}%"
    
    def get_elapsed_time(self):
        return time.time() - self.start_time
    
    def print_status(self):
        elapsed = self.get_elapsed_time()
        print(f"\n{'='*60}")
        print(f"🚀 NAS Progress Status | Elapsed: {elapsed/60:.1f} minutes")
        print(f"Current Stage: {self.current_stage}")
        
        if self.baseline_loss:
            print(f"📊 Baseline Performance: Loss={self.baseline_loss:.6f}")
        
        if self.arch_trials_completed > 0:
            arch_progress = (self.arch_trials_completed / self.arch_trials_total * 100) if self.arch_trials_total > 0 else 0
            print(f"🏗️  Architecture Search: {self.arch_trials_completed}/{self.arch_trials_total} ({arch_progress:.0f}%) | Best Loss: {self.arch_best_loss:.6f}")
        
        if self.hp_trials_completed > 0:
            hp_progress = (self.hp_trials_completed / self.hp_trials_total * 100) if self.hp_trials_total > 0 else 0
            print(f"⚡ Hyperparameter Search: {self.hp_trials_completed}/{self.hp_trials_total} ({hp_progress:.0f}%) | Best Loss: {self.hp_best_loss:.6f}")
        
        print(f"{'='*60}")

# 全局进度追踪器
progress_tracker = ProgressTracker()


def evaluate_baseline_model():
    """评估baseline模型性能"""
    global BASELINE_PERFORMANCE
    
    if BASELINE_PERFORMANCE is not None:
        return BASELINE_PERFORMANCE
    
    logging.info("📏 Evaluating Baseline model performance...")
    progress_tracker.current_stage = "Baseline Model Evaluation in Progress"
    
    try:
        # 使用与NAS相同的设置
        dataset_fname = '/home/byang/BoYang/mNeWRF/dataset/conference_2000STA_4APs.pkl'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 固定随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # 创建baseline模型
        baseline_model = MLP()
        baseline_model.to(device)
        
        # 使用固定的训练配置
        learning_rate = 5e-4
        weight_decay = 0
        batch_size = 1024
        n_iters = 200  # 🚀 大幅减少baseline评估时间
        
        optimizer = torch.optim.Adam(
            baseline_model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9, min_lr=1e-6)
        
        # 数据加载
        loader = DatasetLoader(dataset_fname)
        loader.split_train_val_test(train_ratio=0.8, val_ratio=0.1, seed=42)
        
        base_AP = [1]
        target_AP = [2]
        
        # 训练循环 - 添加进度条
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 50  # 🚀 大幅减少早停耐心
        
        with tqdm(range(n_iters), desc="🔬 Baseline Training", ncols=80) as pbar:
            for i in pbar:
                baseline_model.train()
                
                # 训练步骤
                sta_id = np.random.choice(loader.trainset, batch_size)
                sta_loc = torch.tensor(loader.get_loc_batch("STA", sta_id), device=device, dtype=torch.float32)
                
                input_cfr = torch.tensor(loader.get_cfr_batch(base_AP, sta_id).flatten(), device=device)
                input_cfr = torch.stack([torch.real(input_cfr), torch.imag(input_cfr)], dim=-1)
                
                target_cfr = torch.tensor(loader.get_cfr_batch(target_AP, sta_id).flatten(), device=device)
                
                output_cfr = baseline_model(sta_loc, input_cfr)
                loss = compute_mse_loss(output_cfr, target_cfr)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 更新进度条
                progress = int((i / n_iters) * 100)
                progress_tracker.update_baseline(progress)
                
                # 验证和早停
                if i % 100 == 0 or i == n_iters - 1:
                    baseline_model.eval()
                    with torch.no_grad():
                        val_sta_id = loader.valset
                        val_sta_loc = torch.tensor(loader.get_loc_batch("STA", val_sta_id), device=device, dtype=torch.float32)
                        
                        val_input_cfr = torch.tensor(loader.get_cfr_batch(base_AP, val_sta_id).flatten(), device=device)
                        val_input_cfr = torch.stack([torch.real(val_input_cfr), torch.imag(val_input_cfr)], dim=-1)
                        
                        val_target_cfr = torch.tensor(loader.get_cfr_batch(target_AP, val_sta_id).flatten(), device=device)
                        
                        val_output_cfr = baseline_model(val_sta_loc, val_input_cfr)
                        val_loss = compute_mse_loss(val_output_cfr, val_target_cfr)
                        
                        scheduler.step(val_loss)
                        
                        if val_loss.item() < best_val_loss:
                            best_val_loss = val_loss.item()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        # 更新进度条信息
                        pbar.set_postfix({
                            'Loss': f'{loss.item():.6f}',
                            'ValLoss': f'{val_loss.item():.6f}',
                            'Best': f'{best_val_loss:.6f}',
                            'SNR': f'{compute_snr_db(best_val_loss):.1f}dB'
                        })
                        
                        if patience_counter >= early_stopping_patience:
                            logging.info(f"Baseline early stopping at iteration {i}")
                            break
                
                if i % 200 == 0:
                    torch.cuda.empty_cache()
        
        BASELINE_PERFORMANCE = {
            'loss': best_val_loss,
            'snr': compute_snr_db(best_val_loss),
            'model_params': sum(p.numel() for p in baseline_model.parameters())
        }
        
        progress_tracker.set_baseline_complete(best_val_loss)
        
        logging.info(f"✅ Baseline model evaluation completed!")
        logging.info(f"   Validation Loss: {BASELINE_PERFORMANCE['loss']:.6f}")
        logging.info(f"   SNR: {BASELINE_PERFORMANCE['snr']:.2f} dB")
        logging.info(f"   Parameter Count: {BASELINE_PERFORMANCE['model_params']:,}")
        
        # 清理内存
        del baseline_model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
        
        return BASELINE_PERFORMANCE
        
    except Exception as e:
        logging.error(f"Baseline model evaluation failed: {e}")
        # Set a conservative baseline
        BASELINE_PERFORMANCE = {'loss': 1.0, 'snr': 0.0, 'model_params': 100000}
        progress_tracker.set_baseline_complete(1.0)
        return BASELINE_PERFORMANCE


class IntelligentPruner:
    """Intelligent pruner based on comparison with baseline"""
    
    def __init__(self, baseline_performance, improvement_threshold=0.02):
        self.baseline_loss = baseline_performance['loss']
        self.baseline_snr = baseline_performance['snr']
        self.improvement_threshold = improvement_threshold  # Minimum improvement threshold
        self.trial_history = []
        
    def should_prune(self, trial, step, intermediate_value):
        """Determine whether to prune"""
        
        # Record trial history
        if len(self.trial_history) >= step:
            self.trial_history[step - 1].append(intermediate_value)
        else:
            while len(self.trial_history) < step:
                self.trial_history.append([])
            self.trial_history[step - 1].append(intermediate_value)
        
        # Don't prune in early stages, give model enough training time
        if step < 5:
            return False
        
        # If current loss is much larger than baseline, consider pruning
        if intermediate_value > self.baseline_loss * (1 + self.improvement_threshold * 3):
            logging.info(f"🔪 Trial {trial.number} pruned: loss {intermediate_value:.6f} >> baseline {self.baseline_loss:.6f}")
            return True
        
        # If there's still no significant improvement in mid-to-late stages, prune
        if step >= 10:
            improvement = (self.baseline_loss - intermediate_value) / self.baseline_loss
            if improvement < -self.improvement_threshold:  # Negative improvement (worse)
                logging.info(f"🔪 Trial {trial.number} pruned: no improvement over baseline")
                return True
        
        # Dynamic pruning based on historical performance
        if step >= 15 and len(self.trial_history[step - 1]) >= 5:
            # Get historical best performance at the same step
            historical_best = min(self.trial_history[step - 1])
            if intermediate_value > historical_best * 1.2:  # 20% worse than historical best
                logging.info(f"🔪 Trial {trial.number} pruned: worse than historical best")
                return True
        
        return False

class OptimizedMLP(nn.Module):
    """可配置的 MLP 模型，用于架构搜索 - 每层独立配置"""
    def __init__(self, trial, input_dim=5):  # 默认维度改为12：STA_loc(3) + input_cfr(9)
        super(OptimizedMLP, self).__init__()
        
        # 全局架构参数
        self.n_layers = trial.suggest_int('n_layers', 3, 8)  # 网络深度
        self.use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        self.init_method = trial.suggest_categorical('init_method', ['orthogonal'])
        self.skip_connections = trial.suggest_categorical('skip_connections', [False])
        
        # 每层独立的配置
        self.layer_configs = []
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if self.use_batch_norm else None
        self.dropouts = nn.ModuleList()
        self.activations = []  # 存储每层的激活函数类型
        
        # 可选的隐藏层尺寸和激活函数
        hidden_dim_choices = [32, 64, 96, 128, 160, 192, 256]
        activation_choices = ['relu', 'leaky_relu', 'tanh', 'elu']
        dropout_choices = [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        
        # 第一层（输入层到第一个隐藏层）
        first_hidden_dim = trial.suggest_categorical('layer_0_hidden_dim', hidden_dim_choices)
        first_activation = trial.suggest_categorical('layer_0_activation', activation_choices)
        first_dropout = trial.suggest_categorical('layer_0_dropout', dropout_choices)
        
        self.layer_configs.append({
            'hidden_dim': first_hidden_dim,
            'activation': first_activation,
            'dropout': first_dropout
        })
        self.activations.append(first_activation)
        
        # 构建第一层
        self.layers.append(nn.Linear(input_dim, first_hidden_dim))
        if self.use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(first_hidden_dim))
        self.dropouts.append(nn.Dropout(first_dropout))
        
        # 构建中间隐藏层
        prev_dim = first_hidden_dim
        for i in range(1, self.n_layers - 1):  # -1 因为还需要输出层
            layer_hidden_dim = trial.suggest_categorical(f'layer_{i}_hidden_dim', hidden_dim_choices)
            layer_activation = trial.suggest_categorical(f'layer_{i}_activation', activation_choices)
            layer_dropout = trial.suggest_categorical(f'layer_{i}_dropout', dropout_choices)
            
            self.layer_configs.append({
                'hidden_dim': layer_hidden_dim,
                'activation': layer_activation,
                'dropout': layer_dropout
            })
            self.activations.append(layer_activation)
            
            # 构建层
            self.layers.append(nn.Linear(prev_dim, layer_hidden_dim))
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_hidden_dim))
            self.dropouts.append(nn.Dropout(layer_dropout))
            
            prev_dim = layer_hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, 2)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """根据指定方法初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.init_method == 'he':
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                elif self.init_method == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif self.init_method == 'orthogonal':
                    nn.init.orthogonal_(module.weight)
                elif self.init_method == 'normal':
                    nn.init.normal_(module.weight, mean=0.0, std=0.1)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def _get_activation(self, activation_name):
        """获取激活函数"""
        if activation_name == 'relu':
            return F.relu
        elif activation_name == 'leaky_relu':
            return F.leaky_relu
        elif activation_name == 'tanh':
            return torch.tanh
        elif activation_name == 'elu':
            return F.elu
        elif activation_name == 'swish':
            return lambda x: x * torch.sigmoid(x)
        elif activation_name == 'gelu':
            return F.gelu
        else:
            return F.relu
    
    def forward(self, STA_loc, input_cfr):
        # 拼接输入
        x = torch.cat([STA_loc, input_cfr], dim=-1)
        
        # 前向传播 - 每层使用自己的激活函数
        residual = None
        for i, layer in enumerate(self.layers):
            # 跳跃连接
            if self.skip_connections and i > 0 and residual is not None and residual.shape == x.shape:
                x = x + residual
            
            if self.skip_connections:
                residual = x
            
            # 线性变换
            x = layer(x)
            
            # BatchNorm
            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # 每层使用自己的激活函数
            activation_fn = self._get_activation(self.activations[i])
            x = activation_fn(x)
            
            # Dropout
            if i < len(self.dropouts):
                x = self.dropouts[i](x)
        
        # 输出层
        x = self.output_layer(x)
        
        # 输出复数值
        re_ch = torch.tanh(x[..., 0])
        im_ch = torch.tanh(x[..., 1])
        output = re_ch + 1j * im_ch
        return output
    
    def get_architecture_info(self):
        """获取网络架构的详细信息"""
        info = {
            'total_layers': self.n_layers,
            'use_batch_norm': self.use_batch_norm,
            'init_method': self.init_method,
            'skip_connections': self.skip_connections,
            'layer_configs': self.layer_configs,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        return info
    
    def print_architecture(self):
        """打印网络架构信息"""
        print("🏗️ Network Architecture:")
        print(f"   Total Layers: {self.n_layers}")
        print(f"   Skip Connections: {self.skip_connections}")
        print(f"   Batch Normalization: {self.use_batch_norm}")
        print(f"   Weight Initialization: {self.init_method}")
        print("   Layer Details:")
        
        for i, config in enumerate(self.layer_configs):
            print(f"     Layer {i}: {config['hidden_dim']} neurons, "
                  f"{config['activation']} activation, "
                  f"{config['dropout']:.2f} dropout")
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total Parameters: {total_params:,}")


def display_best_architecture(best_params, stage="Final"):
    """显示最佳架构的详细信息"""
    print("\n" + "=" * 70)
    print(f"🏗️ {stage} 最佳架构详情")
    print("=" * 70)
    
    try:
        # 创建虚拟trial来重建模型
        class DisplayTrial:
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
        
        display_trial = DisplayTrial(best_params)
        model = OptimizedMLP(display_trial)
        
        # 显示架构信息
        model.print_architecture()
        
        # 显示参数分布
        print("\n📋 参数详细分布:")
        architecture_params = {}
        hyperparams = {}
        
        for key, value in best_params.items():
            if any(x in key for x in ['layer_', 'n_layers', 'use_batch_norm', 'init_method', 'skip_connections']):
                architecture_params[key] = value
            else:
                hyperparams[key] = value
        
        if architecture_params:
            print("   🏗️ 架构参数:")
            for key, value in sorted(architecture_params.items()):
                print(f"      {key}: {value}")
        
        if hyperparams:
            print("   ⚙️ 训练超参数:")
            for key, value in sorted(hyperparams.items()):
                print(f"      {key}: {value}")
                
    except Exception as e:
        logging.error(f"显示架构信息时出错: {e}")


def compute_mse_loss(predicted, true):
    """计算相对MSE损失"""
    return torch.sum(torch.abs(predicted - true) ** 2) / torch.sum(torch.abs(true) ** 2)


def compute_snr_db(loss):
    """计算SNR (dB)"""
    return -10.0 * np.log10(loss)


def objective_architecture(trial):
    """第一阶段：网络架构搜索（与baseline比较）"""
    try:
        # 获取baseline性能作为参考
        baseline_perf = evaluate_baseline_model()
        
        # 固定超参数，只搜索网络架构
        batch_size = 1024
        learning_rate = 5e-4
        weight_decay = 0
        scheduler_patience = 10
        scheduler_factor = 0.9
        train_split = 0.8
        val_split = 0.1
        n_iters = 200  # 🚀 大幅减少架构搜索时间
        
        # 2. 数据和设备设置
        dataset_fname = '/home/byang/BoYang/mNeWRF/dataset/conference_2000STA_4APs.pkl'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 固定随机种子确保可复现性
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # 3. 创建模型（只搜索架构参数）
        model = OptimizedMLP(trial)
        model.to(device)
        
        # 🔍 模型复杂度检查
        model_params = sum(p.numel() for p in model.parameters())
        baseline_params = baseline_perf['model_params']
        
        # 如果模型过于复杂但没有明显改进潜力，提前剪枝
        if model_params > baseline_params * 3:  # 参数量超过baseline 3倍
            logging.info(f"🔪 Trial {trial.number} pruned: too complex ({model_params:,} vs {baseline_params:,} params)")
            raise optuna.TrialPruned()
        
        # 4. 优化器设置
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', 
            patience=scheduler_patience, 
            factor=scheduler_factor, 
            min_lr=1e-6
        )
        
        # 5. 数据加载
        loader = DatasetLoader(dataset_fname)
        loader.split_train_val_test(
            train_ratio=train_split, 
            val_ratio=val_split, 
            seed=42
        )
        
        base_AP = [1]
        target_AP = [2]
        
        # 训练循环 - 添加进度条
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 50  # 🚀 减少架构搜索早停耐心
        
        # 📊 性能追踪
        val_losses_history = []
        
        with tqdm(range(n_iters), desc=f"🏗️ Trial{trial.number:3d}", ncols=100, leave=False) as pbar:
            for i in pbar:
                model.train()
                
                # 随机采样训练数据
                sta_id = np.random.choice(loader.trainset, batch_size)
                sta_loc = torch.tensor(loader.get_loc_batch("STA", sta_id), device=device, dtype=torch.float32)
                
                input_cfr = torch.tensor(loader.get_cfr_batch(base_AP, sta_id).flatten(), device=device)
                input_cfr = torch.stack([torch.real(input_cfr), torch.imag(input_cfr)], dim=-1)
                
                target_cfr = torch.tensor(loader.get_cfr_batch(target_AP, sta_id).flatten(), device=device)
                
                # 前向传播
                output_cfr = model(sta_loc, input_cfr)
                loss = compute_mse_loss(output_cfr, target_cfr)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 验证和剪枝检查
                if i % 100 == 0 or i == n_iters - 1:
                    model.eval()
                    with torch.no_grad():
                        val_sta_id = loader.valset
                        val_sta_loc = torch.tensor(loader.get_loc_batch("STA", val_sta_id), device=device, dtype=torch.float32)
                        
                        val_input_cfr = torch.tensor(loader.get_cfr_batch(base_AP, val_sta_id).flatten(), device=device)
                        val_input_cfr = torch.stack([torch.real(val_input_cfr), torch.imag(val_input_cfr)], dim=-1)
                        
                        val_target_cfr = torch.tensor(loader.get_cfr_batch(target_AP, val_sta_id).flatten(), device=device)
                        
                        val_output_cfr = model(val_sta_loc, val_input_cfr)
                        val_loss = compute_mse_loss(val_output_cfr, val_target_cfr)
                        
                        val_snr = compute_snr_db(val_loss.item())
                        val_losses_history.append(val_loss.item())
                        
                        # 学习率调度
                        scheduler.step(val_loss)
                        
                        # 早停检查
                        if val_loss.item() < best_val_loss:
                            best_val_loss = val_loss.item()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        # 更新进度条信息
                        improvement = (baseline_perf['loss'] - val_loss.item()) / baseline_perf['loss'] * 100
                        pbar.set_postfix({
                            'Loss': f'{val_loss.item():.6f}',
                            'Best': f'{best_val_loss:.6f}',
                            'SNR': f'{val_snr:.1f}dB',
                            'Δ': f'{improvement:+.1f}%',
                            'Params': f'{model_params//1000}k'
                        })
                        
                        if patience_counter >= early_stopping_patience:
                            logging.info(f"Early stopping at iteration {i}")
                            break
                        
                        # 报告中间结果给 Optuna
                        step = i // 100 + 1
                        trial.report(val_loss.item(), step)
                        
                        # 🔪 智能剪枝：与baseline比较
                        if step >= 3:  # 给模型一些训练时间
                            # 如果连续多步都比baseline差很多，剪枝
                            recent_losses = val_losses_history[-3:] if len(val_losses_history) >= 3 else val_losses_history
                            avg_recent_loss = sum(recent_losses) / len(recent_losses)
                            
                            improvement = (baseline_perf['loss'] - avg_recent_loss) / baseline_perf['loss']
                            
                            if improvement < -0.05:  # 比baseline差5%以上
                                logging.info(f"🔪 Trial {trial.number} pruned: worse than baseline by {abs(improvement)*100:.1f}%")
                                raise optuna.TrialPruned()
                        
                        # 标准剪枝检查
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                
                # 定期清理GPU内存
                if i % 200 == 0:
                    torch.cuda.empty_cache()
        
        # 7. 最终性能评估
        final_improvement = (baseline_perf['loss'] - best_val_loss) / baseline_perf['loss']
        final_snr = compute_snr_db(best_val_loss)
        
        logging.info(f"🎯 Architecture trial {trial.number} completed:")
        logging.info(f"   Val Loss: {best_val_loss:.6f} (baseline: {baseline_perf['loss']:.6f})")
        logging.info(f"   SNR: {final_snr:.2f} dB (baseline: {baseline_perf['snr']:.2f} dB)")
        logging.info(f"   Improvement: {final_improvement*100:.2f}%")
        logging.info(f"   Params: {model_params:,} (baseline: {baseline_params:,})")
        
        # 清理内存
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
        
        return best_val_loss
    
    except optuna.TrialPruned:
        # 清理内存
        torch.cuda.empty_cache()
        raise
    except Exception as e:
        logging.error(f"Architecture trial failed with error: {e}")
        # 清理内存
        torch.cuda.empty_cache()
        return float('inf')


def objective_hyperparams(trial, best_architecture_params, baseline_perf):
    """第二阶段：超参数搜索（基于最佳架构和baseline比较）"""
    try:
        # 1. 使用最佳架构，搜索超参数
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048, 4096])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        scheduler_patience = trial.suggest_int('scheduler_patience', 5, 20)
        scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.9)
        
        # 固定数据划分和迭代次数
        train_split = 0.8
        val_split = 0.1
        n_iters = 800  # 🚀 减少超参数搜索时间
        
        # 2. 数据和设备设置
        dataset_fname = '/home/byang/BoYang/mNeWRF/dataset/conference_2000STA_4APs.pkl'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 固定随机种子确保可复现性
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # 3. 使用最佳架构创建模型
        class FakeTrial:
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
        
        fake_trial = FakeTrial(best_architecture_params)
        model = OptimizedMLP(fake_trial)
        model.to(device)
        
        # 4. 优化器设置（使用搜索的超参数）
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', 
            patience=scheduler_patience, 
            factor=scheduler_factor, 
            min_lr=1e-6
        )
        
        # 5. 数据加载
        loader = DatasetLoader(dataset_fname)
        loader.split_train_val_test(
            train_ratio=train_split, 
            val_ratio=val_split, 
            seed=42
        )
        
        base_AP = [1]
        target_AP = [2]
        
        # 训练循环 - 添加进度条
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 80  # 🚀 减少超参数搜索早停耐心
        
        with tqdm(range(n_iters), desc=f"⚡HP{trial.number:3d}", ncols=100, leave=False) as pbar:
            for i in pbar:
                model.train()
                
                # 随机采样训练数据
                sta_id = np.random.choice(loader.trainset, batch_size)
                sta_loc = torch.tensor(loader.get_loc_batch("STA", sta_id), device=device, dtype=torch.float32)
                
                input_cfr = torch.tensor(loader.get_cfr_batch(base_AP, sta_id).flatten(), device=device)
                input_cfr = torch.stack([torch.real(input_cfr), torch.imag(input_cfr)], dim=-1)
                
                target_cfr = torch.tensor(loader.get_cfr_batch(target_AP, sta_id).flatten(), device=device)
                
                # 前向传播
                output_cfr = model(sta_loc, input_cfr)
                loss = compute_mse_loss(output_cfr, target_cfr)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 验证和早停检查
                if i % 100 == 0 or i == n_iters - 1:
                    model.eval()
                    with torch.no_grad():
                        val_sta_id = loader.valset
                        val_sta_loc = torch.tensor(loader.get_loc_batch("STA", val_sta_id), device=device, dtype=torch.float32)
                        
                        val_input_cfr = torch.tensor(loader.get_cfr_batch(base_AP, val_sta_id).flatten(), device=device)
                        val_input_cfr = torch.stack([torch.real(val_input_cfr), torch.imag(val_input_cfr)], dim=-1)
                        
                        val_target_cfr = torch.tensor(loader.get_cfr_batch(target_AP, val_sta_id).flatten(), device=device)
                        
                        val_output_cfr = model(val_sta_loc, val_input_cfr)
                        val_loss = compute_mse_loss(val_output_cfr, val_target_cfr)
                        
                        val_snr = compute_snr_db(val_loss.item())
                        
                        # 学习率调度
                        scheduler.step(val_loss)
                        
                        # 早停检查
                        if val_loss.item() < best_val_loss:
                            best_val_loss = val_loss.item()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        # 更新进度条信息
                        improvement = (baseline_perf['loss'] - val_loss.item()) / baseline_perf['loss'] * 100
                        pbar.set_postfix({
                            'Loss': f'{val_loss.item():.6f}',
                            'Best': f'{best_val_loss:.6f}',
                            'SNR': f'{val_snr:.1f}dB',
                            'Δ': f'{improvement:+.1f}%',
                            'LR': f'{optimizer.param_groups[0]["lr"]:.0e}'
                        })
                        
                        if patience_counter >= early_stopping_patience:
                            logging.info(f"Early stopping at iteration {i}")
                            break
                        
                        # 报告中间结果给 Optuna
                        trial.report(val_loss.item(), i)
                        
                        # 检查是否应该剪枝
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                
                # 定期清理GPU内存
                if i % 200 == 0:
                    torch.cuda.empty_cache()
        
        # 7. 最终性能评估与baseline比较
        final_improvement = (baseline_perf['loss'] - best_val_loss) / baseline_perf['loss']
        final_snr = compute_snr_db(best_val_loss)
        
        logging.info(f"🎯 Hyperparameter trial {trial.number} completed:")
        logging.info(f"   Val Loss: {best_val_loss:.6f} (baseline: {baseline_perf['loss']:.6f})")
        logging.info(f"   SNR: {final_snr:.2f} dB (baseline: {baseline_perf['snr']:.2f} dB)")
        logging.info(f"   Total Improvement: {final_improvement*100:.2f}%")
        
        # 清理内存
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
        
        return best_val_loss
    
    except optuna.TrialPruned:
        # 清理内存
        torch.cuda.empty_cache()
        raise
    except Exception as e:
        logging.error(f"Hyperparameter trial failed with error: {e}")
        # 清理内存
        torch.cuda.empty_cache()
        return float('inf')
    
    except Exception as e:
        logging.error(f"Trial failed with error: {e}")
        # 清理内存
        torch.cuda.empty_cache()
        return float('inf')


def create_progress_plot(save_path='nas_progress.png'):
    """Create NAS progress visualization charts"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NAS Progress Monitor', fontsize=16, fontweight='bold')
        
        # Subplot 1: Architecture search progress
        if len(progress_tracker.arch_losses) > 0:
            trials = list(range(1, len(progress_tracker.arch_losses) + 1))
            ax1.plot(trials, progress_tracker.arch_losses, 'b-', marker='o', markersize=4, alpha=0.7, label='Architecture Search')
            if progress_tracker.baseline_loss:
                ax1.axhline(y=progress_tracker.baseline_loss, color='r', linestyle='--', alpha=0.7, label='Baseline')
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('Validation Loss')
            ax1.set_title('Architecture Search Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
        else:
            ax1.text(0.5, 0.5, 'Architecture Search Not Started', ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Architecture Search Progress')
        
        # Subplot 2: Hyperparameter search progress
        if len(progress_tracker.hp_losses) > 0:
            trials = list(range(1, len(progress_tracker.hp_losses) + 1))
            ax2.plot(trials, progress_tracker.hp_losses, 'g-', marker='s', markersize=4, alpha=0.7, label='Hyperparameter Search')
            if progress_tracker.baseline_loss:
                ax2.axhline(y=progress_tracker.baseline_loss, color='r', linestyle='--', alpha=0.7, label='Baseline')
            ax2.set_xlabel('Trial Number')
            ax2.set_ylabel('Validation Loss')
            ax2.set_title('Hyperparameter Search Progress')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'Hyperparameter Search Not Started', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Hyperparameter Search Progress')
        
        # Subplot 3: Overall progress bars
        stages = ['Baseline', 'Architecture Search', 'Hyperparameter Search']
        progress_values = [
            100 if progress_tracker.baseline_loss else progress_tracker.baseline_progress,
            (progress_tracker.arch_trials_completed / progress_tracker.arch_trials_total * 100) if progress_tracker.arch_trials_total > 0 else 0,
            (progress_tracker.hp_trials_completed / progress_tracker.hp_trials_total * 100) if progress_tracker.hp_trials_total > 0 else 0
        ]
        colors = ['red', 'blue', 'green']
        bars = ax3.barh(stages, progress_values, color=colors, alpha=0.7)
        ax3.set_xlim(0, 100)
        ax3.set_xlabel('Progress (%)')
        ax3.set_title('Overall Progress')
        
        # Show percentage on each progress bar
        for bar, value in zip(bars, progress_values):
            width = bar.get_width()
            ax3.text(width/2, bar.get_y() + bar.get_height()/2, f'{value:.1f}%', 
                    ha='center', va='center', fontweight='bold', color='white')
        
        # Subplot 4: Performance comparison
        if progress_tracker.baseline_loss:
            categories = ['Baseline']
            losses = [progress_tracker.baseline_loss]
            colors_bar = ['red']
            
            if progress_tracker.arch_best_loss < float('inf'):
                categories.append('Best Architecture')
                losses.append(progress_tracker.arch_best_loss)
                colors_bar.append('blue')
            
            if progress_tracker.hp_best_loss < float('inf'):
                categories.append('Best Hyperparams')
                losses.append(progress_tracker.hp_best_loss)
                colors_bar.append('green')
            
            bars = ax4.bar(categories, losses, color=colors_bar, alpha=0.7)
            ax4.set_ylabel('Validation Loss')
            ax4.set_title('Performance Comparison')
            ax4.set_yscale('log')
            
            # Show specific values and improvement percentages
            for i, (bar, loss) in enumerate(zip(bars, losses)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height*1.05,
                        f'{loss:.6f}', ha='center', va='bottom', fontweight='bold')
                
                if i > 0:  # Calculate improvement relative to baseline
                    improvement = (progress_tracker.baseline_loss - loss) / progress_tracker.baseline_loss * 100
                    ax4.text(bar.get_x() + bar.get_width()/2., height*0.5,
                            f'{improvement:+.1f}%', ha='center', va='center', 
                            color='white', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Waiting for Results...', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Performance Comparison')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"📊 Progress chart saved: {save_path}")
        
    except Exception as e:
        logging.error(f"Failed to create progress chart: {e}")


def should_save_progress_plot():
    """判断是否应该保存进度图 - 只在关键进度点保存"""
    # 计算总体进度
    total_progress = 0
    completed_stages = 0
    
    # Baseline阶段
    if progress_tracker.baseline_loss is not None:
        completed_stages += 1
        total_progress += 100
    else:
        total_progress += progress_tracker.baseline_progress
    
    # 架构搜索阶段
    if progress_tracker.arch_trials_total > 0:
        arch_progress = (progress_tracker.arch_trials_completed / progress_tracker.arch_trials_total) * 100
        total_progress += arch_progress
        if arch_progress >= 100:
            completed_stages += 1
    
    # 超参数搜索阶段  
    if progress_tracker.hp_trials_total > 0:
        hp_progress = (progress_tracker.hp_trials_completed / progress_tracker.hp_trials_total) * 100
        total_progress += hp_progress
        if hp_progress >= 100:
            completed_stages += 1
    
    # 计算平均进度
    expected_stages = 3  # baseline + arch + hp
    overall_progress = total_progress / expected_stages
    
    # 定义关键进度点 (10%, 20%, 30%, ..., 90%, 100%)
    key_milestones = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # 检查是否达到新的关键点
    if not hasattr(should_save_progress_plot, 'last_saved_milestone'):
        should_save_progress_plot.last_saved_milestone = 0
    
    for milestone in key_milestones:
        if (overall_progress >= milestone and 
            should_save_progress_plot.last_saved_milestone < milestone):
            should_save_progress_plot.last_saved_milestone = milestone
            return True, milestone
    
    return False, 0


def cleanup_old_progress_plots(keep_recent=3):
    """清理旧的进度图，只保留最近的几张"""
    try:
        import glob
        import os
        
        # 找到所有进度图文件 (匹配 nas_progress_XX%_HHMMSS.png 格式)
        progress_files = glob.glob('nas_progress_*%_*.png')
        
        if len(progress_files) > keep_recent:
            # 按修改时间排序，最新的在前
            progress_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # 删除旧的文件，保留最新的 keep_recent 张
            files_to_delete = progress_files[keep_recent:]
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    logging.info(f"🗑️  Cleaned up old progress chart: {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to delete {file_path}: {e}")
                    
    except Exception as e:
        logging.warning(f"Progress chart cleanup failed: {e}")


def periodic_progress_update():
    """定期更新进度并在关键点创建可视化图表"""
    while True:
        try:
            progress_tracker.print_status()
            
            # 只在关键进度点保存图片
            should_save, milestone = should_save_progress_plot()
            if should_save:
                timestamp = datetime.now().strftime("%H%M%S")
                save_path = f'nas_progress_{milestone}%_{timestamp}.png'
                create_progress_plot(save_path)
                logging.info(f"📊 Progress milestone {milestone}% reached! Chart saved: {save_path}")
                
                # 清理旧的进度图
                cleanup_old_progress_plots(keep_recent=5)
            
            time.sleep(30)  # Update every 30 seconds
        except Exception as e:
            logging.error(f"Progress update failed: {e}")
            time.sleep(30)


def run_two_stage_nas(architecture_trials=30, hyperparams_trials=50, storage_url=None):
    """运行两阶段 NAS：第一阶段搜索架构，第二阶段搜索超参数"""
    
    # 重置进度追踪器
    progress_tracker.reset()
    progress_tracker.start_time = time.time()
    
    # 显示图片保存策略
    logging.info("📊 图片保存策略: 只在关键进度点保存 (10%, 20%, 30%, ..., 90%, 100%)")
    logging.info("🗑️  自动清理: 保留最近5张进度图，自动删除旧图片")
    
    # 启动后台进度更新线程
    progress_thread = threading.Thread(target=periodic_progress_update, daemon=True)
    progress_thread.start()
    
    if storage_url is None:
        storage_url = "sqlite:///two_stage_nas.db"
    
    # 🎯 首先评估baseline性能
    baseline_perf = evaluate_baseline_model()
    
    # ======================= 第一阶段：架构搜索 =======================
    logging.info("🏗️  开始第一阶段：神经网络架构搜索")
    logging.info(f"🎯 Baseline性能 - Loss: {baseline_perf['loss']:.6f}, SNR: {baseline_perf['snr']:.2f} dB")
    logging.info(f"目标试验数: {architecture_trials}")
    
    # 更新进度追踪器
    progress_tracker.set_arch_stage(architecture_trials)
    
    # 创建智能剪枝器
    intelligent_pruner = IntelligentPruner(baseline_perf, improvement_threshold=0.02)
    
    # 创建架构搜索研究
    arch_study = optuna.create_study(
        direction='minimize',
        storage=storage_url,
        study_name='architecture_search',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=3  # 更频繁的剪枝检查
        ),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=8,
            multivariate=True
        )
    )
    
    # 自定义回调函数来追踪进度
    def arch_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            progress_tracker.update_arch_trial(len(study.trials), study.best_value)
            logging.info(f"🏗️  架构试验 {trial.number+1}/{architecture_trials} 完成 | 最佳损失: {study.best_value:.6f}")
    
    # 执行架构搜索
    try:
        with tqdm(total=architecture_trials, desc="🏗️ 架构搜索", ncols=80) as pbar:
            def wrapped_objective(trial):
                result = objective_architecture(trial)
                pbar.update(1)
                return result
            
            arch_study.optimize(wrapped_objective, n_trials=architecture_trials, callbacks=[arch_callback])
    except KeyboardInterrupt:
        logging.info("架构搜索被用户中断")
    
    # 分析架构搜索结果
    completed_trials = [t for t in arch_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in arch_study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    if not completed_trials:
        logging.error("❌ 没有完成的架构搜索试验！")
        logging.info("尝试降低搜索标准或增加试验数...")
        return None, None, None
    
    # 获取最佳架构
    best_arch_params = arch_study.best_params
    best_arch_loss = arch_study.best_value
    
    # 检查是否比baseline更好
    improvement = (baseline_perf['loss'] - best_arch_loss) / baseline_perf['loss']
    
    logging.info("=" * 60)
    logging.info("🎯 第一阶段完成！架构搜索结果:")
    logging.info(f"最佳验证损失: {best_arch_loss:.6f}")
    logging.info(f"最佳 SNR: {compute_snr_db(best_arch_loss):.2f} dB")
    logging.info(f"相比Baseline改进: {improvement*100:.2f}%")
    logging.info(f"完成试验: {len(completed_trials)}, 剪枝试验: {len(pruned_trials)}")
    
    if improvement <= 0:
        logging.warning("⚠️  最佳架构未超越baseline，考虑:")
        logging.warning("   1. 增加架构搜索试验数")
        logging.warning("   2. 扩大搜索空间")
        logging.warning("   3. 调整剪枝策略")
    else:
        logging.info(f"✅ 找到更好的架构！改进 {improvement*100:.2f}%")
    
    logging.info("最佳架构参数:")
    for key, value in best_arch_params.items():
        logging.info(f"  {key}: {value}")
    
    # 保存最佳架构
    with open('best_architecture.pkl', 'wb') as f:
        pickle.dump(best_arch_params, f)
    logging.info("最佳架构已保存到 'best_architecture.pkl'")
    
    # ======================= 第二阶段：超参数搜索 =======================
    logging.info("\n" + "=" * 60)
    logging.info("⚡ 开始第二阶段：超参数搜索")
    logging.info(f"使用最佳架构，目标试验数: {hyperparams_trials}")
    
    # 更新进度追踪器
    progress_tracker.set_hp_stage(hyperparams_trials)
    
    # 创建超参数搜索研究
    hp_study = optuna.create_study(
        direction='minimize',
        storage=storage_url,
        study_name='hyperparams_search',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=10
        ),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=15,
            multivariate=True
        )
    )
    
    # 创建带有最佳架构的目标函数
    def objective_with_best_arch(trial):
        return objective_hyperparams(trial, best_arch_params, baseline_perf)
    
    # 自定义回调函数来追踪超参数搜索进度
    def hp_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            progress_tracker.update_hp_trial(len(study.trials), study.best_value)
            logging.info(f"⚡ 超参数试验 {trial.number+1}/{hyperparams_trials} 完成 | 最佳损失: {study.best_value:.6f}")
    
    # 执行超参数搜索
    try:
        with tqdm(total=hyperparams_trials, desc="⚡ 超参数搜索", ncols=80) as pbar:
            def wrapped_hp_objective(trial):
                result = objective_with_best_arch(trial)
                pbar.update(1)
                return result
            
            hp_study.optimize(wrapped_hp_objective, n_trials=hyperparams_trials, callbacks=[hp_callback])
    except KeyboardInterrupt:
        logging.info("超参数搜索被用户中断")
    
    # 获取最终最佳结果
    hp_completed_trials = [t for t in hp_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not hp_completed_trials:
        logging.warning("⚠️  超参数搜索无完成试验，使用架构搜索结果")
        final_best_params = best_arch_params
        final_best_loss = best_arch_loss
    else:
        best_hp_params = hp_study.best_params
        best_hp_loss = hp_study.best_value
        final_best_params = {**best_arch_params, **best_hp_params}
        final_best_loss = best_hp_loss
    
    # 最终结果分析
    final_improvement = (baseline_perf['loss'] - final_best_loss) / baseline_perf['loss']
    
    logging.info("=" * 60)
    logging.info("🏆 两阶段搜索完成！最终结果:")
    logging.info(f"Baseline性能  - Loss: {baseline_perf['loss']:.6f}, SNR: {baseline_perf['snr']:.2f} dB")
    logging.info(f"最终最佳性能 - Loss: {final_best_loss:.6f}, SNR: {compute_snr_db(final_best_loss):.2f} dB")
    logging.info(f"总体改进: {final_improvement*100:.2f}%")
    
    if final_improvement > 0:
        logging.info(f"🎉 成功找到更好的模型配置！")
    else:
        logging.warning(f"⚠️  最终结果未超越baseline")
    
    if len(hp_completed_trials) > 0:
        arch_improvement = (baseline_perf['loss'] - best_arch_loss) / baseline_perf['loss']
        hp_improvement = (best_arch_loss - final_best_loss) / best_arch_loss if best_arch_loss > 0 else 0
        logging.info(f"架构贡献: {arch_improvement*100:.2f}%")
        logging.info(f"超参数贡献: {hp_improvement*100:.2f}%")
    
    # 保存最终结果
    with open('final_best_params.pkl', 'wb') as f:
        pickle.dump(final_best_params, f)
    logging.info("\n最终最佳参数已保存到 'final_best_params.pkl'")
    
    # 搜索统计
    logging.info("\n" + "=" * 60)
    logging.info("📊 搜索统计:")
    logging.info(f"架构搜索 - 完成: {len(completed_trials)}, 剪枝: {len(pruned_trials)}")
    logging.info(f"超参数搜索 - 完成: {len(hp_completed_trials)}, 剪枝: {len([t for t in hp_study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    # 创建最终结果可视化
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_chart_path = f'final_nas_results_{timestamp}.png'
    create_progress_plot(final_chart_path)
    logging.info(f"📊 最终结果图表已保存: {final_chart_path}")
    
    return arch_study, hp_study, final_best_params


def run_optuna_search(n_trials=100, storage_url=None):
    """运行两阶段 Optuna 搜索的入口函数"""
    # 将总试验数分配给两个阶段
    architecture_trials = min(30, n_trials // 3)  # 1/3 用于架构搜索
    hyperparams_trials = n_trials - architecture_trials  # 2/3 用于超参数搜索
    
    return run_two_stage_nas(architecture_trials, hyperparams_trials, storage_url)


def load_and_test_best_model(best_params_file='final_best_params.pkl'):
    """使用最佳参数创建和测试模型"""
    logging.info("加载最佳参数并测试模型...")
    
    with open(best_params_file, 'rb') as f:
        best_params = pickle.load(f)
    
    # 创建一个伪 trial 对象用于模型构建
    class FakeTrial:
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
    
    fake_trial = FakeTrial(best_params)
    
    # 创建最佳模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = OptimizedMLP(fake_trial)
    model.to(device)
    
    logging.info("🏆 最终最佳模型配置:")
    logging.info("架构参数:")
    arch_params = ['n_layers', 'hidden_dim', 'dropout_rate', 'activation', 'use_batch_norm', 'init_method', 'skip_connections']
    for param in arch_params:
        if param in best_params:
            logging.info(f"  {param}: {best_params[param]}")
    
    logging.info("超参数:")
    hp_params = ['batch_size', 'learning_rate', 'weight_decay', 'scheduler_patience', 'scheduler_factor']
    for param in hp_params:
        if param in best_params:
            logging.info(f"  {param}: {best_params[param]}")
    
    return model, best_params

#%%
if __name__ == "__main__":
    # 运行两阶段 NAS
    logging.info("🚀 开始两阶段神经架构搜索")
    
    # 第一阶段：架构搜索 (30 trials)
    # 第二阶段：超参数搜索 (70 trials) 
    arch_study, hp_study, final_params = run_two_stage_nas(
        architecture_trials=30,
        hyperparams_trials=70
    )
    
    # 测试最佳模型
    try:
        best_model, best_params = load_and_test_best_model('final_best_params.pkl')
        logging.info("🎯 最佳模型创建成功！")
        
        # 显示模型信息
        total_params = sum(p.numel() for p in best_model.parameters())
        trainable_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
        logging.info(f"模型总参数量: {total_params:,}")
        logging.info(f"可训练参数量: {trainable_params:,}")
        
        # 如果是最佳模型，也显示它的架构信息
        if hasattr(best_model, 'print_architecture'):
            best_model.print_architecture()
        
        # 显示详细的最佳架构信息
        display_best_architecture(best_params, "Final Best")
        
    except Exception as e:
        logging.error(f"创建最佳模型失败: {e}")
        
    logging.info("✅ 两阶段 NAS 搜索完成！")

# %%
