# 🚀 智能NAS使用指南

## ⏰ 优化后的时间
现在训练时间大幅减少：
- Baseline评估：500 iterations (~1-2分钟)  
- 架构搜索：每trial 500 iterations (~1分钟/trial)
- 超参数搜索：每trial 800 iterations (~1.5分钟/trial)

**总时间：约 10-15 分钟！**

## 📝 三种使用方式

### 1. 快速测试 (推荐首次使用)
```
cd /home/byang/BoYang/mNeWRF
python test_fast_nas.py
```
大约需要 **5 分钟**，测试 3+3=6 个配置

### 2. 标准测试  
```
cd /home/byang/BoYang/mNeWRF
python test_intelligent_nas.py
```
大约需要 **10-15 分钟**，测试 5+8=13 个配置

### 3. 完整搜索
```
cd /home/byang/BoYang/mNeWRF  
python optuna_nas.py
```
大约需要 **30-45 分钟**，测试 30+70=100 个配置

## 🎯 运行完成后的结果文件

- `best_architecture.pkl` - 最佳架构参数
- `final_best_params.pkl` - 完整最佳配置
- `two_stage_nas.db` - Optuna搜索历史
- `*.log` - 详细日志文件

## 📊 如何查看和使用结果

运行以下代码查看最佳配置：
```python
import pickle

# 加载最佳配置
with open('final_best_params.pkl', 'rb') as f:
    best_config = pickle.load(f)
    
print("最佳架构配置:")
arch_keys = ['n_layers', 'hidden_dim', 'dropout_rate', 'activation', 'use_batch_norm', 'init_method', 'skip_connections']
for key in arch_keys:
    if key in best_config:
        print(f"  {key}: {best_config[key]}")

print("最佳超参数:")        
hp_keys = ['batch_size', 'learning_rate', 'weight_decay', 'scheduler_patience', 'scheduler_factor']
for key in hp_keys:
    if key in best_config:
        print(f"  {key}: {best_config[key]}")
```

## ⚡ 如果还想更快

可以修改试验数：
- architecture_trials=2  # 只测试2个架构
- hyperparams_trials=2   # 只测试2个超参数组合

总时间约 **3-5 分钟**！
