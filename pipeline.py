'''BoYang code Cell free network channel prediction'''
#%%
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from tqdm import trange
from loaders import DatasetLoader
import os, random, string
from omegaconf import OmegaConf
from argparse import ArgumentParser
import pandas as pd
import logging
import time
from models import MLP
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from config_function import *
import os

dataset_fname = '/home/byang/BoYang/mNeWRF/dataset/conference_2000STA_4APs.pkl'
file_dir = './ckpt/baseline/NeWRF_method'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class SimpleConfig:
    class training:
        batch_size = 1204
        n_iters = 10000
        display_rate = 20
        train_split = 0.8
        val_split = 0.1
        base_AP = [1]
        target_AP = [2]
        args_env = 'office'
        ap_id = ap_id

    class optimizer:
        lr = 5e-4
        weight_decay = 0
        scheduler_patience = 10
        scheduler_factor = 0.9
        min_lr = 1e-5
    
    class randomseed:
        torch_seed = torch.randint(0, 100, (1,)).item()
        npy_seed = torch.randint(0, 100, (1,)).item()
        dataset_seed = 12

# seed = torch.randint(0, 100, (1,)).item()
# seed = 44  # 固定种子以便复现
# print(f"Random seed: {seed}")
cfg = SimpleConfig()
# 设置种子
torch.manual_seed(cfg.randomseed.torch_seed)
np.random.seed(cfg.randomseed.npy_seed)
random.seed(cfg.randomseed.torch_seed)

#%%
# 创建日志目录
os.makedirs(file_dir, exist_ok=True)
os.makedirs('./logs', exist_ok=True)
# 初始化 TensorBoard SummaryWriter
summary_writer = SummaryWriter(log_dir='./logs')

# 初始化损失记录列表
train_losses = []
val_losses = []
SNRs = []   
val_SNRs = []
iternum = []
# 保存配置文件到 checkpoint 目录
config_file_path = save_config_to_file(cfg, file_dir, dataset_fname, device, cfg.training.base_AP, cfg.training.target_AP)


model = MLP()
model.to(device)
model_params = list(model.parameters())

optimizer = torch.optim.Adam(model_params, lr=cfg.optimizer.lr, weight_decay = cfg.optimizer.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = cfg.optimizer.scheduler_patience, factor=cfg.optimizer.scheduler_factor, min_lr = cfg.optimizer.min_lr)

def compute_mse_loss(predicted, true):
    """计算相对MSE损失 - 归一化版本"""
    return torch.sum(torch.abs(predicted - true) ** 2) / torch.sum(torch.abs(true) ** 2)

def compute_snr_db(loss):
    return -10.0 * np.log10(loss)

loader = DatasetLoader(dataset_fname)
loader.split_train_val_test(train_ratio=cfg.training.train_split, val_ratio=cfg.training.val_split,
                             seed=cfg.randomseed.dataset_seed) 

for i in trange(cfg.training.n_iters, desc = 'Training'):

    sta_id = np.random.choice(loader.trainset, cfg.training.batch_size)

    logging.info("Model in training mode")
    model.train()

    # 🚀 优化10: 直接指定device，避免后续.to()操作
    sta_loc = torch.tensor(loader.get_loc_batch("STA", sta_id), device=device, dtype=torch.float32)
    # AP_loc = torch.tensor(loader.get_loc_batch("AP", sta_id), device=device, dtype=torch.float32) # 多个 

    input_cfr =  torch.tensor(loader.get_cfr_batch(cfg.training.base_AP, sta_id)).flatten().to(device) # 多个AP的信道
    input_cfr = torch.stack([torch.real(input_cfr), torch.imag(input_cfr)], dim=-1)
    output_cfr = model(sta_loc, input_cfr)
    # output_cfr是32，2的tensor，代表着32个STA的预测的与target AP之间的channel的实部和虚部，重新组合成complex value

    target_cfr = torch.tensor(loader.get_cfr_batch(cfg.training.target_AP, sta_id)).flatten().to(device)

    loss = compute_mse_loss(output_cfr, target_cfr) 
    train_losses.append(loss.item())  # 记录训练损失
    summary_writer.add_scalar('train Loss', loss.item(), i)
    SNR = compute_snr_db(loss.item())
    SNRs.append(SNR)  # 记录训练SNR
    summary_writer.add_scalar('train SNR', SNR, i)
    
    # 📊 每次训练都记录配置参数
    summary_writer.add_scalar('Config_Realtime/Batch_Size', cfg.training.batch_size, i)
    summary_writer.add_scalar('Config_Realtime/Learning_Rate_Config', cfg.optimizer.lr, i)
    summary_writer.add_scalar('Config_Realtime/Weight_Decay', cfg.optimizer.weight_decay, i)
    summary_writer.add_scalar('Config_Realtime/Train_Split', cfg.training.train_split, i)
    summary_writer.add_scalar('Config_Realtime/Val_Split', cfg.training.val_split, i)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 🚀 优化8: 定期清理GPU缓存（每100次迭代）
    if i % 100 == 0:
        torch.cuda.empty_cache()
    
    if i % cfg.training.display_rate == 0:
        iternum.append(i)
        with torch.no_grad():
            model.eval()
            val_sta_id = loader.valset
            val_sta_loc = torch.tensor(loader.get_loc_batch("STA", val_sta_id)).to(device)
            val_AP_loc = torch.tensor(loader.get_loc_batch("AP", val_sta_id)).to(device)

            val_input_cfr =  torch.tensor(loader.get_cfr_batch(cfg.training.base_AP, val_sta_id).flatten()).to(device) # 多个AP的信道
            val_input_cfr = torch.stack([torch.real(val_input_cfr), torch.imag(val_input_cfr)], dim=-1)
            val_output_cfr = model(val_sta_loc, val_input_cfr)

            val_target_cfr = torch.tensor(loader.get_cfr_batch(cfg.training.target_AP, val_sta_id)).flatten().to(device)

            val_loss = compute_mse_loss(val_output_cfr, val_target_cfr) 
            val_losses.append(val_loss.item())  # 记录验证损失
            val_SNR = compute_snr_db(val_loss.item())
            val_SNRs.append(val_SNR)  # 记录验证SNR

            print(f"Iteration {i}: train Loss = {loss.item():.6f}, SNR = {SNR:.2f}, val Loss = {val_loss.item():.6f}, val SNR = {val_SNR:.2f}")

            logging.info(f"Iteration {i}:train Loss = {loss.item()}, SNR = {SNR}, val Loss = {val_loss.item()}, val SNR = {val_SNR}")
            summary_writer.add_scalar('Val Loss', val_loss.item(), i)
            summary_writer.add_scalar('Val SNR', val_SNR, i)
            summary_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], i)
            
            # 📊 记录配置相关的动态信息
            summary_writer.add_scalar('Hyperparams/Batch_Size', cfg.training.batch_size, i)
            summary_writer.add_scalar('Hyperparams/Weight_Decay', cfg.optimizer.weight_decay, i)
            summary_writer.add_scalar('Hyperparams/Scheduler_Patience', cfg.optimizer.scheduler_patience, i)
            summary_writer.add_scalar('Hyperparams/Scheduler_Factor', cfg.optimizer.scheduler_factor, i)

            if scheduler is not None:
                prev_lr = scheduler.get_last_lr()[0]
                scheduler.step(loss)
                current_lr = scheduler.get_last_lr()[0]
                if current_lr != prev_lr:
                    print(f"Learning rate changed from {prev_lr:.6f} to {current_lr:.6f}")       

            save_path = os.path.join(file_dir, f'iter_{i}.pt')
            torch.save(model.state_dict(), save_path)
            
            # 📄 更新配置文件，记录当前训练状态

            
            # 更新训练进度
            update_training_progress(cfg, file_dir, i, loss.item(), val_loss.item(),SNR, val_SNR, optimizer.param_groups[0]['lr'])

            # plot the training procedure, the prediction and target difference, the training curves
            fig, ax = plt.subplots(1,3, figsize=(15, 5))
            
            # 第一个子图：损失对比（使用log scale让不同尺度可以观察）
            ax[0].plot(range(len(train_losses)), train_losses, 'b-', label='Train Loss', alpha=0.7, linewidth=1.5)
            ax[0].plot(iternum, val_losses, 'r-', label='Val Loss', alpha=0.7, linewidth=1.5)
            ax[0].set_xlabel('Iterations')
            ax[0].set_ylabel('Loss')
            ax[0].set_yscale('log')  # 使用对数坐标让不同尺度的loss都可见
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)
            ax[0].set_title('Training Procedure (Log Scale)')

            # 第二个子图：复数预测可视化
            ax[1].plot(np.real(val_output_cfr.cpu()), np.imag(val_output_cfr.cpu()), "ro", label="prediction", alpha=0.7, markersize=3)
            ax[1].plot(np.real(val_target_cfr.cpu()), np.imag(val_target_cfr.cpu()), "bo", label="Target", alpha=0.7, markersize=3)
            ax[1].set_xlabel('Real Part')
            ax[1].set_ylabel('Imaginary Part')
            ax[1].set_title(f"AP_{cfg.training.target_AP} predict SNR: {val_SNR:.2f} dB")
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)

            # 第三个子图：SNR对比
            ax[2].plot(range(0, i + 1), SNRs, 'r-', label="Train SNR", alpha=0.7, linewidth=1.5)
            ax[2].plot(iternum, val_SNRs, 'y-', label='Val SNR', alpha=0.7, linewidth=1.5)
            ax[2].set_xlabel('Iterations')
            ax[2].set_ylabel('SNR (dB)')
            ax[2].legend()
            ax[2].grid(True, alpha=0.3)
            ax[2].set_title('SNR Progress')

            plt.tight_layout()



            plt.show()

# 训练结束后关闭 summary_writer
summary_writer.close()
print("Training completed and logs saved to './logs'")




























# %%
