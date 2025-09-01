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
from models import MLP, NeRF2
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from config_function import *
import os
import torch.optim as optim
from ray_gen import *
from samplers import *
from synthesizers import *
from einops import rearrange, repeat
import scipy.constants as sc
import torch.nn.functional as F

dataset_fname = '/home/byang/BoYang/mNeWRF/dataset/conference_2000STA_4APs.pkl'
file_dir = './ckpt/baseline/NeWRF_method'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class SimpleConfig:
    class training:
        batch_size = 32
        n_iters = 10000
        display_rate = 1000
        train_split = 0.8
        val_split = 0.1
        base_AP = [1]
        target_AP = [1]
        args_env = 'office'
        ap_id = ap_id

    class optimizer:
        lr = 1e-3 # 5e-4
        weight_decay = 0
        scheduler_patience = 10
        scheduler_factor = 0.9
        min_lr = 1e-5
    
    class randomseed:
        torch_seed = torch.randint(0, 100, (1,)).item()
        npy_seed = torch.randint(0, 100, (1,)).item()
        dataset_seed = 12

    class sampling:
        alpha_res = 36
        beta_res = 9
        near = 0
        far = 5
        n_samples = 32

# seed = torch.randint(0, 100, (1,)).item()
# seed = 44  # 固定种子以便复现
# print(f"Random seed: {seed}")
cfg = SimpleConfig()
# 设置种子
torch.manual_seed(cfg.randomseed.torch_seed)
np.random.seed(cfg.randomseed.npy_seed)
random.seed(cfg.randomseed.torch_seed)






def raw2outputs_signal(raw, r_vals, rays_d):
    wavelength = sc.c / 2.4e9
    # raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
    # raw2phase = lambda raw, dists: torch.exp(1j*raw*dists)
    raw2phase = lambda raw, dists: raw + 2*np.pi*dists/wavelength
    raw2amp = lambda raw, dists: -raw*dists

    dists = r_vals[...,1:] - r_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [batchsize, chunks, n_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)  # [batchsize,chunks, n_samples, 3].

    att_a, att_p, s_a, s_p = raw[...,0], raw[...,1], raw[...,2], raw[...,3]    # [batchsize,chunks, N_samples]
    att_a = att_a.unsqueeze(-1)
    att_p = att_p.unsqueeze(-1)
    s_a = s_a.unsqueeze(-1)
    s_p = s_p.unsqueeze(-1)

    att_p, s_p = torch.sigmoid(att_p)*np.pi*2-np.pi, torch.sigmoid(s_p)*np.pi*2-np.pi
    att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))

    dists = dists.unsqueeze(-1)

    amp = raw2amp(att_a, dists)  # [batchsize,chunks, N_samples, 26]
    phase = raw2phase(att_p, dists)

    # att_i = torch.cumprod(torch.cat([torch.ones((al_shape[:-1], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    # phase_i = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), phase], -1), -1)[:, :-1]
    amp_i = torch.exp(torch.cumsum(amp, -2))            # [batchsize,chunks, N_samples, 26]
    phase_i = torch.exp(1j*torch.cumsum(phase, -2))                # [batchsize,chunks, N_samples 26]

    recv_signal = torch.sum(s_a*torch.exp(1j*s_p)*amp_i*phase_i, -2)  # integral along line [batchsize,chunks,26]
    recv_signal = torch.sum(recv_signal, 1)   # integral along direction [batchsize, 26]

    return recv_signal














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


synthesizer = Synthesizer()
model = NeRF2().to(device)
params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3,
                                    weight_decay=5e-5,
                                    betas=(0.9, 0.999))
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                T_max=10000, eta_min=1e-6,
                                                                last_epoch=-1)

def compute_mse_loss(predicted, true):
    """计算相对MSE损失 - 归一化版本"""
    return torch.sum(torch.abs(predicted - true) ** 2) / torch.sum(torch.abs(true) ** 2)

def compute_snr_db(loss):
    return -10.0 * np.log10(loss)

loss_fn = lambda pred, true: torch.sum(torch.abs(pred-true)**2)/torch.sum(torch.abs(true)**2) 
snr = lambda loss: -10. * np.log10(loss)


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
    ray_gen = RayGenerator(sta_loc, cfg, device)
    rays_o, rays_d = ray_gen.get_rays() 
    # rays_d is a tensor of 324,3, extend it to batchsize, 324,3
    rays_d = repeat(rays_d, 'b d -> v b d', v=32).to(device)  #[9x36, bs, 3]

    batchsize, viewsize, _ = rays_d.shape
    rays_o = repeat(rays_o, 'b d -> b v d', v=viewsize).to(device)  #[bs, 9x36, 3]

    shape = list(rays_o.shape)
    shape[-1] = 1
    near, far = torch.full(shape, 0), torch.full(shape, 5)
    t_vals = torch.linspace(0., 1., steps=32) * (far - near) + near  # scale t with near and far
    t_vals = t_vals.to(rays_o.device)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * t_vals[...,:,None] # p = o + td, [n_rays, n_samples, 3]
    views = repeat(rays_d, 'b v d -> b v p d', p=32)

    input_cfr =  torch.tensor(loader.get_cfr_batch(cfg.training.base_AP, sta_id)).flatten().to(device) # 多个AP的信道
    input_cfr = torch.stack([torch.real(input_cfr), torch.imag(input_cfr)], dim=-1)
    input_cfr = repeat(input_cfr, 'b d -> b v d', v=viewsize)  
    input_cfr = repeat(input_cfr, 'b v d -> b v p d', p=32)  # [bs, cks, pts, 3]

    raw = model(pts,views, input_cfr)
    recv_signal = raw2outputs_signal(raw, t_vals, rays_d) 
    # output_cfr是32，2的tensor，代表着32个STA的预测的与target AP之间的channel的实部和虚部，重新组合成complex value

    target_cfr = torch.tensor(loader.get_cfr_batch(cfg.training.target_AP, sta_id)).flatten().to(device)

    train_loss = loss_fn(recv_signal, target_cfr)
    train_losses.append(train_loss.item())  # 记录训练损失
    train_SNR = snr(train_loss.item())

    SNRs.append(train_SNR)  # 记录训练SNR

    optimizer.zero_grad()
    train_loss.backward()
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

            val_ray_gen = RayGenerator(val_sta_loc, cfg, device)
            val_rays_o, val_rays_d = val_ray_gen.get_rays() 
            
            # 2. 射线扩展 - 与训练一致
            val_rays_d = repeat(val_rays_d, 'b d -> v b d', v=len(val_sta_id)).to(device)  #[9x36, val_batch_size, 3]
            val_batchsize, val_viewsize, _ = val_rays_d.shape
            val_rays_o = repeat(val_rays_o, 'b d -> b v d', v=val_viewsize).to(device)  #[val_batch_size, 9x36, 3]

            # 3. 采样点生成 - 与训练完全一致
            val_shape = list(val_rays_o.shape)
            val_shape[-1] = 1
            val_near, val_far = torch.full(val_shape, 0), torch.full(val_shape, 5)
            val_t_vals = torch.linspace(0., 1., steps=32) * (val_far - val_near) + val_near
            val_t_vals = val_t_vals.to(val_rays_o.device)
            val_pts = val_rays_o[...,None,:] + val_rays_d[...,None,:] * val_t_vals[...,:,None]
            val_views = repeat(val_rays_d, 'b v d -> b v p d', p=32)

            # 4. 输入CFR准备 - 与训练一致
            val_input_cfr = torch.tensor(loader.get_cfr_batch(cfg.training.base_AP, val_sta_id)).flatten().to(device)
            val_input_cfr = torch.stack([torch.real(val_input_cfr), torch.imag(val_input_cfr)], dim=-1)
            val_input_cfr = repeat(val_input_cfr, 'b d -> b v d', v=val_viewsize)  
            val_input_cfr = repeat(val_input_cfr, 'b v d -> b v p d', p=32)

            # 5. 模型预测 - 与训练一致
            val_raw = model(val_pts, val_views, val_input_cfr)
            val_recv_signal = raw2outputs_signal(val_raw, val_t_vals, val_rays_d)  # [val_batch_size, 26]

            # 6. 目标数据 - 与训练一致
            val_target_cfr = torch.tensor(loader.get_cfr_batch(cfg.training.target_AP, val_sta_id)).flatten().to(device)
            val_loss = loss_fn(val_recv_signal, val_target_cfr)
            val_SNR = snr(val_loss.item())
            val_losses.append(val_loss.item())
            val_SNRs.append(val_SNR)
            # 7. 损失计算 - 与训练一致


            print(f"Iteration {i}: train Loss = {train_loss.item():.6f}, SNR = {train_SNR:.2f}, val Loss = {val_loss.item():.6f}, val SNR = {val_SNR:.2f}")


            save_path = os.path.join(file_dir, f'iter_{i}.pt')
            torch.save(model.state_dict(), save_path)
            
            # 📄 更新配置文件，记录当前训练状态

            
            # 更新训练进度
            update_training_progress(cfg, file_dir, i, train_loss.item(), val_loss.item(), train_SNR, val_SNR, optimizer.param_groups[0]['lr'])

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

            # # 第二个子图：复数预测可视化
            # ax[1].plot(np.real(val_syn_cfr.cpu()), np.imag(val_syn_cfr.cpu()), "ro", label="prediction", alpha=0.7, markersize=3)
            # ax[1].plot(np.real(val_target_cfr.cpu()), np.imag(val_target_cfr.cpu()), "bo", label="Target", alpha=0.7, markersize=3)
            # ax[1].set_xlabel('Real Part')
            # ax[1].set_ylabel('Imaginary Part')
            # ax[1].set_title(f"AP_{cfg.training.target_AP} predict SNR: {val_SNR:.2f} dB")
            # ax[1].legend()
            # ax[1].grid(True, alpha=0.3)

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
