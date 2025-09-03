#%%
import os
import yaml
from shutil import copyfile
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import numpy as np
import torch
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import scipy.io as scio
from dataloader import *
from model import *
from renderer import renderer_dict
from utils.data_painter import paint_spectrum_compare
from utils.logger import logger_config


config_path = 'configs/mimo-csi.yml'
mode='train',  # 或 'test'
dataset_type='mimo'


torch.cuda.set_device('cuda:0')


# 加载配置
with open(config_path) as f:
    kwargs = yaml.safe_load(f)

# 备份配置文件

logdir = os.path.join(kwargs['path']['logdir'], kwargs['path']['expname'])
os.makedirs(logdir, exist_ok=True)
copyfile(config_path, os.path.join(logdir, 'config.yml'))

kwargs_path = kwargs['path']
kwargs_render = kwargs['render']
kwargs_network = kwargs['networks']
kwargs_train = kwargs['train']
expname = kwargs_path['expname']
datadir = kwargs_path['datadir']
logdir = kwargs_path['logdir']
devices = torch.device('cuda')
batch_size = kwargs_train['batch_size']
total_iterations = kwargs_train['total_iterations']
save_freq = kwargs_train['save_freq']


log_filename = "logger.log"
log_savepath = os.path.join(logdir, expname, log_filename)
logger = logger_config(log_savepath=log_savepath, logging_name='nerf2')
writer = SummaryWriter(os.path.join(logdir, expname, 'tensorboard'))



nerf2_network = NeRF2(**kwargs_network).to(devices)
params = list(nerf2_network.parameters())
## Print total number of parameters
total_params = sum(p.numel() for p in params if p.requires_grad)
logger.info("Total number of parameters: %s", total_params)
optimizer = torch.optim.Adam(params, lr=float(kwargs_train['lr']),
                                    weight_decay=float(kwargs_train['weight_decay']),
                                    betas=(0.9, 0.999))
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=float(kwargs_train['T_max']), eta_min=float(kwargs_train['eta_min']),last_epoch=-1)



## Renderer
renderer = renderer_dict['csi'] # renderer_dict = {"spectrum": Renderer_spectrum, "rssi": Renderer_RSSI, "csi": Renderer_CSI}
renderer = renderer(networks_fn=nerf2_network, **kwargs_render)
scale_worldsize = kwargs_render['scale_worldsize']


## Train Settings
current_iteration = 1


## Dataset
dataset = CSI_dataset
train_index = os.path.join(datadir, "train_index.txt")
test_index = os.path.join(datadir, "test_index.txt")

if not os.path.exists(train_index) or not os.path.exists(test_index):
    split_dataset(datadir, ratio=0.8, dataset_type=dataset_type)
logger.info("Loading training set...")
train_set = dataset(datadir, train_index, scale_worldsize)
logger.info("Loading test set...")
test_set = dataset(datadir, test_index, scale_worldsize)

train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
logger.info("Train set size:%d, Test set size:%d", len(train_set), len(test_set))


logger.info("Start training. Current Iteration:%d", current_iteration)
#%%
while current_iteration <= total_iterations:
    with tqdm(total=len(train_iter), desc=f"Iteration {current_iteration}/{total_iterations}") as pbar:
        for train_input, train_label in train_iter:

            train_input, train_label = train_input.to(devices), train_label.to(devices)
            uplink, rays_o, rays_d = train_input[:, :52], train_input[:, 52:55], train_input[:, 55:]
            predict_downlink = renderer.render_csi(uplink, rays_o, rays_d)
            predict_downlink = torch.concat((predict_downlink.real, predict_downlink.imag), dim=-1)
            loss = sig2mse(predict_downlink, train_label)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cosine_scheduler.step()
            current_iteration += 1

            writer.add_scalar('Loss/loss', loss, current_iteration)
            pbar.update(1)
            pbar.set_description(f"Iteration {current_iteration}/{total_iterations}")
            pbar.set_postfix_str('loss = {:.6f}, lr = {:.6f}'.format(loss.item(), optimizer.param_groups[0]['lr']))

            if current_iteration % save_freq == 0:
                torch.save(nerf2_network.state_dict(), os.path.join(logdir, expname, f"checkpoint_{current_iteration}.pth"))

                n_bs = test_set.n_bs    # number of base station antennas
                n_data = len(test_set)  # number of test data

                all_pred_csi = torch.zeros((n_data, 26), dtype=torch.complex64)
                all_gt_csi = torch.zeros((n_data, 26), dtype=torch.complex64)
                with torch.no_grad():
                    for idx, (test_input, test_label) in enumerate(test_iter):
                        test_input, test_label = test_input.to(devices), test_label.to(devices)
                        uplink, rays_o, rays_d = test_input[:, :52], test_input[:, 52:55], test_input[:, 55:]
                        predict_downlink = renderer.render_csi(uplink, rays_o, rays_d)  # [B, 26]
                        gt_downlink = test_label[:, :26] + 1j * test_label[:, 26:]
                        predict_downlink = test_set.denormalize_csi(predict_downlink)
                        gt_downlink = test_set.denormalize_csi(gt_downlink)

                        all_pred_csi[idx*batch_size:(idx+1)*batch_size] = predict_downlink
                        all_gt_csi[idx*batch_size:(idx+1)*batch_size] = gt_downlink

                all_pred_csi = rearrange(all_pred_csi, '(n_data n_bs) channel -> n_data n_bs channel', n_bs=n_bs)
                all_gt_csi = rearrange(all_gt_csi, '(n_data n_bs) channel -> n_data n_bs channel', n_bs=n_bs)
                snr = csi2snr(all_pred_csi, all_gt_csi)
                logger.info("Median SNR:%.2f", torch.median(snr))

                scio.savemat(os.path.join(logdir, expname, "result.mat"), {'pred_csi': all_pred_csi.cpu().numpy(),
                                                                                    'gt_csi': all_gt_csi.cpu().numpy(),
                                                                                    'snr': snr.cpu().numpy()})






        

# %%
