# -*- coding: utf-8 -*-
"""dataset processing and loading
"""
import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange



# def rssi2amplitude(rssi):
#     """convert rssi to amplitude
#     """
#     return 100 * 10 ** (rssi / 20)


# def amplitude2rssi(amplitude):
#     """convert amplitude to rssi
#     """
#     return 20 * np.log10(amplitude / 100)


def rssi2amplitude(rssi):
    """convert rssi to amplitude
    """
    return 1 - (rssi / -100)


def amplitude2rssi(amplitude):
    """convert amplitude to rssi
    """
    return -100 * (1 - amplitude)


def split_dataset(datadir, ratio=0.8, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "mimo":
        csi_dir = os.path.join(datadir, 'csidata.npy')
        index = [i for i in range(np.load(csi_dir).shape[0])]
        random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')




class CSI_dataset(Dataset):

    def __init__(self, datadir, indexdir, scale_worldsize=1):
        """ datasets [datalen*8, up+down+r_o+r_d] --> [datalen*8, 26+26+3+36*3]
        """
        super().__init__()
        self.datadir = datadir
        self.csidata_dir = os.path.join(datadir, 'csidata.npy')
        self.bs_pos_dir = os.path.join(datadir, 'base-station.yml')
        self.rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 9, 36  # resulution of rays

        # load base station position
        with open(os.path.join(self.bs_pos_dir)) as f:
            bs_pos_dict = yaml.safe_load(f)
            self.bs_pos = torch.tensor([bs_pos_dict["base_station"]], dtype=torch.float32).squeeze()
            self.bs_pos = self.bs_pos / scale_worldsize
            self.n_bs = len(self.bs_pos)

        # load CSI data
        csi_data = torch.from_numpy(np.load(self.csidata_dir))  #[N, 8, 52]
        csi_data = self.normalize_csi(csi_data)
        uplink, downlink = csi_data[..., :26], csi_data[..., 26:]
        up_real, up_imag = torch.real(uplink), torch.imag(uplink)
        down_real, down_imag = torch.real(downlink), torch.imag(downlink)
        self.uplink = torch.cat([up_real, up_imag], dim=-1)    # [N, 8, 52]
        self.downlink = torch.cat([down_real, down_imag], dim=-1)    # [N, 8, 52]
        self.uplink = rearrange(self.uplink, 'n g c -> (n g) c')    # [N*8, 52]
        self.downlink = rearrange(self.downlink, 'n g c -> (n g) c')    # [N*8, 52]

        self.nn_inputs, self.nn_labels = self.load_data()


    def normalize_csi(self, csi):
        self.csi_max = torch.max(abs(csi))
        return csi / self.csi_max

    def denormalize_csi(self, csi):
        assert self.csi_max is not None, "Please normalize csi first"
        return csi * self.csi_max


    def load_data(self):
        """load data from datadir to memory for training

        Returns
        --------
        nn_inputs : tensor. [n_samples, 1027]. The inputs for training
                    uplink: 52 (26 real; 26 imag), ray_o: 3, ray_d: 9x36x3, n_samples = n_dataset * n_bs
        nn_labels : tensor. [n_samples, 52]. The downlink channels as labels
        """
        ## NOTE! Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 52+3+3*self.alpha_res*self.beta_res)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 52)), dtype=torch.float32)

        ## generate rays origin at gateways
        bs_ray_o, bs_rays_d = self.gen_rays_gateways()
        bs_ray_o = rearrange(bs_ray_o, 'n g c -> n (g c)')   # [n_bs, 1, 3] --> [n_bs, 3]
        bs_rays_d = rearrange(bs_rays_d, 'n g c -> n (g c)') # [n_bs, n_rays, 3] --> [n_bs, n_rays*3]

        ## Load data
        for data_counter, idx in tqdm(enumerate(self.dataset_index), total=len(self.dataset_index)):
            bs_uplink = self.uplink[idx*self.n_bs: (idx+1)*self.n_bs]    # [n_bs, 52]
            bs_downlink = self.downlink[idx*self.n_bs: (idx+1)*self.n_bs]    # [n_bs, 52]
            nn_inputs[data_counter*self.n_bs: (data_counter+1)*self.n_bs] = torch.cat([bs_uplink, bs_ray_o, bs_rays_d], dim=-1) # [n_bs, 52+3+3*36*9]
            nn_labels[data_counter*self.n_bs: (data_counter+1)*self.n_bs]  = bs_downlink
        return nn_inputs, nn_labels

    def gen_rays_gateways(self):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_bs, 1, 3]. The origin of rays
        r_d : tensor. [n_bs, n_rays, 3]. The direction of rays, unit vector
        """
        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)    # [0,1,2,3,....]
        betas = betas.repeat_interleave(self.alpha_res)    # [0,0,0,0,...]

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (1*360)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 3]
        r_d = r_d.expand([self.n_bs, self.beta_res * self.alpha_res, 3])  # [n_bs, 9*36, 3]
        r_o = self.bs_pos.unsqueeze(1) # [n_bs, 1, 3]
        r_o, r_d = r_o.contiguous(), r_d.contiguous()

        return r_o, r_d


    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]


    def __len__(self):
        return len(self.dataset_index) * self.n_bs


dataset_dict = {"mimo": CSI_dataset}
