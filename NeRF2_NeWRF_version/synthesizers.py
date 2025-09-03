import torch
import numpy as np
import torch.nn.functional as F
import scipy.constants as sc
from einops import rearrange, repeat


def raw2outputs_signal(raw, r_vals, rays_d):
    """Transforms model's predictions to semantically meaningful values.

    Parameters
    ----------
    raw : [batchsize, chunks,n_samples,  4]. Prediction from model.
    r_vals : [batchsize, chunks, n_samples]. Integration distance.
    rays_d : [batchsize,chunks, 3]. Direction of each ray

    Return:
    ----------
    receive_signal : [batchsize, 26]. OFDM singal of each ray
    """
    wavelength = sc.c / 2.4e9
    # raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
    # raw2phase = lambda raw, dists: torch.exp(1j*raw*dists)
    raw2phase = lambda raw, dists: raw + 2*np.pi*dists/wavelength
    raw2amp = lambda raw, dists: -raw*dists

    dists = r_vals[...,1:] - r_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [batchsize, chunks, n_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)  # [batchsize,chunks, n_samples, 3].

    att_a, att_p, s_a, s_p = raw[...,0:1], raw[...,1:2], raw[...,2:3], raw[...,3:4]    # [batchsize,chunks, N_samples]
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
