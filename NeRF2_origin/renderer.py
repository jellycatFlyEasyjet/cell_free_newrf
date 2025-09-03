# -*- coding: utf-8 -*-
"""code for ray marching and signal rendering
"""
import torch
import numpy as np
import torch.nn.functional as F
import scipy.constants as sc
from einops import rearrange, repeat


class Renderer():

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """

        ## Rendering parameters
        self.network_fn = networks_fn
        self.n_samples = kwargs['n_samples']    # 32 
        self.near = kwargs['near']
        self.far = kwargs['far']


    def sample_points(self, rays_o, rays_d):
        """sample points along rays

        Parameters
        ----------
        rays_o : tensor. [n_rays, 3]. The origin of rays
        rays_d : tensor. [n_rays, 3]. The direction of rays

        Returns
        -------
        pts : tensor. [n_rays, n_samples, 3]. The sampled points along rays
        t_vals : tensor. [n_rays, n_samples]. The distance from origin to each sampled point
        """
        shape = list(rays_o.shape)
        shape[-1] = 1
        near, far = torch.full(shape, self.near), torch.full(shape, self.far)
        t_vals = torch.linspace(0., 1., steps=self.n_samples) * (far - near) + near  # scale t with near and far
        t_vals = t_vals.to(rays_o.device)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * t_vals[...,:,None] # p = o + td, [n_rays, n_samples, 3]

        return pts, t_vals




class Renderer_CSI(Renderer):
    """Renderer for CSI (integral from all directions)
    """

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """
        super().__init__(networks_fn, **kwargs)



    def render_csi(self, uplink, rays_o, rays_d):
        """render the RSSI for each gateway.

        Parameters
        ----------
        uplink: tensor. [batchsize, 52]. The uplink CSI (26 real + 26 imag)
        rays_o : tensor. [batchsize, 3]. The origin of rays
        rays_d : tensor. [batchsize, 9x36x3]. The direction of rays
        """

        rays_d = rearrange(rays_d, 'b (v d) -> b v d', d=3)    # [bs, 9x36, 3]
        batchsize, viewsize, _ = rays_d.shape
        rays_o = repeat(rays_o, 'b d -> b v d', v=viewsize) #[bs, 9x36, 3]
        uplink = repeat(uplink, 'b d -> b v d', v=viewsize)        #[bs, 9x36, 52]

        pts, t_vals = self.sample_points(rays_o, rays_d) # [bs, 9x36, pts, 3]
        views = repeat(rays_d, 'b v d -> b v p d', p=self.n_samples)  # [bs, 9x36, pts, 3]
        uplink = repeat(uplink, 'b v d -> b v p d', p=self.n_samples)  # [bs, cks, pts, 3]

        # Run network and compute outputs
        raw = self.network_fn(pts, views, uplink)    # [batchsize, 9x36, pts, 4]
        recv_signal = self.raw2outputs_signal(raw, t_vals, rays_d)  # [bs, 26]

        return recv_signal


    def raw2outputs_signal(self, raw, r_vals, rays_d):
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

        att_a, att_p, s_a, s_p = raw[...,:26], raw[...,26:52], raw[...,52:78], raw[...,78:104]    # [batchsize,chunks, N_samples]
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




renderer_dict = {"csi": Renderer_CSI}