import torch
from typing import Optional, Tuple, List

def sample_points(rays_o, rays_d, near, far, n_samples):
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
    near, far = torch.full(shape, near), torch.full(shape, far)
    t_vals = torch.linspace(0., 1., steps=n_samples) * (far - near) + near  # scale t with near and far
    t_vals = t_vals.to(rays_o.device)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * t_vals[...,:,None] # p = o + td, [n_rays, n_samples, 3]

    return pts, t_vals