import torch
from typing import Optional, Tuple, List, Union, Callable
import logging
import os
from models import * 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from encoders import PositionalEncoder

def get_chunks(
    inputs: torch.Tensor,
    chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Divide an input into chunks.
    Borrowed from: Mason McGough 
    Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def prepare_chunks(
    points: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify points to prepare for NeRF model.
    Borrowed from: Mason McGough 
    Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
    """
    points = points.reshape((-1, 3))
    # points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    return points


def prepare_viewdirs_chunks(
    points: torch.Tensor,
    rays_d: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify viewdirs to prepare for NeRF model.
    Borrowed from: Mason McGough 
    Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
    """
    # Prepare the viewdirs
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    # viewdirs = encoding_function(viewdirs)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs


def save_ckpt(coarse_model, fine_model, optimizer, save_path):
    torch.save({
        'coarse_model_state_dict': coarse_model.state_dict(),
        'fine_model_state_dict':fine_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
