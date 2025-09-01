import torch
from typing import Optional, Tuple, List
import sys
import os

# 添加virtual_tx_improvement模块到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
vtx_module_path = os.path.join(current_dir, 'virtual_tx_improvement')
if vtx_module_path not in sys.path:
    sys.path.insert(0, vtx_module_path)


def sample_stratified(
    rays_o: torch.Tensor, # [n_theta, n_phi, 3]
    rays_d: torch.Tensor, # [n_theta, n_phi, 3]
    near: float,
    far: float,
    n_samples: int,
    perturb: Optional[bool] = True,
    inverse_depth: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample along ray from regularly-spaced bins.
    Borrowed from: Mason McGough 
    Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
    """
    # Grab samples for space integration along ray
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    if not inverse_depth:
        # Sample linearly between `near` and `far`
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)      # 近处更密集，远处更稀疏
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    # Draw uniform samples from bins along ray
    if perturb:
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
        # 这种扰动采样策略是一种简单但有效的技术，通过引入受控的随机性来提高渲染质量和减少伪影。它在保持原始采样分布特征的同时，增加了采样的多样性。
    # Make sure the z_vals starts from 0
    z_vals[0] = 1e-10
    z_vals = z_vals.unsqueeze(0).unsqueeze(0).expand(rays_o.shape[0], rays_d.shape[0], n_samples)
    rays_o_expanded = rays_o.unsqueeze(1).unsqueeze(2)  
    rays_d_expanded = rays_d.unsqueeze(0).unsqueeze(2)
    
    # z_vals: [1024, 324, 32] -> [1024, 324, 32, 1]
    z_vals_expanded = z_vals.unsqueeze(-1)
    
    # Step 6: 计算采样点坐标
    # pts = rays_o + rays_d * z_vals
    pts = rays_o_expanded + rays_d_expanded * z_vals_expanded  # [1024, 324, 32, 3]
    
    # Apply scale from `rays_d` and offset from `rays_o` to samples

    # pts = rays_lastPts[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [n_rays, n_samples, 3]
    # offset = STA2lastPts.to(z_vals.device).unsqueeze(1)  # [n_rays, 1]
    # z_vals = z_vals + offset  

    return pts, z_vals


def sample_around_virtual_tx(
    rays_o: torch.Tensor,  # [n_theta, n_phi, 3] 射线起点
    rays_d: torch.Tensor,  # [n_theta, n_phi, 3] 射线方向
    rays_lastPts: torch.Tensor,  # [n_rays, 3] 最后交点
    STA2lastPts: torch.Tensor,  # [n_rays] STA到最后交点距离
    AP_pos: torch.Tensor,  # [3] AP位置
    STA_pos: torch.Tensor,  # [3] STA位置
    n_samples: int = 128,
    vtx_radius: float = 1.0,  # Virtual Tx周围采样半径
    use_improved_vtx: bool = True,  # 是否使用改进的Virtual Tx计算
    perturb: Optional[bool] = True,
    inverse_depth: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在计算出的Virtual Tx附近进行集中采样，而不是在整条射线上采样
    
    Args:
        rays_o: 射线起点 [n_rays, 3]
        rays_d: 射线方向 [n_rays, 3] 
        rays_lastPts: 最后交点位置 [n_rays, 3]
        STA2lastPts: STA到最后交点的距离 [n_rays]
        AP_pos: AP位置 [3]
        STA_pos: STA位置 [3]
        n_samples: 采样点数量
        vtx_radius: 在Virtual Tx周围的采样半径
        use_improved_vtx: 是否使用改进的Virtual Tx计算方法
        perturb: 是否添加随机扰动
        inverse_depth: 是否使用逆深度采样
    
    Returns:
        pts: 采样点 [n_rays, n_samples, 3]
        z_vals: 采样深度 [n_rays, n_samples]
        vtx_positions: Virtual Tx位置 [n_rays, 3]
    """
    device = rays_o.device
    n_rays = rays_o.shape[0]
    
    # Step 1: 计算Virtual Tx位置
    if use_improved_vtx:
        try:
            from newrf_virtual_tx import replace_original_virtual_tx_calculation
            # 使用改进的Virtual Tx计算方法
            vtx_positions = replace_original_virtual_tx_calculation(
                rays_lastPts, AP_pos, STA_pos, use_improved=True
            )
        except ImportError:
            print("警告: 无法导入改进的Virtual Tx模块，使用原始方法")
            use_improved_vtx = False
    
    if not use_improved_vtx:
        # 使用原始方法计算Virtual Tx（模拟eval.py中的计算）
        # 方向：从STA指向lastPts
        directions = (rays_lastPts - STA_pos.unsqueeze(0))  # [n_rays, 3]
        directions = directions / directions.norm(dim=1, keepdim=True)
        
        # 距离：从lastPts到AP的距离（这里有问题，但为了与原始方法一致）
        distance = torch.norm(rays_lastPts - AP_pos.unsqueeze(0), dim=1)  # [n_rays]
        
        # Virtual Tx：在lastPts基础上沿STA→lastPts方向延长AP到lastPts的距离
        vtx_positions = rays_lastPts + directions * distance.view(-1, 1)
    
    # Step 2: 计算每条射线上Virtual Tx的深度位置
    # Virtual Tx相对于射线起点的距离
    vtx_distances = torch.norm(vtx_positions - rays_o, dim=-1)  # [n_rays]
    
    # Step 3: 在Virtual Tx附近生成采样范围
    # 采样范围: [vtx_distance - radius, vtx_distance + radius]
    near_distances = torch.clamp(vtx_distances - vtx_radius, min=0.01)  # [n_rays]
    far_distances = vtx_distances + vtx_radius  # [n_rays]
    
    # Step 4: 生成采样深度
    t_vals = torch.linspace(0., 1., n_samples, device=device)  # [n_samples]
    
    if not inverse_depth:
        # 线性采样
        z_vals = near_distances.unsqueeze(1) * (1. - t_vals) + far_distances.unsqueeze(1) * t_vals  # [n_rays, n_samples]
    else:
        # 逆深度采样（近处更密集）
        z_vals = 1./(1./near_distances.unsqueeze(1) * (1.-t_vals) + 1./far_distances.unsqueeze(1) * t_vals)
    
    # Step 5: 添加随机扰动
    if perturb:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])  # [n_rays, n_samples-1]
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)  # [n_rays, n_samples]
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)  # [n_rays, n_samples]
        t_rand = torch.rand([n_rays, n_samples], device=device)
        z_vals = lower + (upper - lower) * t_rand
    
    # 确保最小深度
    z_vals[..., 0] = torch.clamp(z_vals[..., 0], min=1e-10)
    
    # Step 6: 计算3D采样点
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # [n_rays, n_samples, 3]
    
    return pts, z_vals, vtx_positions


def sample_mixed_strategy(
    rays_o: torch.Tensor,  # [n_theta, n_phi, 3] 射线起点
    rays_d: torch.Tensor,  # [n_theta, n_phi, 3] 射线方向  
    rays_lastPts: torch.Tensor,  # [n_rays, 3] 最后交点
    STA2lastPts: torch.Tensor,  # [n_rays] STA到最后交点距离
    AP_pos: torch.Tensor,  # [3] AP位置
    STA_pos: torch.Tensor,  # [3] STA位置
    near: float,
    far: float,
    n_samples: int = 128,
    vtx_ratio: float = 0.7,  # Virtual Tx采样占比
    vtx_radius: float = 1.0,  # Virtual Tx采样半径
    use_improved_vtx: bool = True,
    perturb: Optional[bool] = True,
    inverse_depth: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    混合采样策略：结合原始分层采样和Virtual Tx集中采样
    
    Args:
        vtx_ratio: Virtual Tx采样占总采样数的比例，剩余的用于常规采样
        其他参数同上
    
    Returns:
        pts: 混合采样点 [n_rays, n_samples, 3]
        z_vals: 混合采样深度 [n_rays, n_samples]
        vtx_positions: Virtual Tx位置 [n_rays, 3]
    """
    device = rays_o.device
    n_rays = rays_o.shape[0]
    
    # 分配采样数量
    n_vtx_samples = int(n_samples * vtx_ratio)
    n_regular_samples = n_samples - n_vtx_samples
    
    # Step 1: Virtual Tx集中采样
    if n_vtx_samples > 0:
        pts_vtx, z_vals_vtx, vtx_positions = sample_around_virtual_tx(
            rays_o=rays_o,
            rays_d=rays_d, 
            rays_lastPts=rays_lastPts,
            STA2lastPts=STA2lastPts,
            AP_pos=AP_pos,
            STA_pos=STA_pos,
            n_samples=n_vtx_samples,
            vtx_radius=vtx_radius,
            use_improved_vtx=use_improved_vtx,
            perturb=perturb,
            inverse_depth=inverse_depth
        )
    else:
        vtx_positions = None
        pts_vtx = torch.empty((n_rays, 0, 3), device=device)
        z_vals_vtx = torch.empty((n_rays, 0), device=device)
    
    # Step 2: 常规分层采样
    if n_regular_samples > 0:
        pts_regular, z_vals_regular = sample_stratified(
            rays_o=rays_o,
            rays_d=rays_d,
            rays_lastPts=rays_lastPts,
            STA2lastPts=STA2lastPts,
            near=near,
            far=far,
            n_samples=n_regular_samples,
            perturb=perturb,
            inverse_depth=inverse_depth
        )
    else:
        pts_regular = torch.empty((n_rays, 0, 3), device=device)
        z_vals_regular = torch.empty((n_rays, 0), device=device)
    
    # Step 3: 合并采样点并排序
    if n_vtx_samples > 0 and n_regular_samples > 0:
        # 合并z值并排序
        z_vals_combined = torch.cat([z_vals_vtx, z_vals_regular], dim=-1)  # [n_rays, n_samples]
        z_vals_sorted, sort_indices = torch.sort(z_vals_combined, dim=-1)
        
        # 合并并重新排序采样点
        pts_combined = torch.cat([pts_vtx, pts_regular], dim=-2)  # [n_rays, n_samples, 3]
        pts_sorted = torch.gather(pts_combined, -2, sort_indices.unsqueeze(-1).expand(-1, -1, 3))
        
        return pts_sorted, z_vals_sorted, vtx_positions
    
    elif n_vtx_samples > 0:
        return pts_vtx, z_vals_vtx, vtx_positions
    else:
        return pts_regular, z_vals_regular, vtx_positions

def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    perturb: bool = False
) -> torch.Tensor:
    r"""
    Apply inverse transform sampling to a weighted set of points.
    Borrowed from: Mason McGough 
    Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
    """
    # Normalize weights to get PDF.
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True) # [n_rays, weights.shape[-1]]
    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1) # [n_rays, weights.shape[-1]]
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # [n_rays, weights.shape[-1] + 1]

    # Take sample positions to grab from CDF. Linear when perturb == 0.
    if not perturb: #只做一次扰动
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]) # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device) # [n_rays, n_samples]

    # Find indices along CDF where values in u would be placed.
    u = u.contiguous() # Returns contiguous tensor with same values.
    inds = torch.searchsorted(cdf, u, right=True) # [n_rays, n_samples]

    # Clamp indices that are out of bounds.
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1) # [n_rays, n_samples, 2]

    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                         index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                          index=inds_g)

    # Convert samples to ray length.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples # [n_rays, n_samples]

def sample_hierarchical(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    weights: torch.Tensor,
    n_new_samples: int,
    perturb: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Apply hierarchical sampling to the rays.
    Borrowed from: Mason McGough 
    Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
    """
    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])  #z_vals实际已经干扰了 # [N_rays, N_samples_stratified-1] --> N_bins = N_samples_stratified-1
    new_z_samples = sample_pdf(z_vals_mid, 
                               weights[..., 1:-1], #排除头尾两个采样点           # [N_rays, N_samples_stratified-2]
                               n_new_samples,
                               perturb=perturb)
    new_z_samples = new_z_samples.detach()
    
    # Resample points from ray based on PDF.
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]  # [N_rays, N_samples + n_samples, 3]
    return pts, z_vals_combined, rays_o, rays_d
