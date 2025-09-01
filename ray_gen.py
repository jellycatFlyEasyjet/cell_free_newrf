import torch
import matplotlib.pyplot as plt

def euclidian2spherical(x, y, z):
    # Calculate theta
    theta = torch.arctan2(torch.sqrt(x**2 + y**2), z)
    # Calculate phi
    phi = torch.arctan2(y, x)

    return theta, phi

def spherical2euclidian(theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack([x, y, z]).T
 
class RayGenerator: # ray_gen = RayGenerator(sta_loc[i], cfg, device, torch.tensor(aoa, dtype=torch.float32))
    def __init__(self, 
                 center: torch.Tensor, 
                 cfg, 
                 device: torch.device, 
                 aoa:torch.Tensor = None) -> None:
        self.center = center
        self.device = device
        self.cfg = cfg

    def get_rays(self):
        """
        Generate sample rays originating from a single point (e.g., user or center),
        using spherical angles (alpha, beta) to sample directions.

        Returns
        -------
        rays_o : tensor. [1, 3]. The origin of rays
        rays_d : tensor. [n_rays, 3]. The direction of rays (unit vectors)
        """
        alpha_res = self.cfg.sampling.alpha_res  # e.g., 36
        beta_res = self.cfg.sampling.beta_res    # e.g., 9

        alphas = torch.linspace(0, 350, alpha_res) / 180 * torch.pi
        betas = torch.linspace(10, 90, beta_res) / 180 * torch.pi

        alphas = alphas.repeat(beta_res)                      # [alpha_res * beta_res]
        betas = betas.repeat_interleave(alpha_res)            # [alpha_res * beta_res]

        # Compute directions
        x = torch.cos(alphas) * torch.cos(betas)
        y = torch.sin(alphas) * torch.cos(betas)
        z = torch.sin(betas)

        rays_d = torch.stack([x, y, z], dim=1).to(self.device)   # [n_rays, 3]
        rays_o = self.center.to(self.device)     # [1, 3]
        
        return rays_o, rays_d
   
    # def get_rays(self,aoaNum):
    #     r"""
    #     Find origin and direction of rays at each rxpos
    #     """
    #     if self.cfg.sampling.known_aoa:
    #         # use AoA information to generate rays
    #         rays_o = self.center.to(self.device)    # ray_gen = RayGenerator(sta_loc[i], cfg, device, torch.tensor(aoa, dtype=torch.float32))
    #         rays_d = self.aoa.to(self.device)
    #         if self.cfg.sampling.doa_noise > 0:
    #             theta, phi = euclidian2spherical(rays_d[:,0], rays_d[:,1], rays_d[:,2]) # theta: [0, pi], phi: [-pi, pi]
    #             doa_noise_std = self.cfg.sampling.doa_noise/180*torch.pi
    #             theta_noise = (torch.rand([rays_d.shape[0]], device=rays_d.device)-0.5)*doa_noise_std*2
    #             phi_noise = (torch.rand([rays_d.shape[0]], device=rays_d.device)-0.5)*doa_noise_std*2
    #             noisy_theta, noisy_phi = (theta+theta_noise) % torch.pi, ((phi+phi_noise+torch.pi) % (2*torch.pi))-torch.pi
    #             rays_d = spherical2euclidian(noisy_theta, noisy_phi)
            
    #         if self.cfg.sampling.num_extra_rays > 0:    #再多采集10条射线
    #             theta_vec = torch.rand(self.cfg.sampling.num_extra_rays)*torch.pi-torch.pi/2    # [-pi/2, pi/2] 
    #             phi_vec = torch.rand(self.cfg.sampling.num_extra_rays)*torch.pi*2-torch.pi  # [-pi, pi]
    #             ux = torch.cos(theta_vec)*torch.cos(phi_vec)
    #             uy = torch.cos(theta_vec)*torch.sin(phi_vec)
    #             uz = torch.sin(theta_vec)
    #             rays_d_extra = torch.stack([ux, uy, uz], dim=-1).reshape([-1, 3]).to(self.device)
    #             splits = torch.split(rays_d, aoaNum) 
    #             extra_splits = torch.split(rays_d_extra, self.cfg.sampling.num_extra_rays_perAP)
    #             combined = [torch.cat([orig, extra], dim=0) for orig, extra in zip(splits, extra_splits)]
    #             rays_d = torch.cat(combined, dim=0)
    #         rays_o = rays_o.expand(rays_d.shape)    # 将射线的起点扩展到射线的数量

    #     else:
    #         rays_o = self.center.to(self.device)
    #         theta_steps = self.cfg.sampling.num_theta_samples
    #         self.theta_vec = torch.linspace(-torch.pi/2+self.d_theta, torch.pi/2-self.d_theta, steps=theta_steps-2, dtype=torch.float32).to(self.device)
    #         self.phi_vec = torch.arange(-torch.pi, torch.pi, self.d_phi, dtype=torch.float32).to(self.device)
            
    #         self.theta, self.phi = torch.meshgrid(self.theta_vec, self.phi_vec, indexing='ij')

    #         ux = torch.cos(self.theta)*torch.cos(self.phi)
    #         uy = torch.cos(self.theta)*torch.sin(self.phi)
    #         uz = torch.sin(self.theta)
            
    #         rays_d = torch.stack([ux, uy, uz], dim=-1).reshape([-1, 3])
    #         # if not self.perturb:
    #         rays_d = torch.concat([torch.tensor([[0, 0, 1]], device=self.device), 
    #                             rays_d.reshape([-1, 3]), 
    #                             torch.tensor([[0, 0, -1]], device=self.device)], 
    #                             dim = 0)
    #         rays_o = rays_o.expand(rays_d.shape)   # [n_rays, 3]
            
    #     return rays_o, rays_d

    def show_grid(self):
        plt.scatter(self.phi.cpu().numpy(), self.theta.cpu().numpy())
        plt.xlabel("$\phi$")
        plt.ylabel("$\\theta$")
    
    def show_rays(self):
        ray_origin, ray_direction = self.get_rays()
        ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
        _ = ax.quiver(
            ray_origin[...,0].cpu().flatten(),
            ray_origin[...,1].cpu().flatten(),
            ray_origin[...,2].cpu().flatten(),
            ray_direction[...,0].cpu().flatten(),
            ray_direction[...,1].cpu().flatten(),
            ray_direction[...,2].cpu().flatten(), 
            length=0.5, 
            normalize=True
            )
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('z')
        return ax

    