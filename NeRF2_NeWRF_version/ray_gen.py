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

class RayGenerator:
    def __init__(self, 
                 center: torch.Tensor, 
                 cfg, 
                 device: torch.device, 
                 aoa:torch.Tensor = None) -> None:
        self.center = center
        self.d_theta = torch.pi/cfg.sampling.num_theta_samples
        self.d_phi = 2*torch.pi/cfg.sampling.num_phi_samples
        self.device = device
        self.perturb = cfg.sampling.perturb_aoa
        self.cfg = cfg
        self.theta = None
        self.phi = None
        self.aoa = aoa
    
    def get_rays(self):
        r"""
        Find origin and direction of rays at each rxpos
        """

        rays_o = self.center.to(self.device)
        theta_steps = self.cfg.sampling.num_theta_samples
        self.theta_vec = torch.linspace(-torch.pi/2+self.d_theta, torch.pi/2-self.d_theta, steps=theta_steps-2, dtype=torch.float32).to(self.device)
        self.phi_vec = torch.arange(-torch.pi, torch.pi, self.d_phi, dtype=torch.float32).to(self.device)
        
        self.theta, self.phi = torch.meshgrid(self.theta_vec, self.phi_vec, indexing='ij')

        ux = torch.cos(self.theta)*torch.cos(self.phi)
        uy = torch.cos(self.theta)*torch.sin(self.phi)
        uz = torch.sin(self.theta)
        
        rays_d = torch.stack([ux, uy, uz], dim=-1).reshape([-1, 3])
        # if not self.perturb:
        rays_d = torch.concat([torch.tensor([[0, 0, 1]], device=self.device), 
                            rays_d.reshape([-1, 3]), 
                            torch.tensor([[0, 0, -1]], device=self.device)], 
                            dim = 0)

        return rays_o, rays_d

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

    