#%%
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from tqdm import trange
from loaders import DatasetLoader
import os, random, string
from omegaconf import OmegaConf
from ray_gen import RayGenerator
from synthesizers import raw2outputs_signal
from samplers import sample_points
from einops import rearrange, repeat


# Hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

random_num = random.randint(0, 99)
print(f"生成的随机数: {random_num}")
torch.manual_seed(random_num)
np.random.seed(random_num)

######################## Configurations ###################
cfg = OmegaConf.load('/home/byang/BoYang/baseline/NeRF2_NeWRF_version/default.yaml')
n_iters = 100000
base_AP =[1]
target_AP = [1]
env = "conference"

if env == "office":
    cfg.sampling.n_samples = 256+128
    cfg.sampling.n_samples_hierarchical = 128
    cfg.sampling.far = 24
###########################################################
datadir = "./data/"
dataset_fname = '/home/byang/BoYang/baseline/NeRF2_NeWRF_version/dataset/dataset_conference.pkl'

if not os.path.exists(cfg.training.save_dir):
    os.mkdir(cfg.training.save_dir)
file_dir = os.path.join(cfg.training.save_dir, ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)))
os.mkdir(file_dir)
print("Checkpoint will be saved at: ", file_dir)

# Speficify the checkpoint file name to load if retraining
# ckpt_fname = r"ckpt\conference_ckpt_1R.pt"
ckpt_fname = None

loader = DatasetLoader(dataset_fname)
loader.split_train_val_test(train_ratio=0.8, val_ratio=0.1,seed=16)


encoder = PositionalEncoder(cfg.encoder.d_input, cfg.encoder.n_freqs, log_space=cfg.encoder.log_space)
def encode(x): return encoder(x)
encoder_viewdirs = PositionalEncoder(cfg.encoder.d_input, cfg.encoder.n_freqs_views, log_space=cfg.encoder.log_space)
def encode_viewdirs(x): return encoder_viewdirs(x)
d_viewdirs = encoder_viewdirs.d_output


model = NeRF2()
model.to(device)
model_params = list(model.parameters())


# Optimizer
optimizer = torch.optim.Adam(model_params, lr=cfg.optimizer.lr, weight_decay = cfg.optimizer.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = cfg.optimizer.scheduler_patience, factor=cfg.optimizer.scheduler_factor, min_lr = cfg.optimizer.min_lr)


loss_fn = lambda pred, true: torch.sum(torch.abs(pred-true)**2)/torch.sum(torch.abs(true)**2) 
snr = lambda loss: -10. * np.log10(loss)


# Training iterations
train_snrs = []
val_snrs = []
iternum = []
train_losses = []
val_losses = []

for i in trange(n_iters):
    logging.debug(f"Iteration: {i}")
    # Run the training pipeline

    sta_id = np.random.choice(loader.trainset, cfg.training.batch_size)
    model.train()

    kwargs_sample_stratified = {
        'n_samples': cfg.sampling.n_samples,
        'perturb': cfg.sampling.perturb,
        'inverse_depth': cfg.sampling.inverse_depth
    }

    sta_loc = torch.tensor(loader.get_loc_batch("STA", sta_id)).to(device)

    # Get Rays for backtracing
    aoa_lst = loader.get_aoa_batch(sta_id)
    rays_os = []
    rays_ds = []
    n_rays_lst = []
    cfrs = []

    ray_gen = RayGenerator(sta_loc, cfg, device, torch.tensor(aoa_lst[0], dtype=torch.float32))
    cfr = torch.tensor(loader.get_cfr_batch(base_AP[0], sta_id)).to(device)
    rays_o, rays_d = ray_gen.get_rays()
    rays_d = repeat(rays_d, 'b d -> v b d', v = rays_o.size(0))
    rays_o = rays_o.unsqueeze(1).expand(-1, rays_d.size(1), -1)

    fc = torch.tensor(loader.get_freq()).to(device)

    pts, t_vals = sample_points(rays_o, rays_d, cfg.sampling.near, cfg.sampling.far, cfg.sampling.n_samples)
    cfr = repeat(cfr, 'b d -> b v d', v=rays_d.size(1))        #[bs, 9x36, 52]
    cfr = repeat(cfr, 'b v d -> b v p d', p=pts.size(2))  # [bs, cks, pts, 3]
    cfr = torch.cat((cfr.real,cfr.imag), dim=-1)
    views = repeat(rays_d, 'b v d -> b v p d', p=pts.size(2))
    raw = model(pts, views, cfr)
    recv_signal = raw2outputs_signal(raw, t_vals, rays_d) # z_vals:64,324,32 batches_viewdirs:64,324,3
    recv_signal = recv_signal.squeeze()
    # Calculate Loss and backprop
    target_ap_id = 1
    target_cfr = torch.tensor(loader.get_cfr_batch(target_ap_id, sta_id)).flatten().to(device)  # [batch_size, 1]
    coarse_loss = loss_fn(recv_signal, target_cfr) 
    train_losses.append(coarse_loss.item())

    optimizer.zero_grad()
    coarse_loss.backward()
    optimizer.step()
    
    del rays_o, rays_d


    train_snrs.append(snr(coarse_loss.item()))

#####################################################################################################################################################################
#####################################################################################################################################################################
    if i % cfg.training.display_rate == 0:
        with torch.no_grad():
            sta_id = loader.valset
            model.eval()
                
            sta_loc = torch.tensor(loader.get_loc_batch("STA", sta_id)).to(device)
            
            # Get Rays for validation - using same approach as training
            aoa_lst = loader.get_aoa_batch(sta_id)
            
            ray_gen = RayGenerator(sta_loc, cfg, device, torch.tensor(aoa_lst[0], dtype=torch.float32))
            cfr = torch.tensor(loader.get_cfr_batch(base_AP[0], sta_id)).to(device)
            rays_o, rays_d = ray_gen.get_rays()
            rays_d = repeat(rays_d, 'b d -> v b d', v = rays_o.size(0))
            rays_o = rays_o.unsqueeze(1).expand(-1, rays_d.size(1), -1)

            fc = torch.tensor(loader.get_freq()).to(device)

            # Use same sampling approach as training
            pts, t_vals = sample_points(rays_o, rays_d, cfg.sampling.near, cfg.sampling.far, cfg.sampling.n_samples)
            cfr = repeat(cfr, 'b d -> b v d', v=rays_d.size(1))        #[bs, 9x36, 52]
            cfr = repeat(cfr, 'b v d -> b v p d', p=pts.size(2))  # [bs, cks, pts, 3]
            cfr = torch.cat((cfr.real,cfr.imag), dim=-1)
            views = repeat(rays_d, 'b v d -> b v p d', p=pts.size(2))
            
            # Use same model call as training
            raw = model(pts, views, cfr)
            syn_cfr = raw2outputs_signal(raw, t_vals, rays_d)
            syn_cfr = syn_cfr.squeeze()
 
            target_ap_id = target_AP[0]
            target_cfr = torch.tensor(loader.get_cfr_batch(target_ap_id, sta_id)).flatten().to(device)  # [batch_size, 1]
            coarse_loss = loss_fn(syn_cfr, target_cfr) 

            val_losses.append(coarse_loss.item())
            val_snrs.append(snr(coarse_loss.item()))
            iternum.append(i)

            #-------------- Plot results-------------------------------
            fig, ax = plt.subplots(1, 3, figsize=(12,4))

            ax[0].plot(np.real(target_cfr.cpu()), np.imag(target_cfr.cpu()), "ro", label = "GT")
            ax[0].plot(np.real(syn_cfr.cpu()), np.imag(syn_cfr.cpu()), "bo", label = "Coarse")
            ax[0].set_title(f"SNR: {snr(coarse_loss.cpu()).item():.2f} dB")
            ax[0].legend()
    
            #plot the training and val loss curve
            ax[1].plot(range(0, i + 1), train_losses, 'r', label = "Train loss")
            ax[1].plot(iternum, val_losses, 'y', label = 'Val loss')
            ax[1].legend()


            ax[2].plot(range(0, i + 1), train_snrs, 'r', label = "Train SNR")
            ax[2].plot(iternum, val_snrs, 'y', label = 'Val SNR')
            ax[2].legend()
            plt.show()





















































# %%
