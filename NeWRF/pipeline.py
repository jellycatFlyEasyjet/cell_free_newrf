#%%

import torch
from torch import nn
import matplotlib.pyplot as plt
from pipelines import pipeline
import numpy as np
from utils import *
from tqdm import trange
from loaders import DatasetLoader
import os, random, string
from omegaconf import OmegaConf
from ray_gen import RayGenerator

# Hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

random_num = random.randint(0, 99)
print(f"生成的随机数: {random_num}")
torch.manual_seed(random_num)
np.random.seed(random_num)

######################## Configurations ###################
cfg = OmegaConf.load('/workspace/qingwang/BoYang/baseline/NeWRF/default.yaml')
n_iters = 100000

env = "conference"

if env == "office":
    cfg.sampling.n_samples = 256+128
    cfg.sampling.n_samples_hierarchical = 128
    cfg.sampling.far = 24
###########################################################
datadir = "./data/"
dataset_fname = '/workspace/qingwang/BoYang/baseline/NeWRF/dataset/conference_500STA_4APs.pkl'

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


model = create_model(encoder.d_output,n_layers=cfg.models.n_layers,d_filter=cfg.models.d_filter,skip=cfg.models.skip,d_viewdirs=d_viewdirs)
model.to(device)
model_params = list(model.parameters())


synthesizer = Synthesizer()


# Optimizer
optimizer = torch.optim.Adam(model_params, lr=cfg.optimizer.lr, weight_decay = cfg.optimizer.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = cfg.optimizer.scheduler_patience, factor=cfg.optimizer.scheduler_factor, min_lr = cfg.optimizer.min_lr, verbose = True)


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
    for j in range(len(aoa_lst)):
        aoa = aoa_lst[j]
        ray_gen = RayGenerator(sta_loc[j], cfg, device, torch.tensor(aoa, dtype=torch.float32))
        rays_o, rays_d = ray_gen.get_rays()
        rays_os.append(rays_o)
        rays_ds.append(rays_d)
        assert(rays_o.shape == rays_d.shape) # [n_rays, 3]
        n_rays_lst.append(rays_o.shape[0])

    rays_o = torch.cat(rays_os, dim = 0)
    rays_d = torch.cat(rays_ds, dim = 0)

    fc = torch.tensor(loader.get_freq()).to(device)
    
    query_points, z_vals = sample_stratified(rays_o, rays_d, cfg.sampling.near, cfg.sampling.far, **kwargs_sample_stratified)

    batches = prepare_chunks(query_points, encode, chunksize=cfg.training.chunksize)

    batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,encode_viewdirs,chunksize=cfg.training.chunksize)

    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        # print(batch.dtype, batch_viewdirs.dtype)
        predictions.append(model(batch, viewdirs=batch_viewdirs))

    raw = torch.cat(predictions, dim=0).reshape(list(query_points.shape[:2]) + [predictions[0].shape[-1]])
    del predictions
    # ray_batches = torch.cumsum(n_rays_lst, dim=0)
    syn_cfr, depth, amp, weights_coarse = synthesizer.synthesize(raw, z_vals, fc, n_rays_lst)

    del batches, batches_viewdirs, query_points
    
    # Calculate Loss and backprop
    target_ap_id = 1
    target_cfr = torch.tensor(loader.get_cfr_batch(target_ap_id, sta_id)).flatten().to(device)  # [batch_size, 1]
    coarse_loss = loss_fn(syn_cfr, target_cfr) 
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
            kwargs_sample_stratified = {
                'n_samples': cfg.sampling.n_samples,
                'perturb': cfg.sampling.perturb,
                'inverse_depth': cfg.sampling.inverse_depth
            }                
            sta_loc = torch.tensor(loader.get_loc_batch("STA", sta_id)).to(device)
            aoa_lst = loader.get_aoa_batch(sta_id)
            rays_os = []
            rays_ds = []
            n_rays_lst = []
            for j in range(len(aoa_lst)):
                aoa = aoa_lst[j]
                ray_gen = RayGenerator(sta_loc[j], cfg, device, torch.tensor(aoa, dtype=torch.float32))
                rays_o, rays_d = ray_gen.get_rays()
                rays_os.append(rays_o)
                rays_ds.append(rays_d)
                assert(rays_o.shape == rays_d.shape) # [n_rays, 3]
                n_rays_lst.append(rays_o.shape[0])

            rays_o = torch.cat(rays_os, dim = 0)
            rays_d = torch.cat(rays_ds, dim = 0)

            fc = torch.tensor(loader.get_freq()).to(device)
            query_points, z_vals = sample_stratified(rays_o, rays_d, cfg.sampling.near, cfg.sampling.far, **kwargs_sample_stratified)
            batches = prepare_chunks(query_points, encode, chunksize=cfg.training.chunksize)
            batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d, encode_viewdirs, chunksize=cfg.training.chunksize)

            predictions = []
            for batch, batch_viewdirs in zip(batches, batches_viewdirs):
                # print(batch.dtype, batch_viewdirs.dtype)
                predictions.append(model(batch, viewdirs=batch_viewdirs))

            raw = torch.cat(predictions, dim=0).reshape(list(query_points.shape[:2]) + [predictions[0].shape[-1]])
            del predictions
            # ray_batches = torch.cumsum(n_rays_lst, dim=0)
            syn_cfr, depth, amp, weights_coarse = synthesizer.synthesize(raw, z_vals, fc, n_rays_lst)
 
            target_ap_id = 2
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
