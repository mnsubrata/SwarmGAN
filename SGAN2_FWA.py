import torch
from PIL import Image
import numpy as np
import dnnlib
import legacy
from tqdm import tqdm
from glob2 import glob
import os
from importlib import import_module
GAN='SGAN2'
ALGO='FWA_orig'

opti_alg=import_module(ALGO,'*')
fn_name=getattr(opti_alg,ALGO)

network_pkl='stylegan2-celebahq-256x256.pkl'
seeds=0
truncation_psi=1
noise_mode='const'
outdir='out'
class_idx=None
projected_w=None

print(f'Loading networks from {network_pkl}')
device='cuda' if torch.cuda.is_available() else 'cpu'
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
label = torch.zeros([1, G.c_dim], device=device)

pbar=tqdm(range(1,201))
params=[label,truncation_psi,noise_mode]
for i in pbar:
    latent=fn_name(G,*params)
    # latent=calculate_latent_vector(G,label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    latent=torch.from_numpy(latent).unsqueeze(0).to(device)
    img = G(latent, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img=(img.permute(0,2,3,1)*127.5+128).clamp(0, 255).to(torch.uint8)
    image=Image.fromarray(img[0].cpu().numpy(), 'RGB')
    image.save(f"test_res/{ALGO}/{GAN}_{ALGO}{i:05}.png")
    pbar.set_description(f"{GAN}_{ALGO} {i}th Image done")
    pbar.refresh()