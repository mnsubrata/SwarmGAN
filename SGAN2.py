import torch
from PIL import Image
import numpy as np
import dnnlib
import legacy
from tqdm import tqdm

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
for i in pbar:
    z = torch.from_numpy(np.random.RandomState(i).randn(1, G.z_dim)).to(device)
    # z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img=(img.permute(0,2,3,1)*127.5+128).clamp(0, 255).to(torch.uint8)
    image=Image.fromarray(img[0].cpu().numpy(), 'RGB')
    image.save(f"test_res/SGAN2/SGAN2_{i:05}.png")
    # image.save(f"/home/priyo/thinclient_drives/G:/My Drive/RSRCH2/SGAN2_TEST/test_res/SGAN2/SGAN2{i:05}.png")
    pbar.set_description(f"SGAN2 {i}th Image done")
    pbar.refresh()