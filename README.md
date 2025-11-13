1. Download pre-trained stylegan2 model (trained on celebahq-256x256 dataset) and keep it in root folder

https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl

2. run dataset_mean_std.py to calculate mean and standard deviation of the dataset. We take only 200 real samples from celebA-HQ dataset.
3. set mean values to mu list in FWA.py file line no. 5
4. Run SGAN2.py to generate images from baseline StyleGAN2.
5. Run SGAN2_FWA.py to generate images from StyleGAN2 plugged with FWA algorithm.
6. Set paths for real images and generated images in fid_calc.py and run this file to print FID score
N.B. : codes in iqa folder are adopted from https://github.com/chaofengc/IQA-PyTorch/tree/main/pyiqa and that of torch_utils are taken from https://github.com/NVlabs/stylegan2-ada-pytorch
