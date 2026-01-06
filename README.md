1. Download pre-trained StyleGAN2 model (trained on celebahq-256x256 dataset) and keep it in the root folder
https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl
2. Run dataset_mean_std.py to calculate mean and standard deviation of the dataset. We take real samples from the celebA-HQ dataset.
3. Set mean values to mu list in FWA.py file, line no. 5
4. Run SGAN2.py to generate images from baseline StyleGAN2.
5. Run SGAN2_FWA.py to generate images from StyleGAN2 plugged with the FWA algorithm.
6. Set paths for real images and generated images in fid_calc.py and run this file to print the FID score
N.B.: codes in iqa folder are adopted from https://github.com/chaofengc/IQA-PyTorch/tree/main/pyiqa, and those of torch_utils and dnnlib are taken from https://github.com/NVlabs/stylegan2-ada-pytorch
