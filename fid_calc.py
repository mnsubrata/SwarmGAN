import torch
from iqa.fid_pyiqa import FID

device='cuda' if torch.cuda.is_available() else 'cpu'


FID_obj=FID().to(device=device) #####for priya_iqa
f_img_path='test_res/FWA' # path where generated images are stored
r_img_path='CELBAHQ256/'  # path where original images are stored
score_iqa=FID_obj(f_img_path,r_img_path)
print(score)
