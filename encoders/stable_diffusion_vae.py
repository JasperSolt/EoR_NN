import h5py
import numpy as np
import torch
from diffusers import AutoencoderKL
import sys
sys.path.append('../util')
from plotting import plot_image_rows

'''
Helpful reference:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl.py
https://huggingface.co/docs/diffusers/api/models/autoencoderkl
'''

#Initialize
def get_path(model_sim, ws):
    if model_sim == "p21c":
        dp = f"/users/jsolt/data/jsolt/21cmFAST_sims/p21c14/p21c14_subdiv_sliced_ws{ws}.hdf5"
    elif model_sim == "zreion":
        dp = f"/users/jsolt/data/jsolt/zreion_sims/zreion24/zreion24_subdiv_sliced_ws{ws}.hdf5"
    elif model_sim == "ctrpx":
        dp = f"/users/jsolt/data/jsolt/centralpix_sims/centralpix05/centralpix05_subdiv_sliced_ws{ws}.hdf5"
    return dp

sim = "ctrpx"
ws = 0.0
data_dir = f"/users/jsolt/data/jsolt/"
path = get_path(sim, ws)

#Get random sample
n = 8
z = 15
with h5py.File(path, 'r') as f:
    x = f['lightcones/brightness_temp'][n,z]
    
#process
x = np.repeat(x[np.newaxis,:,:], 3, axis=0)
x = torch.from_numpy(x[np.newaxis,:,:,:])

#load VAE
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

#encode
z = vae.encode(x).latent_dist.mean # 1 x 3 x 256 x 256 -> 1 x 4 x 32 x 32

#decode
output = vae.decode(z)

inpt_img = np.mean(x[0].detach().numpy(), axis=0)
outpt_img = np.mean(output.sample[0].detach().numpy(), axis=0)

rowdict = {
    'input':np.array([inpt_img,]),
    'output':np.array([outpt_img,]),
    }

plot_image_rows(rowdict)