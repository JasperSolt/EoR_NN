from diffusers import AutoencoderKL
import h5py
import numpy as np
import torch
from plotting import plot_image_rows


sim="ctrpx"
ws = 3.0

simdict = {
    "p21c":"21cmFAST",
    "zreion":"RLS",
    "ctrpx":"21cmFAST (central-pixel)"
}

pathdict = {
    "p21c":f"/users/jsolt/data/jsolt/21cmFAST_sims/p21c14/p21c14_subdiv_sliced_ws{ws}.hdf5",
    "zreion":f"/users/jsolt/data/jsolt/zreion_sims/zreion24/zreion24_subdiv_sliced_ws{ws}.hdf5",
    "ctrpx":f"/users/jsolt/data/jsolt/centralpix_sims/centralpix05/centralpix05_subdiv_sliced_ws{ws}.hdf5"
}

fname = pathdict[sim]

n = 1
zs = np.linspace(0,29,6, dtype=int)
color_inputs = []
with h5py.File(fname, "r") as f: 
    for z in zs:
        input = f['lightcones/brightness_temp'][n, z].reshape((1,1,256,256))
        color_inputs.append(np.tile(input, (1,3,1,1)))


vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

torch_device = "cpu"
vae.to(torch_device)


encodings = []
for color_input in color_inputs:
    encodings.append(vae.encode(torch.from_numpy(color_input)))
    
rowdict = {f"{j}":[] for j in range(4)}



clean_encodings = [encodings[i].latent_dist.mean.detach().numpy() for i in range(len(encodings))]

for i in range(len(clean_encodings)):
    encoding = clean_encodings[i]
    label_dict = {f"{j}":encoding[0,j] for j in range(4)}
    for label, img in label_dict.items():
        rowdict[label].append(img)

for label in rowdict.keys():
    rowdict[label] = np.array(rowdict[label])
    

plot_image_rows(rowdict)