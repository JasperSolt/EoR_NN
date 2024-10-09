import os
import numpy as np
import argparse
import torch
from variational_autoencoder import train_vae
from hyperparams import DataHyperparameters, ModelHyperparameters

def get_path(model_sim, ws):
    if model_sim == "p21c":
        dp = f"/users/jsolt/data/jsolt/21cmFAST_sims/p21c14/p21c14_ws{ws}_trnspsd_meanremoved_norm.hdf5"
    elif model_sim == "zreion":
        dp = f"/users/jsolt/data/jsolt/zreion_sims/zreion24/zreion24_transposed_ws{ws}.hdf5"
    elif model_sim == "ctrpx":
        dp = f"/users/jsolt/data/jsolt/21cmFAST_centralpix_v05/21cmFAST_centralpix_v05_transposed_ws{ws}.hdf5"
    return dp

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s','--sims', nargs='+', required=True, type=str, help='<Required> List of sims')
parser.add_argument('-n', '--modelnum', required=True, type=float, help='<Required> Model number')
parser.add_argument('-ws', '--wedgeslope', nargs='?', type=float, default=0.0, help='Wedge slope of data (default = 0)')
parser.add_argument('-lr', '--loglr', nargs='?', type=float, default=-3, help='Log learning rate of model (default = -3)')
parser.add_argument('-d','--dims', nargs=3, type=int, help='tuple of dims')
parser.add_argument('--mode', type=str, choices=['run', 'debug'], default='run', help='Run vs debug mode')
args = parser.parse_args()

train_sims = args.sims
ws = args.wedgeslope

#Initializing data hyperparameters
hp_train_data = DataHyperparameters(
    sims=train_sims,
    data_paths=[get_path(ts, ws) for ts in train_sims],
    zindices=np.linspace(0, 511, 30, dtype=int).tolist(),
    boxlength=256,
    param_index=1,
    ztransform=["zoomin"],
    n_channels=1,
)

if args.mode=='debug': hp_train_data.lenlimit=8


#Initializing model hyperparameters
hd1, hd2, ld = args.dims
lr = 10**args.loglr
    
modelnum = args.modelnum
model_name = f"single_channel_vae_v{modelnum:0>4}"

for ts in train_sims:
    model_name += f"_{ts}"
model_name += f"_ws{ws}"
if args.mode=='debug': model_name="debug"
model_dir = "models/" + model_name


hp_model = ModelHyperparameters(
    model_name=model_name, 
    device="cuda" if torch.cuda.is_available() else "cpu",
    training_data_hp=hp_train_data, 
    batchsize=3 if args.mode=='debug' else 128,
    epochs=10 if args.mode=='debug' else 500, 
    initial_lr=lr,
    lr_milestones=[],
    lr_gamma=0.1,
    parent_model=None,
    input_dim=hp_train_data.n_channels, 
    hidden_dim_1=hd1,
    hidden_dim_2=hd2,
    latent_dim=ld,
    mode=args.mode
)

print("Training Loop")
if args.mode=='debug': torch.autograd.set_detect_anomaly(True)
train_vae(hp_model)

