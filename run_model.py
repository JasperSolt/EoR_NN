import os
import numpy as np
import argparse
import torch
from diffusion_unet import train, inference
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
parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False, help='Run vs debug mode')
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

if args.debug: hp_train_data.lenlimit=1

#Initializing model hyperparameters
lr = 2e-5
    
modelnum = args.modelnum
model_name = f"single_channel_diffunet_v{modelnum:0>4}"

for ts in train_sims:
    model_name += f"_{ts}"
model_name += f"_ws{ws}"
if args.debug: model_name="debug"
model_dir = "trained_models/" + model_name


hp_model = ModelHyperparameters(
    model_name=model_name, 
    device="cuda" if torch.cuda.is_available() else "cpu",
    data_hp=hp_train_data, 
    batchsize=4 if args.debug else 64,
    time_steps=1000 if args.debug else 1000,
    epochs=1 if args.debug else 15, 
    initial_lr=lr,
    ema_decay=0.9999,
    checkpoint_path=None,
    input_channels=hp_train_data.n_channels, 
    #layer_channels = [64, 128, 256, 512, 512, 384],
    layer_channels = [16, 32, 64, 128, 128, 96],
    layer_attentions = [False, False, False, False, False, False],
    layer_upscales = [False, False, False, True, True, True],
    num_groups = 8,
    dropout_prob = 0.1,
    num_heads = 1,
    debug_mode = args.debug,
)
'''
print("Training Loop")
if args.debug: torch.autograd.set_detect_anomaly(True)
train(hp_model)
'''
hp_model.checkpoint_path = f'/users/jsolt/FourierNN/trained_models/{model_name}/{model_name}'
inference(hp_model)

