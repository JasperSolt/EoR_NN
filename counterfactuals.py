import os
import h5py
from omegaconf import OmegaConf
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch
from torch import nn, optim
from torch import Tensor
from torch.masked import masked_tensor, as_masked_tensor

import matplotlib.pyplot as plt
from matplotlib import colors
from regressors.latent_cnn import lcnn
from diffusers import AutoencoderKL

from util.plotting import plot_cf_loss, plot_counterfactuals, plot_image_grid







class CounterfactualLoss(nn.Module):
    def __init__(self):
        super(CounterfactualLoss, self).__init__()
        self.epsilon = 0.0001

    '''
    x' = g(z')
    y' = h(z')
    y = h(f(x))
    L = d_x{ g(z'), x } - d_y{ h(z'), h(f(x)) }
    '''
    def forward(self, y, y_prime, x, x_prime):
        return self.d_image(x_prime, x) - self.d_label(y_prime, y)

    '''
    Distance function for the image space. 
    We want to minimize this, i.e., find x' such that x'=g(z') is maximally similar to x.
    '''
    def d_image(self, x_prime, x):
        return ((x_prime - x)**2).mean()
    
    '''
    Distance function for the label space. 
    We want to maximize this, i.e., find x' such that y'=h(z') is maximally different from y=h(z)=h(f(x)).
    '''
    def d_label(self, y_prime, y):
        return (y_prime - y)**2


def get_model(model_cfg):
    # initialize model
    model = lcnn(model_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load model state
    path = f"{model_cfg.model.model_dir}/{model_cfg.model.name}"
    print(f"Loading model state from {path}...")
    checkpoint = torch.load(f"{path}.pth", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    for param in model.parameters():
        param.requires_grad = False
    return model


def get_samples(i_list, data_path, param_index, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dict, label_dict = {}, {}
    with h5py.File(data_path, "r") as f: 
        for i in i_list:
            # Load input tuple
            x = torch.tensor(f['lightcones/brightness_temp'][i], dtype=torch.float)
            x = x.view((1,) + x.shape)
            input_dict[i] = (x, model(x.to(device)))
            
            # Load true label
            label_dict[i] = f['lightcone_params/physparams'][i, param_index]

    return input_dict, label_dict


# ionized = 0, neutral = 1
def estimate_HI_regions(x):
    mask = torch.zeros_like(x)
    for k in range(x.shape[1]):
        zslice = x[0,k,:,:]
        mask[0,k,:,:] = zslice > torch.min(zslice)
    #mask = torch.ones_like(x)
    return mask

def estimate_global_xH(x):
    return torch.sum(~estimate_HI_regions(x), dim=(-1,-2))


def generate_counterfactuals(x, y, lc_index, model: nn.Module, save_dir: str, lr=0.1, epochs=30, verbose=False):
    torch.autograd.set_detect_anomaly(True)

    torch.cuda.memory._record_memory_history()
    
    mask = estimate_HI_regions(x)

    init = torch.randn_like(x) * mask
    x_prime = x + init

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask = mask.to(device)
    x, x_prime = x.to(device), x_prime.to(device)

    x_prime.requires_grad = True
    x.requires_grad = False
    y.requires_grad = False
    
    optimizer = optim.Adam([x_prime], lr=lr) #exclude min values from optimizer?

    lossfn = CounterfactualLoss()
    total_loss, x_loss, y_loss = torch.tensor([]), torch.tensor([]), torch.tensor([])
    for t in range(epochs):
        if t%100==0 or verbose: print(f"\nEpoch {t+1}\n-------------------------------")
        # Backpropagate
        y_prime = model(x_prime)
        
        total_loss_e = lossfn(y, y_prime, x, x_prime)
        x_loss_e = lossfn.d_image(x_prime, x).detach()
        y_loss_e = -lossfn.d_label(y_prime, y).detach()
        if verbose:
            print(f"Total Loss: {total_loss_e.item():.6f}")
            print(f"Input Loss: {x_loss_e.item():.6f}")
            print(f"Label Loss: {y_loss_e.item():.6f}")

        total_loss_e.backward()

        x_prime.grad *= mask #should zero out the grads of out ionized regions, so they don't change

        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

        # Save the loss history
        total_loss = torch.cat((total_loss, total_loss_e.cpu().view((1,))))
        x_loss = torch.cat((x_loss, x_loss_e.cpu().view((1,))))
        y_loss = torch.cat((y_loss, y_loss_e.cpu().view((1,))))
        #torch.cuda.memory._dump_snapshot("snapshot.pickle")

    cf_path = f"{save_dir}/counterfactual_id_{lc_index}"

    y_prime = model(x_prime)
    print("\n-------------------------------")
    print(f"LIGHTCONE INDEX {lc_index}:")
    print(f"    Old label: {y[0]}:")
    print(f"    New label: {y_prime[0]}:")

    state_dict = {
        'index': lc_index,
        'mask': mask,
        'init': init,
        'x': x,
        'y': y,
        'x_prime': x_prime,
        'y_prime': y_prime,
        'total_loss': total_loss,
        'x_loss': x_loss,
        'y_loss': y_loss,
        }
    
    torch.save(state_dict, f"{cf_path}.pth")
    
    loss_dict = {
        "total_loss": total_loss.detach().cpu().numpy(),
        "x_loss": x_loss.detach().cpu().numpy(),
        "y_loss": y_loss.detach().cpu().numpy(),
        }
    
    return x_prime, y_prime, loss_dict







if __name__=="__main__":
    # args
    #i_list = [0, 4, 8, 12, 16]
    i = 4
    i_list = [i]
    n = 10

    names = [
            "cnn_unenc_v02_p21c_ws0.0_2025-01-20T19-46",
            "cnn_unenc_v02_ctrpx_ws0.0_2025-01-20T19-47",
            "cnn_unenc_v02_zreion_ws0.0_2025-01-20T19-47"
            ]
    cf_version = "03"

    for name in names:
        # initialize certain parameters
        def check_substrings(string, substrings):
            return [substring for substring in substrings if substring in string]

        sim = check_substrings(name, ["p21c", "ctrpx", "zreion"])[0]

        start = name.find(sim)
        end = start + len(sim) + 6 #len("_wsX.X") == 6
        id = name[start:end]

        ws = float(id[-3:])

        # Load model config
        model_cfg_path = f"trained_models/{id}/{name}/{name}_config.yaml" 
        model_cfg = OmegaConf.load(model_cfg_path)

        # Load model
        model = get_model(model_cfg)

        # Load sample(s) to generate counterfactual for
        input_dict, label_dict = get_samples(i_list, model_cfg.data.data_paths[sim], model_cfg.data.param_index, model)

        path = f"{model_cfg.model.model_dir}/{model_cfg.model.name}_counterfactuals_{cf_version}"
        fig_path = f"figures/counterfactual_figures/counterfactuals_{cf_version}/{model_cfg.model.name}_counterfactuals_{cf_version}"
        
        #for i in i_list:
        for j in range(n):
            save_dir = f"{path}/lightcone_{i}"
            if not os.path.isdir(save_dir): os.makedirs(save_dir)
            
            fig_save_dir = f"{fig_path}/lightcone_{i}"
            if not os.path.isdir(fig_save_dir): os.makedirs(fig_save_dir)

            ###
            # Generate counterfactual
            ###
            x, y = input_dict[i]
            x_prime, y_prime, loss_dict = generate_counterfactuals(x, y, i, model=model, save_dir=save_dir, epochs=1000, lr=0.005)

            ###
            # Plot
            ###
            x, x_prime = x.detach().cpu().numpy()[0], x_prime.detach().cpu().numpy()[0]
            y, y_prime = y.item(), y_prime.item()

            diff = x-x_prime

            fnames = [f"{save_dir}/counterfactual_id_{i}", f"{fig_save_dir}/counterfactual_id_{i}"]
            title = f"LC {i} ({model_cfg.data.param_name}={label_dict[i]:.3f})"

            plot_image_grid(x, title=title + f": Input (y={y:.3f})", fnames=[f + "_input.jpeg" for f in fnames])
    
            plot_image_grid(x_prime, title=title + f" Iter {j}: Counterfactual (y'={y_prime:.3f})", fnames=[f + f"_iter{j}_cf.jpeg" for f in fnames])

            plot_image_grid(diff, title=title + f" Iter {j}: Difference (y={y:.3f}, y'={y_prime:.3f})", fnames=[f + f"_iter{j}_diff.jpeg" for f in fnames])

            plot_cf_loss(loss_dict, fnames=[f + f"_iter{j}_loss.jpeg" for f in fnames], title=f"Iter {j}: Counterfactual Loss")


