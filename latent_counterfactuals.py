import os
import h5py
from omegaconf import OmegaConf
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch
from torch import nn, optim
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib import colors
from regressors.latent_cnn import lcnn
from diffusers import AutoencoderKL

from util.plotting import plot_cf_loss, plot_counterfactuals






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





class wrapper(nn.Module):
    def __init__(self, vae, cnn):
        super(wrapper, self).__init__()

        self.vae = vae
        self.cnn = cnn
        self.redshifts = cnn.cfg.data.zlength


    def forward(self, z):
        x = decode_sample(z)
        return self.cnn(x)

    


def encode_slice(self, vae, xslice: Tensor):
    xslice = xslice.repeat(3,1)[None, :, :, :] # (1, 3, 256, 256)
    zslice = vae.encode(xslice).latent_dist.mean
    return zslice[0]

def encode_sample(self, x):
    z = torch.Tensor([])
    for i in range(self.redshifts):
        xslice = x[0,i,:,:] # (256, 256)
        print(xslice.shape) 
        zslice = self.encode_slice(self.vae, xslice) # (1, 4, 32, 32)
        print(zslice.shape)
        z = torch.concat((z, zslice), dim=0)
    return z[None, :, :, :, :] # (1, 30, 4, 32, 32)


def decode_slice(vae, zslice):
    return torch.mean(vae.decode(zslice, return_dict=False)[0], dim=1, keepdim=True) # THEORETICALLY, (1, 1, 256, 256)

def decode_sample(z):
    x = torch.Tensor([]).to(device)
    for i in range(redshifts):
        zslice = z[:,i,:,:,:] # THEORETICALLY, (1, 30, 4, 32, 32) -> (1, 4, 32, 32)
        xslice = decode_slice(vae, zslice)
        x = torch.concat((x, xslice), dim=1)
        torch.cuda.memory._dump_snapshot("snapshot.pickle")
    return x




def generate_latent_counterfactuals(z_dict: dict[int, tuple], model: nn.Module, vae: nn.Module, save_dir: str, lr=0.1, epochs=30, verbose=False) -> dict[int, tuple]:
    torch.autograd.set_detect_anomaly(True)


    z_prime_dict = {}
    for index, (z, y) in z_dict.items():
        z_prime = z 

        device = "cuda" if torch.cuda.is_available() else "cpu"
        z, z_prime = z.to(device), z_prime.to(device)
        
        z_prime.requires_grad = True
        z.requires_grad = False
        y.requires_grad = False
        
        optimizer = optim.Adam([z_prime], lr=lr) 

        lossfn = CounterfactualLoss()
        total_loss, z_loss, y_loss = torch.tensor([]), torch.tensor([]), torch.tensor([])

        # Create our "buffer x-prime" so that CUDA doesn't run out of mem
        x_prime = torch.Tensor([]).to(device)
        for i in range(redshifts):
            zslice = z[:,i,:,:,:] # THEORETICALLY, (1, 30, 4, 32, 32) -> (1, 4, 32, 32)
            xslice = decode_slice(vae, zslice).detach()
            x_prime = torch.concat((x_prime, xslice), dim=1)
        x_prime.requires_grad = False

        optimizer.zero_grad()
        model.zero_grad()
        vae.zero_grad()

        torch.cuda.memory._record_memory_history()
        for t in range(epochs):
            if t%100==0 or verbose: print(f"\nEpoch {t+1}\n-------------------------------")

            # compute grad for one slice at a time
            redshift = t % 30
            #x_prime[0,redshift,:,:] = torch.mean(vae.decode(z_prime[:,redshift,:,:,:], return_dict=False)[0], dim=1, keepdim=True) 
            decoder_outpt = vae.decode(z_prime[:,redshift,:,:,:])
            print(type(decoder_outpt))
            x_prime[0,redshift,:,:] = torch.mean(decoder_outpt[0], dim=1, keepdim=True) 
            
            # Backpropagate 
            y_prime = model(x_prime)
            
            total_loss_e = lossfn(y, y_prime, z, z_prime)
            z_loss_e = lossfn.d_image(z_prime, z).detach()
            y_loss_e = -lossfn.d_label(y_prime, y).detach()

            if verbose:
                print(f"Total Loss: {total_loss_e.item():.6f}")
                print(f"Latent Loss: {z_loss_e.item():.6f}")
                print(f"Label Loss: {y_loss_e.item():.6f}")

            total_loss_e.backward(inputs=(z_prime[:,redshift,:,:,:],))
            optimizer.step()

            optimizer.zero_grad()
            model.zero_grad()
            vae.zero_grad()

            # Save the loss history
            total_loss = torch.cat((total_loss, total_loss_e.cpu().view((1,))))
            z_loss = torch.cat((z_loss, z_loss_e.cpu().view((1,))))
            y_loss = torch.cat((y_loss, y_loss_e.cpu().view((1,))))
            torch.cuda.memory._dump_snapshot("snapshot.pickle")

        cf_path = f"{save_dir}/counterfactual_id_{index}"

        y_prime = model(x_prime)
        z_prime_dict[index] = (z_prime, y_prime)
        print("\n-------------------------------")
        print(f"LIGHTCONE INDEX {index}:")
        print(f"    Old label: {y[0]}:")
        print(f"    New label: {y_prime[0]}:")

        state_dict = {
            'index': index,
            'z': z,
            'y': y,
            'z_prime': z_prime,
            'y_prime': y_prime,
            'total_loss': total_loss,
            'z_loss': z_loss,
            'y_loss': y_loss,
            }
        
        torch.save(state_dict, f"{cf_path}.pth")
        
        loss_dict = {
            "total_loss": total_loss.detach().cpu().numpy(),
            "x_loss": z_loss.detach().cpu().numpy(),
            "y_loss": y_loss.detach().cpu().numpy(),
            }
        
        plot_cf_loss(loss_dict,
                    fname=f"{cf_path}_loss.png", 
                    title=f"Counterfactual Loss", 
                    )
        
    return z_prime_dict




if __name__=="__main__":
    # args
    #i_list = [0,4,8,12,16]
    i_list = [0]
    redshifts = 30
    names = ["cnn_unenc_v02_p21c_ws0.0_2025-01-20T19-46",
             "cnn_unenc_v02_ctrpx_ws0.0_2025-01-20T19-47",
             "cnn_unenc_v02_zreion_ws0.0_2025-01-20T19-47"]
    
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
        cnn = get_model(model_cfg)

        # Load VAE
        device="cuda" if torch.cuda.is_available() else "cpu"
        vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        vae.to(device)

        for param in vae.parameters():
            param.requires_grad = False

        # Create EDCNN
        wrpr = wrapper(vae, cnn)

        # Load sample(s) to generate counterfactual for
        data_paths = {
            "p21c": f"/users/jsolt/data/jsolt/21cmFAST_sims/p21c14/p21c14_norm_encoded_ws{model_cfg.data.wedgeslope}.hdf5",
            "zreion": f"/users/jsolt/data/jsolt/zreion_sims/zreion24/zreion24_norm_encoded_ws{model_cfg.data.wedgeslope}.hdf5",
            "ctrpx": f"/users/jsolt/data/jsolt/centralpix_sims/centralpix05/centralpix05_norm_encoded_ws{model_cfg.data.wedgeslope}.hdf5"
        }
        input_dict, label_dict = get_samples(i_list, data_paths[sim], model_cfg.data.param_index, wrpr)
        print(list(label_dict.values()))

        ###
        # Generate counterfactual
        ###
        path = f"{model_cfg.model.model_dir}/{model_cfg.model.name}"
        save_dir = f"{path}/counterfactuals_00"
        if not os.path.isdir(save_dir): os.makedirs(save_dir)

        cf_dict = generate_latent_counterfactuals(input_dict, model=cnn, vae=vae, save_dir=save_dir, epochs=30, lr=0.005, verbose=True)
        torch.cuda.memory._record_memory_history(enabled=None)

        ###
        # Plot
        ###
        meta_rows = 3
        cols = 10
        for i in i_list:
            x, y = input_dict[i]
            x_prime, y_prime = cf_dict[i]
            label = label_dict[i]

            x, x_prime = x.detach().cpu().numpy(), x_prime.detach().cpu().numpy()
            y, y_prime = y.item(), y_prime.item()

            for r in range(meta_rows): #make a seperate figure for each meta row
                zrange = np.arange(r*cols, (r+1)*cols, dtype=int)
                
                rowlabels = []
                rowlabels.append(f"Input (y={y:.3f})")
                rowlabels.append(f"CF (y'={y_prime:.3f})")
                rowlabels.append("Difference")

                title = f"CF for LC {i} with {model_cfg.data.param_name} {label:.3f} ({r}/{meta_rows})"
                fname = f"{save_dir}/counterfactual_id_{i}_{r}.jpeg"

                plot_counterfactuals(input_row=x[0, zrange],
                                    cf_row=x_prime[0, zrange],
                                    diff_row=x[0, zrange]-x_prime[0, zrange],
                                    title=title, 
                                    fname=fname, 
                                    rowlabels=rowlabels,
                                    collabels=zrange)
        