import argparse
import os
import h5py
from omegaconf import OmegaConf
import numpy as np
import torch
from torch import nn, optim
from torchviz import make_dot
from regressors.cnn import cnn
from util.plotting import plot_cf_loss, plot_image_grid, plot_imgrid_with_overlay


# Globals
names = {
        "p21c":"cnn_v03_p21c_ws0.0_2025-02-28T12-28",
        "ctrpx":"cnn_v03_ctrpx_ws0.0_2025-02-28T12-28",
        "zreion":"cnn_v03_zreion_ws0.0_2025-02-28T12-32"
}

data_paths = {
    "ctrpx":"/users/jsolt/data/jsolt/centralpix_sims/centralpix05/centralpix05_norm_subdiv_sliced_ws0.0.hdf5",
    "zreion":"/users/jsolt/data/jsolt/zreion_sims/zreion24/zreion24_norm_subdiv_sliced_ws0.0.hdf5",
    "p21c":"/users/jsolt/data/jsolt/21cmFAST_sims/p21c14/p21c14_norm_subdiv_sliced_ws0.0.hdf5",
}

ids = {
        "p21c":"p21c_ws0.0",
        "ctrpx":"ctrpx_ws0.0",
        "zreion":"zreion_ws0.0"
}

class CounterfactualLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(CounterfactualLoss, self).__init__()
        self.alpha = alpha

        self.loss_dict = {}
        for k in ['total_loss', 'x_loss', 'y_loss']:
            self.loss_dict[k] = torch.tensor([])

    '''
    x' = g(z')
    y' = h(z')
    y = h(f(x))
    L = d_x{ g(z'), x } - d_y{ h(z'), h(f(x)) }
    '''
    def forward(self, y, y_prime, x, x_prime, log=True):
        epoch_loss = {}
        
        epoch_loss.update(self.calc_d_image(x_prime, x))
        epoch_loss.update(self.calc_d_label(y_prime, y))
        epoch_loss['total_loss'] = epoch_loss['x_loss'] + epoch_loss['y_loss']
        
        if log: self.save_loss_history(epoch_loss)
        return epoch_loss

    '''
    Distance function for the image space. 
    We want to minimize this, i.e., find x' such that x'=g(z') is maximally similar to x.
    '''
    def calc_d_image(self, x_prime, x):
        return {"x_loss": torch.square(x_prime - x).mean()}
    
    '''
    Distance function for the label space. 
    We want to maximize this, i.e., find x' such that y'=h(z') is maximally different from y=h(z)=h(f(x)).
    '''
    def calc_d_label(self, y_prime, y):
        return {"y_loss": -self.alpha*torch.square(y_prime - y)}
    
    def save_loss_history(self, epoch_loss_dict:dict):
        for k in self.loss_dict.keys():
            self.loss_dict[k] = torch.cat((self.loss_dict[k], epoch_loss_dict[k].cpu().view((1,))))


class UnderOverCounterfactualLoss(CounterfactualLoss):
    def __init__(self, alpha=1.0, under=True):
        super(UnderOverCounterfactualLoss, self).__init__(alpha)
        if under:
            self.alpha *= -1.0

    def calc_d_label(self, y_prime, y):
        return {'y_loss': -self.alpha*(y_prime-y)}



class XCorrCounterfactualLoss(CounterfactualLoss):
    def __init__(self, alpha=1.0, beta=1.0):
        super(XCorrCounterfactualLoss, self).__init__(alpha)
        self.beta = beta
        for k in ['xcorr_loss', 'mse_loss']:
            self.loss_dict[k] = torch.tensor([])


    def calc_d_image(self, x_prime, x):
        x_loss_dict = {}
        x_loss_dict['mse_loss'] = torch.square(x_prime - x).mean()
        x_loss_dict['xcorr_loss']  = self.corrcoef_loss(x_prime, x)
        x_loss_dict['x_loss'] = x_loss_dict['mse_loss'] + x_loss_dict['xcorr_loss']
        return x_loss_dict
    
    def corrcoef_loss(self, input, target):   
        # For whatever reason, you have to do it in both directions to avoid this weird banding pattern
        #There are probably way more efficient ways to accomplish this. But i'm not confident in my linear algebra, lol
        # Covariance
        X1 = torch.cat((input, target), dim=-2)
        X1 -= torch.mean(X1, -1, keepdim=True)

        X_T1 = torch.transpose(X1, -2, -1)
        c1 = torch.matmul(X1, X_T1) / (X1.shape[-1] - 1)

        # Covariance
        X2 = torch.cat((input, target), dim=-1)
        X2 -= torch.mean(X2, -2, keepdim=True)

        X_T2 = torch.transpose(X2, -2, -1)
        c2 = torch.matmul(X_T2, X2) / (X2.shape[-2] - 1)

        # combine
        c = c1 + c2 / 2.0

        # Correlation Coefficient
        d = torch.diagonal(c, dim1=-1, dim2=-2)
        stddev = torch.sqrt(d)
        stddev = torch.where(stddev == 0, 1, stddev)
        c /= stddev[:,:,:,None]
        c /= stddev[:,:,None,:]

        # 1 - Cross-Correlation Diagonal
        ccd = 1-torch.diagonal(c, offset=c.shape[-1]//2, dim1=-1, dim2=-2)
        
        return self.beta*ccd.mean()




def get_model(model_cfg):
    # initialize model
    model = cnn(model_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load model state
    path = f"{model_cfg.model.model_dir}/{model_cfg.model.name}"
    print(f"Loading model state from {path}...")
    checkpoint = torch.load(f"{path}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model





def get_sample_list(n, data_path, param_index, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with h5py.File(data_path, "r") as f: 
        dataset_len = len(f['lightcones/brightness_temp'])
        if n > dataset_len: n = dataset_len

        x_list = torch.tensor(f['lightcones/brightness_temp'][:n], dtype=torch.float)
        label_list = torch.Tensor(f['lightcone_params/physparams'][:n, param_index])

        y_list = torch.zeros((n, 1)).to(device)
        for i in range(n):
            y_list[i] = model(x_list[i][None,:].to(device))

    return x_list, y_list, label_list


def get_sample(i, data_path, param_index, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with h5py.File(data_path, "r") as f: 
        j = i % len(f['lightcones/brightness_temp'])
        x = torch.tensor(f['lightcones/brightness_temp'][j], dtype=torch.float)[None,:]
        y = model(x.to(device))
        label = f['lightcone_params/physparams'][j, param_index]
    return x, y, label


def get_input_for_index(i, x_list, y_list, label_list):
    j = i % len(x_list)
    return x_list[j][None,:], y_list[j][None,:], label_list[j]



def zslice_mins(x):
    mins = torch.min(torch.min(x, dim=-1)[0], dim=-1)[0][0]
    return mins.view(mins.shape + (1,1))

def zslice_maxs(x):
    maxs = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0][0]
    return maxs.view(maxs.shape + (1,1))

# ionized = 0, neutral = 1
def estimate_zero_xH_zones(x, mins=None):
    if mins == None: mins = zslice_mins(x)
    mask = torch.where(x > mins, 1.0, 0.0) 
    return mask

def find_bubble_edges(x):
    pass



def generate_counterfactual(x:torch.Tensor, 
                             y:torch.Tensor, 
                             lc_index:int, 
                             y_true:str, 
                             model: nn.Module, 
                             lossfn: CounterfactualLoss,
                             save_dir: str = None, 
                             lr=0.1, 
                             epochs=30, 
                             init_weight=1.0,
                             mask=None,
                             clamp=False,
                             verbose=False,
                             ):    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init the init
    init = init_weight*torch.randn_like(x)
    if mask != None: init*= mask
    x_prime = init + x

    # move to GPU
    x, x_prime = x.to(device), x_prime.to(device)
    if mask != None: mask = mask.to(device)

    # Tell pytorch which tensors to optimize
    x_prime.requires_grad = True
    x.requires_grad = False
    y.requires_grad = False

    optimizer = optim.Adam([x_prime], lr=lr) 

    for t in range(epochs):
        if (t%100==0 and verbose): print(f"\nEpoch {t}\n-------------------------------")
        
        optimizer.zero_grad()
        model.zero_grad() #Theoretically, this should not make a difference.
        
        # Forward pass
        if clamp:
            x_prime_prep = torch.clamp(x_prime, x.min(), x.max())
        else:
            x_prime_prep = x_prime*1.0 #dumb thing I'm doing to convince myself the graph is correct

        y_prime = model(x_prime_prep)

        # Backwards pass
        epoch_loss = lossfn(y, y_prime, x, x_prime_prep)
        
        epoch_loss['total_loss'].backward()

        if (t%100==0 and verbose):
            print(f"Total Loss: {epoch_loss['total_loss'].item()}")
            print(f"Input Loss: {epoch_loss['x_loss'].item()}")
            print(f"Label Loss: {epoch_loss['y_loss'].item()}")

        if mask != None: x_prime.grad *= mask #should zero out the grads of out ionized regions, so they don't change

        optimizer.step()

    if clamp: x_prime = torch.clamp(x_prime, x.min(), x.max())
    y_prime = model(x_prime)
    print(f"Final y'={y_prime.item():.3f}")

    state_dict = {
        'index': lc_index,
        'init': init,
        'x': x,
        'x_prime': x_prime,
        'y': y.item(),
        'y_prime': y_prime.item(),
        'y_true': y_true,
        }
    
    if mask != None: state_dict['mask'] = mask
    state_dict.update(lossfn.loss_dict)

    cf_path = f"{save_dir}/counterfactual_id_{lc_index}"
    if save_dir: torch.save(state_dict, f"{cf_path}.pth")
    
    return state_dict





def init_cf_database(fname, n, epochs, lc_shape=(30,256,256)):
    with h5py.File(fname, 'w') as f:
        f.create_group('lightcone_params')
        f['lightcone_params'].create_dataset('y_true', shape=(n,), dtype=np.float32)
        f['lightcone_params'].create_dataset('y', shape=(n,), dtype=np.float32)
        f['lightcone_params'].create_dataset('y_prime', shape=(n,), dtype=np.float32)

        f.create_group('lightcones')
        f['lightcones'].create_dataset('x', shape=(n,)+lc_shape, dtype=np.float32)
        f['lightcones'].create_dataset('x_prime', shape=(n,)+lc_shape, dtype=np.float32)

        f.create_group('regression_info')
        f['regression_info'].create_dataset('total_loss', shape=(n,epochs), dtype=np.float32)
        f['regression_info'].create_dataset('x_loss', shape=(n,epochs), dtype=np.float32)
        f['regression_info'].create_dataset('y_loss', shape=(n,epochs), dtype=np.float32)


detach_params = [
        'x',
        'x_prime',
        'total_loss',
        'x_loss',
        'y_loss',
        'xcorr_loss',
        'mse_loss',
        ]

def save_cf_to_database(fname, i, state_dict):
    with h5py.File(fname, 'a') as f:
        f['lightcone_params/y_true'][i] = state_dict['y_true']
        f['lightcone_params/y'][i] = state_dict['y']
        f['lightcone_params/y_prime'][i] = state_dict['y_prime']

        f['lightcones/x'][i] = state_dict['x']
        f['lightcones/x_prime'][i] = state_dict['x_prime']

        f['regression_info/total_loss'][i] = state_dict['total_loss']
        f['regression_info/x_loss'][i] = state_dict['x_loss']
        f['regression_info/y_loss'][i] = state_dict['y_loss']




def generate_cf_dataset(sim):
    id = ids[sim]
    name = names[sim]

    # PARAMS
    cf_version = f"13"
    n = 3
    epochs = 5000
    alpha = 1e-4 #5e-6
    beta = 1.0
    lr = 1e-4
    init_weight = 1e-4 #1e-1
    clamp = True
    use_mask = False
    
    load_all = True
    plot = True

    # Load model config
    model_cfg_path = f"trained_models/{id}/{name}/{name}_config.yaml" 
    model_cfg = OmegaConf.load(model_cfg_path)

    # Load model
    model = get_model(model_cfg)

    # Load sample(s) to generate counterfactual for
    print("Loading data...")
    if load_all:
        x_list, y_list, label_list = get_sample_list(n, data_paths[sim], model_cfg.data.param_index, model)

    save_dir = f"/users/jsolt/data/jsolt/counterfactuals"
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    fname = f"{save_dir}/{name}_counterfactuals_{cf_version}.hdf5"
    
    init_cf_database(fname, n, epochs)
    
    for i in range(n):
        print("\n-------------------------------")
        print(f"Generating Counterfactual {i} of {n}...")

        if load_all: 
            x, y, y_true = get_input_for_index(i, x_list, y_list, label_list)
        else:
            x, y, y_true = get_sample(i, data_paths[sim], model_cfg.data.param_index, model)

        lossfn = UnderOverCounterfactualLoss(alpha=alpha)

        ###
        # Generate counterfactual
        ###
        mask=None
        if use_mask: mask = estimate_zero_xH_zones(x)
        state_dict = generate_counterfactual(x, 
                                             y, 
                                             i, 
                                             y_true=y_true, 
                                             model=model, 
                                             lossfn=lossfn,
                                             epochs=epochs, 
                                             init_weight=init_weight, 
                                             lr=lr,
                                             mask=mask,
                                             clamp=clamp,
                                             )

        ###
        # Save
        ###

        # Except-pass is bad coding practice. But i don't care.
        for param in state_dict.keys():
            try:
                state_dict[param] = state_dict[param].detach().cpu().numpy()
            except Exception:
                pass
                
        save_cf_to_database(fname, i, state_dict)

        ###
        # Plot
        ###
        if plot:
            fig_path = f"figures/counterfactual_figures/counterfactuals_{cf_version}/{model_cfg.model.name}_counterfactuals_{cf_version}"
            fig_save_dir = f"{fig_path}/lightcone_{i}"
            if not os.path.isdir(fig_save_dir): os.makedirs(fig_save_dir)

            x = state_dict['x'][0]
            x_prime = state_dict['x_prime'][0]
            y = state_dict['y']
            y_prime = state_dict['y_prime']

            diff = x-x_prime

            fnames = [f"{fig_save_dir}/counterfactual_id_{i}"]
            title = f"LC {i} ({model_cfg.data.param_name}={label_list[i]:.3f})"

            plot_image_grid(x, title=title + f": Input (y={y:.3f})", fnames=[f + "_input.jpeg" for f in fnames])

            plot_image_grid(x_prime, title=title + f": Counterfactual (y'={y_prime:.3f})", fnames=[f + f"_cf.jpeg" for f in fnames])
            
            plot_image_grid(diff, title=title + f": Difference (y={y:.3f}, y'={y_prime:.3f})", fnames=[f + f"_diff.jpeg" for f in fnames])

            plot_imgrid_with_overlay(x, diff, title=title + f": Difference (y={y:.3f}, y'={y_prime:.3f})", fnames=[f + f"_diff_ovrly.jpeg" for f in fnames])

            loss_dict = {
                'total_loss': state_dict['total_loss'],
                'x_loss': state_dict['x_loss'],
                'y_loss': state_dict['y_loss'],
            }

            plot_cf_loss(loss_dict, fnames=[f + f"_loss.jpeg" for f in fnames], title=f"Counterfactual Loss")

            


if __name__=="__main__":
    # args
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-s','--sim', required=True, type=str)

    #args = parser.parse_args()
    for sim in ['p21c']:#, 'ctrpx', 'zreion']:
        generate_cf_dataset(sim)