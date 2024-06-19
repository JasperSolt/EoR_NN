import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from EoR_Dataset import EORImageDataset
from hyperparams import Dataset_Hyperparameters
from adversarial_model import encoder, discriminator, regressor


def get_path(model_sim, ws):
    if model_sim == "p21c":
        dp = f"/users/jsolt/data/jsolt/21cmFAST_sims/p21c14/p21c14_ws{ws}_trnspsd_meanremoved_norm.hdf5"
    elif model_sim == "zreion":
        dp = f"/users/jsolt/data/jsolt/zreion_sims/zreion23/zreion23_transposed_ws{ws}.hdf5"
    elif model_sim == "ctrpx":
        dp = f"/users/jsolt/data/jsolt/21cmFAST_centralpix_v04/21cmFAST_centralpix_v04_transposed_ws{ws}.hdf5"
    return dp

###
# LOAD DATA
###
ws=0.0
param = 1
param_index = 0 if param=="mdpt" else 1
scale = 256
zindices = [x*17 for x in range(0, 30)]

for sim in ["zreion", "p21c", "ctrpx"]:
    test_sims = [sim,]
    data_paths = [get_path(ts, ws) for ts in test_sims]
    
    hp_test = Dataset_Hyperparameters(
                                        test_sims, 
                                        data_paths, 
                                        zindices, 
                                        batchsize=8, 
                                        subsample_scale=scale, 
                                        param=param_index,)
    
    data = EORImageDataset("test", hp_test, verbose=False)
    dataloader = DataLoader(data, batch_size=hp_test.BATCHSIZE, shuffle=True)
    
    
    ###
    # NULL MODEL
    ###
    names = {
        "null" : "adversarial_null_v01_p21c_ctrpx_ws0.0_s03",
        "adversarial" : "adversarial_v02_p21c_ctrpx_alpha0.05_lr0.003_ws0.0_s03"
            }
    
    paths = {k : f"models/{name}/{name}" for k, name in names.items()}
    
    mse = {}
    for k, path in paths.items():
        enc, reg = encoder(), regressor()
        
        if torch.cuda.is_available(): 
            enc.cuda()
            reg.cuda()
            enc.load_state_dict(torch.load(f"{path}_enc.pth"))
            reg.load_state_dict(torch.load(f"{path}_reg.pth"))
        
        else:
            enc.load_state_dict(torch.load(f"{path}_enc.pth", map_location=torch.device('cpu')))
            reg.load_state_dict(torch.load(f"{path}_reg.pth", map_location=torch.device('cpu')))
        
        enc.eval()
        reg.eval()
        
        shape = (dataloader.dataset.__len__(), 1)
        predictions, targets = np.zeros(shape), np.zeros(shape)
        i = 0
        
        #predict
        with torch.no_grad():
            for X, y in dataloader:
                if torch.cuda.is_available(): X, y = X.cuda(), y.cuda()
                label = y[:,0]
                
                pred = reg(enc(X))
                
                label = torch.reshape(label, pred.shape)
        
                predictions[i : i + hp_test.BATCHSIZE] = pred.cpu()
                targets[i : i + hp_test.BATCHSIZE] = label.cpu()
                i += hp_test.BATCHSIZE
        
        mse[k] = np.mean((predictions - targets)**2)
    
    print(sim)
    for k, v in mse.items():
        print(f"{k}: {v:.5f}")
    print()