import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from EoR_Dataset import EORImageDataset
from model import Fourier_NN, predict, load

def predict_Fourier_NN(hp_model, hp_test, mode="test"):
    data = EORImageDataset(mode, hp_test)
    loader = DataLoader(data, batch_size=hp_test.BATCHSIZE, shuffle=True)

    net = Fourier_NN(hp_model).cuda()
    load(net)

    predictions, labels = predict(loader, net)
   
    #save prediction
    f = f'{hp_model.MODEL_DIR}/pred_{hp_model.MODEL_NAME}_on_{hp_test.SIMS[0]}_{mode}.npz'
    np.savez(f, targets=labels, predictions=predictions)
    
    print("Prediction saved.")
    return f

