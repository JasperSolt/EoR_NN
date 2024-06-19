import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from EoR_Dataset import EORImageDataset
from model import Fourier_NN, predict, load
from adversarial_model import encoder, discriminator, regressor

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




def predict_adversarial_NN(hp_model, hp_test, mode="test"):
    data = EORImageDataset(mode, hp_test)
    dataloader = DataLoader(data, batch_size=hp_test.BATCHSIZE, shuffle=True)

    enc, reg = encoder(), regressor()
    
    if torch.cuda.is_available(): 
        enc.cuda()
        reg.cuda()
        
    path = f"{hp_model.MODEL_DIR}/{hp_model.MODEL_NAME}"
    enc.load_state_dict(torch.load(f"{path}_enc.pth"))
    reg.load_state_dict(torch.load(f"{path}_reg.pth"))

    enc.eval()
    reg.eval()
    
    shape = (dataloader.dataset.__len__(), 1)
    predictions, targets = np.zeros(shape), np.zeros(shape)
    i = 0

    #predict
    print(f"Predicting on {shape[0]} samples...")
    with torch.no_grad():
        for X, y in dataloader:
            if torch.cuda.is_available(): X, y = X.cuda(), y.cuda()
            label, cls = y[:,0], y[:,1]
            
            pred = reg(enc(X))
            
            label = torch.reshape(label, pred.shape)

            predictions[i : i + hp_test.BATCHSIZE] = pred.cpu()
            targets[i : i + hp_test.BATCHSIZE] = label.cpu()
            i += hp_test.BATCHSIZE
   
    #save prediction
    f = f'{hp_model.MODEL_DIR}/pred_{hp_model.MODEL_NAME}_on_{hp_test.SIMS[0]}_{mode}.npz'
    np.savez(f, targets=targets, predictions=predictions)
    
    print("Prediction saved.")
    return f
