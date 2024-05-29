import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
#from accelerate import Accelerator
from hyperparams import Model_Hyperparameters as hp


#model class
class Fourier_NN(nn.Module):
    def __init__(self, hp):
        self.hp=hp
        super(Fourier_NN, self).__init__()
        
        self.layer_dict = hp.LAYER_DICT
        for name, layer in self.layer_dict.items():
            self.add_module(name, layer)

    # Forward propagation of some batch x. 
    def forward(self, x):
        layer_output = x
        for name, layer in self.layer_dict.items():
            layer_output = layer(layer_output)
        return layer_output


#Training function
def train(dataloader, model, optimizer):
    model.train()
    tot_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        #feed batch through model
        X, y = X.to("cuda"), y.to("cuda")
        pred = model(X)
        y = torch.reshape(y, pred.shape)
        # Compute prediction error
        loss = model.hp.loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        
    #return the average epoch training loss
    avg_loss = tot_loss / len(dataloader)
    print("Average train loss: {}".format(avg_loss))
    return avg_loss


#Validation function
def test(dataloader, model):
    model.eval()
    tot_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cuda"), y.to("cuda")
            pred = model(X)
            y = torch.reshape(y, pred.shape)
            tot_loss += model.hp.loss_fn(pred, y).item()
    avg_loss = tot_loss / len(dataloader)
    print("Average test loss: {}".format(avg_loss))
    return avg_loss
    
#Prediction + save prediction
def predict(dataloader, model):
    model.eval()
    shape = (dataloader.dataset.__len__(), 1)
    predictions, labels = np.zeros(shape), np.zeros(shape)
    i = 0

    #predict
    print(f"Predicting on {shape[0]} samples...")
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cuda"), y.to("cuda")
            batch_pred = model(X)
            batch_size = len(batch_pred)
            y = torch.reshape(y, (batch_size, 1))

            predictions[i : i + batch_size] = batch_pred.cpu()
            labels[i : i + batch_size] = y.cpu()
            i += batch_size

    return predictions, labels
    

#Save model
def save(model, model_save_name):
    model_save_dir=model.hp.MODEL_DIR
    
    # save trained model
    #if accelerator.is_main_process and not os.path.isdir(model_save_dir):
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    
    f = model_save_dir + "/" + model_save_name + ".pth"
    print(f"Saving PyTorch Model State to {f}...")
    
    #accelerator.wait_for_everyone()
    #unwrapped_model = accelerator.unwrap_model(model)
    #accelerator.save(unwrapped_model.state_dict(), f)
    torch.save(model.state_dict(), f)

    print("Model Saved.")


#Save loss
def save_loss(loss, hp):
    model_save_dir=hp.MODEL_DIR
    model_save_name=hp.MODEL_NAME
    
    fl = model_save_dir + "/loss_" + model_save_name + ".npz"
    print("Saving loss data to {}...".format(fl))
    np.savez(fl, train=loss["train"], val=loss["val"])
    print("Loss data saved.")

def load_loss(hp):
    model_save_dir=hp.MODEL_DIR
    model_save_name=hp.MODEL_NAME
    
    fl = model_save_dir + "/loss_" + model_save_name + ".npz"
    return np.load(fl)
    
    

#load model
def load(model):
    model_load_dir=model.hp.MODEL_DIR
    model_load_name=model.hp.MODEL_NAME
    
    f = model_load_dir + "/" + model_load_name + ".pth"
    if os.path.isfile(f):
        print("Loading model state from {}".format(f))
        model.load_state_dict(torch.load(f))
        print("Model loaded.")
    else:
        print("Cannot find model path!")