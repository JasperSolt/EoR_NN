import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
#from accelerate import Accelerator
from hyperparams import Model_Hyperparameters as hp
from EoR_Dataset import EORImageDataset
from plotting import plot_loss

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




def train_Fourier_NN(hp, init_weights=None):
    #make sure we aren't overwriting
    if os.path.isdir(hp.MODEL_DIR):
        print(hp.MODEL_DIR + " already exists. Please rename current model or delete old model directory.")
    else:


        lossplt_fname = hp.MODEL_DIR + "/" + hp.MODEL_NAME + "_loss.png"

        # training & testing datasets
        print("Loading training data...")
        train_data = EORImageDataset("train", hp.TRAINING_DATA_HP)
        print("Loading validation data...")
        val_data = EORImageDataset("val", hp.TRAINING_DATA_HP) 

        # training & testing dataloaders
        train_dataloader = DataLoader(train_data, batch_size=hp.BATCHSIZE, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=hp.BATCHSIZE, shuffle=True)

        # initialize model, optimizer
        model = Fourier_NN(hp).to("cuda")
        optim = hp.optimizer(model)
        
        #initialize scheduler
        scheduler = hp.scheduler(optim)
        
        if init_weights:
            print(f"Loading model state from {init_weights}")
            model.load_state_dict(torch.load(init_weights))
        
        #train / test loop
        loss = { "train" : [], "val" : [] }
        for t in range(hp.EPOCHS):
            print(f"Epoch {t+1}\n-------------------------------")
            loss["train"].append(train(train_dataloader, model, optim))
            loss["val"].append(test(val_dataloader, model))
            if hp.LR_DECAY:
                scheduler.step()
                print("Learning Rate: {}".format(optim.param_groups[0]['lr']))

            if t in hp.SAVE_EPOCHS:
                save(model, f"{hp.MODEL_NAME}_{t}")
                save(model, hp.MODEL_NAME)
                save_loss(loss, hp)
                plot_loss(loss, lossplt_fname, f"{hp.MODEL_NAME} Loss")

        save(model, hp.MODEL_NAME)
        save_loss(loss, hp)
        plot_loss(loss, lossplt_fname, f"{hp.MODEL_NAME} Loss")

        hp.save_hyparam_summary()

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