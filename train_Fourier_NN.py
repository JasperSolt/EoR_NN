import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from EoR_Dataset import EORImageDataset
from model import Fourier_NN, train, test, save, save_loss, load
from adversarial_model import encoder, discriminator, regressor

from plot_model_results import plot_loss


def train_Fourier_NN(hp, init_weights=None):
    #make sure we aren't overwriting
    if os.path.isdir(hp.MODEL_DIR):
        print(hp.MODEL_DIR + " already exists. Please rename current model or delete old model directory.")
    else:

        start_time = datetime.now()

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

            train_loss = train(train_dataloader, model, lossfn, optim)
            loss["train"].append(train_loss)
            print(f"Average train loss: {train_loss}")
            
            val_loss = test(val_dataloader, model, lossfn)
            loss["val"].append()
            print(f"Average validation loss: {val_loss}")

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
        hp.save_time(start_time)





def train_adversarial_NN(hp, init_dict=None):
    #make sure we aren't overwriting
    if os.path.isdir(hp.MODEL_DIR):
        print(hp.MODEL_DIR + " already exists. Please rename current model or delete old model directory.")
    else:
        ###
        # Initialize 
        ###
        
        # training dataset
        print("Loading training data...")
        train_data = EORImageDataset("train", hp.TRAINING_DATA_HP)
        train_dataloader = DataLoader(train_data, batch_size=hp.BATCHSIZE, shuffle=True)

        #validation dataset
        print("Loading validation data...")
        val_data = EORImageDataset("val", hp.TRAINING_DATA_HP) 
        val_dataloader = DataLoader(val_data, batch_size=hp.BATCHSIZE, shuffle=True)

        # initialize modules
        modules = {
            "enc" : encoder(),
            "dis" : discriminator(),
            "reg" : regressor(),
        }
        
        if torch.cuda.is_available(): 
            for module in modules.values():
                module.cuda()

        if init_dict:
            for name, module in modules.items():
                module.load_state_dict(torch.load(init_dict[name]))

        
        # initialize loss functions, optimizers and loss dictionaries
        lossfns = {
            "dis" : nn.BCELoss(),
            "reg" : nn.MSELoss(),
        }

        lossfns["enc"] = lambda pr, yr, pd, yd: lossfns['reg'](pr, yr) - hp.ALPHA*lossfns['dis'](pd, cls)

        optimizers = {k : optim.Adam(v.parameters(), lr=hp.INITIAL_LR) for k, v in modules.items()}
        schedulers = {k : hp.scheduler(v) for k, v in optimizers.items()}

        
        trainloss = {k : np.zeros((hp.EPOCHS,)) for k in modules.keys()}
        valloss = {k : np.zeros((hp.EPOCHS,)) for k in modules.keys()}
        
        for t in range(hp.EPOCHS):
            print(f"Epoch {t+1}\n-------------------------------")
            
            ###
            # TRAINING LOOP
            ###
            for module in modules.values():
                module.train()
            
            # for each batch:
            for batch, (X, y) in enumerate(train_dataloader):
                if torch.cuda.is_available(): X, y = X.cuda(), y.cuda()
                label, cls = y[:,0], y[:,1]
                batchloss = {}
                
                #encode image as vector
                v = modules['enc'](X)
                
                #discriminator
                d = modules['dis'](v)
                cls = torch.reshape(cls, d.shape)
                batchloss['dis'] = lossfns['dis'](d, cls)
                if hp.ALPHA != 0.0:
                    optimizers['dis'].zero_grad()
                    batchloss['dis'].backward(retain_graph=True)
                    optimizers['dis'].step()
                
                #regressor
                r = modules['reg'](v)
                label = torch.reshape(label, r.shape)
                batchloss['reg'] = lossfns['reg'](r, label)
            
                optimizers['reg'].zero_grad()
                batchloss['reg'].backward(retain_graph=True)
                optimizers['reg'].step()
            
                #get adversarial loss
                dummyreg, dummydis = regressor().cuda(), discriminator().cuda()
                dummyreg.load_state_dict(modules['reg'].state_dict())
                dummydis.load_state_dict(modules['dis'].state_dict())
                
                batchloss['enc'] = lossfns['enc'](dummyreg(v), label, dummydis(v), cls)
            
                optimizers['enc'].zero_grad()
                batchloss['enc'].backward()
                optimizers['enc'].step()

                #Save the loss values for plotting later
                for k in batchloss.keys():
                    trainloss[k][t] += batchloss[k].item() / len(train_dataloader)
   
            print(f"Encoder training loss: {trainloss['enc'][t]}")
            #print(f"Discriminator training loss: {trainloss['dis'][t]}")
            #print(f"Regressor training loss: {trainloss['reg'][t]}")
            
            ###
            # VALIDATION LOOP
            ###
            for module in modules.values():
                module.eval()
            
            with torch.no_grad():
                for X, y in val_dataloader:
                    if torch.cuda.is_available(): X, y = X.cuda(), y.cuda()
                    label, cls = y[:,0], y[:,1]
                    batchloss = {}

                    #encode image as vector
                    v = modules['enc'](X)
                    
                    #discriminator
                    d = modules['dis'](v)
                    cls = torch.reshape(cls, d.shape)
                    batchloss['dis'] = lossfns['dis'](d, cls)
                
                    #regressor
                    r = modules['reg'](v)
                    label = torch.reshape(label, r.shape)
                    batchloss['reg'] = lossfns['reg'](r, label)
                
                    #batchloss['enc'] = hp.BETA*torch.exp(lossfns['reg'](dummyreg(v), label) - hp.ALPHA*lossfns['dis'](dummydis(v), cls))
                    batchloss['enc'] = lossfns['enc'](r, label, d, cls)      
                    
                    #Save the loss values for plotting later
                    for k in batchloss.keys():
                        valloss[k][t] += batchloss[k].item() / len(val_dataloader)
            
            print(f"Encoder validation loss: {valloss['enc'][t]}")
            #print(f"Discriminator validation loss: {valloss['dis'][t]}")
            #print(f"Regressor validation loss: {valloss['reg'][t]}")
            if hp.LR_DECAY:
                for scheduler in schedulers.values():
                    scheduler.step()
                print(f"Learning Rate: {schedulers['enc'].get_last_lr()}")

        ###
        # SAVE MODEL
        ###
        os.mkdir(hp.MODEL_DIR)
        for modulename, module in modules.items():
            path = f"{hp.MODEL_DIR}/{hp.MODEL_NAME}_{modulename}"
            #save model
            torch.save(module.state_dict(), f"{path}.pth")
            #save loss
            loss = {"train":trainloss[modulename], "val":valloss[modulename]}
            np.savez(f"{path}_loss.npz", train=loss["train"], val=loss["val"])
