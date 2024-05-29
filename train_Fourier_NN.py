import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from EoR_Dataset import EORImageDataset
from model import Fourier_NN, train, test, save, save_loss, load
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
        hp.save_time(start_time)

