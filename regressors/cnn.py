import importlib
import os
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader
from EoR_Dataset import AugmentedEORImageDataset, EORImageDataset
from util.plotting import plot_loss

#model class
class cnn(nn.Module):
    def __init__(self, cfg):
        self.cfg=cfg
        super(cnn, self).__init__()
        
        # Attributes
        self.n_conv_layers = len(cfg.model.conv_channels)
        self.n_linear_layers = len(cfg.model.linear_features)

        # General layers
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        
        # Conv layers
        self.conv_layers, self.batchnorm_layers = [], []
        prev_out_c = cfg.data.zlength
        for i, out_c in enumerate(cfg.model.conv_channels):
            self.conv_layers.append(nn.Conv2d(prev_out_c, out_c, 3, padding='same'))
            self.batchnorm_layers.append(nn.BatchNorm2d(out_c))
            self.add_module(f"conv{i}", self.conv_layers[i])
            self.add_module(f"batchnorm{i}", self.batchnorm_layers[i])
            prev_out_c = out_c
            
        # Global Maxpool
        xy = cfg.data.boxlength // (2**np.sum(cfg.model.downsample))
        self.global_maxpool = nn.MaxPool2d(xy)

        
        # Linear Layers
        self.linear_layers = []
        prev_out_f = prev_out_c
        for i, out_f in enumerate(cfg.model.linear_features):
            self.linear_layers.append(nn.Linear(prev_out_f, out_f))
            self.add_module(f"linear{i}", self.linear_layers[i])
            prev_out_f = out_f

        # Output layer
        self.out = nn.Linear(prev_out_f, 1)

    # Forward propagation of some batch x. 
    def forward(self, x):
        # Conv layers
        for i in range(self.n_conv_layers):
            x = self.batchnorm_layers[i](self.relu(self.conv_layers[i](x)))
            if self.cfg.model.downsample[i] == True:
                x = self.maxpool(x)
        
        # Global maxpool + flatten
        if self.cfg.model.global_maxpool: x = self.global_maxpool(x)
        x = self.flatten(x)

        # Linear layers
        for i in range(self.n_linear_layers):
            x = self.relu(self.linear_layers[i](self.dropout(x)))
        return self.out(x)




def train_cnn(cfg):
    # training & testing datasets
    print("Loading training data...")
    train_data = EORImageDataset("train", cfg.data)
    print("Loading validation data...")
    val_data = EORImageDataset("val", cfg.data) 
    
    # training & testing dataloaders
    train_dataloader = DataLoader(train_data, batch_size=cfg.model.batchsize, shuffle=True)

    device = cfg.model.device
    assert device=="cuda"

    model = cnn(cfg)
    model.to(device)

    print(model)

    lossfn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg.model.lr) 

    #lr_lambda = lambda epoch: 1 / (epoch*cfg.model.lr_gamma + 1)
    #scheduler = LambdaLR(optimizer, lr_lambda) 
    scheduler = MultiStepLR(optimizer, milestones=cfg.model.lr_milestones, gamma=cfg.model.lr_gamma) 
    
    train_loss, val_loss = torch.tensor([]), torch.tensor([])

    epoch = 0

    # Load checkpoint
    if cfg.model.checkpoint_path:
        # Load checkpoint
        chkpt_pth = cfg.model.checkpoint_path
        print(f"Loading model state from {chkpt_pth}...")
        checkpoint = load_checkpoint(chkpt_pth, model, optimizer, scheduler)
        model, optimizer, scheduler, train_loss, val_loss, epoch = checkpoint
    elif cfg.model.parent_path: 
        # Load parent model
        parent_pth = cfg.model.parent_path
        print(f"Loading parent model from {parent_pth}...")
        model.load_state_dict(torch.load(parent_pth)['model_state_dict'])

    # Initialize dir + some variables
    path = f"{cfg.model.model_dir}/{cfg.model.name}"

    if not cfg.debug: 
        os.mkdir(cfg.model.model_dir)
        OmegaConf.save(config=cfg, f=f'{path}_config.yaml')

    #train / test loop
    for t in range(epoch, cfg.model.epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        ###
        # TRAINING LOOP
        ###
        model.train()
        train_e_losses = []
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            batch_loss = lossfn(model(x), y)
            
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_e_losses.append(batch_loss.item())
        
        train_e_loss = torch.mean(torch.tensor(train_e_losses))
        print(f"Average train loss: {train_e_loss:.6f}")

        train_loss = torch.cat((train_loss, torch.tensor([train_e_loss])))

        ###
        # VALIDATION
        ###
        model.eval()

        with torch.no_grad():
            val_e_loss = lossfn(model(val_data[:][0].to(device)), val_data[:][1].to(device)).item()
        
        print(f"Average validation loss: {val_e_loss:.6f}")

        val_loss = torch.cat((val_loss, torch.tensor([val_e_loss])))


        ###
        # Learning rate decay
        ###
        if cfg.model.lr_decay:
            scheduler.step()
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        ###
        # Log
        ###
        save_path = f"{path}_{t}.pth" if (t%1000==0 and t!=0) else f"{path}.pth"
        save_checkpoint(save_path, model, optimizer, scheduler, train_loss, val_loss, t)
        plot_loss(loss={"train": train_loss.cpu().numpy(), "val": val_loss.cpu().numpy()},
                   fname=f"{path}_loss.png", 
                   title=f"{cfg.model.name} Loss", 
                   transform="log"
                   )







def train_augmented_cnn(cfg):
    # training & testing datasets
    print("Loading training data...")
    train_data = AugmentedEORImageDataset("train", cfg.data, cfg.aug_data)

    print("Loading validation data...")
    val_data = EORImageDataset("val", cfg.data) 
    aug_val_data = EORImageDataset("val", cfg.aug_data) 
    
    # training & testing dataloaders
    train_dataloader = DataLoader(train_data, batch_size=cfg.model.batchsize, shuffle=True)

    device = cfg.model.device
    assert device=="cuda"

    model = cnn(cfg)
    model.to(device)
    print(model)

    lossfn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg.model.lr) 
    scheduler = MultiStepLR(optimizer, milestones=cfg.model.lr_milestones, gamma=cfg.model.lr_gamma) 

    train_loss, val_loss, aug_val_loss = torch.tensor([]), torch.tensor([]), torch.tensor([])
    epoch = 0

    if cfg.model.checkpoint_path:
        # Load checkpoint
        chkpt_pth = cfg.model.checkpoint_path
        print(f"Loading model state from {chkpt_pth}...")
        checkpoint = load_checkpoint(chkpt_pth, model, optimizer, scheduler)
        model, optimizer, scheduler, train_loss, val_loss, epoch = checkpoint
    else: 
        # Load parent model
        parent_pth = cfg.model.parent_path
        print(f"Loading parent model from {parent_pth}...")
        model.load_state_dict(torch.load(parent_pth)['model_state_dict'])

    # Initialize dir + some variables
    path = f"{cfg.model.model_dir}/{cfg.model.name}"

    if not cfg.debug: 
        os.mkdir(cfg.model.model_dir)
        OmegaConf.save(config=cfg, f=f'{path}_config.yaml')

    #train / test loop
    for t in range(epoch, cfg.model.epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        ###
        # TRAINING LOOP
        ###
        model.train()
        train_e_losses = []
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            batch_loss = lossfn(model(x), y)
            
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_e_losses.append(batch_loss.item())
        
        train_e_loss = torch.mean(torch.tensor(train_e_losses))
        print(f"Average train loss: {train_e_loss:.6f}")

        train_loss = torch.cat((train_loss, torch.tensor([train_e_loss])))

        ###
        # VALIDATION
        ###
        model.eval()

        with torch.no_grad():
            val_e_loss = lossfn(model(val_data[:][0].to(device)), val_data[:][1].to(device)).item()
            aug_val_e_loss = lossfn(model(aug_val_data[:][0].to(device)), aug_val_data[:][1].to(device)).item()
        
        print(f"Average validation loss: {val_e_loss:.6f}")
        print(f"Average augmented val loss: {aug_val_e_loss:.6f}")

        val_loss = torch.cat((val_loss, torch.tensor([val_e_loss])))
        aug_val_loss = torch.cat((aug_val_loss, torch.tensor([aug_val_e_loss])))


        ###
        # Learning rate decay
        ###
        if cfg.model.lr_decay:
            scheduler.step()
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        ###
        # Log
        ###
        save_path = f"{path}_{t}.pth" if (t%1000==0 and t!=0) else f"{path}.pth"
        save_checkpoint(save_path, model, optimizer, scheduler, train_loss, val_loss, t)

        lossdict = {
            "train": train_loss.cpu().numpy(), 
            "val": val_loss.cpu().numpy(),
            "aug_val": aug_val_loss.cpu().numpy(),
            }

        plot_loss(lossdict,
                   fname=f"{path}_loss.png", 
                   title=f"{cfg.model.name} Loss", 
                   transform="log"
                   )




def predict_cnn(model_cfg, data_cfg, mode="test"):
    print("Loading test data...")
    test_data = EORImageDataset(mode, data_cfg) 
    test_dataloader = DataLoader(test_data, batch_size=model_cfg.model.batchsize)
    model = cnn(model_cfg)
    model.to(model_cfg.model.device)
    path = f"{model_cfg.model.model_dir}/{model_cfg.model.name}"

    print(f"Loading state dict from {path}.pth...")
    checkpoint = torch.load(f"{path}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    predictions, targets = torch.tensor([]).to(model_cfg.model.device), torch.tensor([]).to(model_cfg.model.device)
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(model_cfg.model.device), y.to(model_cfg.model.device)
            predictions = torch.cat((predictions, model(x)))
            targets = torch.cat((targets, y))

    # Save
    fname = f"{path}_{model_cfg.data.param_name}_predict_{mode}_{data_cfg.sims[0]}.npz"
    np.savez(fname, predictions=predictions.cpu().numpy(), targets=targets.cpu().numpy())
    return fname



def save_checkpoint(path, model, optimizer, scheduler, train_loss, val_loss, epoch):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch': epoch,
            }, path)
    


def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler, train_loss, val_loss, epoch
