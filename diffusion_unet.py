import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange #pip install einops
from typing import List
import random
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from timm.utils import ModelEmaV3 #pip install timm 
from tqdm import tqdm #pip install tqdm
import matplotlib.pyplot as plt #pip install matplotlib
import torch.optim as optim
import numpy as np

from EoR_Dataset import EORImageDataset
from hyperparams import ModelHyperparameters

'''
Edited from the following tutorial:
https://towardsdatascience.com/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946
'''

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)) #TODO: 10000?
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds[:, :, None, None]



# Residual Blocks
class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding='same')
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x


class Attention(nn.Module):
    def __init__(self, C: int, num_heads:int , dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q,k,v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')


class UnetLayer(nn.Module):
    def __init__(self, 
            upscale: bool, 
            attention: bool, 
            num_groups: int, 
            dropout_prob: float,
            num_heads: int,
            C: int):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        #return self.conv(F.pad(x, (0, 1, 0, 1), "constant", 0)), x
        return self.conv(x), x


class UNET(nn.Module):
    def __init__(self, hp: ModelHyperparameters):
        
        super().__init__()
        self.num_layers = len(hp.layer_channels)
        self.shallow_conv = nn.Conv2d(hp.input_channels, hp.layer_channels[0], kernel_size=3, padding='same')
        out_channels = (hp.layer_channels[-1]//2)+hp.layer_channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, hp.input_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=hp.time_steps, embed_dim=max(hp.layer_channels))
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=hp.layer_upscales[i],
                attention=hp.layer_attentions[i],
                num_groups=hp.num_groups,
                dropout_prob=hp.dropout_prob,
                C=hp.layer_channels[i],
                num_heads=hp.num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t):
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings)
            
            residuals.append(r)
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))



class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]
    


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



def train(hp: ModelHyperparameters):
    if not hp.debug_mode: os.mkdir(hp.model_dir)

    set_seed(random.randint(0, 2**32-1))

    train_dataset = EORImageDataset("train", hp.data_hp) 
    train_loader = DataLoader(train_dataset, batch_size=hp.batchsize, shuffle=True, num_workers=0)

    val_dataset = EORImageDataset("val", hp.data_hp) 
    val_loader = DataLoader(val_dataset, batch_size=hp.batchsize, shuffle=True, num_workers=0)

    scheduler = DDPM_Scheduler(num_time_steps=hp.time_steps)
    model = UNET(hp).to(hp.device)
    optimizer = optim.Adam(model.parameters(), lr=hp.initial_lr)
    ema = ModelEmaV3(model, decay=hp.ema_decay) 

    if hp.checkpoint_path is not None:
        checkpoint = torch.load(hp.checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema']) 
        optimizer.load_state_dict(checkpoint['optimizer'])

    criterion = nn.MSELoss(reduction='mean')

    loss_dict = {"train_loss" : torch.zeros((hp.epochs)), "val_loss" : torch.zeros((hp.epochs))}
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # TRAINING
    model.train()
    for i in range(hp.epochs):
        for bidx, (x, *_) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{hp.epochs} (Training)")):
            x = x.to(hp.device)
            this_batchsize = x.shape[0]
            t = torch.randint(0,hp.time_steps,(this_batchsize,))
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(this_batchsize,1,1,1).cuda()
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            output = model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)
            loss_dict['train_loss'][i] += loss.item() / len(train_loader)
            loss.backward()
            optimizer.step()
            ema.update(model)
        print(f"Epoch {i+1} | Training Loss {loss_dict['train_loss'][i]:.5f}")
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # VALIDATION
        model.eval()
        for bidx, (x, *_) in enumerate(tqdm(val_loader, desc=f"Epoch {i+1}/{hp.epochs} (Validation)")):
            with torch.no_grad():
                x = x.to(hp.device)
                this_batchsize = x.shape[0]

                t = torch.randint(0, hp.time_steps, (this_batchsize,))
                e = torch.randn_like(x, requires_grad=False)
                a = scheduler.alpha[t].view(this_batchsize,1,1,1).cuda()
                x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
                output = model(x, t)
                optimizer.zero_grad()
                loss = criterion(output, e)
                loss_dict['val_loss'][i] += loss.item() / len(val_loader)
        print(f"Epoch {i+1} | Validation Loss {loss_dict['val_loss'][i]:.5f}")



    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }

    torch.save(checkpoint, f'{hp.model_dir}/{hp.model_name}')