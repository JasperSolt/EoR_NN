import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

#####
#
# Encoder: 
#
#####
class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, latent_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim_1, kernel_size, padding='same')
        self.conv2 = nn.Conv2d(hidden_dim_1, hidden_dim_2, kernel_size, padding='same')
        self.conv3 = nn.Conv2d(hidden_dim_2, latent_dim, kernel_size, padding='same')
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        #self.batch_norm = nn.LazyBatchNorm2d()
        #self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.pool(self.relu(self.conv1(x)))
        x2 = self.pool(self.relu(self.conv2(x1)))
        x3 = self.pool(self.relu(self.conv3(x2)))
        return x3


#####
#
# Decoder: 
#
#####
class decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim_1, hidden_dim_2, output_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_dim, hidden_dim_1, kernel_size, padding='same')
        self.conv2 = nn.Conv2d(hidden_dim_1, hidden_dim_2, kernel_size, padding='same')
        self.conv3 = nn.Conv2d(hidden_dim_2, output_dim, kernel_size, padding='same')
        
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x):
        x1 = self.relu(self.conv1(self.upsample(x)))
        x2 = self.relu(self.conv2(self.upsample(x1)))
        x3 = self.relu(self.conv3(self.upsample(x2)))
        return x3


#####
#
# Autoencoder: 
#
#####
class slice_autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, latent_dim):
        super().__init__()
        self.encoder = encoder(input_dim, hidden_dim_1, hidden_dim_2, latent_dim)
        self.decoder = decoder(latent_dim, hidden_dim_2, hidden_dim_1, input_dim)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)