import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

#####
#
# Layers
#
#####
def stacked_conv(inchannels, outchannels):
    layer = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(outchannels),
            nn.MaxPool2d(2),
    )
    return layer

#####
#
# Loss Functions: 
#
#####
def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()


def corrcoef_loss(input, target):    
    # Covariance
    X = torch.cat((input, target), dim=-2)
    X -= torch.mean(X, -1, keepdim=True)
    X_T = torch.transpose(X, -2, -1)
    c = torch.matmul(X, X_T) / (X.shape[-1] - 1)

    # Correlation Coefficient
    d = torch.diagonal(c, dim1=-1, dim2=-2)
    stddev = torch.sqrt(d)
    stddev = torch.where(stddev == 0, 1, stddev)
    c /= stddev[:,:,:,None]
    c /= stddev[:,:,None,:]

    #1 - Cross-Correlation Diagonal
    ccd = 1-torch.diagonal(c, offset=c.shape[-1]//2, dim1=-1, dim2=-2)
    
    return ccd.mean()

def ccmse_loss(input, target, alpha=1.0):
    mse = ((input - target) ** 2).mean()
    cc = corrcoef_loss(input, target)
    return  cc + alpha*mse


#####
#
# Autoencoder: 
#
#####
class autoencoder(nn.Module):
    def __init__(self, c=30):
        super().__init__()
        self.encoder = nn.Sequential(
            stacked_conv(c, 16), #30 x 256 x 256 -> 16 x 128 x 128
            stacked_conv(16, 32), #16 x 128 x 128 -> 32 x 64 x 64
            stacked_conv(32, 64), #32 x 64 x 64 -> 64 x 32 x 32 
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, padding='same'),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, 3, padding='same'),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, c, 3, padding='same'),
            nn.Sigmoid()
        )

        
    def forward(self, x):
        return self.decoder(self.encoder(x))

