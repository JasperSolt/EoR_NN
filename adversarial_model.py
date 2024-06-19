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

def stacked_linear(insize, outsize):
    layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(insize, outsize), 
            nn.ReLU(),
    )
    return layer

#####
#
# Modules
#
#####
class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            stacked_conv(30, 16),
            stacked_conv(16, 32),
            stacked_conv(32, 64),
            nn.MaxPool2d(32), 
            nn.Flatten(),
        )

    def forward(self, x):
        return self.main(x)



class regressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            stacked_linear(64, 200),
            stacked_linear(200, 100),
            stacked_linear(100, 20),
            nn.Linear(20, 1),
        )
    
    def forward(self, x):
        return self.main(x)


class discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            stacked_linear(64, 200),
            stacked_linear(200, 100),
            stacked_linear(100, 20),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.main(x)
