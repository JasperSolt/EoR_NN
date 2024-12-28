import os
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from hyperparams import save_hyperparameters
from EoR_Dataset import EORImageDataset
from plot_model_results import plot_loss

#####
#
# Loss Functions
#
#####
def corrcoef_loss(input, target, reduction='mean'):    
    # Covariance
    X = torch.cat((input, target), dim=-2)
    X -= torch.mean(X, -1, keepdim=True)
    X_T = torch.transpose(X, -2, -1)
    c = torch.matmul(X, X_T) / (X.shape[-1] - 1)

    # Correlation Coefficient
    d = torch.diagonal(c, dim1=-1, dim2=-2)
    dd = torch.where(d == 0, 1, d)

    stddev = torch.sqrt(dd)
    c /= stddev[:,:,:,None]
    c /= stddev[:,:,None,:]

    #1 - Cross-Correlation
    ccd = 1-torch.diagonal(c, offset=c.shape[-1]//2, dim1=-1, dim2=-2)

    if reduction == 'mean':
        return ccd.mean()
    elif reduction == 'sum':
        return ccd.sum()
    return ccd

def ccmse_loss(input, target, alpha=1.0):
    mse = ((input - target) ** 2).mean()
    cc = corrcoef_loss(input, target).mean()
    return  cc + alpha*mse


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, x, x_hat, mean, log_var, alpha=1.0):
        BCE = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp())
        return alpha*BCE + KLD



#####
#
# Variational Encoder: 
#
#####

class conv_var_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, latent_dim, kernel_size=3, stride=1, padding='same'):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim_1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(hidden_dim_1, hidden_dim_2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(hidden_dim_2, latent_dim, kernel_size=kernel_size, stride=stride, padding=padding)

        self.conv_mean = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding='same')
        self.conv_var = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding='same')
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        #self.batch_norm = nn.LazyBatchNorm2d()
        #self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        '''
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        '''
        mean = self.conv_mean(x)
        var = self.conv_var(x)
        return mean, var
#####
#
# Decoder: 
#
#####
class conv_decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim_1, hidden_dim_2, output_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_dim, hidden_dim_1, kernel_size, padding='same')
        self.conv2 = nn.Conv2d(hidden_dim_1, hidden_dim_2, kernel_size, padding='same')
        self.conv3 = nn.Conv2d(hidden_dim_2, output_dim, kernel_size, padding='same')
        
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.relu(self.conv1(self.upsample(x)))
        x = self.relu(self.conv2(self.upsample(x)))
        x = torch.sigmoid(self.conv3(self.upsample(x)))
        #x = self.upsample(x)
        #x = self.relu(self.conv1(x))
        #x = self.relu(self.conv2(x))
        #x = torch.sigmoid(self.conv3(x))
        return x



    
#####
#
# Variational Autoencoder: Large bits taken from the tutorial at
# https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
#####
class vae(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.encoder = conv_var_encoder(hp.input_dim, 
                                        hp.hidden_dim_1, 
                                        hp.hidden_dim_2, 
                                        hp.latent_dim,
                                        kernel_size=hp.kernel_size,
                                        stride=hp.stride,
                                        padding=hp.padding
                                       )
        self.decoder = conv_decoder(hp.latent_dim, hp.hidden_dim_2, hp.hidden_dim_1, hp.input_dim)
        self.device = hp.device

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var*epsilon
        return z

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        
        return x_hat, mean, log_var



def train_vae(hp):
    #make sure we aren't overwriting
    if os.path.isdir(hp.model_dir) and hp.mode != 'debug':
        print(hp.model_dir + " already exists. Please rename current model or delete old model directory.")
    else:
        # training & testing datasets
        print("Loading training data...")
        train_data = EORImageDataset("train", hp.training_data_hp)
        print("Loading validation data...")
        val_data = EORImageDataset("val", hp.training_data_hp) 


        
        # training & testing dataloaders
        train_dataloader = DataLoader(train_data, batch_size=hp.batchsize, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=hp.batchsize, shuffle=True)

        model = vae(hp)
        model.to(hp.device)

        lossfn = VAELoss()
        optimizer = optim.Adam(model.parameters(), lr=hp.initial_lr) 
        scheduler = MultiStepLR(optimizer, milestones=hp.lr_milestones, gamma=hp.lr_gamma) 
        
        if hp.parent_model:
            print(f"Loading model state from {hp.parent_model}")
            model.load_state_dict(torch.load(hp.parent_model))


            
        #train / test loop
        loss = { "train" : np.zeros((hp.epochs,)), "val" : np.zeros((hp.epochs,)) }
        for t in range(hp.epochs):
            print(f"Epoch {t+1}\n-------------------------------")

            ###
            # TRAINING LOOP
            ###
            model.train()
            
            for x, *_ in train_dataloader:
                x = x.to(hp.device)
                batchloss = lossfn(x, *model(x))
                
                batchloss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss["train"][t] += batchloss.item()
            loss["train"][t] /= len(train_data)
            
            print(f"Average train loss: {loss['train'][t]}")
            
            ###
            # VALIDATION LOOP
            ###
            model.eval()

            with torch.no_grad():
                for x, *_ in val_dataloader:
                    x = x.to(hp.device)
                    batchloss = lossfn(x, *model(x))
                    loss["val"][t] += batchloss.item()
            loss["val"][t] /= len(val_data)

            print(f"Average validation loss: {loss['val'][t]}")

            ###
            # Learning rate decay
            ###
            scheduler.step()
            print(f"Learning Rate: {scheduler.get_last_lr()}")
        
        ###
        # SAVE MODEL
        ###
        os.mkdir(hp.model_dir)
        path = f"{hp.model_dir}/{hp.model_name}"
        torch.save(model.state_dict(), f"{path}.pth")
        save_hyperparameters(hp)

        ###
        # SAVE + PLOT LOSS
        ###
        np.savez(f"{path}_loss.npz", train=loss["train"], val=loss["val"])

        fname_loss = f"{path}_loss.png"
        title = f"{hp.model_name} Loss"
        plot_loss(loss, fname_loss, title, transform="log")
