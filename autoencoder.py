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



def train_autoencoder(hp, init_weights=None):
    #make sure we aren't overwriting
    if os.path.isdir(hp.MODEL_DIR):
        print(hp.MODEL_DIR + " already exists. Please rename current model or delete old model directory.")
    else:
        # training & testing datasets
        print("Loading training data...")
        train_data = EORImageDataset("train", hp.TRAINING_DATA_HP)
        print("Loading validation data...")
        val_data = EORImageDataset("val", hp.TRAINING_DATA_HP) 

        # training & testing dataloaders
        train_dataloader = DataLoader(train_data, batch_size=hp.BATCHSIZE, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=hp.BATCHSIZE, shuffle=True)

        # initialize model
        model = autoencoder()
        if torch.cuda.is_available(): model.cuda()

        #lossfn = nn.MSELoss
        lossfn = ccmse_loss
        optimizer = optim.Adam(model.parameters(), lr=hp.INITIAL_LR) 
        scheduler = hp.scheduler(optimizer)
        
        if init_weights:
            print(f"Loading model state from {init_weights}")
            model.load_state_dict(torch.load(init_weights))
        
        #train / test loop
        loss = { "train" : np.zeros((hp.EPOCHS,)), "val" : np.zeros((hp.EPOCHS,)) }
        for t in range(hp.EPOCHS):
            print(f"Epoch {t+1}\n-------------------------------")

            ###
            # TRAINING LOOP
            ###
            model.train()
            
            for X, _, _, _ in train_dataloader:
                if torch.cuda.is_available(): X = X.cuda()

                batchloss = lossfn(model(X), X, hp.ALPHA)
                batchloss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss["train"][t] += batchloss.item() / len(train_dataloader)
            
            print(f"Average train loss: {loss['train'][t]}")
            
            ###
            # VALIDATION LOOP
            ###
            model.eval()

            with torch.no_grad():
                for X, _, _, _ in val_dataloader:
                    if torch.cuda.is_available(): X = X.cuda()
                    batchloss = lossfn(model(X), X, hp.ALPHA)
                    loss["val"][t] += batchloss.item() / len(val_dataloader)
            
            print(f"Average validation loss: {loss['val'][t]}")

            if hp.LR_DECAY:
                scheduler.step()
                print(f"Learning Rate: {scheduler.get_last_lr()}")
        print()
        
        ###
        # SAVE MODEL
        ###
        os.mkdir(hp.MODEL_DIR)
        path = f"{hp.MODEL_DIR}/{hp.MODEL_NAME}"
        torch.save(model.state_dict(), f"{path}.pth")

        ###
        # SAVE + PLOT LOSS
        ###
        np.savez(f"{path}_loss.npz", train=loss["train"], val=loss["val"])

        fname_loss = f"{path}_loss.png"
        title = f"{hp.MODEL_NAME} Loss"
        plot_loss(loss, fname_loss, title, transform="log")

        


