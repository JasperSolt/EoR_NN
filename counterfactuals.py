from omegaconf import OmegaConf
import numpy as np
import torch
from torch import nn, optim
from torch import Tensor
import torch.functional as F
from diffusers import AutoencoderKL

from regressors.latent_cnn import lcnn

def encode_slice(vae, x):
    x = np.repeat(x[np.newaxis,:,:], 3, axis=0)
    x = torch.from_numpy(x[np.newaxis,:,:,:]).to(vae.device)
    z = vae.encode(x).latent_dist.mean
    return z.detach().cpu().numpy()[0]

class CounterfactualLoss(nn.Module):
    def __init__(self, encoder, predictor, generator):
        super(CounterfactualLoss, self).__init__()
        self.f = encoder
        self.h = predictor
        self.g = generator


    '''
    x' = g(z')
    y' = h(z')
    y = h(f(x))
    L = d_x{ g(z'), x } - d_y{ h(z'), h(f(x)) }
    '''
    def forward(self, x: Tensor, z_prime: Tensor) -> Tensor:
        return self.d_input(self.g(z_prime), x) - self.d_label(self.h(z_prime), self.h(self.f(x)))

    '''
    Distance function for the input space. 
    We want to minimize this, i.e., find z' such that x'=g(z') is maximally similar to x.
    '''
    def d_input(self, x_prime, x):
        return torch.sqrt((x_prime - x)**2).mean()
    
    '''
    Distance function for the label space. 
    We want to maximize this, i.e., find z' such that y'=h(z') is maximally different from y=h(z)=h(f(x)).
    '''
    def d_label(self, y_prime, y):
        return torch.sqrt((y_prime - y)**2).mean()



def generate_counterfactual(x: Tensor) -> Tensor:
    '''
    Set up our encoder, generator, and predictor
    '''
    device="cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(device)
    
    encoder = vae.encode
    generator = vae.decode

    config_path = '/users/jsolt/FourierNN/configs/lcnn_config.yaml' #TODO
    cfg = OmegaConf.load(config_path)
    predictor = lcnn(cfg)
    predictor.to(device)

    checkpoint_path = '' #TODO
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    predictor.load_state_dict(checkpoint['model_state_dict'])

    #Initialize our counterfactual
    #z = 
    #z_prime = 
    #optimizer = optim.Adam(, lr=cfg.model.lr) 

    return torch.Tensor([0])

'''
def train_lcnn(cfg):
    # training & testing datasets
    print("Loading training data...")
    train_data = EOREncodedImageDataset("train", cfg)
    print("Loading validation data...")
    val_data = EOREncodedImageDataset("val", cfg) 
    
    # training & testing dataloaders
    train_dataloader = DataLoader(train_data, batch_size=cfg.model.batchsize, shuffle=True)

    device = cfg.model.device
    assert device=="cuda"

    model = lcnn(cfg)
    model.to(device)

    print(model)

    lossfn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg.model.lr) 

    lr_lambda = lambda epoch: 1 / (epoch*cfg.model.lr_gamma + 1)
    scheduler = LambdaLR(optimizer, lr_lambda) 
    
    train_loss, val_loss = torch.tensor([]), torch.tensor([])

    epoch = 0

    # Load checkpoint
    if cfg.model.checkpoint_path:
        chkpt_pth = cfg.model.checkpoint_path
        print(f"Loading model state from {chkpt_pth}...")
        checkpoint = load_checkpoint(chkpt_pth, model, optimizer, scheduler)
        model, optimizer, scheduler, train_loss, val_loss, epoch = checkpoint

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
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        ###
        # Log
        ###
        save_checkpoint(f"{path}.pth", model, optimizer, scheduler, train_loss, val_loss, t)
        plot_loss(loss={"train": train_loss.cpu().numpy(), "val": val_loss.cpu().numpy()},
                   fname=f"{path}_loss.png", 
                   title=f"{cfg.model.name} Loss", 
                   transform="log"
                   )

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
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler, train_loss, val_loss, epoch

'''