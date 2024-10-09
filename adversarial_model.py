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





def predict_adversarial_NN(hp_model, hp_test, mode="test"):
    data = EORImageDataset(mode, hp_test)
    dataloader = DataLoader(data, batch_size=hp_test.BATCHSIZE, shuffle=True)

    enc, reg = encoder(), regressor()
    
    if torch.cuda.is_available(): 
        enc.cuda()
        reg.cuda()
        
    path = f"{hp_model.MODEL_DIR}/{hp_model.MODEL_NAME}"
    enc.load_state_dict(torch.load(f"{path}_enc.pth"))
    reg.load_state_dict(torch.load(f"{path}_reg.pth"))

    enc.eval()
    reg.eval()
    
    shape = (dataloader.dataset.__len__(), 1)
    predictions, targets = np.zeros(shape), np.zeros(shape)
    i = 0

    #predict
    print(f"Predicting on {shape[0]} samples...")
    with torch.no_grad():
        for X, y in dataloader:
            if torch.cuda.is_available(): X, y = X.cuda(), y.cuda()
            label, cls = y[:,0], y[:,1]
            
            pred = reg(enc(X))
            
            label = torch.reshape(label, pred.shape)

            predictions[i : i + hp_test.BATCHSIZE] = pred.cpu()
            targets[i : i + hp_test.BATCHSIZE] = label.cpu()
            i += hp_test.BATCHSIZE
   
    #save prediction
    f = f'{hp_model.MODEL_DIR}/pred_{hp_model.MODEL_NAME}_on_{hp_test.SIMS[0]}_{mode}.npz'
    np.savez(f, targets=targets, predictions=predictions)
    
    print("Prediction saved.")
    return f


