import os
import sys
from collections import OrderedDict
import json
import jsonpickle
from datetime import datetime
import torch
from torch import nn

'''
# CPLX package
from cplxmodule import cplx
from cplxmodule.nn import RealToCplx, CplxToReal
from cplxmodule.nn import CplxConv2d, CplxLinear
from cplxmodule.nn import CplxModReLU, CplxBatchNorm2d, CplxDropout, CplxMaxPool2d
from cplxmodule.nn.modules.casting import ConcatenatedRealToCplx, CplxToConcatenatedReal
'''

'''
These should literally never change
'''
class Constant():
    #random constants
    KSZ_CONSTANT = 2.7255 * 1e6

    #indices for each parameter
    PARAM_DICT = {0:"midpoint", 1:"duration", 2:"meanz"}

    
class Dataset_Hyperparameters():
    def __init__(
                    self,
                    data_sims,
                    data_paths,
                    zindices,
                    batchsize=16,
                    subsample_scale=512,
                    tvt_dict={"train":0.8, "val":0.10, "test":0.10},
                    normalize=False,
                    param=0,
                    n_limit=-1
                ):

        # data metadata
        self.SIMS = data_sims
        self.DATA_PATHS = data_paths

        assert len(self.SIMS) == len(self.DATA_PATHS)
        
        self.N_DATASETS = len(data_sims)

        #hyperparameters 
        self.BATCHSIZE = batchsize
        self.SUBSAMPLE_SCALE = subsample_scale        
        self.TVT_DICT = tvt_dict
        self.NORMALIZE = normalize

        #Data params
        self.ZINDICES = zindices
        self.INPUT_CHANNELS = len(self.ZINDICES)

        self.PARAM = param
        self.N_LIMIT=n_limit

'''
class MixedDataset_Hyperparameters():
    def __init__(
                    self,
                    data_sims,
                    data_paths,
                    zindices,
                    batchsize=12,
                    subsample_scale=512,
                    tvt_dict={"train":0.8, "val":0.10, "test":0.10},
                    normalize=True,
                    param=0
                ):

        # data metadata
        self.SIMS = data_sims
        self.DATA_PATHS = data_paths
                
        #hyperparameters 
        #self.N_SAMPLES = n_samples
        self.BATCHSIZE = batchsize
        self.SUBSAMPLE_SCALE = subsample_scale        
        self.TVT_DICT = tvt_dict
        self.NORMALIZE = normalize

        #Data params
        self.ZINDICES = zindices
        self.INPUT_CHANNELS = len(self.ZINDICES)

        self.PARAM = param
'''

'''
Hyperparameters for the model. 
'''
class Model_Hyperparameters():
    
    def __init__(
                    self,
                    model_name,
                    t_data_hp,
                    epochs=400,
                    init_lr=0.1,
                    lr_decay=False,
                    #decay_rt=0.9,
                    lr_milestones=[100,200],
                    lr_gamma=0.1,
                    model_dir=None
                ):

        self.TRAINING_DATA_HP = t_data_hp
        
        # model metadata
        self.MODEL_ID = str(datetime.timestamp(datetime.now())).replace(".","")
        #self.SIM = t_data_hp.SIM
        #self.DATA_PATH = t_data_hp.DATA_PATH
        
        self.MODEL_NAME = model_name
        self.MODEL_DIR = model_dir
        if not model_dir:
            self.MODEL_DIR = "models/" + self.MODEL_NAME
        
        self.HP_JSON_FILENAME = "hp_" + self.MODEL_NAME + ".json"
        
        # training hyperparameters 
        #self.N_SAMPLES = t_data_hp.N_SAMPLES
        self.BATCHSIZE = t_data_hp.BATCHSIZE
        self.SUBSAMPLE_SCALE = t_data_hp.SUBSAMPLE_SCALE
        self.EPOCHS = epochs
        self.SAVE_EPOCHS=[i for i in range(0, self.EPOCHS, 50)]
        
        self.TVT_DICT = t_data_hp.TVT_DICT
        self.NORMALIZE = t_data_hp.NORMALIZE

        #LR decay
        self.INITIAL_LR = init_lr #0.001 #static learning rate if LR_DECAY = False, or initial learning rate if LR_DECAY = True
        self.LR_DECAY = lr_decay
        #self.DECAY_RT = decay_rt
        self.LR_MILESTONES = lr_milestones
        self.LR_GAMMA = lr_gamma
        
        #Data params
        self.ZINDICES = t_data_hp.ZINDICES
        self.INPUT_CHANNELS = len(self.ZINDICES)

        self.PARAM = t_data_hp.PARAM
        
        # Loss function
        self.loss_fn = nn.MSELoss() 
    
        '''
        #complex info
        FFT = False #fourier or image space?
        FFT_SHIFT = False
        INCLUDE_PHASE = False
        IMAGINARY_AXIS = -3 #-1 for cplx package, -3 for channels
        USE_CPLX_PKG = False
        '''
        #Basic model architecture
        self.LAYER_DICT = OrderedDict([
          # batch_size x input_channels x 256 x 256
          ('conv1', nn.Conv2d(self.INPUT_CHANNELS, 16, 3, padding='same')),
          ('relu1_1', nn.ReLU()),
          ('batch1', nn.BatchNorm2d(16)),
          ('maxpool1', nn.MaxPool2d(2)),

          # batch_size x 16 x 128 x 128
          ('conv2', nn.Conv2d(16, 32, 3, padding='same')),
          ('relu1_2', nn.ReLU()),
          ('batch2', nn.BatchNorm2d(32)),
          ('maxpool2', nn.MaxPool2d(2)),

          # batch_size x 32 x 64 x 64
          ('conv3', nn.Conv2d(32, 64, 3, padding='same')),
          ('relu1_3', nn.ReLU()),
          ('batch3', nn.BatchNorm2d(64)),
          ('maxpool3', nn.MaxPool2d(2)),

          # batch_size x 64 x 32 x 32
          # pytorch doesn't have global pooling layers, so I made the kernel the
          # same dimensions as the input
          ('global_maxpool', nn.MaxPool2d(self.SUBSAMPLE_SCALE // 2**3)), 
          ('flat1', nn.Flatten()),

          # batch_size x 64
          ('drop1', nn.Dropout(0.2)),
          ('dense1', nn.Linear(64, 200)), 
          ('relu2_1', nn.ReLU()),

          # batch_size x 200
          ('drop2', nn.Dropout(0.2)),
          ('dense2', nn.Linear(200, 100)),
          ('relu2_2', nn.ReLU()),

          # batch_size x 100
          ('drop3', nn.Dropout(0.2)),
          ('dense3', nn.Linear(100, 20)),
          ('relu2_3', nn.ReLU()),

          # batch_size x 20
          ('output', nn.Linear(20, 1))
        ])

        
    def save_hyparam_summary(self):
        dirr=self.MODEL_DIR
        report_name=self.HP_JSON_FILENAME
        if not os.path.isdir(dirr):
            os.mkdir(dirr)
        print(f"Generating hyperparameter summary at {dirr + '/' + report_name}...")
        with open(dirr + "/" + report_name, 'w') as file:
            json_encode = jsonpickle.encode(self.__dict__.copy(), unpicklable=False, indent=4, max_depth=2)
            json.dump(json_encode, file)
        print("Hyperparameter summary saved.")
    
    
    def optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.INITIAL_LR)
    
    # From docs for MultiStepLR
    # milestones = [30, 80], gamma=0.1
    # Assuming optimizer uses lr = 0.05 for all groups
    # lr = 0.05     if epoch < 30
    # lr = 0.005    if 30 <= epoch < 80
    # lr = 0.0005   if epoch >= 80

    def scheduler(self, opt):
        if self.LR_DECAY:
            #lam = lambda epoch: 1 / (1 + self.DECAY_RT * epoch)
            #scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lam])
            scheduler = MultiStepLR(opt, milestones=self.LR_MILESTONES, gamma=self.LR_GAMMA)
            return scheduler
        return None
    

    def save_time(self, start_time):
        dirr=self.MODEL_DIR
        print(f"\n* * * * * * * *\nPROCESS TIME: {datetime.now() - start_time}\n* * * * * * * *")
        with open(dirr + "/" + "time.txt", 'w') as file:
            file.write("--- %s seconds ---" % (datetime.now() - start_time))
        
    
    '''
    #complex model architecture
    CPLXLAYER_DICT = OrderedDict([
      # batch_size x input_channels x 512 x 512
      ('real2cplx', ConcatenatedRealToCplx()),
      ('conv1', CplxConv2d(INPUT_CHANNELS, 16, 3, padding=1)),
      ('relu1_1', CplxModReLU()),
      ('batch1', CplxBatchNorm2d(16)),
      ('maxpool1', CplxMaxPool2d(2)),

      # batch_size x 16 x 256 x 256
      ('conv2', CplxConv2d(16, 32, 3, padding=1)),
      ('relu1_2', CplxModReLU()),
      ('batch2', CplxBatchNorm2d(32)),
      ('maxpool2', CplxMaxPool2d(2)),

      # batch_size x 32 x 128 x 128
      ('conv3', CplxConv2d(32, 64, 3, padding=1)),
      ('relu1_3', CplxModReLU()),
      ('batch3', CplxBatchNorm2d(64)),
      ('maxpool3', CplxMaxPool2d(2)),

      # batch_size x 64 x 64 x 64
      # pytorch doesn't have global pooling layers, so I made the kernel the
      # same dimensions as the input
      ('global_maxpool', CplxMaxPool2d(64)),
      ('flat1', nn.Flatten()),

      # batch_size x 64 x 1 x 1
      #('drop1', CplxDropout(0.2)),
      ('dense1', CplxLinear(64, 200)),
      ('relu2_1', CplxModReLU()),

      # batch_size x 200 x 1 x 1
      #('drop2', CplxDropout(0.2)),
      ('dense2', CplxLinear(200, 100)),
      ('relu2_2', CplxModReLU()),

      # batch_size x 100 x 1 x 1
      #('drop3', CplxDropout(0.2)),
      ('dense3', CplxLinear(100, 20)),
      ('relu2_3', CplxModReLU()),

      # batch_size x 20 x 1 x 1
      ('output', CplxLinear(20, 1)) #,('cplx2real', CplxToConcatenatedReal())
    ])    
    '''
    
    
if __name__ == "__main__":
    Model_Hyperparameters.save_hyparam_summary()