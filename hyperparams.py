import os
import sys
import json
import jsonpickle
from datetime import datetime
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR


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
                    lr_milestones=[100,200],
                    lr_gamma=0.1,
                    model_dir=None,
                    alpha=0.1,
                ):

        self.TRAINING_DATA_HP = t_data_hp
        
        # model metadata
        self.MODEL_ID = str(datetime.timestamp(datetime.now())).replace(".","")
        self.MODEL_NAME = model_name
        self.MODEL_DIR = model_dir
        if not model_dir:
            self.MODEL_DIR = "models/" + self.MODEL_NAME
        
        self.HP_JSON_FILENAME = "hp_" + self.MODEL_NAME + ".json"
        
        # training hyperparameters 
        self.BATCHSIZE = t_data_hp.BATCHSIZE
        self.SUBSAMPLE_SCALE = t_data_hp.SUBSAMPLE_SCALE
        self.EPOCHS = epochs
        self.SAVE_EPOCHS=[i for i in range(0, self.EPOCHS, 50)]
        
        self.TVT_DICT = t_data_hp.TVT_DICT
        self.NORMALIZE = t_data_hp.NORMALIZE

        #LR decay
        self.INITIAL_LR = init_lr #0.001 #static learning rate if LR_DECAY = False, or initial learning rate if LR_DECAY = True
        self.LR_DECAY = lr_decay
        self.LR_MILESTONES = lr_milestones
        self.LR_GAMMA = lr_gamma
        
        #Data params
        self.ZINDICES = t_data_hp.ZINDICES
        self.INPUT_CHANNELS = len(self.ZINDICES)
        self.PARAM = t_data_hp.PARAM

        #adversarial parameters
        self.ALPHA = alpha

        
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
    
    # From docs for MultiStepLR
    # milestones = [30, 80], gamma=0.1
    # Assuming optimizer uses lr = 0.05 for all groups
    # lr = 0.05     if epoch < 30
    # lr = 0.005    if 30 <= epoch < 80
    # lr = 0.0005   if epoch >= 80

    def scheduler(self, opt):
        if self.LR_DECAY:
            scheduler = MultiStepLR(opt, milestones=self.LR_MILESTONES, gamma=self.LR_GAMMA)
            return scheduler
        return None
    

    def save_time(self, start_time):
        dirr=self.MODEL_DIR
        print(f"\n* * * * * * * *\nPROCESS TIME: {datetime.now() - start_time}\n* * * * * * * *")
        with open(dirr + "/" + "time.txt", 'w') as file:
            file.write("--- %s seconds ---" % (datetime.now() - start_time))
        
if __name__ == "__main__":
    Model_Hyperparameters.save_hyparam_summary()