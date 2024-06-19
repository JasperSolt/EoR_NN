import numpy as np
import h5py
import torch
from torch.utils.data import Dataset



class EORImageDataset(Dataset):
    
    #Load data at initialization. Override from Dataset superclass
    def __init__(self, mode, hp, verbose=True):
        
        self.mode = mode
        self.hp = hp
        
        cubes = {}
        labels = {}
        
        self.subdiv = {}
        self.truesize = {}
        self.n_samples = {}
        self.begin = {}

        self.cube_key = 'lightcones/brightness_temp'
        self.label_key = 'lightcone_params/physparams'
        
        set_nlimit = self.hp.N_LIMIT // self.hp.N_DATASETS

        for i in range(self.hp.N_DATASETS):
            #sometimes I'm dumb and create datasets with different keys to the same type of data, i.e., 'bT_cubes' vs. 'full_bT_cubes'
            with h5py.File(self.hp.DATA_PATHS[i], "r") as h5f: 
                self.truesize[i], _, boxlen, _ = h5f[self.cube_key].shape
                self.subdiv[i] = boxlen // self.hp.SUBSAMPLE_SCALE

            #determine length of dataset based on mode
            size = self.truesize[i] * (self.subdiv[i]**2)
            train_size = int(self.truesize[i] * self.hp.TVT_DICT["train"]) * (self.subdiv[i]**2) #size of training dataset
            val_size = int(self.truesize[i] * self.hp.TVT_DICT["val"]) * (self.subdiv[i]**2) #size of validation dataset
            
            if self.mode == "train":
                self.begin[i], end = 0, train_size # train_percent fraction of samples = training set.
            elif self.mode == "val": 
                self.begin[i], end = train_size, train_size + val_size
            elif self.mode == "test":
                self.begin[i], end = train_size + val_size, size
            elif self.mode == "all":
                self.begin[i], end = 0, size
            else:
                self.begin[i], end = 0, 0 
                print("Invalid mode for dataset")
                
            self.n_samples[i] = end - self.begin[i]
            if set_nlimit > 0 and self.n_samples[i] > set_nlimit:
                self.n_samples[i] = set_nlimit
                
            if verbose: print(f"Sim {i}: {self.n_samples[i]} samples")
        
        self.n_samples_total = sum(self.n_samples.values())
        if verbose: print(f"Total number of samples: {self.n_samples_total}")
        
        self.cubes = torch.zeros((self.n_samples_total, len(self.hp.ZINDICES), self.hp.SUBSAMPLE_SCALE, self.hp.SUBSAMPLE_SCALE), dtype=torch.float)
        self.labels = torch.zeros((self.n_samples_total, 2), dtype=torch.float)
        
        pntr = 0
        for i in range(self.hp.N_DATASETS):
            #load data
            for j in range(self.n_samples[i]): #would be faster loading multiple at a time, but I'm worried I'll break something lol
                if j%100 == 0 and verbose: print(f"Loading cube {j} of {self.n_samples[i]} from sim {i} (pointer = {pntr})...")
                self.cubes[j+pntr] = self.load_cube(self.begin[i] + j, i)
                self.labels[j+pntr][0] = self.load_label(self.begin[i] + j, i)
                self.labels[j+pntr][1] = i
            pntr += self.n_samples[i]
        
    #Override from Dataset
    def __len__(self):
        return self.n_samples_total

    #Override from Dataset
    def __getitem__(self, idx):
        #return self.cubes[idx].cuda(), self.labels[idx].cuda()
        return self.cubes[idx], self.labels[idx]

    #####
    #
    # HELPER FUNCTIONS
    #
    #####
    
    #Load one cube from h5py
    def load_cube(self, idx, sim_index):        
        with h5py.File(self.hp.DATA_PATHS[sim_index], "r") as h5f:
            subscale = self.hp.SUBSAMPLE_SCALE
            i, x, y = self.sub_indices(idx, sim_index)
            cube = torch.tensor(h5f[self.cube_key][i, self.hp.ZINDICES, subscale*x:subscale*(x+1), subscale*y:subscale*(y+1)], dtype=torch.float)
        return cube
   
    #Load labels from h5py
    def load_label(self, idx, sim_index):
        with h5py.File(self.hp.DATA_PATHS[sim_index], "r") as h5f:
            i, _, _ = self.sub_indices(idx, sim_index)
            label = torch.tensor(h5f[self.label_key][i, self.hp.PARAM], dtype=torch.float) 
        return label
    
    # This method ensures a good parameter distribution if you limit n_samples
    def sub_indices(self, idx, sim_index):
        r = idx // self.truesize[sim_index]
        i = idx - r*self.truesize[sim_index]
        x = r % self.subdiv[sim_index]
        y = r // self.subdiv[sim_index]
        return i, x, y
