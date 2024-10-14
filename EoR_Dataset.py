import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

class EORImageDataset(Dataset):
    
    #Load data at initialization. Override from Dataset superclass
    def __init__(self, mode, hp, verbose=True):
        
        self.mode = mode
        self.hp = hp

        self.subdiv = {}
        self.truesize = {}
        self.dataset_len = {}
        self.begin = {}

        self.cube_key = 'lightcones/brightness_temp'
        self.label_key = 'lightcone_params/physparams'
        
        dataset_lenlimit = hp.lenlimit // hp.n_datasets

        for i in range(hp.n_datasets):
            with h5py.File(hp.data_paths[i], "r") as f: 
                self.truesize[i], _, true_boxlength, _ = f[self.cube_key].shape

            #determine length of dataset based on mode
            self.subdiv[i] = true_boxlength // hp.boxlength

            size = self.truesize[i] * (self.subdiv[i]**2)
            train_size = int(self.truesize[i] * hp.tvt_dict["train"]) * (self.subdiv[i]**2) #size of training dataset
            val_size = int(self.truesize[i] * hp.tvt_dict["val"]) * (self.subdiv[i]**2) #size of validation dataset
            
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
                
            self.dataset_len[i] = end - self.begin[i]
            if dataset_lenlimit > 0 and self.dataset_len[i] > dataset_lenlimit:
                self.dataset_len[i] = dataset_lenlimit
                
            if verbose: print(f"Sim {i}: {self.dataset_len[i]} samples")
        
        self.len = sum(self.dataset_len.values())
        if verbose: print(f"Total number of samples: {self.len}")
        
        self.cubes = torch.zeros((self.len, hp.zlength, hp.boxlength, hp.boxlength), dtype=torch.float)
        self.labels = torch.zeros((self.len, 2), dtype=torch.float)
        self.classes = torch.zeros((self.len,), dtype=torch.float)

        #####
        #
        # LOAD DATA
        #
        #####
        pntr = 0
        for i in range(hp.n_datasets):
            for j in range(self.dataset_len[i]): 
                if j%100 == 0 and verbose: print(f"Loading cube {j} of {self.dataset_len[i]} from sim {i} (pointer = {pntr})...")
                
                ###
                # LABELS + CLASSES
                ###
                self.labels[j+pntr] = self.load_label(self.begin[i] + j, i)
                self.classes[j+pntr] = i

                ###
                # CUBES (TODO: make less sloppy)
                ###
                #if "zoomin" in hp.ztransform and self.mode == "train": 
                if "zoomin" in hp.ztransform: 
                    mdpt, dur = self.labels[j+pntr][:]
                    zindrange = np.clip((np.array([mdpt-dur, mdpt+dur])-6.0)*512 / 8.75, 0, 511)
                    zindices = np.linspace(*zindrange, hp.zlength, dtype=int)
                else:
                    zindices = hp.zindices

                self.cubes[j+pntr] = self.load_cube(self.begin[i] + j, i, zindices)

            pntr += self.dataset_len[i]
        
        #####
        #
        # REBATCH
        #
        #####
        #if the input dimension is less than the number of channels, split channels into mini batches
        self.rebatch = hp.zlength // hp.n_channels
        if self.rebatch > 1:
            #self.labels = torch.repeat_interleave(self.labels, self.rebatch, dim=0)
            #self.classes = torch.repeat_interleave(self.classes, self.rebatch, dim=0)
            self.cubes = torch.reshape(self.cubes, (self.len*self.rebatch, hp.n_channels, hp.boxlength, hp.boxlength))
            self.len *= self.rebatch

            
    
    #Override from Dataset
    def __len__(self):
        return self.len

    #Override from Dataset
    def __getitem__(self, idx):
        cube = self.cubes[idx]
        if "shufflez" in self.hp.ztransform and self.mode == "train": 
            rpm = torch.randperm(cube.size()[0])
            cube = cube[rpm]
            
        label, cls = self.labels[idx // self.rebatch], self.classes[idx // self.rebatch]
        
        return cube, label, cls


    #####
    #
    # HELPER FUNCTIONS
    #
    #####
    #returns just the physparams
    def get_physparams(self, idx):
        return self.labels[idx][0:-1]
        
    #Load one cube from h5py
    def load_cube(self, idx, sim_index, zindices):        
        with h5py.File(self.hp.data_paths[sim_index], "r") as h5f:
            subscale = self.hp.boxlength
            i, x, y = self.sub_indices(idx, sim_index)
            cube = torch.tensor(h5f[self.cube_key][i, zindices, subscale*x:subscale*(x+1), subscale*y:subscale*(y+1)], dtype=torch.float)
        return cube
   
    #Load labels from h5py
    def load_label(self, idx, sim_index):
        with h5py.File(self.hp.data_paths[sim_index], "r") as h5f:
            i, _, _ = self.sub_indices(idx, sim_index)
            label = torch.tensor(h5f[self.label_key][i, 0:2], dtype=torch.float) 
        return label
    
    # This method ensures a good parameter distribution if you limit the length
    def sub_indices(self, idx, sim_index):
        r = idx // self.truesize[sim_index]
        i = idx - r*self.truesize[sim_index]
        x = r % self.subdiv[sim_index]
        y = r // self.subdiv[sim_index]
        return i, x, y
