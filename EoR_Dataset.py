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
        self.classes = torch.zeros((self.n_samples_total,), dtype=torch.float)
        self.weights = torch.zeros((self.n_samples_total, len(self.hp.ZINDICES), 1, 1), dtype=torch.float)

        #####
        #
        # LOAD DATA
        #
        #####
        pntr = 0
        for i in range(self.hp.N_DATASETS):
            for j in range(self.n_samples[i]): 
                if j%100 == 0 and verbose: print(f"Loading cube {j} of {self.n_samples[i]} from sim {i} (pointer = {pntr})...")
                
                ###
                # LABELS + CLASSES
                ###
                self.labels[j+pntr] = self.load_label(self.begin[i] + j, i)
                self.classes[j+pntr] = i

                ###
                # CUBES (TODO: make less sloppy)
                ###
                if "zoomin" in self.hp.ZTRANSFORM and self.mode == "train": 
                    mdpt, dur = self.labels[j+pntr][:]
                    r = 6.0
                    zindrange = np.clip((np.array([mdpt-r//2, mdpt+r//2])-6.0)*512 / 8.75, 0, 511)
                    zindices = np.linspace(*zindrange, len(self.hp.ZINDICES), dtype=int)
                else:
                    zindices = self.hp.ZINDICES

                self.cubes[j+pntr] = self.load_cube(self.begin[i] + j, i, zindices)

                ###
                # WEIGHTS
                ###
                if self.hp.WEIGHTS:
                    maxx = np.max(self.cubes[j+pntr].numpy(), axis=(-2,-1), keepdims=True)
                    minn = np.min(self.cubes[j+pntr].numpy(), axis=(-2,-1), keepdims=True)
                    meann = np.mean(self.cubes[j+pntr].numpy(), axis=(-2,-1), keepdims=True)
                    self.weights[j+pntr] = torch.from_numpy(self.hp.M/(1.0-(maxx-meann)*minn) - self.hp.B)
                
            pntr += self.n_samples[i]
        
    #Override from Dataset
    def __len__(self):
        return self.n_samples_total

    #Override from Dataset
    def __getitem__(self, idx):
        cube, label, cls, weight = self.cubes[idx], self.labels[idx], self.classes[idx], self.weights[idx]
        if "shufflez" in self.hp.ZTRANSFORM and self.mode == "train": 
            rpm = torch.randperm(cube.size()[0])
            cube, weight = cube[rpm], weight[rpm]

        return cube, label, cls, weight


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
        with h5py.File(self.hp.DATA_PATHS[sim_index], "r") as h5f:
            subscale = self.hp.SUBSAMPLE_SCALE
            i, x, y = self.sub_indices(idx, sim_index)
            cube = torch.tensor(h5f[self.cube_key][i, zindices, subscale*x:subscale*(x+1), subscale*y:subscale*(y+1)], dtype=torch.float)
        return cube
   
    #Load labels from h5py
    def load_label(self, idx, sim_index):
        with h5py.File(self.hp.DATA_PATHS[sim_index], "r") as h5f:
            i, _, _ = self.sub_indices(idx, sim_index)
            label = torch.tensor(h5f[self.label_key][i, 0:2], dtype=torch.float) 
        return label
    
    # This method ensures a good parameter distribution if you limit n_samples
    def sub_indices(self, idx, sim_index):
        r = idx // self.truesize[sim_index]
        i = idx - r*self.truesize[sim_index]
        x = r % self.subdiv[sim_index]
        y = r // self.subdiv[sim_index]
        return i, x, y
