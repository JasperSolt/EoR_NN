import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

TVT_SPLIT = {
    "train":0.8,
    "val":0.1,
    "test":0.1,
}

def get_dataset_range(mode, size, subdiv=1):
    newsize = size * (subdiv**2)
    train_size = int(size * TVT_SPLIT["train"]) * (subdiv**2) #size of training dataset
    val_size = int(size * TVT_SPLIT["val"]) * (subdiv**2) #size of validation dataset

    range_dict = {
        "train":(0, train_size),
        "val":(train_size, train_size + val_size),
        "test":(train_size + val_size, newsize),
        "all":(0, newsize)
    }

    return range_dict[mode] 



class EORImageDataset(Dataset):
    #Load data at initialization. Override from Dataset superclass
    def __init__(self, mode, data_cfg):
        
        self.mode = mode
        self.data_cfg = data_cfg
            
        cube_key = self.data_cfg.get('cube_key', 'lightcones/brightness_temp')
        label_key = self.data_cfg.get('label_key', 'lightcone_params/physparams')
        
        dataset_lenlimit = self.data_cfg.get('lenlimit', -1)
        
        with h5py.File(self.data_cfg.data_path, "r") as f:             
            begin, end = get_dataset_range(mode, len(f[cube_key]))
            if dataset_lenlimit > 0 and (end - begin) > dataset_lenlimit:
                end = begin + dataset_lenlimit
            self._len = end - begin

            self.cubes = torch.tensor(f[cube_key][begin:end], dtype=torch.float)
            if f[label_key].ndim == 1:
                self.labels = torch.tensor(f[label_key][begin:end], dtype=torch.float)[:,None]
            else:
                self.labels = torch.tensor(f[label_key][begin:end, self.data_cfg.param_index], dtype=torch.float)[:,None]

    #Override from Dataset
    def __len__(self):
        return self._len

    #Override from Dataset
    def __getitem__(self, idx):
        return self.cubes[idx], self.labels[idx]
    



class AugmentedEORImageDataset(Dataset):
    #Load data at initialization. Override from Dataset superclass
    def __init__(self, mode, data_cfg, aug_cfg):
        self.base_data = EORImageDataset(mode, data_cfg)
        self.aug_data = EORImageDataset(mode, aug_cfg)
        
        self.base_len = len(self.base_data)
        self.aug_len = len(self.aug_data)
        self._len = self.base_len + self.aug_len

    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        if idx < self.base_len:
            return self.base_data[idx]
        return self.aug_data[idx-self.base_len]
 
        









class EOREncodedImageDataset(Dataset):
    #Load data at initialization. Override from Dataset superclass
    def __init__(self, mode, cfg):
        
        self.mode = mode
        data_cfg = cfg.data

        cube_key = 'lightcones/brightness_temp'
        label_key = 'lightcone_params/physparams'
        
        n_datasets = len(data_cfg.sims)
        dataset_lenlimit = data_cfg.get('lenlimit', -1) // n_datasets

        set_cubes, set_labels = [], []
        for s in data_cfg.sims:
            with h5py.File(data_cfg.data_paths[s], "r") as f: 
                n_lightcones, old_zlength, encoding_channels, boxlength, *_ = f[cube_key].shape #(n, 30, 4, 32, 32)
                begin, end = get_dataset_range(mode, n_lightcones)
                if dataset_lenlimit > 0 and (end - begin) > dataset_lenlimit:
                    end = begin + dataset_lenlimit
                dataset_len = end - begin

                assert data_cfg.boxlength == boxlength
                newshape = (dataset_len, data_cfg.zlength, data_cfg.boxlength, data_cfg.boxlength) #(n, 120, 32, 32)
                set_cubes.append(torch.tensor(f[cube_key][begin:end], dtype=torch.float).view(newshape))
                set_labels.append(torch.tensor(f[label_key][begin:end, data_cfg.param_index], dtype=torch.float))

        
        self.cubes = torch.cat(set_cubes)
        self.labels = torch.cat(set_labels)[:,None]

        print(self.cubes.shape)
        assert len(self.cubes) == len(self.labels)
        self._len = len(self.labels)

        # Normalize to [0,1)
        if data_cfg.normalize:
            self.cubes = self.norm_to_unit_interval(self.cubes)
        
    
    #Override from Dataset
    def __len__(self):
        return self._len

    #Override from Dataset
    def __getitem__(self, idx):
        return self.cubes[idx], self.labels[idx]
    
    # Helper functions
    def norm_to_unit_interval(self, x):
        return (x - x.min()) / (x.max()-x.min())







    #####
    #
    # Old dataset class
    #
    #####

    '''
class EORImageDataset(Dataset):
    #Load data at initialization. Override from Dataset superclass
    def __init__(self, mode, cfg, verbose=True):
        
        self.mode = mode
        self.data_cfg = cfg.data

        #self.subdiv = {}
        self.size = {}
        self.dataset_len = {}
        self.begin = {}

        self.cube_key = 'lightcones/brightness_temp'
        self.label_key = 'lightcone_params/physparams'
        
        n_datasets = len(self.data_cfg.sims)

        dataset_lenlimit = self.data_cfg.lenlimit // n_datasets
        for i in range(n_datasets):
            sim=self.data_cfg.sims[i]
            self.data_path = self.data_cfg.data_paths[sim]
            with h5py.File(self.data_path, "r") as f: 
                self.size[i], _, true_boxlength, _ = f[self.cube_key].shape

            #determine length of dataset based on mode
            #self.subdiv[i] = true_boxlength // self.data_cfg.boxlength

            self.begin[i], end = get_dataset_range(mode, self.size[i])
            self.dataset_len[i] = end - self.begin[i]
            if dataset_lenlimit > 0 and self.dataset_len[i] > dataset_lenlimit:
                self.dataset_len[i] = dataset_lenlimit
                
            if verbose: print(f"Sim {i}: {self.dataset_len[i]} samples")
        
        self._len = sum(self.dataset_len.values())
        if verbose: print(f"Total number of samples: {self._len}")
        
        self.cubes = torch.zeros((self._len, self.data_cfg.zlength, self.data_cfg.boxlength, self.data_cfg.boxlength), dtype=torch.float)
        self.labels = torch.zeros((self._len, 1), dtype=torch.float)

        #####
        #
        # LOAD DATA
        #
        #####
        pntr = 0
        for i in range(n_datasets):
            for j in range(self.dataset_len[i]): 
                if j%100 == 0 and verbose: print(f"Loading cube {j} of {self.dataset_len[i]} from sim {i} (pointer = {pntr})...")
                
                ###
                # LABELS + CLASSES
                ###
                self.labels[j+pntr] = self.load_label_simple(self.begin[i] + j)

                ###
                # CUBES 
                ###
                self.cubes[j+pntr] = self.load_cube_simple(self.begin[i] + j)

            pntr += self.dataset_len[i]
        


            
    
    #Override from Dataset
    def __len__(self):
        return self._len

    #Override from Dataset
    def __getitem__(self, idx):
        cube = self.cubes[idx]
        label = self.labels[idx]          
        return cube, label


    #####
    #
    # HELPER FUNCTIONS
    #
    #####
    #Load one cube from h5py
    def load_cube(self, idx, sim_index):
        sim=self.cfg.sims[i]
        with h5py.File(self.cfg.data_paths[sim], "r") as h5f:
            subscale = self.cfg.boxlength
            i, x, y = self.sub_indices(idx, sim_index)
            cube = torch.tensor(h5f[self.cube_key][i, :, subscale*x:subscale*(x+1), subscale*y:subscale*(y+1)], dtype=torch.float)
        return cube
   
    #Load labels from h5py
    def load_label(self, idx, sim_index):
        with h5py.File(self.cfg.data_paths[sim_index], "r") as h5f:
            i, _, _ = self.sub_indices(idx, sim_index)
            label = torch.tensor(h5f[self.label_key][i, 0:2], dtype=torch.float) 
        return label
    
    # This method ensures a good parameter distribution if you limit the length
    def sub_indices(self, idx, sim_index):
        r = idx // self.size[sim_index]
        i = idx - r*self.size[sim_index]
        x = r % self.subdiv[sim_index]
        y = r // self.subdiv[sim_index]
        return i, x, y
    

    def load_cube_simple(self, idx):
        with h5py.File(self.data_path, "r") as h5f:
            cube = torch.tensor(h5f[self.cube_key][idx], dtype=torch.float)
        return cube
   
    #Load labels from h5py
    def load_label_simple(self, idx):
        with h5py.File(self.data_path, "r") as h5f:
            label = torch.tensor(h5f[self.label_key][idx, self.data_cfg.param_index], dtype=torch.float) 
        return label



'''