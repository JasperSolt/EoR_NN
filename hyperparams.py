import json
import jsonpickle

class DataHyperparameters():
    def __init__(self, /, **kwargs):

        #Add all kwargs as attributes
        self.__dict__.update(kwargs)

        #Implicit attributes
        self.n_datasets = len(self.data_paths)
        self.zlength = len(self.zindices)

        #Defaults
        self.n_channels = kwargs.get("n_channels", self.zlength)
        self.tvt_dict = kwargs.get("tvt_dict", {"train":0.8, "val":0.10, "test":0.10})
        self.lenlimit = kwargs.get("lenlimit", -1)
    
    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__




class ModelHyperparameters():
    def __init__(self, /, **kwargs):
        #kwargs
        self.__dict__.update(kwargs)
        self.model_dir = f"/users/jsolt/FourierNN/trained_models/{self.model_name}"

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__


    

def save_hyperparameters(hp):
    with open(f"{hp.model_dir}/{hp.model_name}_hyperparameters.json", 'w') as f:
        pickle = jsonpickle.encode(hp, indent=4)
        f.write(pickle)

def load_hyperparameters(path):
    with open(path, 'r') as f:
        jsonstr = f.read()
    return jsonpickle.decode(jsonstr)





