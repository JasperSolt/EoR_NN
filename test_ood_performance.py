from datetime import datetime
import fnmatch
import os
import numpy as np
import argparse
from omegaconf import OmegaConf
from regressors.cnn import predict_cnn
from util.plotting import plot_model_predictions

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--sim', required=True, type=str)
    parser.add_argument('-n','--name', required=True, type=str)
    parser.add_argument('-cf','--cf_version', required=True, type=str)

    return parser

def get_model_name(sim, base_name):
    name = base_name
    name += f"_{sim}_ws0.0_"
    date_format = "%Y-%m-%dT%H-%M"

    most_recent = datetime.min
    for file in os.listdir(f'trained_models/{sim}_ws0.0'):
        if fnmatch.fnmatch(file, f"{name}*"):
            date_str = file[len(name):]
            time = datetime.strptime(date_str, date_format)
            if time > most_recent: most_recent = time

    return name + most_recent.strftime(date_format), name[:-1]

sims = ["p21c", "ctrpx", "zreion"]

parser = get_parser()
args, unknown = parser.parse_known_args()


# Argparse
sim = args.sim
id = sim + "_ws0.0"
base_name = args.name
model_name, short_model_name = get_model_name(sim, base_name)

# model config
cfg_path = f'trained_models/{id}/{model_name}/{model_name}_config.yaml'
base_cfg = OmegaConf.load(cfg_path)

# Additional command line args
cli_cfg = OmegaConf.from_dotlist(unknown)
cfg = OmegaConf.merge(base_cfg, cli_cfg)


# PREDICT
all_prediction_npzs = []
labels = []
data_cfg = cfg.data

#counterfactual config
cf_cfg = OmegaConf.create(data_cfg)
cf_cfg.data_paths[sim] = f"/users/jsolt/data/jsolt/counterfactuals/{cfg.model.parent_model}_counterfactuals_{args.cf_version}.hdf5"
cf_cfg.cube_key = "lightcones/x_prime"
cf_cfg.label_key = "lightcone_params/y_true"

#predict on in domain + ood data

for i in range(len(sims)):
    data_cfg.sims = [sims[i]]
    all_prediction_npzs.append(predict_cnn(cfg, data_cfg))
    labels.append(sims[i])


# Plot predictions
figname = f"{cfg.model.model_dir}/{cfg.model.name}_{data_cfg.param_name}_predictions_ood"

title = f"OOD Performance: {short_model_name}"

plot_model_predictions(all_prediction_npzs, figname, labels=labels, param=data_cfg.param_name, title=title)


#predict on cf
cf_prediction_npzs = [predict_cnn(cfg, cf_cfg, "train"), predict_cnn(cfg, cf_cfg, "val"), predict_cnn(cfg, cf_cfg)]
cf_labels = ["train cf", "val cf", "test cf"]

cf_figname = f"{cfg.model.model_dir}/{cfg.model.name}_{cf_cfg.param_name}_predictions_cf"

cf_title = f"Counterfactual Predictions: {short_model_name}"

plot_model_predictions(cf_prediction_npzs, cf_figname, labels=cf_labels, param=cf_cfg.param_name, title=cf_title)
