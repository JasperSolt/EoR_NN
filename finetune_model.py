import datetime
import numpy as np
import argparse
from omegaconf import OmegaConf
from regressors.cnn import train_augmented_cnn, predict_cnn, train_cnn
from util.plotting import plot_model_predictions



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--sim', required=True, type=str)
    parser.add_argument('--control', action=argparse.BooleanOptionalAction)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    return parser


data_paths = {
    "zreion": "/users/jsolt/data/jsolt/zreion_sims/zreion24/zreion24_norm_subdiv_sliced_ws0.0.hdf5",
    "ctrpx": "/users/jsolt/data/jsolt/centralpix_sims/centralpix05/centralpix05_norm_subdiv_sliced_ws0.0.hdf5",
    "p21c": "/users/jsolt/data/jsolt/21cmFAST_sims/p21c14/p21c14_norm_subdiv_sliced_ws0.0.hdf5"
}

parent_models = {
        "p21c":"cnn_v03_p21c_ws0.0_2025-02-28T12-28",
        "ctrpx":"cnn_v03_ctrpx_ws0.0_2025-02-28T12-28",
        "zreion":"cnn_v03_zreion_ws0.0_2025-02-28T12-32"
}

ids = {
        "p21c":"p21c_ws0.0",
        "ctrpx":"ctrpx_ws0.0",
        "zreion":"zreion_ws0.0" 
}

'''
MAIN
'''
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")

parser = get_parser()
args, unknown = parser.parse_known_args()

'''
SET UP CONFIG
'''
# Argparse
sim = args.sim
id = ids[sim]
parent_model = parent_models[sim]

# Parent model config
parent_cfg_path = f'trained_models/{id}/{parent_model}/{parent_model}_config.yaml'
parent_cfg = OmegaConf.load(parent_cfg_path)

# Fine tuning parameters
aug_cfg_path = f'configs/aug_cnn_config.yaml'
aug_cfg = OmegaConf.load(aug_cfg_path)
base_cfg = OmegaConf.merge(parent_cfg, aug_cfg) 

base_cfg.model.parent_model = parent_model
base_cfg.model.parent_path = f"trained_models/{id}/{parent_model}/{parent_model}.pth"

# Additional command line args
cli_cfg = OmegaConf.from_dotlist(unknown)
cfg = OmegaConf.merge(base_cfg, cli_cfg)

# Default model name
if args.control:
    cfg.model.base_name = cfg.model.base_name + "_ctrl"

suffix = "_".join([f"{word}" for word in cfg.data.sims]) + f"_ws{cfg.data.wedgeslope}"
cfg.model.name = f"{cfg.model.base_name}_{suffix}_{now}"
cfg.model.model_dir = f"trained_models/{suffix}/{cfg.model.name}"

# Default data args
cfg.data.data_path = data_paths[sim]
cfg.data.param_name = "duration" if cfg.data.param_index==1 else "midpoint"
cfg.aug_data.param_name = "duration" if cfg.data.param_index==1 else "midpoint"

print(cfg.pretty())

# Debug config
if args.debug:
    debug_cfg = OmegaConf.load('/users/jsolt/FourierNN/configs/debug_config.yaml')
    cfg = OmegaConf.merge(cfg, debug_cfg)




'''
TRAIN
'''
if args.control:
    train_cnn(cfg)
else:
    train_augmented_cnn(cfg)



'''
PREDICT
'''
all_prediction_npzs = []
all_prediction_npzs.append(predict_cnn(cfg, cfg.data, mode="train"))
all_prediction_npzs.append(predict_cnn(cfg, cfg.data, mode="val"))
test_predictions = predict_cnn(cfg, cfg.data)
all_prediction_npzs.append(test_predictions)


# Plot predictions
figname = f"{cfg.model.model_dir}/{cfg.model.name}_{cfg.data.param_name}_predictions_all"
plot_model_predictions(all_prediction_npzs, figname, param=cfg.data.param_name, labels=["train", "val", "test"], title=None)

figname = f"{cfg.model.model_dir}/{cfg.model.name}_{cfg.data.param_name}_predictions_test"
plot_model_predictions([test_predictions], figname, param=cfg.data.param_name, labels=["test"], title=None)

