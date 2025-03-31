import datetime
import numpy as np
import argparse
from omegaconf import OmegaConf
from regressors.cnn import train_cnn, predict_cnn
from util.plotting import plot_model_predictions


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        const=True,
        default="configs/cnn_config.yaml",
        nargs="?",
        help="desc",
    )
    return parser




'''
MAIN
'''
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")

parser = get_parser()
args, unknown = parser.parse_known_args()

'''
SET UP CONFIG
'''
# Load
base_cfg = OmegaConf.load(args.config)
cli_cfg = OmegaConf.from_dotlist(unknown)

cfg = OmegaConf.merge(base_cfg, cli_cfg)

# Defaults + Interpolation
if not cfg.model.name:
    suffix = "_".join([f"{word}" for word in cfg.data.sims]) + f"_ws{cfg.data.wedgeslope}"
    cfg.model.name = f"{cfg.model.base_name}_{suffix}_{now}"
    cfg.model.model_dir = f"trained_models/{suffix}/{cfg.model.name}"
cfg.data.param_name = "duration" if cfg.data.param_index==1 else "midpoint"

if cfg.debug:
    debug_cfg = OmegaConf.load('/users/jsolt/FourierNN/configs/debug_config.yaml')
    cfg = OmegaConf.merge(cfg, debug_cfg)


print(cfg.pretty())


'''
TRAIN
'''
#train_cnn(cfg)


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

