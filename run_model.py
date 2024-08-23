import os
import numpy as np
import argparse
from train_Fourier_NN import train_Fourier_NN, train_adversarial_NN, train_autoencoder
from predict_Fourier_NN import predict_Fourier_NN, predict_adversarial_NN
from hyperparams import Model_Hyperparameters, Dataset_Hyperparameters
from plot_model_results import plot_model_predictions, plot_loss, plot_loss_grid
from model import load_loss


def get_path(model_sim, ws):
    if model_sim == "p21c":
        dp = f"/users/jsolt/data/jsolt/21cmFAST_sims/p21c14/p21c14_ws{ws}_trnspsd_meanremoved_norm.hdf5"
    elif model_sim == "zreion":
        dp = f"/users/jsolt/data/jsolt/zreion_sims/zreion24/zreion24_transposed_ws{ws}.hdf5"
    elif model_sim == "ctrpx":
        dp = f"/users/jsolt/data/jsolt/21cmFAST_centralpix_v05/21cmFAST_centralpix_v05_transposed_ws{ws}.hdf5"
    return dp

parser = argparse.ArgumentParser()
parser.add_argument('-s','--sim', nargs='+', required=True, type=str, help='<Required> List of sims')
parser.add_argument('-mn', '--modelnum', required=True, type=float, default=0.0, help='<Required> Model number')

parser.add_argument('-ws', '--wedgeslope', nargs='?', const=0.0, type=float, default=0.0, help='Wedge slope of data (default = 0)')
parser.add_argument('-lr', '--learningrt', nargs='?', const=0.0, type=float, default=0.001, help='Static learning rate (default = 0.005)')
parser.add_argument('-a', '--alpha', nargs='?', const=0.0, type=float, default=1.0, help='Discriminator weight (default = 1.0)')
parser.add_argument('-bs', '--batchsize', nargs='?', type=int, default=64, help='Batch size (default = 64)')
parser.add_argument('-p', '--param', choices=['mdpt', 'dur'], nargs='?', type=str, default='dur', help='Parameter for training. mdpt = midpoint, dur = duration')
args = parser.parse_args()

#Initializing
modelnum = args.modelnum
ws = args.wedgeslope
lr = args.learningrt
batchsize = args.batchsize
alpha = args.alpha

param = args.param
param_index = 0 if param=="mdpt" else 1

train_sims = args.sim 

data_paths = []
for ts in train_sims:
    data_paths.append(get_path(ts, ws))

scale = 256
epochs = 2000
zindices = np.linspace(0, 511, 30, dtype=int)
pred_sims = ['p21c', 'zreion', 'ctrpx']


###
# MODEL NAME 
###
model_name = f"autoencoder_v{modelnum:0>4}"
for ts in train_sims:
    model_name += f"_{ts}"
model_name += f"_ws{ws}"
model_dir = "models/" + model_name


###
# WEIGHT INIT
###
init_path = None
'''
init_name = "autoencoder_v13.0"
for ts in train_sims:
    init_name += f"_{ts}"
init_name += f"_ws{ws}"
init_path = f"models/{init_name}/{init_name}.pth" 
'''

###
# INITIALIZE HYPERPARAMETERS
###
hp_train_data = Dataset_Hyperparameters(
                                    train_sims, 
                                    data_paths, 
                                    zindices, 
                                    batchsize, 
                                    subsample_scale=scale, 
                                    param=param_index,
                                    ztransform=["shufflez","zoomin"],
                                    #n_limit=8,
)

hp_model = Model_Hyperparameters(
                                    model_name, 
                                    hp_train_data, 
                                    epochs=epochs, 
                                    init_lr=lr, 
                                    lr_decay=False,
                                    alpha=alpha,
)


###
# TRAIN
###
print("Training Loop")
train_autoencoder(hp_model, init_path)


###
# PREDICT
###
'''
print("Prediction Loop")
pred_set = [Dataset_Hyperparameters([x], [get_path(x, ws)], zindices, batchsize, subsample_scale=scale, param=param_index) for x in pred_sims]
for p in pred_set: 
    predict_adversarial_NN(hp_model=hp_model, hp_test=p, mode="test") 


###
# PLOT
###
print('Plotting...')
pred_files_test = [f"{hp_model.MODEL_DIR}/pred_{hp_model.MODEL_NAME}_on_{p}_test.npz" for p in pred_sims]
psdirs = [f"models/plots/{hp_model.MODEL_NAME}", f"{hp_model.MODEL_DIR}/pred_plots"]
modules = ["enc", "reg", "dis"]



for plot_save_dir in psdirs:
    if not os.path.isdir(plot_save_dir): os.mkdir(plot_save_dir)

    ###
    # Plot loss
    ###
    if hp_model.ALPHA == 0.0:
        path = f"{hp_model.MODEL_DIR}/{hp_model.MODEL_NAME}_reg_loss.npz"
        with np.load(path) as data: 
            loss = {'train': data['train'], 'val': data['val']}
        
        fname_loss = f"{plot_save_dir}/{hp_model.MODEL_NAME}_loss.png"
        title = f"{hp_model.MODEL_NAME} Loss"
    
        plot_loss(loss, fname_loss, title)
    else:
        lossdict = {}
        for module in modules:
            lossdict[module] = {}
            path = f"{hp_model.MODEL_DIR}/{hp_model.MODEL_NAME}_{module}_loss.npz"
            with np.load(path) as data:
                lossdict[module]["train"] = data['train']
                lossdict[module]["val"] = data['val']
        
        fname_loss = f"{plot_save_dir}/{hp_model.MODEL_NAME}_all_loss.png"
        title = f"{hp_model.MODEL_NAME} Loss"
    
        plot_loss_grid(lossdict, fname_loss, title)

    ###
    # Plot predictions
    ###
    fname_pred = f"{plot_save_dir}/duration_{hp_model.MODEL_NAME}"

    for sim in pred_sims:
        fname_pred += f"_{sim}"
    
    title = model_name
    labels = pred_sims
    
    plot_model_predictions(pred_files_test, fname_pred, param, labels, title)
'''