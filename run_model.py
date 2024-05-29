import os
import argparse
from train_Fourier_NN import train_Fourier_NN
from predict_Fourier_NN import predict_Fourier_NN
from hyperparams import Model_Hyperparameters, Dataset_Hyperparameters
from plot_model_results import plot_model_predictions, plot_loss
from model import load_loss

#just a temporary function meant to map parameters to my file names
def get_path(model_sim, ws):
    if model_sim == "p21c":
        dp = f"/users/jsolt/data/jsolt/21cmFAST_sims/p21c14/p21c14_ws{ws}_trnspsd_meanremoved_norm.hdf5"
    elif model_sim == "zreion":
        dp = f"/users/jsolt/data/jsolt/zreion_sims/zreion21/zreion21_transposed_ws{ws}.hdf5"
    elif model_sim == "ctrpx":
        dp = f"/users/jsolt/data/jsolt/21cmFAST_centralpix_v04/21cmFAST_centralpix_v04_transposed_ws{ws}.hdf5"
    return dp

#arg parsing
parser = argparse.ArgumentParser()
#parser.add_argument('sim', type=str)
parser.add_argument('-s','--sim', nargs='+', required=True, type=str, help='<Required> List of sims')
parser.add_argument('-ws', '--wedgeslope', nargs='?', const=0.0, type=float, default=0.0, help='Wedge slope of data (default = 0)')
parser.add_argument('-lr', '--learningrt', nargs='?', const=0.0, type=float, default=0.001, help='Static learning rate (default = 0.001)')
parser.add_argument('-bs', '--batchsize', nargs='?', type=int, default=32, help='Batch size (default = 32)')
parser.add_argument('-p', '--param', choices=['mdpt', 'dur'], nargs='?', type=str, default='dur', help='Parameter for training. mdpt = midpoint, dur = duration')
args = parser.parse_args()

#Initializing
ws=args.wedgeslope
train_sims = args.sim 
lr = args.learningrt
param = args.param
batchsize = args.batchsize

param_index = 0 if param=="mdpt" else 1

data_paths = []
for ts in train_sims:
    data_paths.append(get_path(ts, ws))

scale = 256
epochs = 2000

#initialize weights (comment out if not needed) 
init_model = ""
if len(train_sims) > 1:
    init_model += "mixed_"
for ts in train_sims:
    init_model += f"{ts}_"
init_model += f"m256_dur_ws{ws}_lr0.005_bs64_v02"
init_weights = f"models/{init_model}/{init_model}.pth"
#init_weights = None

model_name = ""
if len(train_sims) > 1:
    model_name += "mixed_"
for ts in train_sims:
    model_name += f"{ts}_"
model_name += f"_ws{ws}_ft02"

model_dir = "models/" + model_name

zindices = [x*17 for x in range(0, 30)]

hp_train_data = Dataset_Hyperparameters(train_sims, data_paths, zindices, batchsize, subsample_scale=scale, param=param_index)
hp_model = Model_Hyperparameters(model_name, hp_train_data, epochs=epochs, init_lr=lr)

#print("Training Loop")
train_Fourier_NN(hp_model, init_weights)

print("Prediction Loop")

#plot_save_dir = f"{hp_model.MODEL_DIR}/pred_plots"
psdirs = [f"models/plots/{hp_model.MODEL_NAME}", f"{hp_model.MODEL_DIR}/pred_plots"]

for plot_save_dir in psdirs:
    if not os.path.isdir(plot_save_dir): os.mkdir(plot_save_dir)
    lossplt_fname = f"{plot_save_dir}/{hp_model.MODEL_NAME}_loss.png"
    plot_loss(load_loss(hp_model), lossplt_fname, f"{hp_model.MODEL_NAME} Loss")

all_sims = ['zreion', 'p21c', 'ctrpx']
for pred_sims in [train_sims, all_sims]:
    pred_set = [Dataset_Hyperparameters([x], [get_path(x, ws)], zindices, batchsize, subsample_scale=scale, param=param_index) for x in pred_sims]
    pred_files_test = [predict_Fourier_NN(hp_model=hp_model, hp_test=p, mode="test") for p in pred_set]
    #pred_files_test = [f'{hp_model.MODEL_DIR}/pred_{hp_model.MODEL_NAME}_on_{x}_test.npz' for x in pred_sims]

    print('Plotting...')
    for plot_save_dir in psdirs:
        fig_name_test = f"{plot_save_dir}/duration_{hp_model.MODEL_NAME}"
    
        for sim in pred_sims:
            fig_name_test += f"_{sim}"
        
        title = model_name
        labels = pred_sims
        
        plot_model_predictions(pred_files_test, fig_name_test, param_index, labels, title)
