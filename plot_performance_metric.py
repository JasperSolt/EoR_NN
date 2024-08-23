import torch
from EoR_Dataset import EORImageDataset
from autoencoder import autoencoder
from hyperparams import Dataset_Hyperparameters
import numpy as np
import matplotlib.pyplot as plt
import argparse


###
# Init
###

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--setnum', type=int)
args = parser.parse_args()

setnum = args.setnum

sets = {
    0: [9.1, 10.0, 12.0, 13.1, 14.0, 15.0, 16.0],
    1: [12.0, 13.1, 9.1, 10.0], #control, shuffle z, zoom-in, shuffle z + zoom-in
    2: [12.0, 16.0], #control, weighted mse
    3: [13.1, 14.0, 15.0], #shuffle z, shufflez + weighted mse x2
    4: [10.0, 17.0], #shuffle z + zoom-in, shufflez + zoom-in + weighted mse x2
    5: [22.0, 23.0, 24.0], #vary alpha
    6: [12.0, 24.0], #mse vs ccmse
    7: [24.0, 25.0, 26.0, 27.0], #ccmse; control, shuffle z, zoom-in, shuffle z + zoom-in
}

labels = {
    "09.1":"zoom-in",
    "10.0":"zoom-in + shuffle z",
    "12.0":"control",
    "13.1":"shuffle z",
    "14.0":"shuffle z + weighted mse (1.0, 0.0)",
    "15.0":"shuffle z + weighted mse (2.0, 0.5)",
    "16.0":"weighted mse (1.0, 0.0)",
    "17.0":"zoom-in + shuffle z + weighted mse",
    "22.0":"ccmse loss (a=0.01)",
    "23.0":"ccmse loss (a=0.1)",
    "24.0":"ccmse loss",
    "25.0":"ccmse loss (shufflez)",
    "26.0":"ccmse loss (zoomin)",
    "27.0":"ccmse loss (shufflez + zoomin)",
}

version_nums = sets[setnum]


###
# Define performance metrics
###

def dgnl_sum_cc(t, p):
    n, c, x, y = t.shape
    cc = np.zeros_like(t)

    for i in range(n):
        for j in range(c):
            cc[i,j] = np.corrcoef(t[i,j], p[i,j])[x:,:y]
    trace = np.nansum(np.diagonal(cc, axis1=-2, axis2=-1), axis=-1)
    return trace

def inv_mse(t, p):
    return -np.mean((t-p)**2,  axis=(-2,-1))


metrics = {
    f"Cross-correlation coefficient trace: ctrpx {setnum:0>2}": dgnl_sum_cc,
    f"Inverse MSE: ctrpx {setnum:0>2}": inv_mse,
}


###
# Load Data
###
ws=0.0
sims = ["ctrpx",] 
data_paths = [f"/users/jsolt/data/jsolt/21cmFAST_centralpix_v05/21cmFAST_centralpix_v05_transposed_ws{ws}.hdf5"]

hp = Dataset_Hyperparameters(
                                    sims, 
                                    data_paths, 
                                    zindices=np.linspace(0, 511, 30, dtype=int), 
                                    batchsize=1, 
                                    subsample_scale=256, 
                                    param=1,
)

data = EORImageDataset("test", hp)



###
# Load Model(s)
###

versions = [f"{v:0>4}" for v in version_nums]
names = {v: f"autoencoder_v{v}_ctrpx_ws{ws}" for v in versions}
paths = {v: f"models/{name}/{name}.pth" for v, name in names.items()}
models = {}

for v, path in paths.items():
    models[v] = autoencoder()
    if torch.cuda.is_available(): models[v].cuda()
    models[v].load_state_dict(torch.load(path, map_location=torch.device('cpu')))



###
# Calculate mean metric per model & save
###

n = len(data)

for metric_name, metric in metrics.items():
    avgscore, stdscore = {}, {}
    for v, model in models.items():
        print(v)
        inpt = data[:][0].detach().numpy()
        outpt = model(data[:][0]).detach().numpy()

        score = metric(inpt, outpt)
        avgscore[v] = np.nanmean(score, axis=0)
        stdscore[v] = np.nanstd(score, axis=0)
    
    np.savez(f'{metric_name}_avg.npz', **avgscore)
    np.savez(f'{metric_name}_std.npz', **stdscore)
    
    
    ###
    # Plot results
    ###
    avgscore = np.load(f'{metric_name}_avg.npz')
    stdscore = np.load(f'{metric_name}_std.npz')
    
    
    for v in versions:
        plt.plot(avgscore[v], label=labels[v])
    plt.legend()
    plt.title(metric_name)
    plt.grid()
    plt.savefig(f"{metric_name}_performance.png")
    plt.close()

    for v in versions:
        plt.plot(stdscore[v], label=labels[v])
    plt.legend()
    plt.title(f"{metric_name} stddev")
    plt.grid()
    plt.savefig(f"{metric_name}_performance_std.png")
    plt.close()

