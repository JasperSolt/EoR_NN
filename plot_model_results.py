import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
matplotlib.use('AGG')
import numpy as np
import os
import h5py
import py21cmfast as p21c
from hyperparams import Model_Hyperparameters as hp, Constant as con

def plot_loss(loss, fname, title=""):
    print("Saving loss plot to {}...".format(fname))
    
    train_loss, val_loss = loss["train"], loss["val"]
    epochs = len(train_loss)
    plt.plot(np.arange(1, epochs+1), np.log10(train_loss), label='Training loss')
    plt.plot(np.arange(1, epochs+1), np.log10(val_loss), label='Evaluation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log MSE Loss')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(fname, dpi=300)
    plt.clf()
    print("Loss plot saved.")


def plot_model_predictions(npz_names, figname, p, labels=None, title=None):
    dlabels = labels
    if not dlabels:
        dlabels = npz_names
    
    n_models = len(npz_names)
    model_colors = ['r','g','b','c','m','y']

    #for p in param_indices:
    param = con.PARAM_DICT[p]
    if not title:
        title = param

    print(f"Plotting results for parameter: {param}...")

    '''
    Layout
    '''
    fig = plt.figure()
    gs1 = gridspec.GridSpec(3,1)
    ax1, ax2 = fig.add_subplot(gs1[:2]), fig.add_subplot(gs1[2])

    #ax1 formatting
    ax1.set_title(title)
    ax1.set_xlabel(r'')
    ax1.set_ylabel(f'Predicted {param}')
    ax1.locator_params(nbins=5)
    ax1.grid(True)

    recs = []
    for i in range(n_models):
        recs.append(mpatches.Rectangle((0,0),0.5,0.5, fc=model_colors[i]))
    ax1.legend(recs, dlabels, fontsize=10)

    #ax2 formatting
    ax2.locator_params(nbins=5)
    ax2.set_ylabel(r'% error')
    ax2.set_xlabel(f'True {param}')
    ax2.grid(True)

    '''
    load and plot the parameter prediction data for each model
    '''
    targets, pred, err = np.array([]), np.array([]), np.array([])
    for i, name in enumerate(npz_names):
        result = np.load(name)

        model_targets = np.array(result['targets'][:, 0])
        model_pred = np.array(result['predictions'][:, 0])
        model_err = 100.0*(1-model_pred/model_targets)

        ax1.scatter(model_targets, model_pred, alpha=0.7, s=1.5, c=model_colors[i])
        ax2.scatter(model_targets, model_err, alpha=0.7, s=1.5, c=model_colors[i])

        targets = np.append(targets,model_targets)
        pred = np.append(pred, model_pred)
        err = np.append(err, model_err)
    '''
    find axis limits 
    '''
    mintargets, maxtargets = np.min(targets), np.max(targets)
    minpred, maxpred = np.min(pred), np.max(pred)
    minerr, maxerr = np.min(err), np.max(err)

    ax1.set_xlim(0.95*mintargets,1.05*maxtargets)
    ax1.set_ylim(minpred*0.95,maxpred*1.05)

    ax2.set_ylim(minerr*0.95,maxerr*1.05)
    ax2.set_xlim(0.95*mintargets,1.05*maxtargets)

    '''
    Plot target line
    '''
    ideal = np.linspace(0.8*mintargets,1.2*maxtargets,10)
    ax1.plot(ideal, ideal, 'k--', alpha=0.3, linewidth=1.5)
    ax2.plot(ideal,len(ideal)*[0.],'k--', alpha=0.3, linewidth=1.5)

    '''
    Save
    '''
    plt.tight_layout()
    plt.savefig(f'{figname}.jpeg',dpi=300)
    plt.close()




