import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
matplotlib.use('AGG')
import numpy as np
import os
import h5py
import pandas as pd

def plot_loss(loss, fname, title="",start=10, steps=[], steplabels=[], transform="tanh"):    
    ax = plt.subplot()
    ax.grid(True)

    if transform == "tanh":
        train_loss = pd.DataFrame(np.tanh(loss['train'][start:])).ewm(com=5.0).mean()
        val_loss = pd.DataFrame(np.tanh(loss['val'][start:])).ewm(com=5.0).mean()
        ax.set_ylabel('MSE Loss (tanh + exp running avg)')
    elif transform == "log":
        train_loss = pd.DataFrame(np.log10(loss['train'][start:])).ewm(com=5.0).mean()
        val_loss = pd.DataFrame(np.log10(loss['val'][start:])).ewm(com=5.0).mean()
        ax.set_ylabel('MSE Loss (log)')
        
    epochs = np.arange(start+1, len(train_loss)+start+1)

    ax.plot(epochs, val_loss, label='Validation', linewidth=0.7)
    ax.plot(epochs, train_loss, label='Training', linewidth=1.0)
    
    for i in range(len(steps)):
        ax.axvline(steps[i], color='red', ls=":", alpha=0.5)
        ax.text(
            steps[i]-0.02*(ax.get_xlim()[1]), 
            ax.get_ylim()[1] - 0.03*(ax.get_ylim()[1]-ax.get_ylim()[0]), 
            steplabels[i],
            horizontalalignment='right',
            verticalalignment='top',
            color='#000000',
            backgroundcolor='#eeeeeec0',)
    
    ax.legend(loc='upper left')
    ax.set_xlabel('Epochs')
    ax.set_title(title)
    
    plt.savefig(fname)
    print("Loss plot saved.")
    plt.close()





def plot_loss_grid(lossdict, fname, title="", start=10, steps=[], steplabels=[], transform="tanh"):
    fig, axs = plt.subplots(len(lossdict), 1, sharex=True, tight_layout=True, figsize=(6, 12))
    fig.suptitle(title)
    
    for r, (label, loss) in enumerate(lossdict.items()):
        axs[r].grid(True)

        if transform == "tanh":
            train_loss = pd.DataFrame(np.tanh(loss['train'][start:])).ewm(com=5.0).mean()
            val_loss = pd.DataFrame(np.tanh(loss['val'][start:])).ewm(com=5.0).mean()
            axs[r].set_ylabel('MSE Loss (tanh + exp running avg)')
        elif transform == "log":
            train_loss = pd.DataFrame(np.log10(loss['train'][start:])).ewm(com=5.0).mean()
            val_loss = pd.DataFrame(np.log10(loss['val'][start:])).ewm(com=5.0).mean()
            axs[r].set_ylabel('MSE Loss (log)')

        epochs = np.arange(start+1, len(train_loss)+start+1)

        axs[r].plot(epochs, val_loss, label='Validation', linewidth=0.7)
        axs[r].plot(epochs, train_loss, label='Training', linewidth=1.0)
        
        for i in range(len(steps)):
            axs[r].axvline(steps[i], color='red', ls=":", alpha=0.5)
            axs[r].text(
                steps[i]-0.02*(axs[r].get_xlim()[1]), 
                axs[r].get_ylim()[1] - 0.03*(axs[r].get_ylim()[1]-axs[r].get_ylim()[0]), 
                steplabels[i],
                horizontalalignment='right',
                verticalalignment='top',
                color='#000000',
                backgroundcolor='#eeeeeec0',)
        
        axs[r].set_title(label)
        axs[r].legend(loc='upper left')
    axs[-1].set_xlabel('Epochs')
        
    plt.savefig(fname)
    print("Loss plot saved.")
    plt.close()





def plot_model_predictions(npz_names, figname, param, labels=None, title=None):
    dlabels = labels
    if not dlabels:
        dlabels = npz_names
    
    n_models = len(npz_names)
    model_colors = ['r','g','b','c','m','y']

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
    plt.savefig(f'{figname}.jpeg')
    plt.close()



def run_multistep_plot():
    modules = ["enc", "reg", "dis"]
    steplabels = ["lr=0.003", "lr=0.001", "lr=0.003"]
    combos = [["p21c", "zreion"], ["p21c", "ctrpx"], ["zreion", "ctrpx"]]
    allsims = ["p21c", "zreion", "ctrpx"]

    for sims in combos:
        for alpha in [0.01, 0.05, 0.1]:
            for ws in [0.0, 3.0]:
                names = [
                    f"adversarial_v02_{sims[0]}_{sims[1]}_alpha{alpha}_lr0.003_ws{ws}",
                    f"adversarial_v02_{sims[0]}_{sims[1]}_alpha{alpha}_lr0.001_ws{ws}_s02",
                    f"adversarial_v02_{sims[0]}_{sims[1]}_alpha{alpha}_lr0.003_ws{ws}_s03",
                ]
                lossdict = {}
                for module in modules:
                    lossdict[module] = {}
                    npz_names = [f"models/{name}/{name}_{module}_loss.npz" for name in names]
                    lossdict[module]["train"]=np.array([])
                    lossdict[module]["val"]=np.array([])
                    steps = []
                    for npz_name in npz_names:
                        with np.load(npz_name) as data:
                            lossdict[module]["train"] = np.concatenate([lossdict[module]["train"], data["train"]])
                            lossdict[module]["val"] = np.concatenate([lossdict[module]["val"], data["val"]])
                            steps.append(len(lossdict[module]["train"]))
    
                
                fname = f"models/{names[-1]}/{names[-1]}_all_loss_all_steps.png"
                fname2 = f"models/plots/{names[-1]}/{names[-1]}_all_loss_all_steps.png"
                title = f"{names[-1]} Loss"
    
                plot_loss_grid(lossdict, fname, title, steps=steps, steplabels=steplabels)
                plot_loss_grid(lossdict, fname2, title, steps=steps, steplabels=steplabels)
                
                npz_names = [f"models/{name}/pred_{name}_on_{s}_test.npz" for s in allsims]
    
                figname=f"models/{name}/duration_{name}_p21c_zreion_ctrpx.jpeg"
                figname2=f"models/plots/{name}/duration_{name}_p21c_zreion_ctrpx.jpeg"
                
                plot_model_predictions(npz_names, figname, "dur", allsims, name)
                plot_model_predictions(npz_names, figname2, "dur", allsims, name)
                



def run_logloss_plots():
    #combos = [["p21c", "zreion"], ["p21c", "ctrpx"], ["zreion", "ctrpx"]]
    combos = [["zreion", "ctrpx"],]

    for sims in combos:
        for ws in [0.0, 3.0]:
            lossdict = {}
            lossdict["train"]=np.array([])
            lossdict["val"]=np.array([])
            
            names = [f"adversarial_null_v01_{sims[0]}_{sims[1]}_ws{ws}_s03",]
            npz_names = [f"models/{name}/{name}_reg_loss.npz" for name in names]
            
            for npz_name in npz_names:
                with np.load(npz_name) as data:
                    lossdict["train"] = np.concatenate([lossdict["train"], data["train"]])
                    lossdict["val"] = np.concatenate([lossdict["val"], data["val"]])

            
            fname = f"models/plots/{names[-1]}/{names[-1]}_logloss.png"
            title = f"{names[-1]} Loss"

            plot_loss(lossdict, fname, title, transform="log")



if __name__=='__main__':
    run_logloss_plots()