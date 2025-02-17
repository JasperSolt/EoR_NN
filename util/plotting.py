import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import colors
matplotlib.use('AGG')

import numpy as np
from numpy.typing import NDArray

import pandas as pd




def plot_loss(loss, fname, title="",start=10, steps=[], steplabels=[], transform=""):    
    ax = plt.subplot()
    ax.grid(True)

    ax.set_ylabel('Loss')
    train_loss, val_loss = np.array(loss['train'])[start:], np.array(loss['val'])[start:]
    
    if transform == "log":
        train_loss = np.log10(train_loss)
        val_loss = np.log10(val_loss)
        ax.set_ylabel('Loss (log)')

    train_loss = pd.DataFrame(train_loss).ewm(com=5.0).mean()
    val_loss = pd.DataFrame(val_loss).ewm(com=5.0).mean()
        
        
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
    plt.close()





def plot_loss_grid(lossdict, fname, title="", start=10, steps=[], steplabels=[], transform=""):
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
        else:
            train_loss = pd.DataFrame(loss['train'][start:]).ewm(com=5.0).mean()
            val_loss = pd.DataFrame(loss['val'][start:]).ewm(com=5.0).mean()
            axs[r].set_ylabel('MSE Loss')

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


def plot_loss_comparison(loss, fname, title="", start=10, steps=[], steplabels=[], transform=None, ylabel="Loss"):    
    ax = plt.subplot()
    ax.grid(True)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Epochs')

    data = {k: v[start:] for k, v in loss.items()}
    if transform:
        data = {k: transform(v) for k, v in data.items()}
    data = {k: pd.DataFrame(v).ewm(com=5.0).mean() for k, v in data.items()}

    
    epochs = np.arange(start+1, len(data[list(data.keys())[0]])+start+1)

    for k, v in data.items():
        ax.plot(epochs, v, label=k, linewidth=0.7)
    
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
    
    ax.legend()
    ax.set_title(title)
    
    plt.savefig(fname)
    print("Loss plot saved.")
    plt.close()



def plot_model_predictions(npz_names, figname, param, labels=None, title=None):
    if not labels:
        labels = npz_names
    
    n_models = len(npz_names)
    model_colors = ['r','g','b','c','m','y']

    if not title:
        title = param

    print(f"Plotting results for parameter: {param}...")

    #Layout
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
    ax1.legend(recs, labels, fontsize=10)

    #ax2 formatting
    ax2.locator_params(nbins=5)
    ax2.set_ylabel(r'% error')
    ax2.set_xlabel(f'True {param}')
    ax2.grid(True)

    
    #load and plot the parameter prediction data for each model
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
    
    #find axis limits 
    mintargets, maxtargets = np.min(targets), np.max(targets)
    minpred, maxpred = np.min(pred), np.max(pred)
    minerr, maxerr = np.min(err), np.max(err)

    ax1.set_xlim(0.95*mintargets,1.05*maxtargets)
    ax1.set_ylim(minpred*0.95,maxpred*1.05)

    ax2.set_ylim(minerr*0.95,maxerr*1.05)
    ax2.set_xlim(0.95*mintargets,1.05*maxtargets)

    #Plot target line
    ideal = np.linspace(0.8*mintargets,1.2*maxtargets,10)
    ax1.plot(ideal, ideal, 'k--', alpha=0.3, linewidth=1.5)
    ax2.plot(ideal,len(ideal)*[0.],'k--', alpha=0.3, linewidth=1.5)

    #Save
    plt.tight_layout()
    plt.savefig(f'{figname}.jpeg')
    plt.close()





def plot_image_rows(rowdict: dict[str, NDArray[np.float32]], 
                    title: str=None, 
                    fname: str=None,
                    collabels: str=None,
                    **kwargs):
    """Saves a plot of rows of images. 
    This function is for comparing rows of images from different sets of data. Use plot_image_grid if you just want to plot lots of images in a grid.
    """
    rowlabels = list(rowdict.keys())
    rows = [rowdict[k] for k in rowlabels]
    
    nrows = len(rows)
    ncols, _, _ = rows[0].shape

    
    imsize = 1.3
    w = ncols*imsize + 0.2
    h = nrows*imsize + 0.9
    
    
    fig, axs = plt.subplots(nrows, ncols, 
                            figsize = (w,h),
                            layout='constrained'
                           )
    if title: fig.suptitle(title)
    
    axs = axs.reshape((nrows,ncols))
    images = [axs[r,c].imshow(rows[r][c]) for r in range(nrows) for c in range(ncols)]

    for ax in axs.flatten():
        ax.set_yticks([])
        ax.set_xticks([])


    if collabels is not None: 
        for c in range(ncols):
            axs[0,c].set_title(collabels[c], fontsize=10)

    for r in range(nrows):
        axs[r,0].set_ylabel(rowlabels[r], fontsize=10)

    
    vmin = kwargs.get('vmin', min(image.get_array().min() for image in images))
    vmax = kwargs.get('vmax', max(image.get_array().max() for image in images))
        
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images: im.set_norm(norm)
    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=0.05, aspect=60)
    
    if fname:
        plt.savefig(fname)
    else:
        plt.show()
    plt.close()


def calc_aspect_ratio(n, y, x):
    cols, rows, i = 1, n, 0
    h, w = y*rows, x*cols
    while w < h:
        i += 1
        if n % i == 0:
            cols = i
            rows = n//cols
            h, w = y*rows, x*cols
    return cols, rows


def plot_image_grid(imgrid:NDArray, title:str=None, fname:str=None, fnames:list[str]=None, colorbar:bool=True, ylbl:bool=True):
    cols, rows = calc_aspect_ratio(*imgrid.shape)

    fig, axs = plt.subplots(rows, cols, layout='constrained')
    if title: fig.suptitle(title)
    images = []

    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            ax = axs[r,c]
            
            images.append(ax.imshow(imgrid[i]))
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        if ylbl:
            axs[r,0].set_ylabel(f"{r*cols}-{(r+1)*cols-1}", x=0, y=0.5, 
                             horizontalalignment='right', 
                             verticalalignment='center',
                             rotation=0)

    ###
    # Colorbar
    ###
    if colorbar:
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
        plt.colorbar(images[0], ax=axs)

    if fname:
        plt.savefig(fname, dpi=300)
    else:
        if fnames:
            for f in fnames: plt.savefig(f, dpi=300)
        else:
            plt.show()
    plt.close()






def plot_cf_loss(loss_dict, fname:str=None, fnames:list[str]=None, title="", start=0, steps=[], steplabels=[]):    
    ax = plt.subplot()
    ax.grid(True)
    ax.set_ylabel('Loss')


    for k in loss_dict.keys():
        loss = np.array(loss_dict[k])[start:]
        loss = pd.DataFrame(loss).ewm(com=5.0).mean()
        
        epochs = np.arange(start+1, len(loss)+start+1)

        ax.plot(epochs, loss, label=k, linewidth=1.0)
    
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
    
    ax.set_xlabel('Epochs')
    ax.set_title(title)
    ax.legend()
    if fname:
        plt.savefig(fname)
    else:
        if fnames:
            for f in fnames: plt.savefig(f)
        else:
            plt.show()
    plt.close()

def plot_counterfactuals(input_row: NDArray[np.float32], 
                         cf_row: NDArray[np.float32], 
                         diff_row: NDArray[np.float32], 
                         title: str=None, 
                         fname: str=None,
                         rowlabels: str=None,
                         collabels: str=None,
                         **kwargs):

    nrows = 3
    ncols, _, _ = input_row.shape

    
    imsize = 1.3
    w = ncols*imsize + 0.2
    h = nrows*imsize + 1.4
    
    
    fig, axs = plt.subplots(nrows, ncols, 
                            figsize = (w,h),
                            layout='constrained'
                           )
    if title: fig.suptitle(title)
    
    axs = axs.reshape((nrows,ncols))
    
    colorbar_group_1 = []
    colorbar_group_1.extend([axs[0,c].imshow(input_row[c]) for c in range(ncols)])
    colorbar_group_1.extend([axs[1,c].imshow(cf_row[c]) for c in range(ncols)])

    colorbar_group_2 = []
    colorbar_group_2.extend([axs[2,c].imshow(diff_row[c]) for c in range(ncols)])

    for ax in axs.flatten():
        ax.set_yticks([])
        ax.set_xticks([])

    if collabels is not None: 
        for c in range(ncols):
            axs[0,c].set_title(collabels[c], fontsize=10)

    for r in range(nrows):
        axs[r,0].set_ylabel(rowlabels[r], fontsize=10)

    
    # colorbar 1
    vmin = min(image.get_array().min() for image in colorbar_group_1)
    vmax = max(image.get_array().max() for image in colorbar_group_1)
        
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in colorbar_group_1: im.set_norm(norm)
    fig.colorbar(colorbar_group_1[0], ax=axs[1], orientation='horizontal', fraction=0.1, aspect=60)
    
    # colorbar 2
    vmin = min(image.get_array().min() for image in colorbar_group_2)
    vmax = max(image.get_array().max() for image in colorbar_group_2)
        
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in colorbar_group_2: im.set_norm(norm)
    fig.colorbar(colorbar_group_2[0], ax=axs[2], orientation='horizontal', fraction=0.1, aspect=60)
    

    if fname:
        plt.savefig(fname, dpi=300)
    else:
        plt.show()
    plt.close()
