import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import py21cmfast as p21c




####
#
# Plots a distribution given a dictionary of {parameter : list of values}
#
#####
def plot_distribution(simname, paramdict):
    fig, axs = plt.subplots(1, len(paramdict), sharey=True, tight_layout=True)
    fig.suptitle(f"Parameter distribution: {simname}")

    for i, (p, d) in enumerate(paramdict.items()):
        axs[i].grid(True)
        axs[i].hist(d, bins=20)
        axs[i].title.set_text(p)
        
    plt.savefig(f"{simname}_param_dist.png")


####
#
# Plots a distribution given a dictionary of {parameter : {sim: list of values}}
#
#####
def plot_multi_distribution(fname, paramdict):
    fig, axs = plt.subplots(1, len(paramdict), sharey=True, tight_layout=True)
    fig.suptitle(f"Parameter distribution")

    for i, (p, simdict) in enumerate(paramdict.items()):
        axs[i].grid(True)
        for label, data in simdict.items():
            axs[i].hist(data, alpha=0.5, label=label)
        axs[i].title.set_text(p)
        axs[i].legend()

    plt.savefig(fname)


####
#
# Makes a bunch of plots on how the parameters in x affect the parameters in y
#
#####
def plot_scatter_grid(simname, xdict, ydict):
    for xlabel, xdata in xdict.items():
        rows = len(ydict)
        fig, axs = plt.subplots(rows, 1, sharex=True, tight_layout=True)
        fig.suptitle(f"Effect of {xlabel} on {simname} ")
        for r, (ylabel, ydata) in enumerate(ydict.items()):
            axs[r].grid(True)
            axs[r].scatter(xdata, ydata, marker='.')
            axs[r].set_ylabel(f"{ylabel}")
        plt.savefig(f"{simname}_{xlabel}_scatter.png")

####
#
# Gets a distribution from an hdf5 file
#
#####
def get_outparams_from_hdf5(fname, param_index):
    paramdict = {}
    with h5py.File(fname, 'r') as f:
        for k, v in param_index.items():
            paramdict[k] = f['lightcone_params/physparams'][:,v]
    return paramdict


####
#
# Gets a distribution from a list of lightcones
#
#####
def get_outparams_from_lightcones(lightcones, param_index):
    n = len(lightcones)
    paramdict = {k : np.zeros((n,)) for k in param_index}

    for i in range(n):
        if i%10 == 0: print(f"Finding output params from {i} of {n} lightcones...")
        lc = p21c.outputs.LightCone.read(lightcones[i], lc_path)
        z25, z50, z75 = np.interp([0.25, 0.5, 0.75], np.flip(lc.global_xH), np.flip(lc.node_redshifts))
        params = [z50, z75-z25, (z75+z25)/2]
        for k, v in param_index.items():
            paramdict[k][i] = params[v]
    return paramdict



####
#
# Gets a distribution of input params from a list of lightcones
#
#####
def get_inparams_from_hdf5_zreion(fname, inpt_param_index):
    paramdict = {}
    with h5py.File(fname, 'r') as f:
        for k, v in inpt_param_index.items():
            paramdict[k] = f['lightcone_params/input_params'][:,v]
    return paramdict


####
#
# Gets a distribution of input params from a list of lightcones
#
#####
def get_inparams_from_lightcones(lightcones, inpt_param_list):

    paramdict = {p:[] for p in inpt_param_list.keys()}
    n=len(lightcones)
    for i in range(n):
        if i%10 == 0: print(f"Finding input params from {i} of {n} lightcones...")

        lc = p21c.outputs.LightCone.read(lightcones[i], lc_path)
        for p, linlog in inpt_param_list.items():
            if linlog == 'log':
                paramdict[p].append(np.log10(lc.astro_params.defining_dict[p]))
            else:
                paramdict[p].append(lc.astro_params.defining_dict[p])
    return paramdict


    
##############################################################################################################

####
#
# Run program
#
#####
if __name__ == '__main__':
    param_index = {'midpoint' : 0, 'duration' : 1}
    '''
    inpt_param_list = {
                        'F_STAR10':'log',
                        'ALPHA_STAR':'lin',
                        'F_ESC10':'log',
                        'ALPHA_ESC':'lin',
                        'M_TURN':'log',
                        'L_X':'log',
                        't_STAR':'lin',
                        'NU_X_THRESH':'lin',
                    }
    '''
    inpt_param_index = {
                            'zmean':0,
                            'alpha':1,
                            'k0':2,
                        }
    fname_dict = {
        "21cmFAST (full-sphere)":"/users/jsolt/data/jsolt/21cmFAST_sims/p21c14/p21c14_ws0.0_trnspsd_meanremoved_norm.hdf5",
        "21cmFAST (central-pixel)":"/users/jsolt/data/jsolt/21cmFAST_centralpix_v04/21cmFAST_centralpix_v04_processed_ws0.0.hdf5",
        "zreion":"/users/jsolt/data/jsolt/zreion_sims/zreion21/zreion21_processed_ws0.0.hdf5",
    }
    paramdict = {p:{} for p in param_index.keys()}
    for sim, fname in fname_dict.items():
        temp = get_outparams_from_hdf5(fname, param_index)
        for p in param_index.keys():
            paramdict[p][sim] = temp[p]

    
    abrev_list = ["p21c", "ctrpx", "zreion"]
    plot_fname = f"{'_'.join(abrev_list)}_param_dist.png"

    plot_multi_distribution(plot_fname, paramdict)