import os
import numpy as np
import glob
import h5py
import argparse
from zreion import apply_zreion
import py21cmfast as p21c
from astropy.cosmology import z_at_value
from tools import xHcube, bTcube, t0
from scipy.stats import truncnorm


default_dict = { 
                'zmean': 8.0,
                'k0': 0.5, #0.185,
                'alpha': 0.564,
            }

bound_dict = {
                'zmean': (7.0, 13.0),
                'k0':(0.1, 1.0),
                'alpha':(-0.2, 2.0), #'alpha':(0.2, 2.0),
            }


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def random_params():
    params = {}
    for key, val in default_dict.items():
        low, upp = bound_dict[key]
        sigma = upp-low
        X = get_truncated_normal(mean=val, sd=sigma, low=low, upp=upp)
        params[key] = X.rvs()
    return params

def set_up_database(fname, n, shape, test_lc):
    print("Setting up database...")
    _,_,z = shape
    
    with h5py.File(fname, "w") as f:
        f.create_group('lightcones')
        f['lightcones'].create_dataset("brightness_temp", (n,) + shape)
        f['lightcones'].create_dataset("zreion_output", (n,) + shape)
        f['lightcones'].create_dataset("density", (n,) + shape)
        f['lightcones'].create_dataset("xH_box", (n,) + shape)
        
        f.create_group('lightcone_params')
        f['lightcone_params'].create_dataset("source_lightcone_file", (n,), dtype=h5py.string_dtype(encoding='utf-8'))
        f['lightcone_params'].create_dataset("random_seed", (n,))
        f['lightcone_params'].create_dataset("input_params", (n, 3))
        f['lightcone_params'].create_dataset("global_xH", (n, z))
        f['lightcone_params'].create_dataset("physparams", (n, 3))
    
        f.attrs['cell_size'] = test_lc.cell_size
        f.attrs['lightcone_coords'] = test_lc.lightcone_coords
        f.attrs['lightcone_dimensions'] = test_lc.lightcone_dimensions
        f.attrs['lightcone_distances'] = test_lc.lightcone_distances
        f.attrs['lightcone_redshifts'] = np.load("redshifts.npy")[:z]
        f.attrs['node_redshifts'] = test_lc.node_redshifts
        f.attrs['shape'] = test_lc.shape


def run_zreion(fname, lightcones):    
    #iterate and save, redoing lightcones if necessary
    i, good_lc, bad_lc = 0, 0, 0
    while i < len(lightcones):

        if i%10 == 0: print(f"{good_lc} lightcones saved, {bad_lc} lightcones discarded")

        with h5py.File(lightcones[i], 'r') as f:
            density = f['lightcones/density'][:,:,:]
            random_seed = f.attrs['random_seed']
            if i == 0: 
                _,_,z=density.shape
                redshifts = np.load('redshifts.npy')[:z]

            
        #run zreion
        param_dict = random_params()

        zre_cube = apply_zreion(
                            density=density, 
                            zmean=param_dict['zmean'],
                            alpha=param_dict['alpha'], 
                            k0=param_dict['k0'],
                            boxsize= 1000 
                            )
        
        #find xH_box, global_xH, and physparams
        xH_cube = xHcube(zre_cube, redshifts)

        global_xH = np.mean(xH_cube, axis=(0,1))

        z25, z50, z75 = np.interp([0.25, 0.5, 0.75], global_xH, redshifts)
        mdpt, dur, meanz = z50, z75-z25, (z75+z25)/2

        print(f"zmean: {param_dict['zmean']:.2f}, alpha: {param_dict['alpha']:.4f}, k0 = {param_dict['k0']:.2f}")
        print(f"midpoint: {mdpt:.2f}, duration: {dur:.4f}, zmean = {meanz:.2f}")
        print()
        
        #if lightcone is 'good',
        if z25 > redshifts[0] and z75 < redshifts[-1]:
            #find bT box
            bT_cube = bTcube(density, xH_cube, redshifts)
            
            #save everything
            with h5py.File(fname,"a") as f:
                f['lightcones/density'][i] = density
                f['lightcones/zreion_output'][i,:,:,:] = zre_cube
                f['lightcones/xH_box'][i,:,:,:] = xH_cube
                f['lightcones/brightness_temp'][i,:,:,:] = bT_cube
    
                f['lightcone_params/source_lightcone_file'][i] = lightcones[i]
                f['lightcone_params/random_seed'][i] = random_seed
                f['lightcone_params/input_params'][i,:] = [param_dict['zmean'], param_dict['alpha'], param_dict['k0']]
                f['lightcone_params/global_xH'][i] = global_xH
                f['lightcone_params/physparams'][i,:] = z50, z75-z25, (z75+z25)/2
            good_lc += 1
        else:
            #if lightcone is not good, try again
            lightcones.append(lightcones[i])
            bad_lc += 1
        i += 1

            

    
if __name__ == "__main__":
    #simulation parameters
    #sim_name = "zreion22"
    #lc_path = '/users/jsolt/data/jsolt/zreion_density_lightcones_v01'

    lightcones = glob.glob(f"{lc_path}/Light*")
    
    savedir = f"/users/jsolt/data/jsolt/zreion_sims/{sim_name}"
    if not os.path.isdir(savedir): os.mkdir(savedir)
    
    fname = f"{savedir}/{sim_name}.hdf5"
    
    n = len(lightcones)
    shape = (256, 256, 552)

    test_lc_dir = "/users/jsolt/data/jsolt/21cmFAST_lightcones_centralpix_v04"
    test_lc = p21c.outputs.LightCone.read(os.listdir(test_lc_dir)[0], test_lc_dir)

    set_up_database(fname, n, shape, test_lc)
    run_zreion(fname, lightcones)
