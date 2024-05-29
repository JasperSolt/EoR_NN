import os
import shutil
import h5py
import argparse
import numpy as np
import py21cmfast as p21c
from tools import find_filter, apply_constant_wedge_filter

def save_filters(slopes, shape, filename):
    filtrs = {}
    for m in slopes:
        filtrs[m] = find_filter(shape, m)
    np.savez(filename, filtrs=filtrs) 

def save_good_lightcones_list(lightcones, lc_path):
    good_lc = []
    redshifts=np.load('redshifts.npy')

    for i, lc in enumerate(lightcones):
        if i%10 == 0: print(f"Lightcone {i} of {len(lightcones)} (Good light cones so far: {len(good_lc)})")
    
        lightcone = p21c.outputs.LightCone.read(lc, lc_path)
        z25, z50, z75 = np.interp([0.25, 0.5, 0.75], np.flip(lightcone.global_xH), np.flip(lightcone.node_redshifts))
        if z25 > redshifts[0] and z75 < redshifts[-1]:
            good_lc.append(lc)
            
    with open(f"good_lightcones.txt", "w") as txt_file:
        for fn in good_lc:
            txt_file.write(fn + "\n")
    
    print(f"{len(good_lc)} lightcone filenames saved to {lc_path}/good_lightcones.txt")
    return good_lc

def load_good_lightcones_list():
    good_lc = []
    with open(f"good_lightcones.txt", 'r') as txt_file:
        for x in txt_file:
            fn = x.replace('\n','')
            good_lc.append(fn)
    return good_lc

def set_up_database(fname, n, shape, test_lc):
    print("Setting up database...")
    n_nodes = len(test_lc.node_redshifts)

    with h5py.File(fname, "w") as f:
        f.create_group('lightcones')
    
        f['lightcones'].create_dataset("density", (n,) + shape)
        f['lightcones'].create_dataset("xH_box", (n,) + shape)
        f['lightcones'].create_dataset("brightness_temp", (n,) + shape)

        f.create_group('lightcone_params')

        f['lightcone_params'].create_dataset("physparams", (n, 3))
        f['lightcone_params'].create_dataset("source_lightcone_file", (n,), dtype=h5py.string_dtype(encoding='utf-8'))
        f['lightcone_params'].create_dataset("astro_params", (n,), dtype=h5py.string_dtype(encoding='utf-8'))
        f['lightcone_params'].create_dataset("random_seed", (n,))
        f['lightcone_params'].create_dataset("global_brightness_temp", (n, n_nodes))
        f['lightcone_params'].create_dataset("global_density", (n, n_nodes))
        f['lightcone_params'].create_dataset("global_xH", (n, n_nodes))
        
        f.attrs['user_params'] = str(test_lc.user_params)
        f.attrs['cosmo_params'] = str(test_lc.cosmo_params)
        f.attrs['flag_options'] = str(test_lc.flag_options)
        f.attrs['global_params'] = str(test_lc.global_params)
        f.attrs['cell_size'] = test_lc.cell_size
        f.attrs['lightcone_coords'] = test_lc.lightcone_coords
        f.attrs['lightcone_dimensions'] = test_lc.lightcone_dimensions
        f.attrs['lightcone_distances'] = test_lc.lightcone_distances
        f.attrs['lightcone_redshifts'] = np.load("redshifts.npy")[:]
        f.attrs['node_redshifts'] = test_lc.node_redshifts
        f.attrs['shape'] = test_lc.shape
        
def save_lightcones_to_hdf5(fname, lightcones, lc_path):
    test_lc = p21c.outputs.LightCone.read(lightcones[0], lc_path)
    n = len(lightcones)
    shape = test_lc.shape
    
    set_up_database(fname, n, shape, test_lc)
    
    with h5py.File(fname, "a") as f:
        for i in range(n):
            if i%10==0: print(f"Saving lightcone {i} of {n}...")
            lightcone = p21c.outputs.LightCone.read(lightcones[i], lc_path)
            
            #save lightcones
            f['lightcones/brightness_temp'][i] = lightcone.lightcones['brightness_temp']
            f['lightcones/density'][i] = lightcone.lightcones['density']
            f['lightcones/xH_box'][i] = lightcone.lightcones['xH_box']
            
            #save params
            f['lightcone_params/source_lightcone_file'][i] = lightcones[i]
            f['lightcone_params/astro_params'][i] = str(lightcone.astro_params)
            f['lightcone_params/random_seed'][i] = lightcone.random_seed
            f['lightcone_params/global_brightness_temp'][i] = lightcone.global_brightness_temp
            f['lightcone_params/global_density'][i] = lightcone.global_density
            f['lightcone_params/global_xH'][i] = lightcone.global_xH
            
            z25, z50, z75 = np.interp([0.25, 0.5, 0.75], np.flip(lightcone.global_xH), np.flip(lightcone.node_redshifts))

            f['lightcone_params/physparams'][i,:] = [z50, z75-z25, (z75+z25)/2]

            
    

def process_lightcones(src, dest, shape, ws): 
    if ws > 0.0:
        npfile = np.load("wedge_filter_256.npz", allow_pickle=True)
        filtr = npfile['filtrs'].item()[ws]
    x,y,z = shape
    
    with h5py.File(src, "r") as f1, h5py.File(dest, "w") as f2:
        f2.create_group('lightcone_params')
        f2["lightcone_params/physparams"] = f1["lightcone_params/physparams"][:]
        f2["lightcone_params/source_lightcone_file"] = f1["lightcone_params/source_lightcone_file"][:]
        
        f2.create_group('lightcones')
        n,_,_,_ = f1['lightcones/brightness_temp'].shape
        f2['lightcones'].create_dataset("brightness_temp", (n,) + shape)

        for i in range(n):
            if i%10==0: print(f"Processing lightcone {i} of {n}...")
    
            ###
            # Mean Removal
            ###
            bT_cube = f1['lightcones/brightness_temp'][i,:x,:y,:z]
            bT_cube = bT_cube - np.mean(bT_cube, axis=(0,1)) # remove mean
    
            ###
            # Wedge
            ###
            if ws > 0.0: bT_cube = apply_constant_wedge_filter(bT_cube, filtr) #apply wedge 
    
            ###
            # Normalization
            ###
            minn, maxx = np.min(bT_cube), np.max(bT_cube)
            bT_cube_norm = (bT_cube - minn) / (maxx - minn)
            f2['lightcones/brightness_temp'][i] = bT_cube_norm


def transpose_lightcones(src, dest):
    with h5py.File(src, "r") as f1, h5py.File(dest, "w") as f2:
        f2.create_group('lightcone_params')
        f2["lightcone_params/physparams"] = f1["lightcone_params/physparams"][:]
        f2["lightcone_params/source_lightcone_file"] = f1["lightcone_params/source_lightcone_file"][:]
        
        f2.create_group('lightcones')
        n, x, y, z = f1['lightcones/brightness_temp'].shape
        newshape = (z, x, y)
        f2['lightcones'].create_dataset("brightness_temp", (n,) + newshape)

        for i in range(n):
            if i%10==0: print(f"Transposing lightcone {i} of {n}...")
            f2['lightcones/brightness_temp'][i] = np.transpose(f1['lightcones/brightness_temp'][i], (2,0,1))



if __name__ == "__main__":
    
    lcdir = "/users/jsolt/data/jsolt/21cmFAST_lightcones_centralpix_v04"
    #good_lc = save_good_lightcones_list(os.listdir(lcdir), lcdir)

    #simname = "21cmFAST_centralpix_v04"
    #savedir = f"/users/jsolt/data/jsolt/{simname}"
    simname = "zreion21"
    savedir = f"/users/jsolt/data/jsolt/zreion_sims/{simname}"
    if not os.path.isdir(savedir): os.mkdir(savedir)

    ws=3.0
    fname1 = f"{savedir}/{simname}.hdf5"
    fname2 = f"{savedir}/{simname}_processed_ws{ws}.hdf5"
    fname3 = f"{savedir}/{simname}_transposed_ws{ws}.hdf5"
    

    # Collate lightcones into full hdf5 file
    #save_lightcones_to_hdf5(fname1, good_lc, lcdir)

    # make a reduced, processed file
    reduced_shape = (256, 256, 512)
    
    process_lightcones(fname1, fname2, reduced_shape, ws)
    
    # Transpose data
    transpose_lightcones(fname2, fname3)
