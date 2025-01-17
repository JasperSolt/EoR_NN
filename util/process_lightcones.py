import os
import h5py
import numpy as np
import torch
import argparse
import py21cmfast as p21c
from diffusers import AutoencoderKL

from tools import find_filter, apply_constant_wedge_filter

def save_filters(slopes, shape, filename):
    filtrs = {}
    for m in slopes:
        filtrs[m] = find_filter(shape, m)
    np.savez(filename, filtrs=filtrs) 



def save_good_lightcones_list(lc_path):
    lightcones = os.listdir(lc_path)
    good_lc = []

    for i, lc in enumerate(lightcones):
        if i%10 == 0: print(f"Lightcone {i} of {len(lightcones)} (Good light cones so far: {len(good_lc)})")
    
        with h5py.File(f"{lc_path}/{lc}", 'r') as f:
            global_xH = np.flip(f['global_quantities/xH_box'][:])
            node_redshifts = np.flip(f['node_redshifts'][:])
        z25, z50, z75 = np.interp([0.25, 0.5, 0.75], global_xH, node_redshifts)
        if z25 > node_redshifts[0] and z75 < node_redshifts[-1]:
            good_lc.append(lc)
    
    txt_pathname = f"{lc_path}/good_lightcones.txt"
    with open(txt_pathname, "w") as txt_file:
        for fn in good_lc:
            txt_file.write(fn + "\n")
    
    print(f"{len(good_lc)} lightcone filenames saved to {txt_pathname}")
    return good_lc




def load_good_lightcones_list():
    good_lc = []
    with open(f"good_lightcones.txt", 'r') as txt_file:
        for x in txt_file:
            fn = x.replace('\n','')
            good_lc.append(fn)
    return good_lc






def set_up_database(fname, n, test_lc, shape=None, lite=False):
    print("Setting up database...")

    shape = test_lc.shape if not shape else shape
    n_nodes = len(test_lc.node_redshifts)

    with h5py.File(fname, "w") as f:
        f.create_group('lightcones')
        f['lightcones'].create_dataset("brightness_temp", (n,) + shape, dtype=np.float32)

        f.create_group('lightcone_params')
        f['lightcone_params'].create_dataset("physparams", (n, 3))
        f['lightcone_params'].create_dataset("source_lightcone_file", (n,), dtype=h5py.string_dtype(encoding='utf-8'))
        
        if not lite:
            f['lightcones'].create_dataset("density", (n,) + shape, dtype=np.float32)
            f['lightcones'].create_dataset("xH_box", (n,) + shape, dtype=np.float32)

            f['lightcone_params'].create_dataset("astro_params", (n,), dtype=h5py.string_dtype(encoding='utf-8'))
            f['lightcone_params'].create_dataset("random_seed", (n,))
            f['lightcone_params'].create_dataset("global_brightness_temp", (n, n_nodes))
            f['lightcone_params'].create_dataset("global_density", (n, n_nodes))
            f['lightcone_params'].create_dataset("global_xH", (n, n_nodes))
        
            f.attrs['lightcone_redshifts'] = np.load("redshifts.npy")[:]
            f.attrs['node_redshifts'] = test_lc.node_redshifts
            f.attrs['shape'] = shape
            f.attrs['user_params'] = str(test_lc.user_params)
            f.attrs['cosmo_params'] = str(test_lc.cosmo_params)
            f.attrs['flag_options'] = str(test_lc.flag_options)
            f.attrs['global_params'] = str(test_lc.global_params)
            f.attrs['cell_size'] = test_lc.cell_size
            f.attrs['lightcone_coords'] = test_lc.lightcone_coords
            f.attrs['lightcone_dimensions'] = test_lc.lightcone_dimensions
            f.attrs['lightcone_distances'] = test_lc.lightcone_distances

        






def save_lightcones_to_hdf5(fname, lightcones, lc_path, lite=False):
    test_lc = p21c.outputs.LightCone.read(lightcones[0], lc_path)
    n = len(lightcones)
    
    set_up_database(fname, n, test_lc, lite)
    
    with h5py.File(fname, "a") as f:
        for i in range(n):
            if i%10==0: print(f"Saving lightcone {i} of {n}...")
            lightcone = p21c.outputs.LightCone.read(lightcones[i], lc_path)
            
            #save lightcones
            f['lightcones/brightness_temp'][i] = lightcone.lightcones['brightness_temp']
            if not lite:
                f['lightcones/density'][i] = lightcone.lightcones['density']
                f['lightcones/xH_box'][i] = lightcone.lightcones['xH_box']
            
            #save params
            z25, z50, z75 = np.interp([0.25, 0.5, 0.75], np.flip(lightcone.global_xH), np.flip(lightcone.node_redshifts))
            f['lightcone_params/physparams'][i,:] = [z50, z75-z25, (z75+z25)/2]
            f['lightcone_params/source_lightcone_file'][i] = lightcones[i]

            if not lite:
                f['lightcone_params/astro_params'][i] = str(lightcone.astro_params)
                f['lightcone_params/random_seed'][i] = lightcone.random_seed
                f['lightcone_params/global_brightness_temp'][i] = lightcone.global_brightness_temp
                f['lightcone_params/global_density'][i] = lightcone.global_density
                f['lightcone_params/global_xH'][i] = lightcone.global_xH


            
    

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
        f2['lightcones'].create_dataset("brightness_temp", (n,) + shape, dtype=np.float32)

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
        f2['lightcones'].create_dataset("brightness_temp", (n,) + newshape, dtype=np.float32)

        for i in range(n):
            if i%10==0: print(f"Transposing lightcone {i} of {n}...")
            f2['lightcones/brightness_temp'][i] = np.transpose(f1['lightcones/brightness_temp'][i], (2,0,1))



def process_lightcones_subdiv(src, dest, new_xy, new_z, test=False, zoomin=False): 
    with h5py.File(src, "r") as f1, h5py.File(dest, "w") as f2:
        n, z, xy, _ = f1['lightcones/brightness_temp'].shape
        if test: n = 3
        subdiv = int(xy // new_xy)

        new_n = n*subdiv**2 
        newshape = (new_n, new_z, new_xy, new_xy)

        f2.create_group('lightcone_params')
        f2['lightcone_params'].create_dataset("physparams", (new_n, 3))
        f2['lightcone_params'].create_dataset("source_lightcone_file", (new_n,), dtype=h5py.string_dtype(encoding='utf-8'))

        f2.create_group('lightcones')
        f2['lightcones'].create_dataset("brightness_temp", newshape, dtype=np.float32)

        redshifts = np.load("redshifts.npy")[:z]
        zmin, zmax = min(redshifts), max(redshifts)

        for l1 in range(n):
            ###
            # Split into slices
            ###
            if zoomin:
                mdpt, dur, _ = f1['lightcone_params/physparams'][l1,:]
                zindrange = z_to_i(np.array([mdpt-dur, mdpt+dur]), zmin, zmax, 0, z-1)
            else:
                zindrange = 0, z-1
            zsliceinds = np.linspace(*zindrange, new_z, dtype=int)
            ###
            # subdivide 
            ###
            for i in range(subdiv):
                for j in range(subdiv):
                    l2 = l1*subdiv**2 + subdiv*j + i
                    if l2%50==0: print(f"Saving lightcone {l2} of {new_n}...")

                    f2['lightcones/brightness_temp'][l2] = f1['lightcones/brightness_temp'][l1, zsliceinds, i*new_xy:(i+1)*new_xy, j*new_xy:(j+1)*new_xy]
                    f2['lightcone_params/physparams'][l2] = f1['lightcone_params/physparams'][l1]
                    f2['lightcone_params/source_lightcone_file'][l2] = f1['lightcone_params/source_lightcone_file'][l1]


def z_to_i(zarr, zmin, zmax, imin, imax):
    iarr = ((zarr-zmin) / (zmax-zmin))*(imax-imin)+imin
    iarr = np.clip(iarr, imin, imax)
    return iarr






def process_all(fname, ws, new_z, new_xy, lightcones, debug=False):
    
    n = len(lightcones) if not debug else 3
    
    with h5py.File(lightcones[0]) as lc_file:
        oldshape = lc_file['lightcones/brightness_temp'].shape
        node_redshifts = np.flip(lc_file['node_redshifts'][:])

    old_xy, _, old_z = oldshape
    old_z = 512
    oldshape = old_xy, old_xy, old_z
    
    subdiv = int(old_xy // new_xy)
    new_n = n*subdiv**2 
    newshape = (new_n, new_z, new_xy, new_xy)
    
    if ws > 0.0:
        filter_npz = f"wedge_filter_xy{old_xy}_z{old_z}.npz"
        save_filters([ws], oldshape, filter_npz)
        npfile = np.load(filter_npz, allow_pickle=True)
        filtr = npfile['filtrs'].item()[ws]

    with h5py.File(fname, "w") as f:
        # Initialize file
        f.create_group('lightcone_params')
        f['lightcone_params'].create_dataset("physparams", (new_n, 3))
        f['lightcone_params'].create_dataset("source_lightcone_file", (new_n,), dtype=h5py.string_dtype(encoding='utf-8'))

        f.create_group('lightcones')
        f['lightcones'].create_dataset("brightness_temp", newshape, dtype=np.float32)

        for ni in range(n):
            if ni%10==0 or debug: print(f"Processing lightcone file {ni} of {n}...")

            # Load Lightcone
            with h5py.File(lightcones[ni], 'r') as lc_file:
                bT_cube = lc_file['lightcones/brightness_temp'][:,:,:old_z]
                global_xH = np.flip(lc_file['global_quantities/xH_box'][:])
            
            # Find params for later
            z25, z50, z75 = np.interp([0.25, 0.5, 0.75], global_xH, node_redshifts)
            
            # Mean Removal
            bT_cube = bT_cube - np.mean(bT_cube, axis=(0,1)) # remove mean
    
            # Wedge 
            if ws > 0.0: bT_cube = apply_constant_wedge_filter(bT_cube, filtr) #apply wedge 
            
            # Slice
            zsliceinds = np.linspace(0, old_z-1, new_z, dtype=int)
            bT_cube = bT_cube[:,:,zsliceinds]

            # Transpose
            bT_cube = np.transpose(bT_cube, (2,0,1))

            # Subdiv
            for i in range(subdiv):
                for j in range(subdiv):
                    l2 = ni*subdiv**2 + subdiv*j + i

                    #Save lightcone
                    f['lightcones/brightness_temp'][l2] = bT_cube[:, i*new_xy:(i+1)*new_xy, j*new_xy:(j+1)*new_xy]

                    #save params
                    f['lightcone_params/physparams'][l2,:] = [z50, z75-z25, (z75+z25)/2]
                    f['lightcone_params/source_lightcone_file'][l2] = lightcones[ni]





def find_good_lightcones_hdf5(src_path):
    low, high = 6.0, 16.06496507732184
    with h5py.File(f"{src_path}", 'r') as f:
        n = len(f['lightcones/brightness_temp'])
        good_inds = []
        for i in range(n):
            if i%10 == 0: print(f"Lightcone {i} of {n} (Good light cones so far: {len(good_inds)})")
        
            _, dur, meanz = f['lightcone_params/physparams'][i]
            z25, z75 = meanz - dur/2, meanz + dur/2
            if z25 > low and z75 < high:
                good_inds.append(i)
                if i%10 == 0: print(z25, z75)
    return good_inds
                




def process_all_from_hdf5(dest_fname, ws, new_z, new_xy, src_fname, debug=False):
    with h5py.File(src_fname) as src_file, h5py.File(dest_fname, "w") as dest_file:
        oldshape = src_file['lightcones/brightness_temp'].shape
        n, old_xy, _, old_z = oldshape

        subdiv = int(old_xy // new_xy)
        new_n = n*subdiv**2 
        newshape = (new_n, new_z, new_xy, new_xy)
        
        if ws > 0.0:
            filter_npz = f"wedge_filter_xy{old_xy}_z{old_z}.npz"
            save_filters([ws], (old_xy, old_xy, old_z), filter_npz)
            npfile = np.load(filter_npz, allow_pickle=True)
            filtr = npfile['filtrs'].item()[ws]

        # Initialize file
        dest_file.create_group('lightcone_params')
        dest_file["lightcone_params/physparams"] = src_file["lightcone_params/physparams"][:]
        dest_file["lightcone_params/source_lightcone_file"] = src_file["lightcone_params/source_lightcone_file"][:]
        
        dest_file.create_group('lightcones')
        dest_file['lightcones'].create_dataset("brightness_temp", newshape, dtype=np.float32)

        for ni in range(n):
            if ni%10==0 or debug: print(f"Processing lightcone file {ni} of {n}...")

            # Load Lightcone
            bT_cube = src_file['lightcones/brightness_temp'][ni,:,:old_z]

            # Mean Removal
            bT_cube = bT_cube - np.mean(bT_cube, axis=(0,1)) # remove mean
    
            # Wedge 
            if ws > 0.0: bT_cube = apply_constant_wedge_filter(bT_cube, filtr) #apply wedge 
            
            # Slice
            zsliceinds = np.linspace(0, old_z-1, new_z, dtype=int)
            bT_cube = bT_cube[:,:,zsliceinds]

            # Transpose
            bT_cube = np.transpose(bT_cube, (2,0,1))

            # Subdiv
            for i in range(subdiv):
                for j in range(subdiv):
                    l2 = ni*subdiv**2 + subdiv*j + i

                    #Save lightcone
                    dest_file['lightcones/brightness_temp'][l2] = bT_cube[:, i*new_xy:(i+1)*new_xy, j*new_xy:(j+1)*new_xy]








def encode_slice(vae, x):
    x = np.repeat(x[np.newaxis,:,:], 3, axis=0)
    x = torch.from_numpy(x[np.newaxis,:,:,:]).to(vae.device)
    z = vae.encode(x).latent_dist.mean
    return z.detach().cpu().numpy()[0]

def encode_lightcones(src, dest, prenorm=False, test=False): 
    #load VAE
    device="cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(device)

    with h5py.File(src, "r") as f1, h5py.File(dest, "w") as f2:
        f2.create_group('lightcone_params')
        f2["lightcone_params/physparams"] = f1["lightcone_params/physparams"][:]
        f2["lightcone_params/source_lightcone_file"] = f1["lightcone_params/source_lightcone_file"][:]
        
        f2.create_group('lightcones')
        n, z, xy, _ = f1['lightcones/brightness_temp'].shape
        if test: n = 3
        
        test_slice = f1['lightcones/brightness_temp'][0,0]
        test_encoding = encode_slice(vae, test_slice)
        newshape = (n, z) + test_encoding.shape
        print(f"New shape: {newshape}")

        maxx, minn = -np.inf,np.inf
        if prenorm:
            for ni in range(n):
                this_max = np.max(f1['lightcones/brightness_temp'][ni])
                this_min = np.min(f1['lightcones/brightness_temp'][ni])
                if this_max > maxx: maxx = this_max
                if this_min < minn: minn = this_min


        lightcones = np.empty(newshape, dtype=np.float32)
        for ni in range(n):
            if ni%10==0 or test: print(f"Encoding lightcone {ni} of {n}...")
            for zi in range(z):
                slicee = f1['lightcones/brightness_temp'][ni,zi]
                if prenorm:
                    slicee =  (slicee - minn) / (maxx - minn)
                lightcones[ni, zi] = encode_slice(vae, slicee)

        f2['lightcones'].create_dataset("brightness_temp", newshape, dtype=np.float32)
        f2['lightcones/brightness_temp'][:] = lightcones




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--sim', required=True, type=str)
    parser.add_argument('-ws', '--wedgeslope', nargs='?', type=float, default=0.0, help='Wedge slope of data (default = 0)')
    args = parser.parse_args()

    get_name = {
        "p21c" : "p21c14",
        "zreion" : "zreion24",
        "ctrpx" : "centralpix05"
        }

    get_src_dir = {
        "p21c" : "/users/jsolt/data/jsolt/lightcones/21cmFAST_lightcones_v04",
        "zreion": f"/users/jsolt/data/jsolt/zreion_sims/{get_name['zreion']}",
        "ctrpx" : "/users/jsolt/data/jsolt/lightcones/21cmFAST_lightcones_centralpix_v05"
        }

    get_dir = {
        "p21c" : "/users/jsolt/data/jsolt/21cmFAST_sims",
        "zreion" : "/users/jsolt/data/jsolt/zreion_sims",
        "ctrpx" : "/users/jsolt/data/jsolt/centralpix_sims"
        }

    sim, ws = args.sim, args.wedgeslope

    sim_name = get_name[sim]
    save_dir = f"{get_dir[sim]}/{sim_name}"
    src_dir = get_src_dir[sim]

    fname1 = f"{save_dir}/{sim_name}_subdiv_sliced_ws{ws}.hdf5"
    fname2 = f"{save_dir}/{sim_name}_norm_encoded_ws{ws}.hdf5"

    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    '''
    if sim in ["p21c", "ctrpx"]:
        # Find good lc files (ctrpx and p21c only)
        good_lc = save_good_lightcones_list(src_dir)
        good_lc = [f"{src_dir}/{lc}" for lc in good_lc]
        process_all(fname1, ws, new_z=30, new_xy=256, lightcones=good_lc)

    elif sim == "zreion":
        src_fname = f"{src_dir}/{sim_name}.hdf5"
        process_all_from_hdf5(fname1, ws, new_z=30, new_xy=256, src_fname=src_fname)
    '''
    # Encode and save lightcones
    encode_lightcones(fname1, fname2, prenorm=True)