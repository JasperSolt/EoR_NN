import os
import glob
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
    lightcones = glob.glob(f"{lc_path}/*.h5")
    good_lc = []

    for i, lc in enumerate(lightcones):
        if i%10 == 0: print(f"Lightcone {i} of {len(lightcones)} (Good light cones so far: {len(good_lc)})")
    
        with h5py.File(lc, 'r') as f:
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


def process_all(fname, ws, new_z, new_xy, lightcones, norm=False, debug=False):
    n = len(lightcones) if not debug else 3
    
    with h5py.File(lightcones[0]) as lc_file:
        oldshape = lc_file['lightcones/brightness_temp'].shape

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
                node_redshifts = np.flip(lc_file['node_redshifts'][:])

            
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

            if norm:
                minn, maxx = np.min(bT_cube), np.max(bT_cube)
                bT_cube = (bT_cube - minn) / (maxx - minn)

            # Subdiv
            for i in range(subdiv):
                for j in range(subdiv):
                    l2 = ni*subdiv**2 + subdiv*j + i

                    #Save lightcone
                    f['lightcones/brightness_temp'][l2] = bT_cube[:, i*new_xy:(i+1)*new_xy, j*new_xy:(j+1)*new_xy]

                    #save params
                    f['lightcone_params/physparams'][l2,:] = [z50, z75-z25, (z75+z25)/2]
                    f['lightcone_params/source_lightcone_file'][l2] = lightcones[ni]







def process_all_from_hdf5(dest_fname, ws, new_z, new_xy, src_fname, norm=False, debug=False):
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

            if norm:
                minn, maxx = np.min(bT_cube), np.max(bT_cube)
                bT_cube = (bT_cube - minn) / (maxx - minn)

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


def encode_lightcones(src, dest, prenorm=True, append=False, test=False): 
    #load VAE
    device="cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(device)
    
    mode = 'a' if append else 'r'
    with h5py.File(src, mode) as f1:
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
        
        '''
        Write
        '''
        if append:
            f1['lightcones'].create_dataset("brightness_temp_encoded", newshape, dtype=np.float32)
            f1['lightcones/brightness_temp_encoded'][:] = lightcones
        else: 
            with h5py.File(dest, "w") as f2:
                f2.create_group('lightcone_params')
                f2["lightcone_params/physparams"] = f1["lightcone_params/physparams"][:]
                f2["lightcone_params/source_lightcone_file"] = f1["lightcone_params/source_lightcone_file"][:]
                
                f2.create_group('lightcones')

                f2['lightcones'].create_dataset("brightness_temp", newshape, dtype=np.float32)
                f2['lightcones/brightness_temp'][:] = lightcones



def decode_slice(vae, x):
    x = torch.from_numpy(x[np.newaxis,:]).to(vae.device)
    z = vae.decode(x, return_dict=False)[0][0]
    z = torch.mean(z, dim=0)
    return z.detach().cpu().numpy()


def decode_lightcones(src, dest, append=False, test=False): 
    #load VAE
    device="cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(device)

    mode = 'a' if append else 'r'
    with h5py.File(src, mode) as f1:
        n, z, l, xy, _ = f1['lightcones/brightness_temp'].shape
        if test: n = 3
        
        test_slice = f1['lightcones/brightness_temp'][0,0]

        test_decoding = decode_slice(vae, test_slice)
        newshape = (n, z) + test_decoding.shape
        print(f"New shape: {newshape}")

        if append:
            f1['lightcones'].create_dataset("brightness_temp_decoded", newshape, dtype=np.float32)
        else:
            with h5py.File(dest, "w") as f2:
                f2.create_group('lightcone_params')
                f2["lightcone_params/physparams"] = f1["lightcone_params/physparams"][:]
                f2["lightcone_params/source_lightcone_file"] = f1["lightcone_params/source_lightcone_file"][:]

                f2.create_group('lightcones')
                f2['lightcones'].create_dataset("brightness_temp", newshape, dtype=np.float32)

        
        for ni in range(n):
            if ni%10==0 or test: print(f"Decoding lightcone {ni} of {n}...")
            for zi in range(z):
                slicee = decode_slice(vae, f1['lightcones/brightness_temp'][ni,zi])
                if append:
                    f2['lightcones/brightness_temp_decoded'][ni, zi] = slicee

                else:
                    with h5py.File(dest, "a") as f2:
                        f2['lightcones/brightness_temp'][ni, zi] = slicee



        







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--sim', required=True, type=str)
    parser.add_argument('-ws', '--wedgeslope', nargs='?', type=float, default=0.0, help='Wedge slope of data (default = 0)')
    args = parser.parse_args()

    get_name = {
        "zreion" : "zreion24",
        "ctrpx" : "centralpix05",
        "ctrpxbig" : "centralpixbig01",
        "fllsphr" : "fullsphere02"
        }

    get_src_dir = {
        "zreion": f"/users/jsolt/data/jsolt/zreion_sims/{get_name['zreion']}",
        "ctrpx" : "/users/jsolt/data/jsolt/lightcones/21cmFAST_lightcones_centralpix_v05",
        "ctrpxbig" : "/users/jsolt/data/jsolt/lightcones/21cmFAST_lightcones_centralpix_big_v01",
        "fllsphr" : "/users/jsolt/data/jsolt/lightcones/21cmFAST_lightcones_fullsphere_v04",        
        }

    get_dir = {
        "zreion" : "/users/jsolt/data/jsolt/zreion_sims",
        "ctrpx" : "/users/jsolt/data/jsolt/centralpix_sims",
        "ctrpxbig" : "/users/jsolt/data/jsolt/centralpix_sims",
        "fllsphr" : "/users/jsolt/data/jsolt/fullsphere_sims"
        }

    sim, ws = args.sim, args.wedgeslope
    norm = False

    sim_name = get_name[sim]
    save_dir = f"{get_dir[sim]}/{sim_name}"
    src_dir = get_src_dir[sim]

    if norm: sim_name += '_norm'
    fname1 = f"{save_dir}/{sim_name}_subdiv_sliced_ws{ws}.hdf5"
    fname2 = f"{save_dir}/{sim_name}_encoded_ws{ws}.hdf5"
    fname3 = f"{save_dir}/{sim_name}_decoded_ws{ws}.hdf5"

    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    
    if sim == "zreion":
        src_fname = f"{src_dir}/{sim_name}.hdf5"
        process_all_from_hdf5(fname1, ws, new_z=30, new_xy=256, norm=True, src_fname=src_fname)
    else:
        good_lc = save_good_lightcones_list(src_dir)
        process_all(fname1, ws, new_z=30, new_xy=256, norm=norm, lightcones=good_lc)


    # Encode and save lightcones
    encode_lightcones(fname1, fname2, prenorm=False)
    
    decode_lightcones(fname2, fname3)