import argparse
import numpy as np
import h5py


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

    get_dir = {
        "p21c" : "/users/jsolt/data/jsolt/21cmFAST_sims",
        "zreion" : "/users/jsolt/data/jsolt/zreion_sims",
        "ctrpx" : "/users/jsolt/data/jsolt/centralpix_sims"
        }

    sim, ws = args.sim, args.wedgeslope
    
    sim_name = get_name[sim]
    save_dir = f"{get_dir[sim]}/{sim_name}"

    fname1 = f"{save_dir}/{sim_name}_subdiv_sliced_ws{ws}.hdf5"
    fname2 = f"{save_dir}/{sim_name}_norm_decoded_ws{ws}.hdf5"

    npz_fname = f"{save_dir}/{sim_name}_ws{ws}_vae_stats.npz"
    


    with h5py.File(fname1, "r") as f1, h5py.File(fname2, "r") as f2:
        
        n, z, *_ = f1['lightcones/brightness_temp'].shape
        stats = {
            "mse":np.zeros((n,z)),
            "corrcoef":np.zeros((n,z)),
        }

        print(f"Finding normalization constants...")
        maxx, minn = -np.inf,np.inf
        for ni in range(n):
            this_max = np.max(f1['lightcones/brightness_temp'][ni])
            this_min = np.min(f1['lightcones/brightness_temp'][ni])
            if this_max > maxx: maxx = this_max
            if this_min < minn: minn = this_min
        
        for ni in range(n):
            if ni%100==0: print(f"Finding stats for sample {ni} of {n}...")
            original = (f1['lightcones/brightness_temp'][ni] - minn) / (maxx-minn)
            decoded = f2['lightcones/brightness_temp'][ni]
            stats["mse"][ni,:] = ((original-decoded)**2).mean(axis=(-1,-2))
            
            ccd = []
            for zi in range(z):
                c = np.corrcoef(original[zi], decoded[zi])

                stats["corrcoef"][ni,zi] = np.nanmean(np.diagonal(c, offset=c.shape[-1]//2, axis1=-1, axis2=-2)) 

    np.savez(npz_fname, **stats)

