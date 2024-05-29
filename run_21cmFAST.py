import os
import glob
import argparse
import numpy as np
from datetime import datetime

import py21cmfast as p21c
from py21cmfast.inputs import AstroParams
from py21cmfast.cache_tools import clear_cache
from py21cmfast import global_params

from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

'''
#if USE_MASS_DEPENDENT_ZETA=True
bound_dict = {
                'F_STAR10': (0.0, 0.2),
                'ALPHA_STAR':(-0.5, 1.0),
                'F_ESC10': (0.0, 0.5),
                'ALPHA_ESC': (-1.0, 0.5),
                'M_TURN': (8.0, 10.0),
                'L_X': (38, 42),
                't_STAR': (0.1, 1.0),
                'NU_X_THRESH': (100, 1500)
             }
'''
bound_dict = {}

def random_astro_params():
    astro_params = AstroParams._defaults_
    for key, val in bound_dict.items():
        mu, sigma = AstroParams._defaults_[key], val[1]-val[0]
        X = get_truncated_normal(mean=mu, sd=sigma, low=val[0], upp=val[1])
        astro_params.update({key: X.rvs()}) 
    return astro_params

#set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--box_length', type=int)
parser.add_argument('-s', '--seed', type=int)
parser.add_argument('-smm', '--sampler_min_mass', type=float)
args = parser.parse_args()

#Simulation meta parameters
n_samples = 1
n_voxels = args.box_length #512
random_seed = args.seed
smm=args.sampler_min_mass * 1e6
vox_length = 3.9
sim_volume = vox_length*n_voxels 
sim_name = "smm_test_2"
user_params = {"HII_DIM": n_voxels, "BOX_LEN": sim_volume, "USE_INTERPOLATION_TABLES":True}
lightcone_quantities = ('brightness_temp', 'density', 'xH_box', 'halo_mass')

#format save path
save_path = f'/users/jsolt/data/jsolt/halo_lightcones_bigmem_{sim_name}'

if not os.path.exists(save_path):
    os.mkdir(save_path)

#Format Cache
local_dir = f'/users/jsolt/scratch'
cache_path = f'{local_dir}/21cmFAST_cache'

if not os.path.exists(cache_path):
    os.mkdir(cache_path)

p21c.config['direc'] = cache_path

#run sims
start_time = datetime.now()

flag_options = p21c.inputs.FlagOptions(
                                        USE_HALO_FIELD=True, #this makes it DexM
                                        HALO_STOCHASTICITY=True, #this uses the new beta branch halo sampler
                                        USE_MASS_DEPENDENT_ZETA=True
                                    )
astro_params = random_astro_params()

#with global_params.use(sampler_min_mass=smm, maxhalo_factor=2.0):
with global_params.use(sampler_min_mass=smm):
    
    #initial conditions
    print("Initial Conditions")
    init_box = p21c.initial_conditions(
                                    user_params=user_params,
                                    random_seed=random_seed
                                    )
    '''
    #testing halo sampler
    print("Halo Sampler")
    halolist_init = p21c.determine_halo_list(redshift=16.0,
                                         init_boxes=init_box,
                                         astro_params=astro_params,
                                         flag_options=flag_options,
                                         random_seed=random_seed)
    print(halolist_init.buffer_size)
    print(halolist_init.n_halos)
    '''

    lightcone = p21c.run_lightcone(
                                    redshift = 6.0,
                                    max_redshift = 16.0,
                                    astro_params = astro_params,
                                    flag_options = flag_options,
                                    lightcone_quantities = lightcone_quantities,
                                    global_quantities = lightcone_quantities,
                                    init_box=init_box,
                                )
    lightcone.save(direc=save_path)
    

end_time = datetime.now()
print(f"*** TOTAL TIME: {end_time - start_time}***")
