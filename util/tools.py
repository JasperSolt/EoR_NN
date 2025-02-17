import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import py21cmfast as p21c
from astropy.cosmology import Planck18

##################################################################################################################
#                                    ~  LIGHTCONE PROCESSING  ~
##################################################################################################################

#Cosmo params
OMb = 0.045
OMm = 0.3096

def t0(z):
    return 38.6*Planck18.h*(OMb/0.045)*np.sqrt((0.27/OMm)*(1+z)/10)

##################################################################################################################
#                                    ~  LIGHTCONE PROCESSING: zreion  ~
##################################################################################################################


def xHcube(cube, redshifts):
    xH_cube = np.zeros_like(cube)
    for z in range(len(redshifts)):
        xH_cube[:,:,z] = np.where(cube[:,:,z] <= redshifts[z], 1.0, 0.0)
    return xH_cube

def bTcube(density, xH_cube, redshifts):
    bTcube = np.zeros_like(xH_cube)
    for z in range(len(redshifts)):
        bTcube[:,:,z] = t0(redshifts[z]) * (1 + density[:,:,z]) * xH_cube[:,:,z] # * spin_temperature
    return bTcube


##################################################################################################################
#                                         ~  WEDGE FILTERING  ~
##################################################################################################################

# Global Vars:
Mpc2cm = 3.08560e24
lam21_cgs = 21.106
c_cgs = 2.99792e10
OmegaM  = 0.27 #0.315823
Tcmb0 = 2.7255
hubble0 = 0.673212
OmegaR = 0 #4.48e-7*(1+0.69)*Tcmb0**4/hubble0**2
OmegaL  = 0.73 #1-OmegaM
wde = -1.0
H0_cgs = 3.24086e-18

def E_of_a(a):
    # Calculate Hubble parameter E(a)
    return np.sqrt(OmegaM/a**3 + OmegaR/a**2 + OmegaL) #OmegaR/a**4

def wedge_slope(z):
    # Init
    a = 1.0/(1+z)

    # Compute comoving distance of redshift
    D = Planck18.comoving_transverse_distance(z).value * Mpc2cm * Planck18.h 
    
    # Compute E(z)
    Ez = E_of_a(a)

    # Compute wavelength of redshifted 21cm line
    lam = lam21_cgs*(1+z)

    # Compute rest-frame frequency of 21cm line
    f21 = c_cgs/lam21_cgs

    # Put it all together
    wedge_slope = lam*D*f21*H0_cgs*Ez/(c_cgs**2 * (1+z)**2)
    return wedge_slope

def apply_timevar_wedge_filter(cube, filtrs, nzpart, t):
    cube = apodize(cube)
        
    #fft
    cube = np.fft.fftn(cube)

    #iteratively find the wedge filtered data at different slope "chunks"
    filtr_cube = np.zeros_like(cube)
    for zin in range(nzpart):
        filtr_cube[:,:,zin*t:(zin+1)*t] = (cube * filtrs[zin])[:,:,zin*t:(zin+1)*t]

    # Inverse FFT
    filtr_cube = np.fft.ifftn(filtr_cube)

    return filtr_cube.real

def apply_constant_wedge_filter(cube, filtr):
    cube = apodize(cube)
        
    #fft
    cube = np.fft.fftn(cube)

    #iteratively find the wedge filtered data at different slope "chunks"
    filtr_cube = cube * filtr

    # Inverse FFT
    filtr_cube = np.fft.ifftn(filtr_cube)

    return filtr_cube.real

def apodize(t21_box):
    N_grid, _, Nsightpix = t21_box.shape

    # Apodize window along line-of-sight axis
    for k in range(0, Nsightpix):
        wz = np.cos(np.pi*(k+0.5)/Nsightpix - np.pi/2)
        for i, j in zip(range(0, N_grid), range(0, N_grid)):
            t21_box[i,j,k] = t21_box[i,j,k]*wz
    return t21_box

def find_filter(shape, slp):
    filtr = np.ones(shape)

    N_grid, _, Nsightpix = shape
    
    print("Calculating wedge filter...")
    # Apply wedge filter
    freq_perp = 2*np.pi*np.fft.fftfreq(N_grid)
    freq_para = 2*np.pi*np.fft.fftfreq(Nsightpix)
    for k in range(Nsightpix):
        kpara = abs(freq_para[k])
        for j in range(N_grid):
            ky = freq_perp[j]
            for i in range(N_grid):
                kx = freq_perp[i]
                kperp = np.sqrt(kx**2 + ky**2)
                if (kpara <= kperp*slp):
                    filtr[i,j,k] = 0.0
    return filtr

