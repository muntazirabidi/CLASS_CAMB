#####################################################
#
# Cosmological parameters and other CLASS parameters
#
#####################################################
from classy import Class
#import matplotlib
#from matplotlib import pyplot as plt
import numpy as np

from scipy.interpolate import interp1d

omch2 = 0.12038
ombh2 = 0.022032
h = 0.67556 #H0/100
cspeed = 299792.458 # km/s
A_s = 2.215e-9
n_s = 0.9619
tau_reion = 0.079
neff=3.046

ellmax = 2000
nell   = 400
zmax = 2.0


mu_z1    = 0.8
deltazf = 0.22
#sigma_z1 = 0.1
# redshift bin 2
mu_z2    = 1.2
deltazb = 0.2
#sigma_z2 = 0.1
#
ns = 4.0 # number of sigma in integration
nz = 400 #number of steps to use for the radial/redshift integration
#
def norm_gaussian(x,mu,sigma):
    return 1.0/np.sqrt(2.0*np.pi*sigma**2)*np.exp(-0.5*(x-mu)**2/sigma**2)

# Define your cosmology (what is not specified will be set to CLASS default parameters)
pars = {
    'output': 'tCl lCl mPk',
    'l_max_scalars': 4000,
    'lensing': 'yes',
    'non linear': 'halofit',
    'omega_b' : ombh2,
    'omega_cdm' : omch2,
    'h' : h,
    'ln10^{10}A_s' : np.log(10**10 *A_s),
    'n_s' : n_s,
    'tau_reio' : tau_reion,
    'N_eff': neff,
    'YHe' : 0.24,
    'N_ncdm' : 0,
    'halofit_k_per_decade' : 3000.,
    'l_switch_limber' : 40.,
    'accurate_lensing':1,
    'num_mu_minus_lmax' : 1000.,
    'delta_l_max' : 1000.,
    'z_max_pk':zmax}

k_max = 1;

#Initialize the cosmology andcompute everything
cosmo = Class()
cosmo.set(pars)
cosmo.compute()

#Specify k and z
k = np.logspace(-5, np.log10(k_max), num=1000) #Mpc^-1
z = 0.0

#Call these for the nonlinear and linear matter power spectra
#Pnonlin = np.array([cosmo.pk(ki, z) for ki in k])
Plin = np.array([cosmo.pk_lin(ki, z) for ki in k])


#NOTE: You will need to convert these to h/Mpc and (Mpc/h)^3
#to use in the toolkit. To do this you would do:
#k /= h
#Plin *= h**3
#Pnonlin *= h**3


background  = cosmo.get_background()
# comivng distance is in units Mpc
comoving_dist = background['comov. dist.']
ztab = background['z']

# interpolation function of comoving distance
comoving_dist_intp = interp1d(ztab,comoving_dist,kind='cubic')

Hubble = background['H [1/Mpc]']
Hubble_intp = interp1d(ztab,Hubble,kind='cubic')

growth_factor = background['gr.fac. D']
growth_factor_intp = interp1d(ztab,growth_factor,kind='cubic')



deltazf = 0.22;
deltazb = 0.2;
mu_z1 = 0.8;
mu_z2 = 1.2;
z1  = np.linspace(mu_z1-deltazf/2,mu_z1+deltazf/2,num=nz,endpoint=False) # should use better integrator formula
z2  = np.linspace(mu_z2-deltazb/2,mu_z2+deltazb/2,num=nz,endpoint=False)
dz1 = z1[1]-z1[0]
dz2 = z2[1]-z2[0]


ellarr_class = np.linspace(2,ellmax,nell)
resarr_class = np.zeros(np.size(ellarr_class))

resarr_class2 = np.zeros(np.size(ellarr_class))

for il, ell in enumerate(ellarr_class):
    res_c = 0.0
    for iz in z1:
        chi_iz = comoving_dist_intp(iz) # returns comoving radial distance chi in Mpc
        kz     = (0.5+ell)/chi_iz
        Pkz    = cosmo.pk_lin(kz, iz)
        W_iz = 1/deltazf

        H_iz = Hubble_intp(iz)

        res_c    = res_c + W_iz*Pkz*W_iz*(H_iz)/(chi_iz**2)

    #res_c = res_c*cspeed # units 1/Mpc^2
    resarr_class2[il]=res_c*dz1

    print(ell,res_c)

np.savez_compressed('cl_dd_limber_usingCLASSfunctions.npz',l_limber=ellarr_class,cl_limber=resarr_class2)
