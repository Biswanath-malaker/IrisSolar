import numpy as np
from astropy.constants import c
import astropy.units as u



def get_flux_gaussian(a,sigma):
    """
    a and sigma may be numpy array.
    """
    Flux = np.sqrt(2*np.pi)*a*sigma
    return Flux

def get_fwhm(sigma:u,lam0:u):
    """
    Return fwhm in km/s
    """

    fwhm = (c/lam0)*2*np.sqrt(2*np.log(2))*sigma
    fwhm = fwhm.to(u.km/u.s).value
    return fwhm

def get_doppler_shift(mu:u,mu_bg:u):
    """
    mu might be 2d array.\n
    """
    doppler_shift = ((mu-mu_bg)/mu_bg)*c.to('km/s').value
    return doppler_shift


def get_wavelength2dopplershift(lam:u,lam0:u):
    """
    lam might be 2d array.\n
    lam0 is reference wavelength\n
    lam,lam0 must be of same unit.\n
    returned in km/s
    """
    doppler_shift = (lam-lam0)*c.to('km/s').value/lam0
    return doppler_shift


def get_dopplershift2wavelength(v:u,lam0:u):
    """
    lam0 is reference wavelength\n
    v is speed in astropy unit\n
    lam0 is wavelength in astropy units\n
    returned in angstrom
    """
    doppler_shift = (lam0+lam0*v/c)
    return doppler_shift


def get_nonthermal_vel_SiIV1394(fwhm):
    """
    fwhm can be numpy array but must be in km/s.
    Return in km/s
    # https://hesperia.gsfc.nasa.gov/ssw/hinode/eis/idl/atest/hwarren/eis_width2velocity.dat # thermal width data
    # https://link.springer.com/article/10.1007/s11207-014-0485-y  # instrumental width paper
    # https://iris.lmsal.com/itn38/analysis_lines_iris.html # https://iris.lmsal.com/itn38/analysis_lines_iris.html
    """
        
    dv_instr = 5.59  # km/s
    dv_th = 6.63    # km/s
    w_nth_sq = fwhm**2-dv_th**2-dv_instr**2 
    w_nth_sq[w_nth_sq<0]=np.nan
    w_nth = np.sqrt(w_nth_sq)
    return w_nth


def get_nonthermal_vel_CII1334(fwhm):
    """
    fwhm can be numpy array but must be in km/s.
    Return in km/s
    # https://hesperia.gsfc.nasa.gov/ssw/hinode/eis/idl/atest/hwarren/eis_width2velocity.dat # thermal width data
    # https://link.springer.com/article/10.1007/s11207-014-0485-y  # instrumental width paper
    # https://iris.lmsal.com/itn38/analysis_lines_iris.html # https://iris.lmsal.com/itn38/analysis_lines_iris.html
    """
        
    dv_instr = 5.59  # km/s
    dv_th = 5.7    # km/s
    w_nth_sq = fwhm**2-dv_th**2-dv_instr**2 
    w_nth_sq[w_nth_sq<0]=np.nan
    w_nth = np.sqrt(w_nth_sq)
    return w_nth





if __name__=="__main__":
    # doppl_shift = get_dopplershift2wavelength(-50*u.km/u.s,1394.755*u.angstrom)
    # print(1394.755-doppl_shift.value)
    # print(doppl_shift)
    # v = get_fwhm(0.1*u.angstrom,1393.7588*u.angstrom)
    # l1 = get_dopplershift2wavelength(75 *u.km/u.s,1393.7588*u.angstrom)
    # l0 = get_dopplershift2wavelength(-75 *u.km/u.s,1393.7588*u.angstrom)
    # v = get_doppler_shift(1393.786*u.angstrom,1393.76*u.angstrom)
    v1 = get_wavelength2dopplershift(1393.7074*u.angstrom,1393.7248*u.angstrom)

    # S = np.array([[0.1,1],[10,100]])*u.angstrom
    # v = get_fwhm(0.1*u.angstrom,1500*u.angstrom)
    # V = get_fwhm(S,1500*u.angstrom)

    print(v1)






