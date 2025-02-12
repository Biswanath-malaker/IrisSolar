import os
import numpy as np
from sunpy.map import Map as sm
from Functions.sunpy_map_cut import sunpy_map_cut
from fig_templates import intensity_fwhm_fig1
import glob
from Functions.utilities import get_extent
import matplotlib.pyplot as plt
import astropy.units as u
from models import Models


def plot_spectrum(wavelength,data,err):

    fig = plt.figure(figsize = (15,6))

    ax = fig.add_subplot(1,1,1,)

    # ax.plot(wavelength , data,drawstyle='steps-mid')

    ax.set_xlabel('Wavelength ($\AA$)',fontsize=20,fontweight='medium')
    ax.set_ylabel('Intensity',fontsize=20,fontweight='medium')

    plt.errorbar(
        wavelength,
        data,
        yerr=err,
        drawstyle='steps-mid',
        color='black',
        lolims=0,
        capsize=5,             # Size of error bar caps
        capthick=1,            # Thickness of caps
        ecolor='red',         # Color of error bars
        marker='o',            # Add markers at data points
        markersize=3,
        markerfacecolor='white',
        markeredgecolor='blue',
        label='Data with Errors'
        )
    # plt.tight_layout()
    return fig


def get_spectra_by_indices(row_p,col_p,m_dummy_raster,datacube,errorcube,popt,reduced_chisq):
    """
    m_dummpy_raster is the original dummy raster without any cut.
    
    r = row index   (along y axis)
    c = col index   (along x axis)
    """ 
    
    S = m_dummy_raster.pixel_to_world(col_p*u.pix,row_p*u.pix)
    X = S.Tx.value
    Y = S.Ty.value
    I = popt[...,0][col_p,row_p]
    Mu = popt[...,1][col_p,row_p]
    Sigma = popt[...,2][col_p,row_p]
    Slope = popt[...,3][col_p,row_p]
    Const = popt[...,4][col_p,row_p]
    red_chisq_point = reduced_chisq[row_p,col_p]
    spec = datacube[col_p,row_p]
    
    spec[spec<0] = 0
    noise = errorcube[col_p,row_p]
    
    return spec,noise,I,Mu,Sigma,Slope,Const,red_chisq_point,X,Y

def get_spectra_by_indices_double(row_p,col_p,m_dummy_raster,datacube,errorcube,popt,reduced_chisq):
    """
    m_dummpy_raster is the original dummy raster without any cut.
    
    r = row index   (along y axis)
    c = col index   (along x axis)
    """ 
    
    S = m_dummy_raster.pixel_to_world(col_p*u.pix,row_p*u.pix)
    X = S.Tx.value
    Y = S.Ty.value
    I1 = popt[...,0][col_p,row_p]
    I2 = popt[...,1][col_p,row_p]

    Mu1 = popt[...,2][col_p,row_p]
    Mu2 = popt[...,3][col_p,row_p]

    Sigma1 = popt[...,4][col_p,row_p]
    Sigma2 = popt[...,5][col_p,row_p]

    Slope = popt[...,6][col_p,row_p]
    Const = popt[...,7][col_p,row_p]
    
    red_chisq_point = reduced_chisq[row_p,col_p]
    spec = datacube[col_p,row_p]
    
    spec[spec<0] = 0
    noise = errorcube[col_p,row_p]
    
    return spec,noise,I1,I2,Mu1,Mu2,Sigma1,Sigma2,Slope,Const,red_chisq_point,X,Y

m = 2
n = 1
D = 'D044'
file_name = "iris_l2_20160319_145728_3623010639_raster_t000_r00000"
ext_y = [320,400]
x0,y0 = 12,201     # Initial spectral point (px) to show!




siiv_results_dir = f"./FITTING/SIV_1394/Gaussian1_Linear_Bg/m_{m}-n_{n}/{file_name}"
siiv_results_dir1 = f"./FITTING/SIV_1394/Gaussian2_Linear_Bg/m_{m}-n_{n}/{file_name}"

file_171 = glob.glob(f"data_scanwise_m-{m}_n-{n}/{D}/{file_name}/mid/171/*.fits")[0]
file_hmi = glob.glob(f"data_scanwise_m-{m}_n-{n}/{D}/{file_name}/mid/hmi/*.fits")[0]

si_iv_results = {
    'data':np.load(os.path.join(siiv_results_dir,'data.npy'),allow_pickle=True),
    'dopplershift':np.load(os.path.join(siiv_results_dir,'dopplershift.npy'),allow_pickle=True).T,
    'dummy_raster_2d':sm(os.path.join(siiv_results_dir,'dummy_raster_2d.fits')),
    'err':np.load(os.path.join(siiv_results_dir,'err.npy'),allow_pickle=True),
    'flux':np.load(os.path.join(siiv_results_dir,'flux.npy'),allow_pickle=True).T,
    'fwhm':np.load(os.path.join(siiv_results_dir,'fwhm.npy'),allow_pickle=True).T,
    'non_th_vel':np.load(os.path.join(siiv_results_dir,'non_th_vel.npy'),allow_pickle=True).T,
    'popt':np.load(os.path.join(siiv_results_dir,'popt.npy'),allow_pickle=True),
    'red_chisq':np.load(os.path.join(siiv_results_dir,'red_chisq.npy'),allow_pickle=True).T,
    'wavelength':np.load(os.path.join(siiv_results_dir,'wavelength.npy'),allow_pickle=True),
    'popt_double':np.load(os.path.join(siiv_results_dir1,'popt.npy'),allow_pickle=True),
    'red_chisq_double':np.load(os.path.join(siiv_results_dir1,'red_chisq.npy'),allow_pickle=True).T,
} 



def cut_map(m,m_dummy_raster):

    bl = [float(m_dummy_raster.bottom_left_coord.Tx.value),float(m_dummy_raster.bottom_left_coord.Ty.value)]
    tr = [float(m_dummy_raster.top_right_coord.Tx.value),float(m_dummy_raster.top_right_coord.Ty.value)]
    m_sub = sunpy_map_cut(m,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
    
    return m_sub

    
datacube = si_iv_results['data']
errorcube = si_iv_results['err']
wavelength = si_iv_results['wavelength']

m_dummy_raster_2d = si_iv_results['dummy_raster_2d']

m_dopplershift = sm((si_iv_results['dopplershift'],m_dummy_raster_2d.fits_header))
m_flux = sm((si_iv_results['flux'],m_dummy_raster_2d.fits_header))
m_fwhm = sm((si_iv_results['fwhm'],m_dummy_raster_2d.fits_header))
m_non_th_vel = sm((si_iv_results['non_th_vel'],m_dummy_raster_2d.fits_header))
m_red_chisq = sm((si_iv_results['red_chisq'],m_dummy_raster_2d.fits_header))
m_red_chisq_double = sm((si_iv_results['red_chisq_double'],m_dummy_raster_2d.fits_header))

m_171 = sm(file_171)
m_hmi = sm(file_hmi)




m_dopplershift_sub = cut_map(m_dopplershift,m_dummy_raster_2d)
m_flux_sub = cut_map(m_flux,m_dummy_raster_2d)
m_fwhm_sub = cut_map(m_fwhm,m_dummy_raster_2d)
m_non_th_vel_sub = cut_map(m_non_th_vel,m_dummy_raster_2d)
m_red_chisq_sub = cut_map(m_red_chisq,m_dummy_raster_2d)
m_red_chisq_double_sub = cut_map(m_red_chisq,m_dummy_raster_2d)
m_171_sub = cut_map(m_171,m_dummy_raster_2d)
m_hmi_sub = cut_map(m_hmi,m_dummy_raster_2d)


print(si_iv_results['popt'].data.shape)
popt_double = si_iv_results['popt_double']

a1 = popt_double[:,:,0]
a2 = popt_double[:,:,1]

mu1 = popt_double[:,:,2]
mu2 = popt_double[:,:,3]

s1 = popt_double[:,:,4]
s2 = popt_double[:,:,5]


C = mu1>mu2

mu_red = np.where(C,mu1,mu2).T

mu_blue = np.where(C,mu2,mu1).T




from physics import get_wavelength2dopplershift

dopp_red = get_wavelength2dopplershift(mu_red*u.angstrom,1393.76*u.angstrom).value
dopp_blue = get_wavelength2dopplershift(mu_blue*u.angstrom,1393.76*u.angstrom).value

Dopp_red_chisq_gt1 = np.where(si_iv_results['red_chisq']>1.1,dopp_red,si_iv_results['dopplershift'])
Dopp_blue_chisq_gt1 = np.where(si_iv_results['red_chisq']>1.1,dopp_blue,si_iv_results['dopplershift'])


np.save(os.path.join(siiv_results_dir1,'dopp_red.npy'),Dopp_red_chisq_gt1,allow_pickle=True)
np.save(os.path.join(siiv_results_dir1,'dopp_blue.npy'),Dopp_blue_chisq_gt1,allow_pickle=True)


