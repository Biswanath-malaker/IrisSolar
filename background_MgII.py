from iris_raster import IrisRaster
import glob
import numpy as np
import matplotlib.pyplot as plt
from models import Models
import os
from Functions.utilities import get_extent
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord, SpectralCoord
from sunpy.coordinates import frames
import time
# from ndcube import NDCube
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.constants import c
from Functions.sunpy_map_cut import sunpy_map_cut
from astropy.io import fits
from sunpy.map import Map as sm
from utilities import utilities as utils
from physics import get_dopplershift2wavelength
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy import integrate

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

def get_lam_I1(wavelength_k3,data_k3,kline=True):
    
    try:
        
        status = 1
        
        f = interp1d(wavelength_k3,data_k3,kind='cubic')

        wavelength_k3_dense = np.linspace(wavelength_k3[0],wavelength_k3[-1],300)
        data_k3_interpolated = f(wavelength_k3_dense)
        data_k3_interpolated = gaussian_filter1d(data_k3_interpolated,1)
        
        peaks, _ = find_peaks(data_k3_interpolated, height=0)
        valleys, _ = find_peaks(-data_k3_interpolated+np.max(data_k3_interpolated), height=0)
        

        # Get central dip

        # Check that there must be 2 peaks and 1 dip
        if len(peaks)<2:
            print(f"Only {len(peaks)} peaks found and not a good data!")
            status=0
        if len(valleys)<1:
            print(f"Only {len(valleys)} dips found and not a good data!")
            status=0

        if status!=0:
            dip_heights = data_k3_interpolated[valleys]
            central_dip_ind = np.argmax(dip_heights)

            # if kline:
            #     central_dip_ind = np.argmin(np.abs(valleys*u.angstrom-k3_init))

            central_dip_wavelength = wavelength_k3_dense[valleys][central_dip_ind]
            central_dip_height = data_k3_interpolated[valleys][central_dip_ind]
            
        
            # Get nearest peaks around central dip
            
            # get k2v
            violet_peaks_index = wavelength_k3_dense[peaks]<central_dip_wavelength
            violet_peaks = wavelength_k3_dense[peaks][violet_peaks_index]
            violet_peaks_heights = data_k3_interpolated[peaks][violet_peaks_index]
            violet_peak_index = np.argmax(violet_peaks_heights)
            violet_peak = violet_peaks[violet_peak_index]
            violet_peak_height = violet_peaks_heights[violet_peak_index]
                    
            # get k2r
            red_peaks_index = wavelength_k3_dense[peaks]>central_dip_wavelength
            red_peaks = wavelength_k3_dense[peaks][red_peaks_index]
            red_peaks_heights = data_k3_interpolated[peaks][red_peaks_index]
            red_peak_index = np.argmax(red_peaks_heights)
            red_peak = red_peaks[red_peak_index]
            red_peak_height = red_peaks_heights[red_peak_index]

            
            # Check that peak height is greater than dips
            ch1 = red_peak_height>central_dip_height 
            ch2 = violet_peak_height>central_dip_height 
            
            if not (ch1 and ch2):
                if not ch1:
                    print("Red peak is lesser height than cantral dip")
                    status=0
                if not ch2:
                    print("Violet peak is lesser height than cantral dip")
                    status=0
            
            if status!=0:
                # Get violet dip
                violet_end_ind = wavelength_k3_dense<violet_peak
                violet_end = wavelength_k3_dense[violet_end_ind]
                data_violet_end = data_k3_interpolated[violet_end_ind]
                
                violet_dip_ind = np.argmin(data_violet_end)
                violet_dip = violet_end[violet_dip_ind]
                violet_dip_height = data_violet_end[violet_dip_ind]
                
                # Get red dip
                red_end_ind = wavelength_k3_dense>red_peak
                red_end = wavelength_k3_dense[red_end_ind]
                data_red_end = data_k3_interpolated[red_end_ind]
                
                red_dip_ind = np.argmin(data_red_end)
                red_dip = red_end[red_dip_ind]
                red_dip_height = data_red_end[red_dip_ind]
                    
        if status != 1:
            return [np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan]
        else:
            return [violet_peak,red_peak,central_dip_wavelength],[violet_peak_height,red_peak_height,central_dip_height],[violet_dip,red_dip],[violet_dip_height,red_dip_height]
    except:
        return [np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan]

def fit_MgII_lines_2gaussian_linearbg(args):
    """
    wavelength_k3,wavelength_h3,data_k3,data_h3,error_k3,error_h3,I = args \n;
    I is pixel index for identifications!
    
    """
    
    wavelength_k3,wavelength_h3,data_k3,data_h3,error_k3,error_h3,I = args

    init_a = [10,30,30,10,30,30,10]
    k3_init , h3_init = 2796.36 * u.angstrom,2803.54 * u.angstrom

    init_mu = [
        float(get_dopplershift2wavelength(-200*u.km/u.s,k3_init).value), \
        float(get_dopplershift2wavelength(-15*u.km/u.s,k3_init).value), \
        float(get_dopplershift2wavelength(+15*u.km/u.s,k3_init).value), \
        (k3_init.value+h3_init.value)/2, \
        float(get_dopplershift2wavelength(-15*u.km/u.s,h3_init).value), \
        float(get_dopplershift2wavelength(+15*u.km/u.s,h3_init).value), \
        float(get_dopplershift2wavelength(200*u.km/u.s,h3_init).value) \
        ]
    
    init_sigma = [1,0.15,0.15,1.5,0.15,0.15,1]
        
    init_p_k3 = init_a[1:3]+init_mu[1:3]+init_sigma[1:3]+[0.0001,0.001]
    init_p_h3 = init_a[4:6]+init_mu[4:6]+init_sigma[4:6]+[0.0001,0.001]


    popt_k3,_ = curve_fit(Models.Gauss2linearbackground,wavelength_k3,data_k3,p0=init_p_k3,sigma=error_k3,absolute_sigma=True,maxfev = 10000)
    popt_h3,_ = curve_fit(Models.Gauss2linearbackground,wavelength_h3,data_h3,p0=init_p_h3,sigma=error_h3,absolute_sigma=True,maxfev = 10000)

    chi_sq_k3 = (data_k3-Models.Gauss2linearbackground1(wavelength_k3,popt_k3))**2/error_k3**2
    red_chisq1_k3 = np.sum(chi_sq_k3)/(len(data_k3)-8)
    
    
    chi_sq_h3 = (data_h3-Models.Gauss2linearbackground1(wavelength_h3,popt_h3))**2/error_h3**2
    red_chisq1_h3 = np.sum(chi_sq_h3)/(len(data_h3)-8)
    
    return popt_k3,popt_h3,red_chisq1_k3,red_chisq1_h3,I


def plot_MgII_spectra(wavelength_k3,data_k3,error_k3,popt_k3,wavelength_h3,data_h3,error_h3,popt_h3):
    
    # k3_init , h3_init = 2796.36 * u.angstrom,2803.54 * u.angstrom
    # v = 70 *u.km/u.s    # +-v range from h3 and k3 line.

    # k3_range = [float(get_dopplershift2wavelength(-v,k3_init).value),float(get_dopplershift2wavelength(v,k3_init).value)]
    # h3_range = [float(get_dopplershift2wavelength(-v,h3_init).value),float(get_dopplershift2wavelength(v,h3_init).value)]

    # wavelength_k3,data_k3,error_k3 = raster.get_spectrum(8,raster_timestep,ypixel,fuv=False,wmin=k3_range[0],wmax=k3_range[1])
    # wavelength_h3,data_h3,error_h3 = raster.get_spectrum(8,raster_timestep,ypixel,fuv=False,wmin=h3_range[0],wmax=h3_range[1])

    wavelength_k3_dense = np.linspace(wavelength_k3[0],wavelength_k3[-1],300)
    wavelength_h3_dense = np.linspace(wavelength_h3[0],wavelength_h3[-1],300)

    fig = raster.plot_spectrum(wavelength_k3,data_k3,error_k3)
    fig1 = raster.plot_spectrum(wavelength_h3,data_h3,error_h3)

    ax_fig = fig.axes[0]
    ax_fig1 = fig1.axes[0]
    
    fitted_k3_dense = Models.Gauss2linearbackground1(wavelength_k3_dense,popt_k3)
    fitted_h3_dense = Models.Gauss2linearbackground1(wavelength_h3_dense,popt_h3)

    ax_fig.plot(wavelength_k3_dense,fitted_k3_dense,'--')
    ax_fig1.plot(wavelength_h3_dense,fitted_h3_dense,'--')

    ax_fig.plot(wavelength_k3_dense,Models.Gauss1linearbackground1(wavelength_k3_dense,[popt_k3[0],popt_k3[2],popt_k3[4],popt_k3[6],popt_k3[7]]))
    ax_fig.plot(wavelength_k3_dense,Models.Gauss1linearbackground1(wavelength_k3_dense,[popt_k3[1],popt_k3[3],popt_k3[5],popt_k3[6],popt_k3[7]]))

    ax_fig1.plot(wavelength_h3_dense,Models.Gauss1linearbackground1(wavelength_h3_dense,[popt_h3[0],popt_h3[2],popt_h3[4],popt_h3[6],popt_h3[7]]))
    ax_fig1.plot(wavelength_h3_dense,Models.Gauss1linearbackground1(wavelength_h3_dense,[popt_h3[1],popt_h3[3],popt_h3[5],popt_h3[6],popt_h3[7]]))

    return fig,fig1



def get_lam_I(wavelength_k3,popt_k3):
    """
    Returns lam_k2v,lam_k2r,lam_k3] , [I_k2v,I_k2r,I_k3]
    Applicable for h3 line and k3 line although arguments are in terms of k3 only;
    
    """
    wavelength_k3_dense = np.linspace(wavelength_k3[0],wavelength_k3[-1],300)
    fitted_k3_dense = Models.Gauss2linearbackground1(wavelength_k3_dense,popt_k3)
    peaks_k,_ = find_peaks(fitted_k3_dense)
    
    if len(peaks_k) == 2:

        fitted_k2v_t0_k2r = fitted_k3_dense[peaks_k[0]:peaks_k[-1]]
        wavelength_dense_k2v_t0_k2r = wavelength_k3_dense[peaks_k[0]:peaks_k[-1]]
        
        k3_ind = np.argmin(fitted_k2v_t0_k2r)
        lam_k3 = wavelength_dense_k2v_t0_k2r[k3_ind]
        I_k3 = fitted_k2v_t0_k2r[k3_ind]
        lam_k2v,lam_k2r = wavelength_k3_dense[peaks_k]
        I_k2v,I_k2r = fitted_k3_dense[peaks_k]
        return np.array([lam_k2v,lam_k2r,lam_k3]) , np.array([I_k2v,I_k2r,I_k3])
    else:
        return np.array([np.nan,np.nan,np.nan]) , np.array([np.nan,np.nan,np.nan])


hdu_index = 8
raster_file = "./../iris_obj/data_aligned/D044/RASTER_FILES/iris_l2_20160319_145728_3623010639_raster_t000_r00000.fits"
aia_171_file = "./../iris_obj/data_aligned/D044/171/2016-03-19T14_59_23.fits"
error_file = f"./../iris_obj/Errors/D044/iris_l2_20160319_145728_3623010639_raster_t000_r00000/ERR_hdu_{hdu_index}.npy"

def slice_spectralcube_by_spatial_pixel(spectral_cube,row_l,row_u,col_l,col_u):
    """
    row_l is lower row index in spatial coord
    """
    D2 = spectral_cube[col_l:col_u+1,row_l:row_u+1]

    return D2


with IrisRaster(raster_file) as raster:

    BG_ext = [12,23,345,352]        # ***** Extension for the background in arcsec.
    y0,y1 = 320,400                 # lower and upper limit of the scans to be shown in the paper

    wrange_k = raster.get_wrange('MgII_k')
    wrange_h = raster.get_wrange('MgII_h')


    m = raster.get_spectroheliogram(hdu_index,np.average(np.array(wrange_k)))  # It is only for finding background corner indices!

    data,hd = raster.get_data_header(hdu_index,print_line=True)

    ERR = np.load(error_file)

    coordinate_lower = SkyCoord(BG_ext[0]*u.arcsec,BG_ext[2]*u.arcsec,frame=m.coordinate_frame)
    coordinate_upper = SkyCoord(BG_ext[1]*u.arcsec,BG_ext[3]*u.arcsec,frame=m.coordinate_frame)

    col_l,row_l = m.world_to_pixel(coordinate_lower)
    col_u,row_u = m.world_to_pixel(coordinate_upper)

    col_l,col_u,row_l,row_u = int(np.rint(col_l.value)),int(np.rint(col_u.value)),int(np.rint(row_l.value)),int(np.rint(row_u.value))

    bg_cube = slice_spectralcube_by_spatial_pixel(data,row_l,row_u,col_l,col_u)
    bg_err = slice_spectralcube_by_spatial_pixel(ERR,row_l,row_u,col_l,col_u)
    wavelengths = raster.get_wavelengths(hdu_index).value
    fuv_exp_time = raster.get_nuv_exposure_time()       #********
    bg_cube = bg_cube/fuv_exp_time[:,np.newaxis,np.newaxis][col_l:col_u+1]  #**********

    mean_bg_spectra = np.mean(bg_cube,axis=(0,1))
    bg_err_sq = bg_err**2
    mean_bg_err = np.sqrt(np.sum(bg_err_sq,axis=(0,1)))/(bg_err.shape[0]*bg_err.shape[1])


    wavelength_ind_k = np.logical_and(wavelengths>=wrange_k[0],wavelengths<=[wrange_k[1]])
    wavelengths_k = wavelengths[wavelength_ind_k]
    mean_bg_spectra_k = mean_bg_spectra[wavelength_ind_k]
    mean_bg_err_k = mean_bg_err[wavelength_ind_k]
    
    wavelength_ind_h = np.logical_and(wavelengths>=wrange_h[0],wavelengths<=[wrange_h[1]])
    wavelengths_h = wavelengths[wavelength_ind_h]
    mean_bg_spectra_h = mean_bg_spectra[wavelength_ind_h]
    mean_bg_err_h = mean_bg_err[wavelength_ind_h]

    LAM_k3,I_k3,RANGE_k3,_ = get_lam_I1(wavelengths_k,mean_bg_spectra_k)
    LAM_h3,I_h3,RANGE_h3,_ = get_lam_I1(wavelengths_h,mean_bg_spectra_h,kline=False)

    f1 = interp1d(wavelengths_k,mean_bg_spectra_k,kind='cubic')
    f2 = interp1d(wavelengths_h,mean_bg_spectra_h,kind='cubic')

    wavelength_k3_dense = np.linspace(wavelengths_k[0],wavelengths_k[-1],300)
    data_k3_interpolated = f1(wavelength_k3_dense)
    data_k3_interpolated = gaussian_filter1d(data_k3_interpolated,1)


    wavelength_h3_dense = np.linspace(wavelengths_h[0],wavelengths_h[-1],300)
    data_h3_interpolated = f2(wavelength_h3_dense)
    data_h3_interpolated = gaussian_filter1d(data_h3_interpolated,1)
    
    
    fig1 = plot_spectrum(wavelengths_k,mean_bg_spectra_k,mean_bg_err_k)
    fig1.get_axes()[0].plot(wavelength_k3_dense,data_k3_interpolated)
    fig1.get_axes()[0].plot(LAM_k3,I_k3,'x',markersize=20)
    
    fig2 = plot_spectrum(wavelengths_h,mean_bg_spectra_h,mean_bg_err_h)
    fig2.get_axes()[0].plot(wavelength_h3_dense,data_h3_interpolated)
    fig2.get_axes()[0].plot(LAM_h3,I_h3,'x',markersize=20)

    print(f"LAM k = {LAM_k3}")
    print(f"LAM h = {LAM_h3}")


    plt.show()
