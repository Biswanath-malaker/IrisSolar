"""
Extension and modified of MG_9 to accomodate finding peaks and dips methods rather fitting by gaussian.
"""
from iris_raster import IrisRaster
import matplotlib.pyplot as plt
import numpy as np
from physics import get_dopplershift2wavelength
import astropy.units as u
from models import Models
from scipy.optimize import curve_fit
import os
from tqdm import tqdm
import glob
import time
from sunpy.map import Map as sm
from fig_templates import plot_MgII_spectra
from Functions.utilities import get_extent
from Functions.sunpy_map_cut import sunpy_map_cut
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from fig_templates import get_MgII_plot

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

import os
import numpy as np
from sunpy.map import Map as sm
from Functions.sunpy_map_cut import sunpy_map_cut
from fig_templates import intensity_fwhm_fig,intensity_fwhm_fig1
import glob
from Functions.utilities import get_extent
import matplotlib.pyplot as plt
import astropy.units as u
from models import Models
from scipy.integrate import trapezoid


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

m = 2
n = 1
ext_y = [320,400]
x0,y0 = 12,201     # Initial spectral point (px) to show!

D = 'D044'
file = f"./data_aligned/{D}/RASTER_FILES/iris_l2_20160319_145728_3623010639_raster_t000_r00000.fits"
files = glob.glob(f"./../iris_obj/data_aligned/{D}/RASTER_FILES/*.fits")
files = sorted(files)

file = files[0]
file_name = os.path.basename(file).replace(".fits","")

def plot_MgII_spectra(wavelength_k3,data_k3,error_k3,wavelength_h3,data_h3,error_h3):
    

    
    f = interp1d(wavelength_k3,data_k3,kind='cubic')
    f1 = interp1d(wavelength_h3,data_h3,kind='cubic')


    wavelength_k3_dense = np.linspace(wavelength_k3[0],wavelength_k3[-1],300)
    data_k3_interpolated = f(wavelength_k3_dense)
    data_k3_interpolated = gaussian_filter1d(data_k3_interpolated,3)
    
    wavelength_h3_dense = np.linspace(wavelength_h3[0],wavelength_h3[-1],300)
    data_h3_interpolated = f1(wavelength_h3_dense)
    data_h3_interpolated = gaussian_filter1d(data_h3_interpolated,3)



    fig = plot_spectrum(wavelength_k3,data_k3,error_k3)
    fig1 = plot_spectrum(wavelength_h3,data_h3,error_h3)

    ax_fig = fig.axes[0]
    ax_fig1 = fig1.axes[0]
    
    ax_fig.plot(wavelength_k3_dense,data_k3_interpolated)
    
    ax_fig1.plot(wavelength_h3_dense,data_h3_interpolated)


    return fig,fig1

ext_y = [320,400]   # lower and upper limit of the plots.


        


raster_timestep = 12
ypixel = 182        # 10,182 is nan
m , n = 2,1
file_name = os.path.basename(file).replace('.fits','')
aia171_file = glob.glob(f"./../iris_obj/data_scanwise_m-{m}_n-{n}/{D}/{file_name}/mid/171/*.fits")[0]
hmi_file = glob.glob(f"./../iris_obj/data_scanwise_m-{m}_n-{n}/{D}/{file_name}/mid/hmi/*.fits")[0]

# m = sm

target_dir = f"./FITTING/MgII/{D}/m_{m}-n_{n}/{file_name}"
    
# popt_k3_path = os.path.join(target_dir,'popt_k3.npy')
# popt_h3_path = os.path.join(target_dir,'popt_h3.npy')
# red_chisq_k3_path = os.path.join(target_dir,'red_chisq_k3.npy')
# red_chisq_h3_path = os.path.join(target_dir,'red_chisq_h3.npy')
dummy_raster_path = os.path.join(target_dir,"dummy_raster_2d.fits")
datacube_k3_path = os.path.join(target_dir,'datacube_k3.npy')
datacube_h3_path = os.path.join(target_dir,'datacube_h3.npy')
errcube_k3_path = os.path.join(target_dir,'errcube_k3.npy')
errcube_h3_path = os.path.join(target_dir,'errcube_h3.npy')

wavelength_k3_path = os.path.join(target_dir,'wavelength_k3.npy')
wavelength_h3_path = os.path.join(target_dir,'wavelength_h3.npy')

LAM_K3_path = os.path.join(target_dir,'LAM_K3.npy')
LAM_H3_path = os.path.join(target_dir,'LAM_H3.npy')
LAM_K2V_path = os.path.join(target_dir,'LAM_K2V.npy')
LAM_K2R_path = os.path.join(target_dir,'LAM_K2R.npy')
LAM_H2V_path = os.path.join(target_dir,'LAM_H2V.npy')
LAM_H2R_path = os.path.join(target_dir,'LAM_H2R.npy')

I_K3_path = os.path.join(target_dir,'I_K3.npy')
I_H3_path = os.path.join(target_dir,'I_H3.npy')
I_K2V_path = os.path.join(target_dir,'I_K2V.npy')
I_K2R_path = os.path.join(target_dir,'I_K2R.npy')
I_H2V_path = os.path.join(target_dir,'I_H2V.npy')
I_H2R_path = os.path.join(target_dir,'I_H2R.npy')

FLUX_K_path = os.path.join(target_dir,'FLUX_K.npy')
FLUX_H_path = os.path.join(target_dir,'FLUX_H.npy')

DOPPSHIFT_k_path = os.path.join(target_dir,'DOPPSHIFT_k.npy')
DOPPSHIFT_h_path = os.path.join(target_dir,'DOPPSHIFT_h.npy')
DOPPSHIFT_av_path = os.path.join(target_dir,'DOPPSHIFT_av.npy')
GRAD_V_path = os.path.join(target_dir,'GRAD_V.npy')
TEMP_k_path = os.path.join(target_dir,'TEMP_k.npy')
TEMP_h_path = os.path.join(target_dir,'TEMP_h.npy')

dummy_raster_m = sm(dummy_raster_path)
# POPT_k3 = np.load(popt_k3_path,allow_pickle=True)
# POPT_h3 = np.load(popt_h3_path,allow_pickle=True)
# RED_CHISQ_k3 = np.load(red_chisq_k3_path,allow_pickle=True)
# RED_CHISQ_h3 = np.load(red_chisq_h3_path,allow_pickle=True)
datacube_k3 = np.load(datacube_k3_path,allow_pickle=True)
datacube_h3 = np.load(datacube_h3_path,allow_pickle=True)
errcube_k3 = np.load(errcube_k3_path,allow_pickle=True)
errcube_h3 = np.load(errcube_h3_path,allow_pickle=True)
wavelength_k3 = np.load(wavelength_k3_path,allow_pickle=True)
wavelength_h3 = np.load(wavelength_h3_path,allow_pickle=True)
DOPPSHIFT_av = np.load(DOPPSHIFT_av_path,allow_pickle=True).T
DOPPSHIFT_k = np.load(DOPPSHIFT_k_path,allow_pickle=True).T
DOPPSHIFT_h = np.load(DOPPSHIFT_h_path,allow_pickle=True).T
TEMP_k = np.load(TEMP_k_path,allow_pickle=True).T
TEMP_h = np.load(TEMP_h_path,allow_pickle=True).T
FLUX_K = np.load(FLUX_K_path,allow_pickle=True).T
FLUX_H = np.load(FLUX_H_path,allow_pickle=True).T

LAM_K2V = np.load(LAM_K2V_path,allow_pickle=True).T
LAM_K2R = np.load(LAM_K2R_path,allow_pickle=True).T
LAM_K3 = np.load(LAM_K3_path,allow_pickle=True).T
LAM_H2V = np.load(LAM_H2V_path,allow_pickle=True).T
LAM_H2R = np.load(LAM_H2R_path,allow_pickle=True).T
LAM_H3 = np.load(LAM_H3_path,allow_pickle=True).T

I_K2V = np.load(I_K2V_path,allow_pickle=True).T
I_K2R = np.load(I_K2R_path,allow_pickle=True).T
I_K3 = np.load(I_K3_path,allow_pickle=True).T
I_H2V = np.load(I_H2V_path,allow_pickle=True).T
I_H2R = np.load(I_H2R_path,allow_pickle=True).T
I_H3 = np.load(I_H3_path,allow_pickle=True).T



data_k3 = datacube_k3[raster_timestep,ypixel]
data_h3 = datacube_h3[raster_timestep,ypixel]
error_k3 = errcube_k3[raster_timestep,ypixel]
error_h3 = errcube_h3[raster_timestep,ypixel]

fig_sp_k,fig_sp_h = plot_MgII_spectra(wavelength_k3,data_k3,error_k3,wavelength_h3,data_h3,error_h3)
ax_k = fig_sp_k.axes[0]
ax_h = fig_sp_h.axes[0]
ax_k.plot([LAM_K2V[ypixel,raster_timestep]],[I_K2V[ypixel,raster_timestep]],'X',markersize=14,color='violet')
ax_k.plot([LAM_K3[ypixel,raster_timestep]],[I_K3[ypixel,raster_timestep]],'X',markersize=14,color='black')
ax_k.plot([LAM_K2R[ypixel,raster_timestep]],[I_K2R[ypixel,raster_timestep]],'X',markersize=14,color='red')

ax_h.plot([LAM_H2V[ypixel,raster_timestep]],[I_H2V[ypixel,raster_timestep]],'X',markersize=14,color='violet')
ax_h.plot([LAM_H3[ypixel,raster_timestep]],[I_H3[ypixel,raster_timestep]],'X',markersize=14,color='black')
ax_h.plot([LAM_H2R[ypixel,raster_timestep]],[I_H2R[ypixel,raster_timestep]],'X',markersize=14,color='red')


m171 = sm(aia171_file)
mhmi = sm(hmi_file)
m_dummy_raster = sm(dummy_raster_path)
m_doppshift_av = sm((DOPPSHIFT_av,m_dummy_raster.meta))
m_doppshift_k = sm((DOPPSHIFT_k,m_dummy_raster.meta))
m_doppshift_h = sm((DOPPSHIFT_h,m_dummy_raster.meta))
m_temp_k = sm((TEMP_k,m_dummy_raster.meta))
m_temp_h = sm((TEMP_h,m_dummy_raster.meta))
m_flux_k = sm((FLUX_K,m_dummy_raster.meta))
m_flux_h = sm((FLUX_H,m_dummy_raster.meta))



# mhmi.quicklook()

bl = [float(m_dummy_raster.bottom_left_coord.Tx.value),float(m_dummy_raster.bottom_left_coord.Ty.value)]
tr = [float(m_dummy_raster.top_right_coord.Tx.value),float(m_dummy_raster.top_right_coord.Ty.value)]


m171_sub = sunpy_map_cut(m171,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
mhmi_sub = sunpy_map_cut(mhmi,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
m_doppshift_av_sub = sunpy_map_cut(m_doppshift_av,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
m_doppshift_k_sub = sunpy_map_cut(m_doppshift_k,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
m_temp_k_sub = sunpy_map_cut(m_temp_k,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
m_doppshift_h_sub = sunpy_map_cut(m_doppshift_h,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
m_temp_h_sub = sunpy_map_cut(m_temp_h,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
m_flux_k_sub = sunpy_map_cut(m_flux_k,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
m_flux_h_sub = sunpy_map_cut(m_flux_h,[bl[0],ext_y[0]],[tr[0],ext_y[1]])



ext_171 = get_extent(m171_sub)
ext_hmi = get_extent(mhmi_sub)
ext_raster = get_extent(m_doppshift_av_sub)

fig_physics = get_MgII_plot(
    m171_sub.data,
    mhmi_sub.data,
    m_doppshift_av_sub.data,
    m_temp_k_sub.data,
    m_flux_k_sub.data,
    ext_171,
    ext_hmi,
    ext_raster
)


AXS = fig_physics.get_axes()
IMS = []
S = m_dummy_raster.pixel_to_world(raster_timestep*u.pix,ypixel*u.pix)
X = S.Tx.value
Y = S.Ty.value
for ax in AXS:
    im = ax.plot([X],[Y],'X',color='red',markersize=4)
    IMS.append(im)

# For Si IV





siiv_results_dir = f"./FITTING/SIV_1394/Gaussian1_Linear_Bg/m_{m}-n_{n}/{file_name}"
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
m_171 = sm(file_171)
m_hmi = sm(file_hmi)




m_dopplershift_sub = cut_map(m_dopplershift,m_dummy_raster_2d)
m_flux_sub = cut_map(m_flux,m_dummy_raster_2d)
m_fwhm_sub = cut_map(m_fwhm,m_dummy_raster_2d)
m_non_th_vel_sub = cut_map(m_non_th_vel,m_dummy_raster_2d)
m_red_chisq_sub = cut_map(m_red_chisq,m_dummy_raster_2d)
m_171_sub = cut_map(m_171,m_dummy_raster_2d)
m_hmi_sub = cut_map(m_hmi,m_dummy_raster_2d)



# PLOTTING
fig = intensity_fwhm_fig1(
    m_dopplershift_sub.data,
    m_flux_sub.data,
    m_171_sub.data,
    m_red_chisq_sub.data,
    m_hmi_sub.data,
    m_fwhm_sub.data,
    m_non_th_vel_sub.data,
    get_extent(m_flux_sub),
    get_extent(m_171_sub),
    get_extent(m_hmi_sub)  
)

spec,noise,I,Mu,Sigma,Slope,Const,red_chisq_point,X,Y = get_spectra_by_indices(
    y0,x0,m_dummy_raster_2d,datacube,errorcube,si_iv_results['popt'],m_red_chisq.data
)
fig_spec = plot_spectrum(wavelength,spec,noise)
fitted_spectra = Models.Gauss1linearbackground(wavelength,I,Mu,Sigma,Slope,Const)
axes_spec = fig_spec.get_axes()
axes_spec[0].plot(wavelength,fitted_spectra)


fig_axes = fig.get_axes()
im0 = fig_axes[0].plot([X],[Y],'X',color='k')
im1 = fig_axes[1].plot([X],[Y],'X',color='k')
im2 = fig_axes[2].plot([X],[Y],'X',color='k')
im3 = fig_axes[3].plot([X],[Y],'X',color='k')
im4 = fig_axes[4].plot([X],[Y],'X',color='k')
im5 = fig_axes[5].plot([X],[Y],'X',color='k')
im6 = fig_axes[6].plot([X],[Y],'X',color='k')
im7 = fig_axes[7].plot([X],[Y],'X',color='k')

IM= [im0,im1,im2,im3,im4,im5,im6,im7]


# UPDATING

def on_key(event):
    global raster_timestep,ypixel,x0,y0
    global AXS
    global ax_k
    global ax_h


    global fig_axes
    global fig
    global fig_spec


    if event.key == "up":
        ypixel+=1
    elif event.key == "down":
        ypixel-=1
    elif event.key == "right":
        raster_timestep+=1
    elif event.key == "left":
        raster_timestep-=1
        
    x0 = raster_timestep
    y0 = ypixel


    if event.inaxes in AXS:

        S = m_dummy_raster.pixel_to_world(raster_timestep*u.pix,ypixel*u.pix)
        X = S.Tx.value
        Y = S.Ty.value
        for im_marker in IMS:
            im_marker[0].set_xdata([X])
            im_marker[0].set_ydata([Y])            
        
        data_k3 = datacube_k3[raster_timestep,ypixel]
        data_h3 = datacube_h3[raster_timestep,ypixel]
        error_k3 = errcube_k3[raster_timestep,ypixel]
        error_h3 = errcube_h3[raster_timestep,ypixel]
        
        ax_k.clear()
        ax_k.set_xlabel('Wavelength ($\AA$)',fontsize=20,fontweight='medium')
        ax_k.set_ylabel('Intensity',fontsize=20,fontweight='medium')
        ax_k.errorbar(
            wavelength_k3,
            data_k3,
            yerr=error_k3,
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
        
        ax_k.plot([LAM_K2V[ypixel,raster_timestep]],[I_K2V[ypixel,raster_timestep]],'X',markersize=14,color='violet')
        ax_k.plot([LAM_K3[ypixel,raster_timestep]],[I_K3[ypixel,raster_timestep]],'X',markersize=14,color='black')
        ax_k.plot([LAM_K2R[ypixel,raster_timestep]],[I_K2R[ypixel,raster_timestep]],'X',markersize=14,color='red')
    

        ax_h.clear()
        ax_h.set_xlabel('Wavelength ($\AA$)',fontsize=20,fontweight='medium')
        ax_h.set_ylabel('Intensity',fontsize=20,fontweight='medium')
        ax_h.errorbar(
            wavelength_h3,
            data_h3,
            yerr=error_h3,
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
                
        ax_h.plot([LAM_H2V[ypixel,raster_timestep]],[I_H2V[ypixel,raster_timestep]],'X',markersize=14,color='violet')
        ax_h.plot([LAM_H3[ypixel,raster_timestep]],[I_H3[ypixel,raster_timestep]],'X',markersize=14,color='black')
        ax_h.plot([LAM_H2R[ypixel,raster_timestep]],[I_H2R[ypixel,raster_timestep]],'X',markersize=14,color='red')
        
        
        
    # For Si IV
    
 

    # if event.inaxes in fig_axes:

        spec,noise,I,Mu,Sigma,Slope,Const,red_chisq_point,X,Y = get_spectra_by_indices(y0,x0,m_dummy_raster_2d,datacube,errorcube,si_iv_results['popt'],m_red_chisq.data)

        for im_marker in IM:
            im_marker[0].set_xdata([X])
            im_marker[0].set_ydata([Y])

        spec_fit = Models.Gauss1linearbackground(wavelength,I,Mu,Sigma,Slope,Const)

        axes_spec[0].clear()
        axes_spec[0].set_xlabel('Wavelength ($\AA$)',fontsize=20,fontweight='medium')
        axes_spec[0].set_ylabel('Intensity',fontsize=20,fontweight='medium')
        axes_spec[0].errorbar(
            wavelength,
            spec,
            yerr=noise,
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

        axes_spec[0].plot(wavelength,spec_fit)
        axes_spec[0].set_title(f"$\\tilde{{\chi}}^2$ = {red_chisq_point}")

        fig.canvas.draw_idle()
        fig_spec.canvas.draw_idle()

        fig_sp_k.canvas.draw_idle()
        fig_sp_h.canvas.draw_idle()

        fig_physics.canvas.draw_idle()


fig_physics.canvas.mpl_connect("key_press_event", on_key)
fig_sp_k.canvas.mpl_connect("key_press_event", on_key)
fig_sp_h.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()




