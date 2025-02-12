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
from skimage import measure
from skimage.draw import polygon2mask


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
ext_y = [350,380]
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
    'dopp_red': np.load(os.path.join(siiv_results_dir1,'dopp_red.npy'),allow_pickle=True),
    'dopp_blue': np.load(os.path.join(siiv_results_dir1,'dopp_blue.npy'),allow_pickle=True)
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

m_dopplershift_red = sm((si_iv_results['dopp_red'],m_dummy_raster_2d.fits_header))
m_dopplershift_blue = sm((si_iv_results['dopp_blue'],m_dummy_raster_2d.fits_header))


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
m_dopplershift_red_cut = cut_map(m_dopplershift_red,m_dummy_raster_2d)
m_dopplershift_blue_cut = cut_map(m_dopplershift_blue,m_dummy_raster_2d)


from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# PLOTTING
fig1 = intensity_fwhm_fig1(
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

fig_dopp = plt.figure(figsize=(6,8))
ax_dopp_red = fig_dopp.add_subplot(1,3,1)
ax_dopp_blue = fig_dopp.add_subplot(1,3,2)
ax_dopp = fig_dopp.add_subplot(1,3,3)


norm_br = TwoSlopeNorm(0,-15,+15)

im_dopp_red = ax_dopp_red.imshow(m_dopplershift_red_cut.data,origin="lower",aspect='auto',cmap = 'bwr',extent=get_extent(m_dopplershift_red_cut),norm=norm_br)
im_dopp_blue = ax_dopp_blue.imshow(m_dopplershift_blue_cut.data,origin="lower",aspect='auto',cmap = 'bwr',extent=get_extent(m_dopplershift_blue_cut),norm=norm_br)
im_dopp = ax_dopp.imshow(m_dopplershift_sub.data,origin="lower",aspect='auto',cmap = 'bwr',extent=get_extent(m_dopplershift_sub),norm=norm_br)


divider_dopp_red = make_axes_locatable(ax_dopp_red)
divider_dopp_blue = make_axes_locatable(ax_dopp_blue)
divider_dopp = make_axes_locatable(ax_dopp)


cax_dopp_red = divider_dopp_red.append_axes("top","3%",pad=0.1)
cax_dopp_blue = divider_dopp_blue.append_axes("top","3%",pad=0.1)
cax_dopp = divider_dopp.append_axes("top","3%",pad=0.1)


cbardopp_red = fig_dopp.colorbar(im_dopp_red, cax=cax_dopp_red, orientation='horizontal', extend='both')
cbardopp_red.set_ticks([-15,-7,0,7,15])
cbardopp_red.set_label("Valocity (km/s) Red", fontsize=14)
cbardopp_red.ax.tick_params(labelsize=14)

cbardopp_blue = fig_dopp.colorbar(im_dopp_blue, cax=cax_dopp_blue, orientation='horizontal', extend='both')
cbardopp_blue.set_ticks([-15,-7,0,7,15])
cbardopp_blue.set_label("Valocity (km/s) Blue when red chisq > 1", fontsize=14)
cbardopp_blue.ax.tick_params(labelsize=14)

cbardopp = fig_dopp.colorbar(im_dopp, cax=cax_dopp, orientation='horizontal', extend='both')
cbardopp.set_ticks([-15,-7,0,7,15])
cbardopp.set_label("Valocity (km/s) Blue when red chisq > 1", fontsize=14)
cbardopp.ax.tick_params(labelsize=14)

cax_dopp_red.xaxis.set_ticks_position('top')
cax_dopp_red.xaxis.set_label_position('top')

cax_dopp_blue.xaxis.set_ticks_position('top')
cax_dopp_blue.xaxis.set_label_position('top')

cax_dopp.xaxis.set_ticks_position('top')
cax_dopp.xaxis.set_label_position('top')



spec,noise,I,Mu,Sigma,Slope,Const,red_chisq_point,X,Y = get_spectra_by_indices(
    y0,x0,m_dummy_raster_2d,datacube,errorcube,si_iv_results['popt'],m_red_chisq.data
)

_,_,I1,I2,Mu1,Mu2,Sigma1,Sigma2,Slope,Const,red_chisq_point_double,_,_ = get_spectra_by_indices_double(
    y0,x0,m_dummy_raster_2d,datacube,errorcube,si_iv_results['popt_double'],m_red_chisq_double.data
)

fig_spec = plot_spectrum(wavelength,spec,noise)
fitted_spectra = Models.Gauss1linearbackground(wavelength,I,Mu,Sigma,Slope,Const)
fitted_spectra_double1 = Models.Gauss1linearbackground(wavelength,I1,Mu1,Sigma1,Slope,Const)
fitted_spectra_double2 = Models.Gauss1linearbackground(wavelength,I2,Mu2,Sigma2,Slope,Const)
fitted_spectra_double = Models.Gauss2linearbackground(wavelength,I1,I2,Mu1,Mu2,Sigma1,Sigma2,Slope,Const)



axes_spec = fig_spec.get_axes()
axes_spec[0].plot(wavelength,fitted_spectra,color='red')

axes_spec[0].plot(wavelength,fitted_spectra_double,color='k')
axes_spec[0].plot(wavelength,fitted_spectra_double1,color='green')
axes_spec[0].plot(wavelength,fitted_spectra_double2,color='blue')


fig_axes = fig1.get_axes()
im0 = fig_axes[0].plot([X],[Y],'X',color='k')
im1 = fig_axes[1].plot([X],[Y],'X',color='k')
im2 = fig_axes[2].plot([X],[Y],'X',color='k')
im3 = fig_axes[3].plot([X],[Y],'X',color='k')
im4 = fig_axes[4].plot([X],[Y],'X',color='k')
im5 = fig_axes[5].plot([X],[Y],'X',color='k')
im6 = fig_axes[6].plot([X],[Y],'X',color='k')
im7 = fig_axes[7].plot([X],[Y],'X',color='k')

im_dopp_red = ax_dopp_red.plot([X],[Y],'X',color='k')
im_dopp_blue = ax_dopp_blue.plot([X],[Y],'X',color='k')
im_dopp = ax_dopp.plot([X],[Y],'X',color='k')




# UPDATING

IM0= [im0,im1,im2,im3,im4,im5,im6,im7,im_dopp_red,im_dopp_blue,im_dopp]
def on_key(event):
    global y0,x0
    global fig_axes
    global fig1
    global fig_spec

    if event.key == "up":
        # print("Up arrow key pressed!")
        y0+=1
    elif event.key == "down":
        # print("Down arrow key pressed!")
        y0-=1
    elif event.key == "right":
        # print("Right arrow key pressed!")
        x0+=1
    elif event.key == "left":
        # print("Left arrow key pressed!")
        x0-=1

    if event.inaxes in fig_axes or event.inaxes in [ax_dopp_blue,ax_dopp_red,ax_dopp]:

        spec,noise,I,Mu,Sigma,Slope,Const,red_chisq_point,X,Y = get_spectra_by_indices(y0,x0,m_dummy_raster_2d,datacube,errorcube,si_iv_results['popt'],m_red_chisq.data)

        _,_,I1,I2,Mu1,Mu2,Sigma1,Sigma2,Slope,Const,red_chisq_point_double,_,_ = get_spectra_by_indices_double(
            y0,x0,m_dummy_raster_2d,datacube,errorcube,si_iv_results['popt_double'],m_red_chisq_double.data
        )
        
        for im_marker in IM0:
            im_marker[0].set_xdata([X])
            im_marker[0].set_ydata([Y])

        spec_fit = Models.Gauss1linearbackground(wavelength,I,Mu,Sigma,Slope,Const)

        fitted_spectra_double1 = Models.Gauss1linearbackground(wavelength,I1,Mu1,Sigma1,Slope,Const)
        fitted_spectra_double2 = Models.Gauss1linearbackground(wavelength,I2,Mu2,Sigma2,Slope,Const)
        fitted_spectra_double = Models.Gauss2linearbackground(wavelength,I1,I2,Mu1,Mu2,Sigma1,Sigma2,Slope,Const)
        
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

        axes_spec[0].plot(wavelength,fitted_spectra_double,color='k')
        axes_spec[0].plot(wavelength,fitted_spectra_double1,color='green')
        axes_spec[0].plot(wavelength,fitted_spectra_double2,color='blue')
        
        
        axes_spec[0].plot(wavelength,spec_fit,color='red',linestyle='-.')
        axes_spec[0].set_title(f"$\\tilde{{\chi}}^2$ = {red_chisq_point:.2f}\n$\\tilde{{\chi}}^2 double$ = {red_chisq_point_double:.2f}")

        fig1.canvas.draw_idle()
        fig_spec.canvas.draw_idle()
        fig_dopp.canvas.draw_idle()


fig1.canvas.mpl_connect("key_press_event", on_key)
fig_dopp.canvas.mpl_connect("key_press_event", on_key)



plt.show()
