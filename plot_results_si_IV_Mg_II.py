import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from sunpy.map import Map as sm
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Example imports for your own helper functions/models
from Functions.utilities import get_extent
from Functions.sunpy_map_cut import sunpy_map_cut
from models import Models

##############################################################################
# 1. HELPER FUNCTIONS
##############################################################################

def plot_spectrum(wavelength, data, err):
    """
    Generic function to plot a single spectrum with error bars.
    Returns the figure object.
    """
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Intensity',     fontsize=12)

    ax.errorbar(
        wavelength,
        data,
        yerr=err,
        drawstyle='steps-mid',
        color='black',
        capsize=3,
        marker='o',
        markersize=3,
        markerfacecolor='white',
        markeredgecolor='blue',
        ecolor='red'
    )
    return fig

def get_spectra_siiv(row_p, col_p, m_dummy_raster, datacube, errorcube, popt, reduced_chisq):
    """
    Your existing function that extracts the Si IV spectrum at (row_p, col_p),
    along with the best-fit parameters, etc.
    """
    # Convert pixel -> world coords
    S = m_dummy_raster.pixel_to_world(col_p * u.pix, row_p * u.pix)
    X = S.Tx.value
    Y = S.Ty.value

    spec  = datacube[col_p, row_p]
    noise = errorcube[col_p, row_p]
    spec[spec < 0] = 0

    I      = popt[..., 0][col_p, row_p]
    Mu     = popt[..., 1][col_p, row_p]
    Sigma  = popt[..., 2][col_p, row_p]
    Slope  = popt[..., 3][col_p, row_p]
    Const  = popt[..., 4][col_p, row_p]
    redchi = reduced_chisq[row_p, col_p]

    return spec, noise, I, Mu, Sigma, Slope, Const, redchi, X, Y


def plot_mgii_spectra(wave_k3, data_k3, err_k3,
                      wave_h3, data_h3, err_h3,
                      # optional: K2V etc. below
                      lam_k2v, lam_k3, lam_k2r,
                      i_k2v,  i_k3,  i_k2r,
                      lam_h2v, lam_h3_, lam_h2r,
                      i_h2v,   i_h3_,  i_h2r):
    """
    Example function that produces two separate figures:
      - Fig for Mg II K-line (K3, plus K2v/K2r peaks)
      - Fig for Mg II H-line (H3, plus H2v/H2r peaks)

    Returns (fig_k, fig_h) 
    """
    # Plot K-line
    fig_k = plot_spectrum(wave_k3, data_k3, err_k3)
    ax_k = fig_k.axes[0]

    # Mark K2v, K3, K2r
    ax_k.plot([lam_k2v], [i_k2v], 'X', markersize=10, color='violet')
    ax_k.plot([lam_k3],  [i_k3],  'X', markersize=10, color='black')
    ax_k.plot([lam_k2r], [i_k2r], 'X', markersize=10, color='red')
    ax_k.set_title("Mg II K-line")

    # Plot H-line
    fig_h = plot_spectrum(wave_h3, data_h3, err_h3)
    ax_h = fig_h.axes[0]

    # Mark H2v, H3, H2r
    ax_h.plot([lam_h2v], [i_h2v], 'X', markersize=10, color='violet')
    ax_h.plot([lam_h3_], [i_h3_], 'X', markersize=10, color='black')
    ax_h.plot([lam_h2r], [i_h2r], 'X', markersize=10, color='red')
    ax_h.set_title("Mg II H-line")

    return fig_k, fig_h


##############################################################################
# 2. LOAD ALL DATA (Si IV + Mg II + AIA/HMI, etc.)
##############################################################################

# --------------------
# (A) Load your Si IV data
# --------------------
m = 2
n = 1
D = 'D044'
file_name_siiv = "iris_l2_20160319_145728_3623010639_raster_t000_r00000"

siiv_results_dir = f"./FITTING/SIV_1394/Gaussian1_Linear_Bg/m_{m}-n_{n}/{file_name_siiv}"
datacube_siiv   = np.load(os.path.join(siiv_results_dir,'data.npy'), allow_pickle=True)
errorcube_siiv  = np.load(os.path.join(siiv_results_dir,'err.npy'),  allow_pickle=True)
wavelength_siiv = np.load(os.path.join(siiv_results_dir,'wavelength.npy'), allow_pickle=True)
popt_siiv       = np.load(os.path.join(siiv_results_dir,'popt.npy'), allow_pickle=True)
redchisq_siiv   = np.load(os.path.join(siiv_results_dir,'red_chisq.npy'), allow_pickle=True).T

m_dummy_siiv_2d = sm(os.path.join(siiv_results_dir,'dummy_raster_2d.fits'))

# Example submaps (like Dopplershift, flux, fwhm, etc.)
m_dopplershift_siiv = sm((np.load(os.path.join(siiv_results_dir,'dopplershift.npy'), allow_pickle=True).T,
                          m_dummy_siiv_2d.fits_header))
m_flux_siiv         = sm((np.load(os.path.join(siiv_results_dir,'flux.npy'), allow_pickle=True).T,
                          m_dummy_siiv_2d.fits_header))

m_red_chisq_siiv         = sm((np.load(os.path.join(siiv_results_dir,'red_chisq.npy'), allow_pickle=True).T,
                          m_dummy_siiv_2d.fits_header))

m_non_th_vel_siiv         = sm((np.load(os.path.join(siiv_results_dir,'non_th_vel.npy'), allow_pickle=True).T,
                          m_dummy_siiv_2d.fits_header))

m_fwhm_siiv         = sm((np.load(os.path.join(siiv_results_dir,'fwhm.npy'), allow_pickle=True).T,
                          m_dummy_siiv_2d.fits_header))


# ... load more if needed (fwhm, red_chisq map, etc.)

# Path to AIA/HMI for context
file_171_siiv = glob.glob(f"data_scanwise_m-{m}_n-{n}/{D}/{file_name_siiv}/mid/171/*.fits")[0]
file_hmi_siiv = glob.glob(f"data_scanwise_m-{m}_n-{n}/{D}/{file_name_siiv}/mid/hmi/*.fits")[0]
m_171_siiv    = sm(file_171_siiv)
m_hmi_siiv    = sm(file_hmi_siiv)

# --------------------
# (B) Load your Mg II data
# --------------------
file_name_mgii = "iris_l2_20160319_145728_3623010639_raster_t000_r00000.fits"
# Or rename as needed

target_dir_mg = f"./FITTING/MgII/{D}/m_{m}-n_{n}/{os.path.splitext(os.path.basename(file_name_mgii))[0]}"
dummy_raster_mg  = sm(os.path.join(target_dir_mg, "dummy_raster_2d.fits"))
datacube_k3 = np.load(os.path.join(target_dir_mg,'datacube_k3.npy'), allow_pickle=True)
datacube_h3 = np.load(os.path.join(target_dir_mg,'datacube_h3.npy'), allow_pickle=True)
errcube_k3  = np.load(os.path.join(target_dir_mg,'errcube_k3.npy'),  allow_pickle=True)
errcube_h3  = np.load(os.path.join(target_dir_mg,'errcube_h3.npy'),  allow_pickle=True)
wave_k3     = np.load(os.path.join(target_dir_mg,'wavelength_k3.npy'), allow_pickle=True)
wave_h3     = np.load(os.path.join(target_dir_mg,'wavelength_h3.npy'), allow_pickle=True)

# Example “physics” arrays for Mg II 
doppshift_k = np.load(os.path.join(target_dir_mg,'DOPPSHIFT_k.npy'), allow_pickle=True).T
doppshift_h = np.load(os.path.join(target_dir_mg,'DOPPSHIFT_h.npy'), allow_pickle=True).T
flux_k      = np.load(os.path.join(target_dir_mg,'FLUX_K.npy'),      allow_pickle=True).T
flux_h      = np.load(os.path.join(target_dir_mg,'FLUX_H.npy'),      allow_pickle=True).T
temp_h      = np.load(os.path.join(target_dir_mg,'TEMP_h.npy'),      allow_pickle=True).T
temp_k      = np.load(os.path.join(target_dir_mg,'TEMP_k.npy'),      allow_pickle=True).T


# Etc.

# Example arrays for peak/dip wavelengths (K2v, K3, K2r, etc.)
LAM_K2V = np.load(os.path.join(target_dir_mg,'LAM_K2V.npy'), allow_pickle=True).T
LAM_K2R = np.load(os.path.join(target_dir_mg,'LAM_K2R.npy'), allow_pickle=True).T
LAM_K3  = np.load(os.path.join(target_dir_mg,'LAM_K3.npy'),  allow_pickle=True).T
LAM_H2V = np.load(os.path.join(target_dir_mg,'LAM_H2V.npy'), allow_pickle=True).T
LAM_H2R = np.load(os.path.join(target_dir_mg,'LAM_H2R.npy'), allow_pickle=True).T
LAM_H3  = np.load(os.path.join(target_dir_mg,'LAM_H3.npy'),  allow_pickle=True).T

# Example arrays for intensities at those peaks/dips
I_K2V = np.load(os.path.join(target_dir_mg,'I_K2V.npy'), allow_pickle=True).T
I_K2R = np.load(os.path.join(target_dir_mg,'I_K2R.npy'), allow_pickle=True).T
I_K3  = np.load(os.path.join(target_dir_mg,'I_K3.npy'),  allow_pickle=True).T
I_H2V = np.load(os.path.join(target_dir_mg,'I_H2V.npy'), allow_pickle=True).T
I_H2R = np.load(os.path.join(target_dir_mg,'I_H2R.npy'), allow_pickle=True).T
I_H3  = np.load(os.path.join(target_dir_mg,'I_H3.npy'),  allow_pickle=True).T


##############################################################################
# 3. MAKE SUBMAPS
##############################################################################

# Suppose you want to cut both Si IV and Mg II to the same region in y:
# ext_y = [320, 400]
ext_y = [350, 380]

def cut_map(m_sunpy, m_dummy):
    """Helper to cut a sub-region based on dummy raster bottom-left & top-right."""
    bl = [float(m_dummy.bottom_left_coord.Tx.value),
          float(m_dummy.bottom_left_coord.Ty.value)]
    tr = [float(m_dummy.top_right_coord.Tx.value),
          float(m_dummy.top_right_coord.Ty.value)]
    # Now override the y-limits
    return sunpy_map_cut(m_sunpy, [bl[0], ext_y[0]], [tr[0], ext_y[1]])

# -- Si IV submaps --
m_dopplershift_siiv_sub = cut_map(m_dopplershift_siiv, m_dummy_siiv_2d)
m_flux_siiv_sub         = cut_map(m_flux_siiv,         m_dummy_siiv_2d)
m_171_siiv_sub          = cut_map(m_171_siiv,          m_dummy_siiv_2d)
m_hmi_siiv_sub          = cut_map(m_hmi_siiv,          m_dummy_siiv_2d)
m_red_chisq_siiv_sub    = cut_map(m_red_chisq_siiv,    m_dummy_siiv_2d)
m_fwhm_siiv_sub    = cut_map(m_fwhm_siiv,    m_dummy_siiv_2d)
m_non_th_vel_siiv_sub    = cut_map(m_non_th_vel_siiv,    m_dummy_siiv_2d)

# etc.

# -- Mg II submaps (some examples) --
m_doppshift_k_map = sm((doppshift_k, dummy_raster_mg.meta))
m_doppshift_h_map = sm((doppshift_h, dummy_raster_mg.meta))
m_flux_k_map      = sm((flux_k,      dummy_raster_mg.meta))
m_flux_h_map      = sm((flux_h,      dummy_raster_mg.meta))
m_temp_k_map      = sm((temp_k,      dummy_raster_mg.meta))
m_temp_h_map      = sm((temp_h,      dummy_raster_mg.meta))


m_doppshift_k_sub = cut_map(m_doppshift_k_map, dummy_raster_mg)
m_doppshift_h_sub = cut_map(m_doppshift_h_map, dummy_raster_mg)
m_flux_k_sub      = cut_map(m_flux_k_map,      dummy_raster_mg)
m_flux_h_sub      = cut_map(m_flux_h_map,      dummy_raster_mg)
m_temp_k_sub      = cut_map(m_temp_k_map,      dummy_raster_mg)
m_temp_h_sub      = cut_map(m_temp_h_map,      dummy_raster_mg)

# etc.

##############################################################################
# 4. PLOT OVERVIEW MAPS (Si IV + Mg II)
#    (You can do separate figures or combine them; for brevity, let's do two)
##############################################################################

# For demonstration, let's assume you have a function to produce an 8-panel
# figure for Si IV. We'll call it fig_siiv. Or just do a simple example:


norm_br = TwoSlopeNorm(0,-15,+15)
norm_mag = TwoSlopeNorm(0,-20,+700)

fig_siiv, axs_siiv = plt.subplots(1, 7, figsize=(20, 8))

im1 = axs_siiv[0].imshow(m_dopplershift_siiv_sub.data,origin="lower", extent=get_extent(m_dopplershift_siiv_sub),norm=norm_br,cmap='bwr',aspect='auto')
# axs_siiv[0].set_title("Si IV Dopplershift")

im2 = axs_siiv[1].imshow(m_flux_siiv_sub.data,origin="lower", extent=get_extent(m_flux_siiv_sub),vmin=0,vmax=2,aspect='auto')
# axs_siiv[1].set_title("Si IV Flux")

im3 = axs_siiv[2].imshow(m_171_siiv_sub.data,origin="lower", extent=get_extent(m_171_siiv_sub),vmin=40,vmax=500,aspect='auto',cmap='sdoaia171')
# axs_siiv[2].set_title("AIA 171")

im4 = axs_siiv[3].imshow(m_red_chisq_siiv_sub.data,origin="lower", extent=get_extent(m_red_chisq_siiv_sub),vmin=0,vmax=2,aspect='auto')
# axs_siiv[3].set_title("Si IV red xhisq")

im5 = axs_siiv[4].imshow(m_hmi_siiv_sub.data,origin="lower", extent=get_extent(m_hmi_siiv_sub),aspect='auto',norm=norm_mag,cmap='hmimag')
# axs_siiv[4].set_title("HMI Blos")

im6 = axs_siiv[5].imshow(m_fwhm_siiv_sub.data,origin="lower", extent=get_extent(m_fwhm_siiv_sub),vmin=0,vmax=50,aspect='auto')
# axs_siiv[5].set_title("Si IV fwhm")

im7 = axs_siiv[6].imshow(m_non_th_vel_siiv_sub.data,origin="lower", extent=get_extent(m_non_th_vel_siiv_sub),vmin=0,vmax=50,aspect='auto')
# axs_siiv[6].set_title("Si IV Non th vel")




divider1 = make_axes_locatable(axs_siiv[0])
divider2 = make_axes_locatable(axs_siiv[1])
divider3 = make_axes_locatable(axs_siiv[2])
divider4 = make_axes_locatable(axs_siiv[3])
divider5 = make_axes_locatable(axs_siiv[4])
divider6 = make_axes_locatable(axs_siiv[5])
divider7 = make_axes_locatable(axs_siiv[6])


cax1 = divider1.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
cax2 = divider2.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
cax3 = divider3.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
cax4 = divider4.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
cax5 = divider5.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
cax6 = divider6.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
cax7 = divider7.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar

cbar1 = fig_siiv.colorbar(im1, cax=cax1, orientation='horizontal', extend='both')
cbar1.set_ticks([-15,-7,0,7,15])
cbar1.set_label("Valocity (km/s)", fontsize=14)
cbar1.ax.tick_params(labelsize=14)

cbar2 = fig_siiv.colorbar(im2, cax=cax2, orientation='horizontal', extend='max')
cbar2.set_label("Flux (DN/s)", fontsize=14)
cbar2.ax.tick_params(labelsize=14)

cbar3 = fig_siiv.colorbar(im3, cax=cax3, orientation='horizontal', extend='both')
cbar3.set_ticks([40,150,250,350,500])
cbar3.set_label("AIA 171 (DN/s)", fontsize=14)
cbar3.ax.tick_params(labelsize=14)

cbar4 = fig_siiv.colorbar(im4, cax=cax4, orientation='horizontal', extend='both')
cbar4.set_label("$\\tilde{\chi}^2$", fontsize=14)
cbar4.ax.tick_params(labelsize=14)

cbar5 = fig_siiv.colorbar(im5, cax=cax5, orientation='horizontal', extend='both')
cbar5.set_ticks([-20,0,300,700])
cbar5.set_label("$B_{los}$", fontsize=14)
cbar5.ax.tick_params(labelsize=14)

cbar6 = fig_siiv.colorbar(im6, cax=cax6, orientation='horizontal', extend='max')
# cbar6.set_ticks([-20,0,300,700])
cbar6.set_label("fwhm (km/s)", fontsize=14)
cbar6.ax.tick_params(labelsize=14)

cbar7 = fig_siiv.colorbar(im7, cax=cax7, orientation='horizontal', extend='max')
# cbar7.set_ticks([-20,0,300,700])
cbar7.set_label("$V_{n\_{th}}$ (km/s)", fontsize=14)
cbar7.ax.tick_params(labelsize=14)

cax1.xaxis.set_ticks_position('top')
cax1.xaxis.set_label_position('top')

cax2.xaxis.set_ticks_position('top')
cax2.xaxis.set_label_position('top')

cax3.xaxis.set_ticks_position('top')
cax3.xaxis.set_label_position('top')

cax4.xaxis.set_ticks_position('top')
cax4.xaxis.set_label_position('top')

cax5.xaxis.set_ticks_position('top')
cax5.xaxis.set_label_position('top')

cax6.xaxis.set_ticks_position('top')
cax6.xaxis.set_label_position('top')

cax7.xaxis.set_ticks_position('top')
cax7.xaxis.set_label_position('top')

# Similarly, a figure for the Mg II submaps:
fig_mgii, axs_mg = plt.subplots(1, 5, figsize=(18, 8))

im_ax1_fig_physics = axs_mg[0].imshow(m_171_siiv_sub.data,origin="lower", extent=get_extent(m_171_siiv_sub))
# axs_mg[0].set_title("AIA 171")

im_ax2_fig_physics = axs_mg[1].imshow(m_hmi_siiv_sub.data,origin="lower", extent=get_extent(m_hmi_siiv_sub))
# axs_mg[1].set_title("HMI")


im_ax3_fig_physics = axs_mg[2].imshow(m_doppshift_k_sub.data,origin="lower", extent=get_extent(m_doppshift_k_sub),vmin=-5,vmax= 5,cmap='RdBu_r')
# axs_mg[2].set_title("Mg II K Dopplershift")

im_ax4_fig_physics = axs_mg[3].imshow(m_temp_k_sub.data,origin="lower", extent=get_extent(m_temp_k_sub))
# axs_mg[3].set_title("Temp")

im_ax5_fig_physics = axs_mg[4].imshow(m_flux_k_sub.data,origin="lower",extent=get_extent(m_flux_k_sub))
# axs_mg[4].set_title("Mg II K Flux")


divider1_mg = make_axes_locatable(axs_mg[0])
divider2_mg = make_axes_locatable(axs_mg[1])
divider3_mg = make_axes_locatable(axs_mg[2])
divider4_mg = make_axes_locatable(axs_mg[3])
divider5_mg = make_axes_locatable(axs_mg[4])


cax1_mg = divider1_mg.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
cax2_mg = divider2_mg.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
cax3_mg = divider3_mg.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
cax4_mg = divider4_mg.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
cax5_mg = divider5_mg.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar

cbar1_mg = fig_mgii.colorbar(im_ax1_fig_physics, cax=cax1_mg, orientation='horizontal', extend='both')
# cbar1.set_ticks([40,150,250,350,500])
cbar1_mg.set_label("AIA 171 (DN/s)", fontsize=14)
cbar1_mg.ax.tick_params(labelsize=14)


cbar2_mg = fig_mgii.colorbar(im_ax2_fig_physics, cax=cax2_mg, orientation='horizontal', extend='max')
# cbar2.set_label("Flux (DN/s)", fontsize=14)
# cbar2.ax.tick_params(labelsize=14)
cbar2_mg.set_ticks([-20,0,300,700])
cbar2_mg.set_label("$B_{los}$", fontsize=14)


cbar3_mg = fig_mgii.colorbar(im_ax3_fig_physics, cax=cax3_mg, orientation='horizontal', extend='both')
# cbar3.set_ticks([40,150,250,350,500])
cbar3_mg.set_label("Dopplershift", fontsize=14)
cbar3_mg.ax.tick_params(labelsize=14)


cbar4_mg = fig_mgii.colorbar(im_ax4_fig_physics, cax=cax4_mg, orientation='horizontal', extend='both')
# cbar6.set_ticks([-20,0,300,700])
cbar4_mg.set_label("$T_B$", fontsize=14)
cbar4_mg.ax.tick_params(labelsize=14)


cbar5_mg = fig_mgii.colorbar(im_ax5_fig_physics, cax=cax5_mg, orientation='horizontal', extend='both')
# cbar5.set_ticks([-20,0,300,700])
cbar5_mg.set_label("Flux", fontsize=14)
cbar5_mg.ax.tick_params(labelsize=14)




cax1_mg.xaxis.set_ticks_position('top')
cax1_mg.xaxis.set_label_position('top')

cax2_mg.xaxis.set_ticks_position('top')
cax2_mg.xaxis.set_label_position('top')

cax3_mg.xaxis.set_ticks_position('top')
cax3_mg.xaxis.set_label_position('top')

cax4_mg.xaxis.set_ticks_position('top')
cax4_mg.xaxis.set_label_position('top')

cax5_mg.xaxis.set_ticks_position('top')
cax5_mg.xaxis.set_label_position('top')



fig_dopplershifts ,axs_dopplershift = plt.subplots(1,2,figsize=(4,8))

im_mg_dopp = axs_dopplershift[0].imshow(m_doppshift_k_sub.data,origin="lower", extent=get_extent(m_doppshift_k_sub),vmin=-5,vmax= 5,cmap='RdBu_r')
im_si_dopp = axs_dopplershift[1].imshow(m_dopplershift_siiv_sub.data,origin="lower", extent=get_extent(m_dopplershift_siiv_sub),norm=norm_br,cmap='bwr',aspect='auto')



# Markers that show the chosen pixel in each figure
marker_siiv_1 = axs_siiv[0].plot([], [], 'X', color='red')[0]
marker_siiv_2 = axs_siiv[1].plot([], [], 'X', color='red')[0]
marker_siiv_3 = axs_siiv[2].plot([], [], 'X', color='red')[0]
marker_siiv_4 = axs_siiv[3].plot([], [], 'X', color='red')[0]
marker_siiv_5 = axs_siiv[4].plot([], [], 'X', color='red')[0]
marker_siiv_6 = axs_siiv[5].plot([], [], 'X', color='red')[0]
marker_siiv_7 = axs_siiv[6].plot([], [], 'X', color='red')[0]



marker_mg_1   = axs_mg[0].plot([], [], 'X', color='red')[0]
marker_mg_2   = axs_mg[1].plot([], [], 'X', color='red')[0]
marker_mg_3   = axs_mg[2].plot([], [], 'X', color='red')[0]
marker_mg_4   = axs_mg[3].plot([], [], 'X', color='red')[0]
marker_mg_5   = axs_mg[4].plot([], [], 'X', color='red')[0]

marker_mg_dopp   = axs_dopplershift[0].plot([], [], 'X', color='red')[0]
marker_si_dopp   = axs_dopplershift[1].plot([], [], 'X', color='red')[0]


# We will have 4 submaps here, but you can expand to more.

##############################################################################
# 5. PLOT THE SPECTRA FIGURES (One for Si IV, Two for Mg II K & H)
##############################################################################

# (A) Si IV
fig_siiv_spec = plot_spectrum(wavelength_siiv, datacube_siiv[0,0], errorcube_siiv[0,0])
ax_siiv_spec  = fig_siiv_spec.axes[0]
ax_siiv_spec.set_title("Si IV Spectrum")

# (B) Mg II K/H 
# We can produce two separate figures or one figure with two subplots. Here let's do separate:
fig_mg_k = plot_spectrum(wave_k3, datacube_k3[0, 0], errcube_k3[0, 0])
ax_mg_k   = fig_mg_k.axes[0]
ax_mg_k.set_title("Mg II K")

fig_mg_h = plot_spectrum(wave_h3, datacube_h3[0, 0], errcube_h3[0, 0])
ax_mg_h   = fig_mg_h.axes[0]
ax_mg_h.set_title("Mg II H")

# Mark the K2v, K3, K2r lines
mk_k2v = ax_mg_k.plot([], [], 'X', markersize=10, color='violet')[0]
mk_k3  = ax_mg_k.plot([], [], 'X', markersize=10, color='black')[0]
mk_k2r = ax_mg_k.plot([], [], 'X', markersize=10, color='red')[0]

mh_h2v = ax_mg_h.plot([], [], 'X', markersize=10, color='violet')[0]
mh_h3  = ax_mg_h.plot([], [], 'X', markersize=10, color='black')[0]
mh_h2r = ax_mg_h.plot([], [], 'X', markersize=10, color='red')[0]


##############################################################################
# 6. INTERACTIVE CALLBACK
##############################################################################
# We'll keep track of a single global (x0, y0) pixel
x0, y0 = 12, 201  # example initial pixel

def on_key(event):
    global x0, y0

    if event.key == "up":
        y0 += 1
    elif event.key == "down":
        y0 -= 1
    elif event.key == "right":
        x0 += 1
    elif event.key == "left":
        x0 -= 1

    # 1) Update coordinates in world space for both rasters.
    #    NOTE: If your raster dims are row=y, col=x, be consistent with indexing:
    #    Some IRIS data are stored as [x, y], some as [y, x], so watch out carefully!
    #    Here we'll match your get_spectra_siiv: row=y, col=x
    #    For the dummy maps, you might also do: S_siiv = m_dummy_siiv_2d.pixel_to_world(x0*u.pix, y0*u.pix)

    # --- Si IV coordinate ---
    S_siiv = m_dummy_siiv_2d.pixel_to_world(x0*u.pix, y0*u.pix)
    X_siiv = S_siiv.Tx.value
    Y_siiv = S_siiv.Ty.value

    # --- Mg II coordinate ---
    S_mg = dummy_raster_mg.pixel_to_world(x0*u.pix, y0*u.pix)
    X_mg = S_mg.Tx.value
    Y_mg = S_mg.Ty.value

    # 2) Update the 'X' marker positions on each map
    marker_siiv_1.set_xdata([X_siiv])
    marker_siiv_1.set_ydata([Y_siiv])
    marker_siiv_2.set_xdata([X_siiv])
    marker_siiv_2.set_ydata([Y_siiv])
    marker_siiv_3.set_xdata([X_siiv])
    marker_siiv_3.set_ydata([Y_siiv])
    marker_siiv_4.set_xdata([X_siiv])
    marker_siiv_4.set_ydata([Y_siiv])
    marker_siiv_5.set_xdata([X_siiv])
    marker_siiv_5.set_ydata([Y_siiv])
    marker_siiv_6.set_xdata([X_siiv])
    marker_siiv_6.set_ydata([Y_siiv])
    marker_siiv_7.set_xdata([X_siiv])
    marker_siiv_7.set_ydata([Y_siiv])
     

    marker_mg_1.set_xdata([X_mg])
    marker_mg_1.set_ydata([Y_mg])
    marker_mg_2.set_xdata([X_mg])
    marker_mg_2.set_ydata([Y_mg])
    marker_mg_3.set_xdata([X_mg])
    marker_mg_3.set_ydata([Y_mg])
    marker_mg_4.set_xdata([X_mg])
    marker_mg_4.set_ydata([Y_mg])
    marker_mg_5.set_xdata([X_mg])
    marker_mg_5.set_ydata([Y_mg])


    # 3) Fetch the new spectra for Si IV
    siiv_spec, siiv_err, I, Mu, Sigma, Slope, Const, red_chi, X_, Y_ = \
        get_spectra_siiv(y0, x0, m_dummy_siiv_2d,
                         datacube_siiv, errorcube_siiv,
                         popt_siiv, redchisq_siiv)
    # Recompute best-fit
    fitted_siiv = Models.Gauss1linearbackground(wavelength_siiv, I, Mu, Sigma, Slope, Const)

    # Update the Si IV spectrum figure
    ax_siiv_spec.clear()
    ax_siiv_spec.set_xlabel('Wavelength (Å)')
    ax_siiv_spec.set_ylabel('Intensity')
    ax_siiv_spec.errorbar(
        wavelength_siiv, siiv_spec, yerr=siiv_err,
        drawstyle='steps-mid',
        color='black', capsize=3, marker='o',
        markersize=3, markerfacecolor='white',
        markeredgecolor='blue', ecolor='red'
    )
    ax_siiv_spec.plot(wavelength_siiv, fitted_siiv, color='green')
    ax_siiv_spec.set_title(f"Si IV Spectrum\nred. χ²={red_chi:.3f}")

    # 4) Fetch and update the Mg II spectra (K & H)
    # Make sure you’re consistent with array shape: e.g. datacube_k3[time, y, x], or [x, y]? 
    # Below assumes [x, y], so be sure to adapt if needed.
    mg_k3  = datacube_k3[x0, y0]
    err_k3 = errcube_k3 [x0, y0]
    mg_h3  = datacube_h3[x0, y0]
    err_h3 = errcube_h3 [x0, y0]

    # Update K-line figure
    ax_mg_k.clear()
    ax_mg_k.set_xlabel("Wavelength (Å)")
    ax_mg_k.set_ylabel("Intensity")
    ax_mg_k.errorbar(
        wave_k3, mg_k3, yerr=err_k3,
        drawstyle='steps-mid',
        color='black', capsize=3, marker='o',
        markersize=3, markerfacecolor='white',
        markeredgecolor='blue', ecolor='red'
    )
    # Mark the K2v, K3, K2r
    ax_mg_k.plot(
        [LAM_K2V[y0, x0]], [I_K2V[y0, x0]], 'X', color='violet', markersize=10
    )
    ax_mg_k.plot(
        [LAM_K3 [y0, x0]], [I_K3 [y0, x0]], 'X', color='black',  markersize=10
    )
    ax_mg_k.plot(
        [LAM_K2R[y0, x0]], [I_K2R[y0, x0]], 'X', color='red',    markersize=10
    )
    ax_mg_k.set_title("Mg II K-line")

    # Update H-line figure
    ax_mg_h.clear()
    ax_mg_h.set_xlabel("Wavelength (Å)")
    ax_mg_h.set_ylabel("Intensity")
    ax_mg_h.errorbar(
        wave_h3, mg_h3, yerr=err_h3,
        drawstyle='steps-mid',
        color='black', capsize=3, marker='o',
        markersize=3, markerfacecolor='white',
        markeredgecolor='blue', ecolor='red'
    )
    # Mark the H2v, H3, H2r
    ax_mg_h.plot(
        [LAM_H2V[y0, x0]], [I_H2V[y0, x0]], 'X', color='violet', markersize=10
    )
    ax_mg_h.plot(
        [LAM_H3 [y0, x0]], [I_H3 [y0, x0]], 'X', color='black',  markersize=10
    )
    ax_mg_h.plot(
        [LAM_H2R[y0, x0]], [I_H2R[y0, x0]], 'X', color='red',    markersize=10
    )
    ax_mg_h.set_title("Mg II H-line")

    # 5) Redraw all
    fig_siiv.canvas.draw_idle()
    fig_mgii.canvas.draw_idle()
    fig_siiv_spec.canvas.draw_idle()
    fig_mg_k.canvas.draw_idle()
    fig_mg_h.canvas.draw_idle()


##############################################################################
# 7. CONNECT THE CALLBACK TO ALL FIGURES & SHOW
##############################################################################
fig_siiv.canvas.mpl_connect("key_press_event",   on_key)
fig_mgii.canvas.mpl_connect("key_press_event",   on_key)
fig_siiv_spec.canvas.mpl_connect("key_press_event", on_key)
fig_mg_k.canvas.mpl_connect("key_press_event",   on_key)
fig_mg_h.canvas.mpl_connect("key_press_event",   on_key)

plt.show()
