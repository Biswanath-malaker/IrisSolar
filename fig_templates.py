from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as  plt
import numpy as np
from models import Models




def intensity_fwhm_fig(doppler_shift,flux,m171_data,red_chisq_data,mhmi_data,ext_spectrogram,ext_m171,ext_mhmi):


    fig = plt.figure(figsize=(18,8))
    ax1 = fig.add_subplot(1,5,1)
    ax2 = fig.add_subplot(1,5,2)
    ax3 = fig.add_subplot(1,5,3)
    ax4 = fig.add_subplot(1,5,4)
    ax5 = fig.add_subplot(1,5,5)


    norm_br = TwoSlopeNorm(0,-15,+15)
    norm_mag = TwoSlopeNorm(0,-20,+700)


    im1 = ax1.imshow(doppler_shift,origin="lower",aspect="auto",norm=norm_br,cmap='bwr',extent=ext_spectrogram)

    im2 = ax2.imshow(flux,origin="lower",aspect="auto",vmin=0,vmax=2,extent=ext_spectrogram)
    im3 = ax3.imshow(m171_data,origin="lower",cmap='sdoaia171',extent=ext_m171,vmin=40,vmax=500,aspect="auto")
    im4 = ax4.imshow(red_chisq_data,origin="lower",extent=ext_m171,vmax=2,vmin=0,aspect="auto")
    im5 = ax5.imshow(mhmi_data.data,origin="lower",cmap='hmimag',extent=ext_mhmi,aspect="auto",norm=norm_mag)

    # ax1.plot([X],[Y],"X")
    # ax2.plot([X],[Y],"X")
    # ax3.plot([X],[Y],"X")
    # ax4.plot([X],[Y],"X")
    # ax5.plot([X],[Y],"X")


    ax2.set_yticks([])
    ax3.set_yticks([])
    ax4.set_yticks([])
    ax5.set_yticks([])


    # ax1.plot(bg_rect[0],bg_rect[1],'-k')
    # ax2.plot(bg_rect[0],bg_rect[1],'-y')
    # ax3.plot(bg_rect[0],bg_rect[1],'-k')
    # ax4.plot(bg_rect[0],bg_rect[1],'-k')
    # ax5.plot(bg_rect[0],bg_rect[1],'-c')

    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    divider3 = make_axes_locatable(ax3)
    divider4 = make_axes_locatable(ax4)
    divider5 = make_axes_locatable(ax5)


    cax1 = divider1.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax2 = divider2.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax3 = divider3.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax4 = divider4.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax5 = divider5.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar

    cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal', extend='both')
    cbar1.set_ticks([-15,-7,0,7,15])
    cbar1.set_label("Valocity (km/s)", fontsize=14)
    cbar1.ax.tick_params(labelsize=14)

    cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal', extend='max')
    cbar2.set_label("Flux (DN/s)", fontsize=14)
    cbar2.ax.tick_params(labelsize=14)

    cbar3 = fig.colorbar(im3, cax=cax3, orientation='horizontal', extend='both')
    cbar3.set_ticks([40,150,250,350,500])
    cbar3.set_label("AIA 171 (DN/s)", fontsize=14)
    cbar3.ax.tick_params(labelsize=14)

    cbar4 = fig.colorbar(im4, cax=cax4, orientation='horizontal', extend='both')
    cbar4.set_label("$\\tilde{\chi}^2$", fontsize=14)
    cbar4.ax.tick_params(labelsize=14)

    cbar5 = fig.colorbar(im5, cax=cax5, orientation='horizontal', extend='both')
    cbar5.set_ticks([-20,0,300,700])
    cbar5.set_label("$B_{los}$", fontsize=14)
    cbar5.ax.tick_params(labelsize=14)


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


    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax2.get_xticklabels(), fontsize=14)
    plt.setp(ax3.get_xticklabels(), fontsize=14)
    plt.setp(ax4.get_xticklabels(), fontsize=14)
    plt.setp(ax5.get_xticklabels(), fontsize=14)

    plt.setp(ax1.get_yticklabels(), fontsize=14)

    fig.supxlabel("Solar X (arcsec)",fontsize = 16)
    fig.supylabel("Solar Y (arcsec)",fontsize = 16)

    return fig







def intensity_fwhm_fig1(doppler_shift,flux,m171_data,red_chisq_data,mhmi_data,fwhm_data,nth_vel_data,ext_spectrogram,ext_m171,ext_mhmi):


    fig = plt.figure(figsize=(20,8))
    ax1 = fig.add_subplot(1,7,1)
    ax2 = fig.add_subplot(1,7,2,sharey=ax1)
    ax3 = fig.add_subplot(1,7,3,sharey=ax1)
    ax4 = fig.add_subplot(1,7,4,sharey=ax1)
    ax5 = fig.add_subplot(1,7,5,sharey=ax1)
    ax6 = fig.add_subplot(1,7,6,sharey=ax1)
    ax7 = fig.add_subplot(1,7,7,sharey=ax1)


    norm_br = TwoSlopeNorm(0,-15,+15)
    norm_mag = TwoSlopeNorm(0,-20,+700)


    im1 = ax1.imshow(doppler_shift,origin="lower",aspect="auto",norm=norm_br,cmap='bwr',extent=ext_spectrogram)

    im2 = ax2.imshow(flux,origin="lower",aspect="auto",vmin=0,vmax=2,extent=ext_spectrogram)
    im3 = ax3.imshow(m171_data,origin="lower",cmap='sdoaia171',extent=ext_m171,vmin=40,vmax=500,aspect="auto")
    im4 = ax4.imshow(red_chisq_data,origin="lower",extent=ext_m171,vmax=2,vmin=0,aspect="auto")
    im5 = ax5.imshow(mhmi_data.data,origin="lower",cmap='bwr',extent=ext_mhmi,aspect="auto",norm=norm_mag)

    im6 = ax6.imshow(fwhm_data,origin="lower",cmap=None,extent=ext_spectrogram,aspect="auto",vmax=50,vmin=0)
    im7 = ax7.imshow(nth_vel_data,origin="lower",cmap=None,extent=ext_spectrogram,aspect="auto",vmax=50,vmin=0)

    # ax1.plot([X],[Y],"X")
    # ax2.plot([X],[Y],"X")
    # ax3.plot([X],[Y],"X")
    # ax4.plot([X],[Y],"X")
    # ax5.plot([X],[Y],"X")


    ax2.set_yticks([])
    ax3.set_yticks([])
    ax4.set_yticks([])
    ax5.set_yticks([])
    ax6.set_yticks([])
    ax7.set_yticks([])


    # ax1.plot(bg_rect[0],bg_rect[1],'-k')
    # ax2.plot(bg_rect[0],bg_rect[1],'-y')
    # ax3.plot(bg_rect[0],bg_rect[1],'-k')
    # ax4.plot(bg_rect[0],bg_rect[1],'-k')
    # ax5.plot(bg_rect[0],bg_rect[1],'-c')

    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    divider3 = make_axes_locatable(ax3)
    divider4 = make_axes_locatable(ax4)
    divider5 = make_axes_locatable(ax5)
    divider6 = make_axes_locatable(ax6)
    divider7 = make_axes_locatable(ax7)


    cax1 = divider1.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax2 = divider2.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax3 = divider3.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax4 = divider4.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax5 = divider5.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax6 = divider6.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax7 = divider7.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar

    cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal', extend='both')
    cbar1.set_ticks([-15,-7,0,7,15])
    cbar1.set_label("Valocity (km/s)", fontsize=14)
    cbar1.ax.tick_params(labelsize=14)

    cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal', extend='max')
    cbar2.set_label("Flux (DN/s)", fontsize=14)
    cbar2.ax.tick_params(labelsize=14)

    cbar3 = fig.colorbar(im3, cax=cax3, orientation='horizontal', extend='both')
    cbar3.set_ticks([40,150,250,350,500])
    cbar3.set_label("AIA 171 (DN/s)", fontsize=14)
    cbar3.ax.tick_params(labelsize=14)

    cbar4 = fig.colorbar(im4, cax=cax4, orientation='horizontal', extend='both')
    cbar4.set_label("$\\tilde{\chi}^2$", fontsize=14)
    cbar4.ax.tick_params(labelsize=14)

    cbar5 = fig.colorbar(im5, cax=cax5, orientation='horizontal', extend='both')
    cbar5.set_ticks([-20,0,300,700])
    cbar5.set_label("$B_{los}$", fontsize=14)
    cbar5.ax.tick_params(labelsize=14)

    cbar6 = fig.colorbar(im6, cax=cax6, orientation='horizontal', extend='max')
    # cbar6.set_ticks([-20,0,300,700])
    cbar6.set_label("fwhm (km/s)", fontsize=14)
    cbar6.ax.tick_params(labelsize=14)
    
    cbar7 = fig.colorbar(im7, cax=cax7, orientation='horizontal', extend='max')
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
    
    
    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax2.get_xticklabels(), fontsize=14)
    plt.setp(ax3.get_xticklabels(), fontsize=14)
    plt.setp(ax4.get_xticklabels(), fontsize=14)
    plt.setp(ax5.get_xticklabels(), fontsize=14)

    plt.setp(ax1.get_yticklabels(), fontsize=14)

    fig.supxlabel("Solar X (arcsec)",fontsize = 16)
    fig.supylabel("Solar Y (arcsec)",fontsize = 16)

    return fig




def combined_intensity_wavelength_figs(
    doppler_shift,flux,m171_data,red_chisq_data,mhmi_data,ext_spectrogram,ext_m171,ext_mhmi
):
    figs = plt.figure(layout='constrained',figsize=(15,8))

    fig1,fig2 = figs.subfigures(2,1,height_ratios=[3,1])

    fig1.set_facecolor('white')
    fig1.suptitle('subfigs[0]')

    fig2.set_facecolor('white')
    fig2.suptitle('subfigs[1]')



    
    
    # subfigs0 = subfigs[0].subfigures(2,1,height_ratios=[1,1])
    
    # subfigs0[0].set_facecolor('black')
    # subfigs0[0].suptitle('subfigs0[0]')
    
    # ax_spectra = subfigs0[0].add_subplot(1,1,1)
    
    ax1 = fig1.add_subplot(1,5,1)
    ax2 = fig1.add_subplot(1,5,2)
    ax3 = fig1.add_subplot(1,5,3)
    ax4 = fig1.add_subplot(1,5,4)
    ax5 = fig1.add_subplot(1,5,5)





    norm_br = TwoSlopeNorm(0,-15,+15)
    norm_mag = TwoSlopeNorm(0,-20,+700)


    im1 = ax1.imshow(doppler_shift,origin="lower",aspect="auto",norm=norm_br,cmap='bwr',extent=ext_spectrogram)

    im2 = ax2.imshow(flux,origin="lower",aspect="auto",vmin=0,vmax=2,extent=ext_spectrogram)
    im3 = ax3.imshow(m171_data,origin="lower",cmap='sdoaia171',extent=ext_m171,vmin=40,vmax=500,aspect="auto")
    im4 = ax4.imshow(red_chisq_data,origin="lower",extent=ext_m171,vmax=2,vmin=0,aspect="auto")
    im5 = ax5.imshow(mhmi_data.data,origin="lower",cmap='hmimag',extent=ext_mhmi,aspect="auto",norm=norm_mag)

    # ax1.plot([X],[Y],"X")
    # ax2.plot([X],[Y],"X")
    # ax3.plot([X],[Y],"X")
    # ax4.plot([X],[Y],"X")
    # ax5.plot([X],[Y],"X")


    ax2.set_yticks([])
    ax3.set_yticks([])
    ax4.set_yticks([])
    ax5.set_yticks([])


    # ax1.plot(bg_rect[0],bg_rect[1],'-k')
    # ax2.plot(bg_rect[0],bg_rect[1],'-y')
    # ax3.plot(bg_rect[0],bg_rect[1],'-k')
    # ax4.plot(bg_rect[0],bg_rect[1],'-k')
    # ax5.plot(bg_rect[0],bg_rect[1],'-c')

    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    divider3 = make_axes_locatable(ax3)
    divider4 = make_axes_locatable(ax4)
    divider5 = make_axes_locatable(ax5)


    cax1 = divider1.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax2 = divider2.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax3 = divider3.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax4 = divider4.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax5 = divider5.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar

    cbar1 = fig1.colorbar(im1, cax=cax1, orientation='horizontal', extend='both')
    cbar1.set_ticks([-15,-7,0,7,15])
    cbar1.set_label("Valocity (km/s)", fontsize=14)
    cbar1.ax.tick_params(labelsize=14)

    cbar2 = fig1.colorbar(im2, cax=cax2, orientation='horizontal', extend='max')
    cbar2.set_label("Flux (DN/s)", fontsize=14)
    cbar2.ax.tick_params(labelsize=14)

    cbar3 = fig1.colorbar(im3, cax=cax3, orientation='horizontal', extend='both')
    cbar3.set_ticks([40,150,250,350,500])
    cbar3.set_label("AIA 171 (DN/s)", fontsize=14)
    cbar3.ax.tick_params(labelsize=14)

    cbar4 = fig1.colorbar(im4, cax=cax4, orientation='horizontal', extend='both')
    cbar4.set_label("$\\tilde{\chi}^2$", fontsize=14)
    cbar4.ax.tick_params(labelsize=14)

    cbar5 = fig1.colorbar(im5, cax=cax5, orientation='horizontal', extend='both')
    cbar5.set_ticks([-20,0,300,700])
    cbar5.set_label("$B_{los}$", fontsize=14)
    cbar5.ax.tick_params(labelsize=14)


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


    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax2.get_xticklabels(), fontsize=14)
    plt.setp(ax3.get_xticklabels(), fontsize=14)
    plt.setp(ax4.get_xticklabels(), fontsize=14)
    plt.setp(ax5.get_xticklabels(), fontsize=14)

    plt.setp(ax1.get_yticklabels(), fontsize=14)

    # fig.supxlabel("Solar X (arcsec)",fontsize = 16)
    # fig.supylabel("Solar Y (arcsec)",fontsize = 16)

    return figs





def plot_MgII_spectra(wavelength_k3,data_k3,error_k3,popt_k3,wavelength_h3,data_h3,error_h3,popt_h3,raster):
    
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




def get_MgII_plot(m171_data,mhmi_data,dopplershift_data,temp_data,flux_data,ext_171,ext_hmi,ext_raster):


    fig_physics = plt.figure(figsize=(18,8))
    ax1_fig_physics = fig_physics.add_subplot(1,5,1)
    ax2_fig_physics = fig_physics.add_subplot(1,5,2)
    ax3_fig_physics = fig_physics.add_subplot(1,5,3)
    ax4_fig_physics = fig_physics.add_subplot(1,5,4)
    ax5_fig_physics = fig_physics.add_subplot(1,5,5)

    
    norm_mag = TwoSlopeNorm(0,-20,+700)


    im_ax1_fig_physics = ax1_fig_physics.imshow(m171_data,origin="lower",extent=ext_171)
    im_ax2_fig_physics = ax2_fig_physics.imshow(mhmi_data.data,origin="lower",extent=ext_hmi,cmap='hmimag',norm=norm_mag)
    im_ax3_fig_physics = ax3_fig_physics.imshow(dopplershift_data,origin="lower",extent=ext_raster,vmin=-5,vmax= 5,cmap='RdBu_r')
    im_ax4_fig_physics = ax4_fig_physics.imshow(temp_data,origin="lower",extent=ext_raster)
    im_ax5_fig_physics = ax5_fig_physics.imshow(flux_data,origin="lower",extent=ext_raster)

    
    # IMS = [im_ax1_fig_physics,im_ax2_fig_physics,im_ax3_fig_physics,im_ax4_fig_physics,im_ax5_fig_physics,im_ax6_fig_physics,im_ax7_fig_physics,]
    # AXS = [ax1_fig_physics,ax2_fig_physics,ax3_fig_physics,ax4_fig_physics,ax5_fig_physics,ax6_fig_physics,ax7_fig_physics]
    # S = m_dummy_raster.pixel_to_world(raster_timestep*u.pix,ypixel*u.pix)
    # X = S.Tx.value
    # Y = S.Ty.value
    # for ax in AXS:
    #     ax.plot([X],[Y],'X',color='red',markersize=4)
    
    divider1 = make_axes_locatable(ax1_fig_physics)
    divider2 = make_axes_locatable(ax2_fig_physics)
    divider3 = make_axes_locatable(ax3_fig_physics)
    divider4 = make_axes_locatable(ax4_fig_physics)
    divider5 = make_axes_locatable(ax5_fig_physics)


    cax1 = divider1.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax2 = divider2.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax3 = divider3.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax4 = divider4.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    cax5 = divider5.append_axes("top", size="3%", pad=0.1)  # Use "bottom" for horizontal colorbar
    
    cbar1 = fig_physics.colorbar(im_ax1_fig_physics, cax=cax1, orientation='horizontal', extend='both')
    # cbar1.set_ticks([40,150,250,350,500])
    cbar1.set_label("AIA 171 (DN/s)", fontsize=14)
    cbar1.ax.tick_params(labelsize=14)


    cbar2 = fig_physics.colorbar(im_ax2_fig_physics, cax=cax2, orientation='horizontal', extend='max')
    # cbar2.set_label("Flux (DN/s)", fontsize=14)
    # cbar2.ax.tick_params(labelsize=14)
    cbar2.set_ticks([-20,0,300,700])
    cbar2.set_label("$B_{los}$", fontsize=14)


    cbar3 = fig_physics.colorbar(im_ax3_fig_physics, cax=cax3, orientation='horizontal', extend='both')
    # cbar3.set_ticks([40,150,250,350,500])
    cbar3.set_label("Dopplershift", fontsize=14)
    cbar3.ax.tick_params(labelsize=14)


    cbar4 = fig_physics.colorbar(im_ax4_fig_physics, cax=cax4, orientation='horizontal', extend='both')
    # cbar6.set_ticks([-20,0,300,700])
    cbar4.set_label("$T_B$", fontsize=14)
    cbar4.ax.tick_params(labelsize=14)


    cbar5 = fig_physics.colorbar(im_ax5_fig_physics, cax=cax5, orientation='horizontal', extend='both')
    # cbar5.set_ticks([-20,0,300,700])
    cbar5.set_label("Flux", fontsize=14)
    cbar5.ax.tick_params(labelsize=14)




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
    
    return fig_physics



if __name__=="__main__":
    combined_intensity_wavelength_figs()

    plt.show()