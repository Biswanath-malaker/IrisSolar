"""
extension of fit_MgII_1.py. It fit mass spectra and save them in a particular file similar to raster!
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
from scipy.signal import find_peaks
from astropy.constants import c
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy import integrate


D = 'D044'
file = f"./data_aligned/{D}/RASTER_FILES/iris_l2_20160319_145728_3623010639_raster_t000_r00000.fits"
files = glob.glob(f"./../iris_obj/data_aligned/{D}/RASTER_FILES/*.fits")
files = sorted(files)

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


for file in files[1:]:
    with IrisRaster(file) as raster:

        # raster_timestep = 15
        # ypixel = 200
        
        k3_init , h3_init = 2796.36 * u.angstrom,2803.54 * u.angstrom

        v = 150 *u.km/u.s    # +-v range from h3 and k3 line.

        k3_range = [float(get_dopplershift2wavelength(-v,k3_init).value),float(get_dopplershift2wavelength(v,k3_init).value)]
        h3_range = [float(get_dopplershift2wavelength(-v,h3_init).value),float(get_dopplershift2wavelength(v,h3_init).value)]
        
        # k3_range = raster.get_wrange('MgII_k')
        # h3_range = raster.get_wrange('MgII_h')

        m , n = 2,1
        datacube_k3 , errcube_k3 , dummy_raster_2d , wavelength_k3 , MS_sub = raster.get_spectral_cube(8,f'./Errors/{D}',fuv=False,wrange=k3_range,rebinn=[[m,n]])  # I don't need to rebinn other images like aia and hmi as the have been already doen!
        datacube_h3 , errcube_h3 , dummy_raster_2dp , wavelength_h3 , MS_subp = raster.get_spectral_cube(8,f'./Errors/{D}',fuv=False,wrange=h3_range,rebinn=[[m,n]])  # I don't need to rebinn other images like aia and hmi as the have been already doen!

        s = datacube_k3.shape
        datacube_k3_linear = datacube_k3.reshape((s[0]*s[1],s[2]))
        datacube_h3_linear = datacube_h3.reshape((s[0]*s[1],s[2]))
        errcube_k3_linear = errcube_k3.reshape((s[0]*s[1],s[2]))
        errcube_h3_linear = errcube_h3.reshape((s[0]*s[1],s[2]))

        
        # POPT_k3 = np.zeros((s[0]*s[1],8))
        # POPT_h3 = np.zeros((s[0]*s[1],8))
        # RED_CHISQ_k3 = np.zeros((s[0]*s[1],))
        # RED_CHISQ_h3 = np.zeros((s[0]*s[1],))
        
        LAM_K3 = np.zeros((s[0]*s[1],))
        LAM_H3 = np.zeros((s[0]*s[1],))
        LAM_K2V = np.zeros((s[0]*s[1],))
        LAM_K2R = np.zeros((s[0]*s[1],))
        LAM_H2V = np.zeros((s[0]*s[1],))
        LAM_H2R = np.zeros((s[0]*s[1],))
        
        I_K3 = np.zeros((s[0]*s[1],))
        I_H3 = np.zeros((s[0]*s[1],))
        I_K2V = np.zeros((s[0]*s[1],))
        I_K2R = np.zeros((s[0]*s[1],))
        I_H2V = np.zeros((s[0]*s[1],))
        I_H2R = np.zeros((s[0]*s[1],))
        
        FLUX_K = np.zeros((s[0]*s[1],))
        FLUX_H = np.zeros((s[0]*s[1],))


        DOPPSHIFT_k = np.zeros((s[0]*s[1],))
        DOPPSHIFT_h = np.zeros((s[0]*s[1],))
        DOPPSHIFT_av = np.zeros((s[0]*s[1],))
        GRAD_V = np.zeros((s[0]*s[1],))
        TEMP_k = np.zeros((s[0]*s[1],))
        TEMP_h = np.zeros((s[0]*s[1],))


        wavelength_k3_dense = np.linspace(wavelength_k3[0],wavelength_k3[-1],300)
        wavelength_h3_dense = np.linspace(wavelength_h3[0],wavelength_h3[-1],300)

        with tqdm(total=s[0]*s[1],desc="progressing") as pbar:
            for i,(data_k3,data_h3,error_k3,error_h3) in enumerate(zip(datacube_k3_linear,datacube_h3_linear,errcube_k3_linear,errcube_h3_linear)):
                args = wavelength_k3,wavelength_h3,data_k3,data_h3,error_k3,error_h3,i
                LAM_k3,I_k3,RANGE_k3,_ = get_lam_I1(wavelength_k3,data_k3)
                LAM_h3,I_h3,RANGE_h3,_ = get_lam_I1(wavelength_h3,data_h3,kline=False)
                
                flux_k = integrate.trapezoid(data_k3,wavelength_k3)
                flux_h = integrate.trapezoid(data_h3,wavelength_h3)
                FLUX_K[i] = flux_k
                FLUX_H[i] = flux_h
                
                if not np.all(np.isnan(LAM_K3)) or np.all(np.isnan(LAM_h3)):
                
                    lam_k0 = 2796.38
                    lam_h0 = 2803.56
                
                    del_v_k3 = ((LAM_k3[2]-lam_k0)*c/lam_k0).to(u.km/u.s).value
                    del_v_h3 = ((LAM_h3[2]-lam_h0)*c/lam_h0).to(u.km/u.s).value
                    
                    vel_grad = del_v_k3-del_v_h3

                    del_v_k0 = ((((LAM_k3[0]+LAM_k3[1])/2-lam_k0)/lam_k0)*c).to(u.km/u.s).value# Average dopplershift

                    TEMP_k[i] = np.average(I_k3[0:2])
                    TEMP_h[i] = np.average(I_h3[0:2])
                    
                    # I_K3[i] = I_k3[2]
                    # I_H3[i] = I_h3[2]
                    # I_K2V[i] = I_k3[0]
                    # I_K2R[i] = I_k3[1]
                    # I_H2V[i] = I_h3[0]
                    # I_H2R[i] = I_h3[1]
                else:

                    del_v_k3 = np.nan
                    del_v_h3 = np.nan
                    
                    vel_grad = np.nan

                    del_v_k0 = np.nan

                    
                    TEMP_k[i] = np.nan
                    TEMP_h[i] = np.nan

                    # I_K3[i] = np.nan
                    # I_H3[i] = np.nan
                    # I_K2V[i] = np.nan
                    # I_K2R[i] = np.nan
                    # I_H2V[i] = np.nan
                    # I_H2R[i] = np.nan

                DOPPSHIFT_k[i] = del_v_k3
                DOPPSHIFT_h[i] = del_v_h3

                DOPPSHIFT_av[i] = del_v_k0
                GRAD_V[i] = vel_grad

                LAM_K3[i] = LAM_k3[2]
                LAM_H3[i] = LAM_h3[2]
                LAM_K2V[i] = LAM_k3[0]
                LAM_K2R[i] = LAM_k3[1]
                LAM_H2V[i] = LAM_h3[0]
                LAM_H2R[i] = LAM_h3[1]
                 
                 
                I_K3[i] = I_k3[2]
                I_H3[i] = I_h3[2]
                I_K2V[i] = I_k3[0]
                I_K2R[i] = I_k3[1]
                I_H2V[i] = I_h3[0]
                I_H2R[i] = I_h3[1]


                # fitted_k3_dense = Models.Gauss2linearbackground1(wavelength_k3_dense,popt_k3)
                # fitted_h3_dense = Models.Gauss2linearbackground1(wavelength_h3_dense,popt_h3)


                # RED_CHISQ_k3[i] = red_chisq1_k3
                # RED_CHISQ_h3[i] = red_chisq1_h3
                pbar.update()

        # POPT_k3 = np.reshape(POPT_k3,s[0:2]+(8,))
        # POPT_h3 = np.reshape(POPT_h3,s[0:2]+(8,))
        # RED_CHISQ_k3 = np.reshape(RED_CHISQ_k3,s[0:2])
        # RED_CHISQ_h3 = np.reshape(RED_CHISQ_h3,s[0:2])


        LAM_K3 = np.reshape(LAM_K3,s[0:2])
        LAM_H3 = np.reshape(LAM_H3,s[0:2])
        LAM_K2V = np.reshape(LAM_K2V,s[0:2])
        LAM_K2R = np.reshape(LAM_K2R,s[0:2])
        LAM_H2V = np.reshape(LAM_H2V,s[0:2])
        LAM_H2R = np.reshape(LAM_H2R,s[0:2])
        
        I_K3 = np.reshape(I_K3,s[0:2])
        I_H3 = np.reshape(I_H3,s[0:2])
        I_K2V = np.reshape(I_K2V,s[0:2])
        I_K2R = np.reshape(I_K2R,s[0:2])
        I_H2V = np.reshape(I_H2V,s[0:2])
        I_H2R = np.reshape(I_H2R,s[0:2])
        
        FLUX_K = np.reshape(FLUX_K,s[0:2])
        FLUX_H = np.reshape(FLUX_H,s[0:2])


        DOPPSHIFT_k = np.reshape(DOPPSHIFT_k,s[0:2])
        DOPPSHIFT_h = np.reshape(DOPPSHIFT_h,s[0:2])
        DOPPSHIFT_av = np.reshape(DOPPSHIFT_av,s[0:2])
        GRAD_V = np.reshape(GRAD_V,s[0:2])
        TEMP_k = np.reshape(TEMP_k,s[0:2])
        TEMP_h = np.reshape(TEMP_h,s[0:2])

        
        target_dir = f"./FITTING/MgII/{D}/m_{m}-n_{n}/{os.path.basename(file).replace('.fits','')}"
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
            
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

        dummy_raster_2d.save(dummy_raster_path,overwrite=True)
        # np.save(popt_k3_path,POPT_k3,allow_pickle=True)
        # np.save(popt_h3_path,POPT_h3,allow_pickle=True)
        # np.save(red_chisq_k3_path,RED_CHISQ_k3,allow_pickle=True)
        # np.save(red_chisq_h3_path,RED_CHISQ_h3,allow_pickle=True)
        np.save(datacube_k3_path,datacube_k3,allow_pickle=True)
        np.save(datacube_h3_path,datacube_h3,allow_pickle=True)
        np.save(errcube_k3_path,errcube_k3,allow_pickle=True)
        np.save(errcube_h3_path,errcube_h3,allow_pickle=True)
        np.save(wavelength_k3_path,wavelength_k3,allow_pickle=True)
        np.save(wavelength_h3_path,wavelength_h3,allow_pickle=True)
        
        np.save(LAM_K3_path,LAM_K3,allow_pickle=True)
        np.save(LAM_H3_path,LAM_H3,allow_pickle=True)
        np.save(LAM_K2V_path,LAM_K2V,allow_pickle=True)
        np.save(LAM_K2R_path,LAM_K2R,allow_pickle=True)
        np.save(LAM_H2V_path,LAM_H2V,allow_pickle=True)
        np.save(LAM_H2R_path,LAM_H2R,allow_pickle=True)
        
        
        np.save(I_K3_path,I_K3,allow_pickle=True)
        np.save(I_H3_path,I_H3,allow_pickle=True)
        np.save(I_K2V_path,I_K2V,allow_pickle=True)
        np.save(I_K2R_path,I_K2R,allow_pickle=True)
        np.save(I_H2V_path,I_H2V,allow_pickle=True)
        np.save(I_H2R_path,I_H2R,allow_pickle=True)

        np.save(FLUX_K_path,FLUX_K,allow_pickle=True)
        np.save(FLUX_H_path,FLUX_H,allow_pickle=True)

        
        np.save(DOPPSHIFT_k_path,DOPPSHIFT_k,allow_pickle=True)
        np.save(DOPPSHIFT_h_path,DOPPSHIFT_h,allow_pickle=True)
        np.save(DOPPSHIFT_av_path,DOPPSHIFT_av,allow_pickle=True)
        np.save(GRAD_V_path,GRAD_V,allow_pickle=True)
        np.save(TEMP_k_path,TEMP_k,allow_pickle=True)
        np.save(TEMP_h_path,TEMP_h,allow_pickle=True)

        # args = wavelength_k3,wavelength_h3,datacube_k3[raster_timestep,ypixel],datacube_h3[raster_timestep,ypixel],errcube_k3[raster_timestep,ypixel],errcube_h3[raster_timestep,ypixel],[raster_timestep,ypixel]

        # popt_k3,popt_h3,red_chisq1_k3,red_chisq1_h3,I = fit_MgII_lines_2gaussian_linearbg(args)
        # print(popt_k3.shape)
        # plot_MgII_spectra(wavelength_k3,data_k3,error_k3,popt_k3,wavelength_h3,data_h3,error_h3,popt_h3)
        # plot_MgII_spectra(wavelength_k3,datacube_k3[raster_timestep,ypixel],errcube_k3[raster_timestep,ypixel],popt_k3,wavelength_h3,datacube_h3[raster_timestep,ypixel],errcube_h3[raster_timestep,ypixel],popt_h3)

        # plt.show()
    print(f"{file} done!")
    
    time.sleep(5)




