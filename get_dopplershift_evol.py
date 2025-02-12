"""
That code defines a center of a box with width and height,\n
find informations of the central time of each raster file and then find centres of that box as if rotation with the sun.\n

given lower left and upper right as mentioned above,\n
    \t we cut the sunpy maps of A , Sigma , non-thermal vel, Mu , aia and hmi and create flux as well as BR map for each raster scan \n
    save them as 1d array with proper time_stamp , with non_th arr , sigma_arr etc... with figures in ./Figs/slit dir.

"""
import glob
from iris_raster import IrisRaster
import glob
import numpy as np
import matplotlib.pyplot as plt
from models import Models
import os
from Functions.utilities import get_extent
from astropy.wcs import WCS
import datetime
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
from physics import get_doppler_shift , get_fwhm , get_flux_gaussian
from fig_templates import intensity_fwhm_fig , intensity_fwhm_fig1 
from utilities import utilities as utils
from sunpy.time import TimeRange
from Functions.misc import write_json,update_json

def de_rotate(aiamap , ref_coordinate , initial_time , end_time , plot_point = False,model = 'howard'):
    """
    file is here to get coordinate frame. and plotting purpose and nowhere it has role.
    coodinate   --->    tuple (x,y) in arcsecs ; reference coordinate.
    time        --->    int or float in sec; +ve for future time coordinate to rotate
     , -ve for past time coodinate to rotate.
    plot_point  --->    default : False , If true will plot points on the map provided in file.
    return      --->    tuple ; (x,y) rotated in arcsec
    """

    import matplotlib.pyplot as plt
    import numpy as np

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    import sunpy.map
    from sunpy.coordinates import RotatedSunFrame

    import datetime

    # initial_time = datetime.datetime.strptime(initial_time,"%Y-%m-%dT%H:%M:%S.%f")
    # end_time = datetime.datetime.strptime(end_time,"%Y-%m-%dT%H:%M:%S.%f")


    d_time = end_time-initial_time
    d_time = d_time.total_seconds()
    # aiamap = sunpy.map.Map(file)
    point = SkyCoord(ref_coordinate[0]*u.arcsec, ref_coordinate[1]*u.arcsec, frame=aiamap.coordinate_frame)
    # print(point.observer)
    durations = np.array([d_time])*u.s
    # print(point.frame)
    diffrot_point = RotatedSunFrame(base=point, duration=durations , rotation_model=model)

    transformed_diffrot_point = diffrot_point.transform_to(aiamap.coordinate_frame)

    if plot_point:
        ax = plt.subplot(projection=aiamap)
        aiamap.plot(clip_interval=(1., 99.95)*u.percent)

        ax.plot_coord(point, 'ro', fillstyle='none', label='Original')
        ax.plot_coord(transformed_diffrot_point, 'bo', fillstyle='none',
                    label='Rotated')
        plt.legend()

        plt.show()

    return transformed_diffrot_point.Tx.value[0] , transformed_diffrot_point.Ty.value[0]


D = 'D044'

files = glob.glob(f"./../iris_obj/data_aligned/{D}/RASTER_FILES/*.fits")
files = sorted(files)

raster_file = files[0]


# slit parameters   # reference (raster0)
xcen0 = 21
ycen0 = 370

dx = 8
dy = 8

m = 2
n=1

# FRAME Y LIMIT
ext_y = [320,400]



        
with IrisRaster(files[0]) as raster0:
    obs_times0 = raster0.get_observation_times()

    DT = TimeRange(obs_times0[0],obs_times0[-1])
    cent_time = DT.center.strftime('%Y-%m-%dT%H_%M_%S')
    init_time = datetime.datetime.strptime(cent_time,'%Y-%m-%dT%H_%M_%S')
    
for raster_file in files:

    file_name = os.path.basename(raster_file).replace(".fits","")
    fitting_type = "Gaussian1_Linear_Bg"
    line_name = "MgII"
    # dummy_raster_path = f"./FITTING/{line_name}/{D}/m_{m}-n_{n}/{file_name}/dummy_raster_2d.fits"
    dummy_raster_path = f"./FITTING/MgII/D044/m_2-n_1/iris_l2_20160319_145728_3623010639_raster_t000_r00000/dummy_raster_2d.fits"
    print(os.path.isfile(dummy_raster_path))
    doppler_shift_path = f"./FITTING/{line_name}/{D}/m_{m}-n_{n}/{file_name}/DOPPSHIFT_k.npy"
    # doppler_shift_path = f"./FITTING/MgII/D044/m_2-n_1/iris_l2_20160319_145728_3623010639_raster_t000_r00000/DOPPSHIFT_k.npy"
    # popt_path = f"./FITTING/{line_name}/{fitting_type}/m_{m}-n_{n}/{file_name}/popt.npy"
    # popt1_path = f"./FITTING/{line_name}/{fitting_type}/m_{m}-n_{n}/{file_name}/popt.npy"
    # # red_chisq_path = f"./FITTING/{line_name}/{fitting_type}/m_{m}-n_{n}/{file_name}/red_chisq.npy"
    # red_chisq1_path = f"./FITTING/{line_name}/{fitting_type}/m_{m}-n_{n}/{file_name}/red_chisq.npy"
    # wavelength_path = f"./FITTING/{line_name}/{fitting_type}/m_{m}-n_{n}/{file_name}/wavelength.npy"
    # err_path = f"./FITTING/{line_name}/{fitting_type}/m_{m}-n_{n}/{file_name}/err.npy"

    aia171_file = glob.glob(f"./../iris_obj/data_scanwise_m-2_n-1/{D}/iris_l2_20160319_145728_3623010639_raster_t000_r00000/mid/171/*")[0]
    # hmi_file = glob.glob(f"./data_scanwise_m-{m}_n-{n}/{D}/{file_name}/mid/hmi/*.fits")[0]

    aux_data_dir = f"./result_data/{D}/light_curves/{line_name}/m{m}_n{n}"
    if not os.path.isdir(aux_data_dir):
        os.makedirs(aux_data_dir)
        
    aux_data_path = os.path.join(aux_data_dir,f'{file_name}.json')
    if not os.path.isfile(aux_data_path):
        write_json(aux_data_path,{})

    with IrisRaster(raster_file) as raster:
        obs_times = raster.get_observation_times()
        DT = TimeRange(obs_times[0],obs_times[-1])
        cent_time = DT.center.strftime('%Y-%m-%dT%H_%M_%S')
        end_time = datetime.datetime.strptime(cent_time,'%Y-%m-%dT%H_%M_%S')


        # data = np.load(data_path,allow_pickle=True)
        # # popt = np.load(popt_path,allow_pickle=True)
        # popt1 = np.load(popt1_path,allow_pickle=True)
        # # red_chisq = np.load(red_chisq_path,allow_pickle=True)
        # red_chisq1 = np.load(red_chisq1_path,allow_pickle=True)
        # wavelength = np.load(wavelength_path,allow_pickle=True)
        # err = np.load(err_path,allow_pickle=True)
        m_dummy_raster = sm(dummy_raster_path)
        m171 = sm(aia171_file)
        # mhmi = sm(hmi_file)

        # a = popt1[...,0].T
        # mu = popt1[...,1].T
        # sigma = popt1[...,2].T

        # ext_m171 = get_extent(m171)
        # ext_mhmi = get_extent(mhmi)
        # ext_spectrogram = get_extent(m_dummy_raster)

        # doppler_shift = get_doppler_shift(mu,1393.758753571523)
        # flux = get_flux_gaussian(a,sigma)
        # fwhm = get_fwhm(sigma)
        # red_chisq1_data = red_chisq1.T
        # mhmi_data = mhmi.data
        # m171_data = m171.data
        doppler_shift = np.load(doppler_shift_path,allow_pickle=True).T 

        m_doppler_shift = sm((doppler_shift,m_dummy_raster.fits_header))
        # m_flux = sm((flux,m_dummy_raster.fits_header))
        # m_fwhm = sm((fwhm,m_dummy_raster.fits_header))
        # m_red_chisq1 = sm((red_chisq1_data,m_dummy_raster.fits_header))

        bl = [float(m_dummy_raster.bottom_left_coord.Tx.value),float(m_dummy_raster.bottom_left_coord.Ty.value)]
        tr = [float(m_dummy_raster.top_right_coord.Tx.value),float(m_dummy_raster.top_right_coord.Ty.value)]

        m_doppler_shift_sub = sunpy_map_cut(m_doppler_shift,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
        # m_flux_sub = sunpy_map_cut(m_flux,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
        # m_fwhm_sub = sunpy_map_cut(m_fwhm,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
        # m_red_chisq1_sub = sunpy_map_cut(m_red_chisq1,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
        m171_sub = sunpy_map_cut(m171,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
        # mhmi_sub = sunpy_map_cut(mhmi,[bl[0],ext_y[0]],[tr[0],ext_y[1]])
        

        
        xcen,ycen = de_rotate(m171_sub,[xcen0,ycen0],init_time,end_time)
        ext_slit = [xcen-dx/2,xcen+dx/2,ycen-dy/2,ycen+dy/2]
        rect_slit_x,rect_slit_y = utils.ext2rect(ext_slit)
        bl_slit,tr_slit = utils.ext2bltr(ext_slit)
        
        m_doppler_shift_slit = sunpy_map_cut(m_doppler_shift_sub,bl_slit,tr_slit)
        # m_flux_slit = sunpy_map_cut(m_flux_sub,bl_slit,tr_slit)
        # m_fwhm_slit = sunpy_map_cut(m_fwhm_sub,bl_slit,tr_slit)
        # m_red_chisq1_slit = sunpy_map_cut(m_red_chisq1_sub,bl_slit,tr_slit)
        m171_slit = sunpy_map_cut(m171_sub,bl_slit,tr_slit)
        # mhmi_slit = sunpy_map_cut(mhmi_sub,bl_slit,tr_slit)


        ext_m171 = get_extent(m171_sub)
        # ext_mhmi = get_extent(mhmi_sub)
        # ext_spectrogram = get_extent(m_flux_sub)

        ext_m171_slit = get_extent(m171_slit)
        # ext_mhmi_slit = get_extent(mhmi_slit)
        # ext_spectrogram_slit = get_extent(m_flux_slit)
        
        doppler_shift_sub = m_doppler_shift_sub.data
        # flux_sub = m_flux_sub.data
        # fwhm_sub = m_fwhm_sub.data
        # fwhm_sub = fwhm_sub*c/1393.7587
        # fwhm_sub = fwhm_sub.to(u.km/u.s).value
        # red_chisq1_data_sub = m_red_chisq1_sub.data
        # mhmi_data_sub = mhmi_sub.data
        m171_data_sub = m171_sub.data
        
        # dv_instr = (26*(10**-3)*c)/1393.7587
        
        # dv_instr = dv_instr.to(u.km/u.s)# https://iris.lmsal.com/itn38/analysis_lines_iris.html
        # dv_instr = 5.59  # km/s
        # dv_th = 6.83    # km/s
        # w_nth_sq = fwhm_sub**2-dv_th**2-dv_instr**2 
        # w_nth_sq[w_nth_sq<0]=0
        # w_nth = np.sqrt(w_nth_sq)
        
        # m_w_nth = sm(w_nth,m_fwhm_sub.fits_header)
        # m_w_nth_slit = sunpy_map_cut(m_w_nth,bl_slit,tr_slit)

        # fig = intensity_fwhm_fig1(
        #     doppler_shift_sub,
        #     flux_sub,
        #     m171_data_sub,
        #     red_chisq1_data_sub,
        #     mhmi_data_sub,
        #     fwhm_sub,
        #     w_nth,
        #     ext_spectrogram,
        #     ext_m171,
        #     ext_mhmi
        # )
        
        # fig1 = intensity_fwhm_fig1(
        #     m_doppler_shift_slit.data,
        #     m_flux_slit.data,
        #     m171_slit.data,
        #     m_red_chisq1_slit.data,
        #     mhmi_slit.data,
        #     m_fwhm_slit.data,
        #     m_w_nth_slit.data,
        #     ext_spectrogram_slit,
        #     ext_m171_slit,
        #     ext_mhmi_slit
        # )

        # fig.suptitle(f"{DT.center.strftime('%Y-%m-%d %H:%M:%S UT')}",fontsize=16)
        
        # fig_axes = fig.get_axes()
        # for ax_fig in fig_axes:
        #     ax_fig.plot(rect_slit_x,rect_slit_y,color='red')
            
        update_json(aux_data_path,{'doppler_shift':m_doppler_shift_slit.data.ravel().tolist()})
        # update_json(aux_data_path,{'flux':m_flux_slit.data.ravel().tolist()})
        update_json(aux_data_path,{'L_171':m171_slit.data.ravel().tolist()})
        # update_json(aux_data_path,{'red_chisq':m_red_chisq1_slit.data.ravel().tolist()})
        # update_json(aux_data_path,{'L_hmi':mhmi_slit.data.ravel().tolist()})
        # update_json(aux_data_path,{'L_fwhm':m_fwhm_slit.data.ravel().tolist()})
        # update_json(aux_data_path,{'L_n_th':m_w_nth_slit.data.ravel().tolist()})

        # plt.show()


    # time.sleep(5)









