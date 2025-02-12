
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import datetime
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

import warnings
warnings.filterwarnings('ignore')

from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore',category=VerifyWarning)

from utilities import utilities as utils
from sunpy.map import Map as sm
from Functions.utilities import get_extent
from models import Models
from scipy.optimize import curve_fit
from spectral_cube import SpectralCube
import os
import time

class IrisRaster(utils,Models):
    '''
    urls :\n 
            https://www.lmsal.com/iris_science/doc?cmd=dcur&proj_num=IS0554&file_type=pdf \n
            https://iris.lmsal.com/itn41/introduction.html \n
    '''

    def __init__(self,file_path:str):

        self.file_path = file_path
        self.hdu_list = None
        self.N = None    # Number of HDU's in the list


    def open_file(self):
        self.hdu_list = fits.open(self.file_path) 
        self.N =len(self.hdu_list)

    def close_file(self):
        if self.hdu_list:
            self.hdu_list.close()
            self.hdu_list = None
            self.N = None

    def __enter__(self):
        self.open_file()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close_file()


    def get_methods_attrs(self):
        methods = dir(self)
        methods = [m for m in methods if '__' not in m]
        return methods


    def get_info(self):
        # from astropy.io import fits
        self.info = fits.info(self.file_path)
        return ''


    def get_hdu(self,hdu_index:int):
        if hdu_index>=len(self.hdu_list):
            print(f"hdu_index must be less than {len(self.hdu_list)}")
        else:
            return self.hdu_list[hdu_index]
        
    def __str__(self):
        l = self.N  # length of the hdu list
        R = f'HDU_Index     Line desc.      Detector\n\n'

        for i in range(1,l-2):
            # print(i)
            line = self.get_hdu(0).header[f'TDESC{i}']
            detector = self.get_hdu(0).header[f'TDET{i}']

            R+=f'{i}             {line}     {detector}\n'


        return R
    

    def get_line_desc(self,hdu_index):
        return self.get_hdu(0).header[f'TDESC{hdu_index}']

    def get_natural_wavelengths(self):
        """
        https://iris.lmsal.com/itn39/Mg_diagnostics.html#table-lines-nist\n
        For MgII simply call MW['MgIIk] or MW["MgIIh"]
        """
        NW = {
            "Cii_1334":{'chianti':1334.5320,'nist':1334.5326,'transitions':['2s2 2p 2P1/2','2s 2p2 2D3/2']},
            "Cii_1335.66":{'chianti':1335.6620,'nist':1335.6626,'transitions':['2s2 2p 2P3/2','2s 2p2 2D3/2']},
            "Cii_1335.70":{'chianti':1335.7070,'nist':1335.7077,'transitions':['2s2 2p 2P3/2','2s 2p2 2D5/2']},
            "SiIV1394":{'chianti':1393.7550,'nist':1393.7546,'transitions':['2p6 3s 2S1/2','2p6 3p 2P3/2']},
            "SiIV1402":{'chianti':1402.7700,'nist':1402.7697,'transitions':['2p6 3s 2S1/2','2p6 3p 2P1/2']},
            "MgIIk":2795.528,
            "MgIIh":2802.704
        }
        return NW
    
        

    def get_wrange(self,line):
        """
        Provide wavelength range of the line to be fitted!\n
        line = 'SiIV_1394' [ or SiIV_1402, Cii_1334]
        """
        wrange = [1393.4,1394.1]
        if line == 'SiIV_1394':
            wrange = [1393.4,1394.1]
        elif line == 'CII_1334':
            wrange = [1334.2, 1334.8]
        elif line == 'MgII_k':
            wrange = [2795.70, 2797.00]
        elif line == 'MgII_h':
            wrange = [2802.88, 2804.20]

        else:
            wrange = [None,None]
            raise ValueError(f"{line} doesn't exist!")
        
        return wrange


    def get_data_header(self,hdu_index,print_line = False):
        """
        return data , hd in tuple.
        """


        hd = self.get_hdu(hdu_index).header
        data = self.get_hdu(hdu_index).data
        if print_line:
            try:
                line = self.get_hdu(0).header[f'TDESC{hdu_index}']
                print(f'This is {line} line.')
            except:
                print("That is not a line (TDESC couldn't be found)")

        return data,hd
    
    def get_obs_desc(self):
        """
        return observation descriptions.
        """


        obs_desc = self.get_hdu(0).header['OBS_DESC']
        return obs_desc

    def get_wcs(self,hdu_index):
        _ , hd = self.get_data_header(hdu_index)
        wcs = WCS(hd)
        return wcs

    def get_fuv_exposure_time(self):
        '''
        ndarray of exposure times of length number of raster exposure.
        '''
        hdu_ind = self.N-2
        hdu = self.get_hdu(hdu_index=hdu_ind)

        exposure_index_for_fuv = hdu.header['exptimef']

        exposure_times = hdu.data[:,exposure_index_for_fuv]  

        return exposure_times
    
    def get_nuv_exposure_time(self):
        '''
        ndarray of exposure times of length number of raster exposure.
        '''
        hdu_ind = self.N-2
        hdu = self.get_hdu(hdu_index=hdu_ind)

        exposure_index_for_fuv = hdu.header['exptimen']

        exposure_times = hdu.data[:,exposure_index_for_fuv]  

        return exposure_times      

    
    def get_error(self,data,fuv=True):
        """
        data is in DN\n
        https://hesperia.gsfc.nasa.gov/ssw/iris/idl/nrl/iris_getwindata.pro\n
        NOTE: EXPOSURE TIME NOT CORRECTED! call from get_spectrum for exposure_time corrected one!
        """


        dn2ph = 4.0
        dark_unc = 3.1

        if not fuv:
            dn2ph = 18
            dark_unc = 1.2
        
        data_ph = data*dn2ph
        dark_unc_ph=dark_unc*dn2ph

        k = np.where(data_ph<0)[0]
        data_ph[k] = 0.0
        x = data_ph+dark_unc_ph**2
        err = np.sqrt(x)/dn2ph
        return err
    

    def write_error_cube(self,save_dir):
        """
        That errorcube is normalized to exposure time\n
        save_dir is where all the errors calculated will be saved.
        """

        for hdu_ind in range(1,self.N-2,1):
            file_path = os.path.join(save_dir,f"ERR_hdu_{hdu_ind}.npy")
            if not os.path.isfile(file_path):

                data,_= self.get_data_header(hdu_ind)
                # print(data.shape)
                wave = self.get_wavelengths(hdu_ind)
                exp_time = self.get_fuv_exposure_time()
                fuv= True
                if wave[-1] > 1600 *u.angstrom:
                    exp_time = self.get_nuv_exposure_time()
                    fuv=False
                t0 = time.perf_counter()
                err = self.get_error(data,fuv)/exp_time[:,np.newaxis,np.newaxis]
                t1 = time.perf_counter()

                np.save(file_path,err,allow_pickle=True)
                print(f"{file_path}     saved and took {t1-t0} seconds")
            else:
                print(f"{file_path} already exists!")



    def get_observation_times(self , relative = False):
        '''
        ndarray of exposure times of length number of raster exposure.\n
        If relative = True ; returns --->  (initial_datetime_obj , seconds from initial_date_time_obj , 0 included.)
        '''
        hdu_ind = self.N-2
        hdu = self.get_hdu(hdu_index=hdu_ind)
        hdu0 = self.get_hdu(0)

        time_sec_ind = hdu.header['time']

        times_sec = hdu.data[:,time_sec_ind] 

        obs_start = hdu0.header['STARTOBS']
        obs_start = datetime.datetime.strptime(obs_start,'%Y-%m-%dT%H:%M:%S.%f')
        
        time_obs = np.array([obs_start+datetime.timedelta(seconds=t) for t in times_sec])
        if not relative:
            return  time_obs
        else:
            t0 = time_obs[0]
            time_obs1 = time_obs-t0
            time_obs1_sec = [t.total_seconds() for t in time_obs1]
            time_obs1_sec = np.array(time_obs1_sec)

            return t0 , time_obs1_sec


    def get_wavelengths(self,hdu_index):
        '''
        return wavelengths(ndarray) of a particular line provided by hdu_index
        '''
        hdu = self.get_hdu(hdu_index=hdu_index)
        hd = hdu.header


        unit = hd['CUNIT1']
        axis = hd['NAXIS1']
        crval = hd['CRVAL1']
        crpix = hd['CRPIX1']
        cdelt = hd['CDELT1']

        n = np.arange(axis)

        wavelengths = crval+cdelt*(n-(crpix-1))
        wavelengths = np.array(wavelengths)*u.Unit(unit.lower())
        return wavelengths
    

    def get_spectrum(self,hdu_index,raster_time_step , y_pixel_value,fuv=True,**kwargs):
        """
        kwargs == wmin,wmax,\n
        Assuming fuv = True, if not provide False!\n
        All are normalized.
        """
        hdu = self.get_hdu(hdu_index)
        data = hdu.data[raster_time_step,y_pixel_value]
        wavelength = self.get_wavelengths(hdu_index)

        wavelength = np.array(wavelength)
        data = np.array(data)

        exp_time = self.get_fuv_exposure_time()[raster_time_step]
        if fuv:
            exp_time = self.get_fuv_exposure_time()[raster_time_step]
        else:
            exp_time = self.get_nuv_exposure_time()[raster_time_step]



        if 'wmin' in kwargs:
            wavelength_ind = wavelength>=kwargs['wmin']
            wavelength = wavelength[wavelength_ind]
            data = data[wavelength_ind]
        if 'wmax' in kwargs:
            wavelength_ind = wavelength<=kwargs['wmax']
            wavelength = wavelength[wavelength_ind]
            data = data[wavelength_ind]

        err = self.get_error(data,fuv=fuv)/exp_time

        return wavelength , data/exp_time , err
    


    def get_spectral_cube(self,hdu_index,error_file_dir ='./Errors/D044',fuv=True,**kwargs):
        """
        kwargs:
            wrange = [wmin,wmax].\n
            rebinn = [[m,n],[m1,m2..];
                m pixels along row and n pixels along column are averaged.\n
                [m1,m2,m3] etc are list of sunpy map that require to exclude excessive array because\n
                rebinn shape array must be multiple of (m,n)  \n

        Assuming fuv = True, if not provide False!\n
        return normalized data , normalized error , wcs coordinate , wavelength, list of other maps that\n
            have been extracted from normalized data coordinate. ;\n 
        data[...,].T gives spectroheliograms at different wavelengths ; data[...,].T[0] is typical spectroheliogram with wcs as its wcs coordinate.\n
        """
        file_path = self.file_path
        file_name = os.path.basename(file_path).replace(".fits","")

        error_file_dir = os.path.join(error_file_dir,file_name)
        error_file_path = os.path.join(error_file_dir,f"ERR_hdu_{hdu_index}.npy")

        hdu = self.get_hdu(hdu_index)
        wcs = self.get_wcs(hdu_index)
        data = hdu.data
        wavelength = self.get_wavelengths(hdu_index)

        wavelength = np.array(wavelength)
        data = np.array(data)

        exp_time = self.get_fuv_exposure_time()
        if not fuv:
            exp_time = self.get_nuv_exposure_time()

        exp_time = exp_time[:,np.newaxis,np.newaxis]
        data = data/exp_time

        wcs = wcs.dropaxis(0)
        w_ind = np.argmin(np.abs(wavelength - wavelength[len(wavelength)//2]))
        data_dummpy_2d = hdu.data[...,w_ind]

        hd = wcs.to_header()
        new_hdu = fits.PrimaryHDU(data_dummpy_2d,header=hd)
        new_wcs = WCS(new_hdu.header)

        new_wcs = new_wcs.swapaxes(1,0)
        m_dummy = sm((data_dummpy_2d.T,new_wcs.to_header()))

        if os.path.isfile(error_file_path):
            err = np.load(error_file_path,allow_pickle=True)
        else:
            print(f"{error_file_path} doesn't exist! returning None!")
            err = None

        if 'wrange' in kwargs:
            wavelength_ind = np.logical_and(wavelength>=kwargs['wrange'][0],wavelength<=kwargs['wrange'][-1])
            wavelength = wavelength[wavelength_ind]
        
            wavelength_ind0 = np.where(wavelength_ind)[0]

            data = data[:,:,wavelength_ind0[0]:wavelength_ind0[-1]+1]
            err = err[:,:,wavelength_ind0[0]:wavelength_ind0[-1]+1]


        MS_sub = []
        if 'rebinn' in kwargs:

            m,n = kwargs['rebinn'][0]
            s = data.shape

            DN_x = s[0]-s[0]%n
            DN_y = s[1]-s[1]%m
            data_cut = data[0:DN_x,0:DN_y]
            wcs_cut = m_dummy.wcs[0:DN_x,0:DN_y]


            data_reshape = data[0:DN_x,0:DN_y].reshape((s[0]//n,n,s[1]//m,m,s[2]))
            err_reshape = err[0:DN_x,0:DN_y].reshape((s[0]//n,n,s[1]//m,m,s[2]))

            data = data_reshape.mean(axis=(1,3))
            err = np.mean(np.sqrt(err_reshape*err_reshape),axis=(1,3))  # No error file and rebinn will cause error here!
            m_dummpy_extracted = sm((data_cut[...,].T[0],wcs_cut.to_header()))

            m_dummy = m_dummpy_extracted.resample(data.shape[0:2]*u.pix)  # data.shape[0:2] is (new dim along x axis,new dim along y axis) 
            print(len(kwargs['rebinn']))
            if len(kwargs['rebinn']) == 2:
                MS = kwargs['rebinn'][1]    # list of maps that requires excessive data array to be removed!
                tr = m_dummpy_extracted.top_right_coord    # coordinate in HPC beyond which data has been stripped!
                bl = m_dummpy_extracted.bottom_left_coord
                for ms in MS:
                    bottom_left = SkyCoord(bl.Tx.value*bl.Tx.unit,bl.Ty.value*bl.Ty.unit,frame=ms.coordinate_frame)
                    top_right = SkyCoord(tr.Tx.value*tr.Tx.unit,tr.Ty.value*tr.Ty.unit,frame=ms.coordinate_frame)
                    ms_sub = ms.submap(bottom_left,top_right=top_right)
                    MS_sub.append(ms_sub)
            # else:
            #     m_dummy_resampled = m_dummy
        else:
            MS_sub = None

        # Correct for data<0 = 0
        data[data<0] = 0
        return data,err,m_dummy,wavelength,MS_sub




    def plot_spectrum(self,wavelength,data,err):

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
    
    def get_spectroheliogram(self,hdu_index,wavelength,fuv=True):
        """
        Return sunpy map of the spectroheliogram\n
        assuming fuv = True\n
        if False do mention!
        """

        exp_time = self.get_fuv_exposure_time()
        if fuv:
            exp_time = self.get_fuv_exposure_time()
        else:
            exp_time = self.get_nuv_exposure_time()


        wavelengths = self.get_wavelengths(hdu_index).value
        if wavelength>=np.min(wavelengths) and wavelength<=np.max(wavelengths):
            hdu = self.get_hdu(hdu_index)
            hd = hdu.header
            wcs = WCS(hd)
            wcs = wcs.dropaxis(0)

            index =  np.argmin(np.abs(wavelengths - wavelength))
            data = hdu.data[...,index]

            hd = wcs.to_header()
            new_hdu = fits.PrimaryHDU(data,header=hd)
            new_wcs = WCS(new_hdu.header)

            new_wcs = new_wcs.swapaxes(1,0)
            m = sm((data.T/exp_time,new_wcs.to_header()))

            return m
        else:
            print(f"wavelength is out of range.\n range = [{wavelengths[0]} , {wavelengths[-1]}]")

    def plot_spectroheliogram(self,m,vmin=0,vmax=10):
        """
        m is spectroheliogram map\n 
        """
        m = raster.get_spectroheliogram(4,1393.7)
        fig1 = plt.figure(figsize=(10,7))
        ax1 = fig1.add_subplot(1,2,1,projection=m)
        ax2 = fig1.add_subplot(1,2,2)
        ax1.imshow(m.data,origin='lower',vmin=vmin,vmax=vmax , aspect = 0.1)
        ax2.imshow(m.data,origin='lower',vmin=vmin,vmax=vmax , aspect = 0.1)

        ax1.set_xlabel('Time',fontsize=20,fontweight='medium')
        ax1.set_ylabel('Solar - Y',fontsize=20,fontweight='medium')

        return fig1


    def fit_datacube_gaussian1_with_linear_bg(self,args):
        """
        data is spectra with wavelength and error all 1d array\n
        i is just to track index to re assign array of popt etc!\n
        """
        data,err,wavelength,guess_wave_min,guess_wave_max,bg_peak_wave,i = args
        data = np.array(data)
        err = np.array(err)
        wavelength = np.array(wavelength)
        
        peak_height_init = float(np.max(data))
        ind_peak = np.where(data==peak_height_init)[0]
        w_cen = float(wavelength[ind_peak][0])

        # if w_cen<guess_wave_min or w_cen>guess_wave_max:
        #     w_cen = bg_peak_wave
            # print(f"wcen = {w_cen}, which is out of bound (1393.4,1394.0)!")
            
        w_cen = bg_peak_wave
        peak_height = 3
        init_value=[peak_height_init,w_cen,0.1,0.0,0.0]

        bnd_l = [0,1393.5,0,-np.inf,-np.inf]
        bnd_u = [np.inf,1394.0,2,np.inf,np.inf]
        bnd = (bnd_l,bnd_u)

        try:
            popt,pcov_false = curve_fit(Models.Gauss1linearbackground,wavelength,data,p0=init_value,sigma=err,bounds=bnd,absolute_sigma=False)
            popt1,pcov_true = curve_fit(Models.Gauss1linearbackground,wavelength,data,p0=init_value,sigma=err,bounds=bnd,absolute_sigma=True)
            red_chisq = pcov_false/pcov_true

            G = Models.Gauss1linearbackground(wavelength,popt[0],popt[1],popt[2],popt[3],popt[4])
            chi_sq = (data-G)**2/err**2
            red_chisq1 = np.sum(chi_sq)/(len(data)-5)
        except:
            popt = np.array([np.nan]*5)
            popt1 = np.array([np.nan]*5)
            red_chisq = np.array([np.nan]*25).reshape((5,5))
            red_chisq1 = np.nan


        return wavelength,data,popt,popt1,red_chisq,red_chisq1,i




    def fit_si_IV1394(self,raster_time_step=15,ypixel=400,return_only_popt = False):
        """
        NOTE: assuming hdu_index for si_IV1394 is 4.\n
        If not modify the line, wavelength,data = raster.get_spectrum(4,15,400,wmin=1393.2,wmax = 1394.4)\n
        Optimize for constant background.
        """
        wmin = self.get_wrange("SiIV_1394")[0]
        wmax = self.get_wrange("SiIV_1394")[1]
        hdu_ind = 4

        wavelength,data,err = self.get_spectrum(hdu_ind,raster_time_step,ypixel,wmin=wmin,wmax = wmax)


        peak_height_init = float(np.max(data))
        ind_peak = np.where(data==peak_height_init)[0]
        w_cen = float(wavelength[ind_peak][0])

        if w_cen<1393.4 or w_cen>1394.00:
            w_cen = 1393.7587
            # print(f"wcen = {w_cen}, which is out of bound (1393.4,1394.0)!")

        init_value=[peak_height_init,w_cen,0.1,0.0,0.0]

        try:
            popt,pcov_false = curve_fit(Models.Gauss1linearbackground,wavelength,data,p0=init_value,sigma=err,absolute_sigma=False)
            popt1,pcov_true = curve_fit(Models.Gauss1linearbackground,wavelength,data,p0=init_value,sigma=err,absolute_sigma=True)
            red_chisq = pcov_false/pcov_true

            G = Models.Gauss1linearbackground(wavelength,popt[0],popt[1],popt[2],popt[3],popt[4])
            chi_sq = (data-G)**2/err**2
            red_chisq1 = np.sum(chi_sq)/(len(data)-5)
        except:
            popt = np.array([np.nan]*5)
            popt1 = np.array([np.nan]*5)
            red_chisq = np.array([np.nan]*25).reshape((5,5))
            red_chisq1 = np.nan

        if not return_only_popt:
            if int(np.sum(~np.isnan(popt))) != 0:
                G = Models.Gauss1linearbackground(wavelength,popt[0],popt[1],popt[2],popt[3],popt[4])
            else:
                G = np.array([np.nan]*len(wavelength))
            return wavelength,data,G
        else:
            return wavelength,data,popt,popt1,red_chisq,red_chisq1



if __name__=="__main__":
    import glob
    import astropy.units as u
    from astropy.coordinates import SkyCoord, SpectralCoord
    from sunpy.coordinates import frames
    from astropy.wcs.utils import wcs_to_celestial_frame
    from ndcube import NDCube
    from sunpy.coordinates import frames

    def slice_spectralcube_by_spatial_pixel(spectral_cube,row_l,row_u,col_l,col_u):
        """
        row_l is lower row index in spatial coord
        """
        D2 = spectral_cube[col_l:col_u+1,row_l:row_u+1]

        return D2

    file1 = './data/iris_l2_20160319_145728_3623010639_raster_t000_r00000.fits'
    file2 = './data/iris_l2_20160319_145728_3623010639_raster_t000_r00000.fits'

    raster_files = glob.glob("./data_aligned/D044/RASTER_FILES/*.fits")
    raster_files = sorted(raster_files)

    file1 = raster_files[0]
    with IrisRaster(file1) as raster:

        data,hd = raster.get_data_header(4)
        m = raster.get_spectroheliogram(4,1393.75)
        wcs = WCS(hd)
        coordinate = SkyCoord(13*u.arcsec,330*u.arcsec,frame=m.coordinate_frame)
        P = m.world_to_pixel(coordinate)[0].value
        print(P)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(m.data,origin="lower",extent=get_extent(m))
        plt.show()



        # exp_time = self.get_fuv_exposure_time()
        # if fuv:
        #     exp_time = self.get_fuv_exposure_time()
        # else:
        #     exp_time = self.get_nuv_exposure_time()


        # wavelengths = self.get_wavelengths(4).value
        # hdu = self.get_hdu(hdu_index)
        # hd = hdu.header
        # wcs = WCS(hd)
        # wcs = wcs.dropaxis(0)

        # index =  np.argmin(np.abs(wavelengths - 1393.75))
        # data_cropped = cropped_cube.data[...,index]

        # hd = wcs.to_header()
        # new_hdu = fits.PrimaryHDU(data,header=hd)
        # new_wcs = WCS(new_hdu.header)

        # new_wcs = new_wcs.swapaxes(1,0)
        # m_subcube = sm((data_cropped.T/exp_time,new_wcs.to_header()))





        # plt.show()

            



