from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import sunpy.map

import warnings
warnings.filterwarnings('ignore')

from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore',category=VerifyWarning)


class utilities:
    def __init__(self):
        pass
    
    @staticmethod
    def simple_plot(data,ax,vmin=None,vmax=None,extent=None):
        ax.imshow(data,origin='lower',vmin=vmin,vmax=vmax,extent=extent)
        return ax
    
    @staticmethod
    def datetime_obj_to_str(t):
        x = t.strftime('%b %d %Y %H:%M:%S.%f UT')
        return x
    
    @staticmethod
    def sunpy_map_cut(file:str,lower_left:list,upper_right:list):
        """
        file ---> either fits file or sunpy map.
        return sunpy map
        lower_left and upper_right are in arcsec.
        """
        if getattr(file,"__module__",None)=='sunpy.map.sources.sdo':
            m = file
        else:
            m = sunpy.map.Map(file)

        top_right = SkyCoord(upper_right[0] * u.arcsec, upper_right[1] * u.arcsec, frame=m.coordinate_frame)
        bottom_left = SkyCoord(lower_left[0] * u.arcsec, lower_left[1] * u.arcsec, frame=m.coordinate_frame)
        m_submap = m.submap(bottom_left, top_right=top_right)
        return m_submap
    
    @staticmethod
    def get_nearest_time_index(time_list , time):
        """
        time_list is a list of datetime object\n
        time is datetime object\n
        return index of the time_list nearest to t\n
        """
        import numpy as np

        time_list = np.array(time_list)
        dt = np.abs(time_list-time)
        i = np.argmin(dt)
        
        return i
    
    @staticmethod
    def ext2rect(ext):
        return [ext[0],ext[1],ext[1],ext[0],ext[0]],[ext[2],ext[2],ext[3],ext[3],ext[2]]
    
    @staticmethod
    def ext2bltr(ext):
        return [ext[0],ext[2]],[ext[1],ext[3]]
    
    @staticmethod
    def bltr2ext(bl,tr):
        return [bl[0],tr[0],bl[1],tr[1]]

if __name__ == "__main__":
    import datetime

    time_list = []
    t0 = datetime.datetime(year=2023,month=8,day=5,hour=0,minute=0,second=0)
    dt = datetime.timedelta(minutes=3)

    for i in range(30):
        time_list.append(t0+i*dt)


    t_test = datetime.datetime(year=2023,month=8,day=5,hour=0,minute=58,second=30)

    x = utilities()

    i = x.get_nearest_time_index(time_list,t_test)
    print(time_list[i])

    
def get_vmin_vmax(data):
    """
    return rough vmin = median-std , vmax = median+std\nval<=0 omitted.
    """
    d = data.flatten()

    l = d<0
    d = d[~l]
    median = np.median(d)
    sd = np.std(d)
    return median-sd , median+sd

