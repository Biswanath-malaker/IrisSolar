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
import os
import datetime
from astropy.io import fits
from sunpy.time import TimeRange
from Functions.misc import read_json
import matplotlib.pyplot as plt


D = 'D044'
m = 2
n = 1

files = glob.glob(f"./../iris_obj/data_aligned/{D}/RASTER_FILES/*.fits")
files = sorted(files)

# raster_file = files[0]

T = []
for file in files:
    with IrisRaster(file) as raster0:
        obs_times0 = raster0.get_observation_times()

        DT = TimeRange(obs_times0[0],obs_times0[-1])
        cent_time = DT.center.strftime('%Y-%m-%dT%H_%M_%S')
        init_time = datetime.datetime.strptime(cent_time,'%Y-%m-%dT%H_%M_%S')
        T.append(init_time)
        
result_files = glob.glob(f"./result_data/D044/light_curves/MgII/m2_n1/*.json")
result_files = sorted(result_files)

# B_POS = []
# B_NEG = []
# B_SIGNED = []
# B_UNSIGNED = []

RED_SHIFT = []
BLUE_SHIFT = []
SIGNED_SHIFT = []
UNSIGNED_SHIFT = []
# print(result_files)

for f in result_files:
    df = read_json(f)
    
    doppler_shift = df['doppler_shift']
    # B = df['L_hmi']
    
    doppler_shift_arr = np.array(doppler_shift)
    # B_arr = np.array(B)
    
    
    red_shift_arr = doppler_shift_arr[doppler_shift_arr>=0]
    blue_shift_arr = doppler_shift_arr[doppler_shift_arr<0]
    # B_pos = B_arr[B_arr>=0]
    # B_neg = B_arr[B_arr<=0]

    
    # B_POS.append(np.average(B_pos))
    # B_NEG.append(np.average(B_neg))
    # B_SIGNED.append(np.average(B_arr))
    # B_UNSIGNED.append(np.average(np.abs(B_arr)))
    
    RED_SHIFT.append(np.average(red_shift_arr))
    BLUE_SHIFT.append(np.average(blue_shift_arr))
    SIGNED_SHIFT.append(np.average(doppler_shift_arr))
    UNSIGNED_SHIFT.append(np.average(np.abs(doppler_shift_arr)))


RED_SHIFT = np.array(RED_SHIFT)
BLUE_SHIFT = np.array(BLUE_SHIFT)
SIGNED_SHIFT = np.array(SIGNED_SHIFT)
UNSIGNED_SHIFT = np.array(UNSIGNED_SHIFT)

fig1 = plt.figure(figsize=(12,6))
ax1_f1 = fig1.add_subplot(1,1,1)
# ax2_f1 = fig1.add_subplot(2,1,2)
ax_t_ax1_f1 = ax1_f1.twinx()
# ax_t_ax2_f1 = ax2_f1.twinx()

pl1 = ax1_f1.plot(T,RED_SHIFT,color='red',label='RED_SHIFT')
pl11 = ax_t_ax1_f1.plot(T,BLUE_SHIFT,color='blue',label='BLUE_SHIFT')
# pl21 = ax2_f1.plot(T,SIGNED_SHIFT,color='red',label='SIGNED_SHIFT')
# pl22 = ax_t_ax2_f1.plot(T,UNSIGNED_SHIFT,color='blue',label='UNSIGNED_SHIFT')

ax1_f1.legend(['RED_SHIFT'])
ax_t_ax1_f1.legend(['BLUE_SHIFT'])

# ax2_f1.legend(['SIGNED_SHIFT'])
# ax_t_ax2_f1.legend(['UNSIGNED_SHIFT'])

ax1_f1.set_title(f"RED_SHIFT - BLUE_SHIFT")
# ax2_f1.set_title(f"SIGNED_SHIFT - UNSIGNED_SHIFT")






# B_POS = np.array(B_POS)
# B_NEG = np.array(B_NEG)
# B_SIGNED = np.array(B_SIGNED)
# B_UNSIGNED = np.array(B_UNSIGNED)

# print(B_POS)

# print(B_POS.shape,B_NEG.shape,B_SIGNED.shape,B_UNSIGNED.shape)

# fig2 = plt.figure(figsize=(12,6))
# ax1_f2 = fig2.add_subplot(1,1,1)
# # ax2_f2 = fig2.add_subplot(2,1,2)
# ax_t_ax1_f2 = ax1_f2.twinx()
# ax_t_ax2_f2 = ax2_f2.twinx()

# ax1_f2.plot(T,B_POS,color='red',linestyle='-',label='B_POS')
# ax_t_ax1_f2.plot(T,B_NEG,linestyle='-.',color='blue',label='B_NEG')
# ax2_f2.plot(T,B_SIGNED,color='red',label='B_SIGNED')
# ax_t_ax2_f2.plot(T,B_UNSIGNED,color='blue',label='B_UNSIGNED')

# ax1_f2.legend(['B_POS'])
# ax_t_ax1_f2.legend(['B_NEG'])

# ax2_f2.legend(['B_SIGNED'])
# ax_t_ax2_f2.legend(['B_UNSIGNED'])

# ax1_f2.set_title(f"B_POS - B_NEG")
# ax2_f2.set_title(f"B_SIGNED - B_UNSIGNED")



# fig3 = plt.figure(figsize=(15,8))
# ax1_f3 = fig3.add_subplot(3,1,1)
# ax2_f3 = fig3.add_subplot(3,1,2)
# ax3_f3 = fig3.add_subplot(3,1,3)

# ax1_f3.plot(T,RED_SHIFT,color='red',label='RED_SHIFT')
# ax1_f3.plot(T,BLUE_SHIFT,color='blue',label='BLUE_SHIFT')
# ax1_f3.plot(T,np.abs(BLUE_SHIFT),color='k',label='ABS_BLUE_SHIFT')

# ax1_f3.plot(T,RED_SHIFT+np.abs(BLUE_SHIFT),linestyle='-.',color='k',label='UNSIGNED SHIFT')
# ax1_f3.plot(T,RED_SHIFT+BLUE_SHIFT,linestyle='--',color='k',label='SIGNED SHIFT')


# ax1_f3.legend(['redshift','blueshift','absolute blueshift','unsigned shift','signed shift'])

# ax2_f3.plot(T,SIGNED_SHIFT,color='red',label='signed shift')
# ax2_f3.plot(T,UNSIGNED_SHIFT,color='k',label='unsigned shift')
# ax2_f3.legend(['signed shift','unsigned shift'])

# ax3_f3.plot(T,UNSIGNED_SHIFT,color='red',label='signed shift dierctly')
# ax3_f3.plot(T,RED_SHIFT+np.abs(BLUE_SHIFT),linestyle='--',color='k',label='SIGNED SHIFT indirectly')
# ax3_f3.legend(['signed shift dierctly','signed shift indirectly'])


fig1.savefig("vel.jpeg")
# fig2.savefig("Mag.jpeg")
# fig3.savefig("rough.jpeg")

plt.show()
    

