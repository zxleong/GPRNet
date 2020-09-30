'''@author: Zi Xian Leong (zxleong@psu.edu) '''
#%%

import numpy as np
import matplotlib.pyplot as plt
from random import uniform as rand
import scipy.io as sio
import os
from sklearn import preprocessing as pp
from sklearn.preprocessing import StandardScaler
from random import randrange
from skimage.transform import rescale, resize
import scipy.ndimage.filters as fil
from scipy.misc import imresize
from random import uniform as rand
import random
from numpy.matlib import repmat
from scipy.signal import tukey

from utils.random_vel_generator import vel_generator

from tqdm import tqdm
from utils.smooth import smooth

from utils.gpr_generate_model import generate_interface_gpr
from scipy.interpolate import interp1d


'''Generates a total of 50,000 random (unique) velocity profiles

    - 42,000 from totally random 1D list realization
    - 8,000 came from selecting 10 random 1D profile from 800 2D 'wavy' subsurface models 

'''

# %%

def total_time(input_arr,diffdepth):
    """"Computes the time taken for wave to pass each layer """
    
    return 2*np.sum(diffdepth/input_arr)

def add_air(input_arr):
    '''Adds 10 grid rows of air, '''
    c = 299792500;
    return np.insert(input_arr,0,(c,c,c,c,c,c,c,c,c,c))



# %% Making 42k random 1D velocity models
#479 depth, 29 window
# # 
c=299792500 #speed of light, in m/s
epmin=2
epmax=50

computeVelMin = c/np.sqrt(epmax)/1e6
computeVelMax = c/np.sqrt(epmin)/1e6    


# Create velocities
n=42000
pbar = tqdm(total=n, position=0, leave=True)    
arr = []
tt_list = []
while len(arr) < n:
    
    vegas = np.random.randint(0,2)
    lyr = vel_generator(gamble=vegas, velmin=computeVelMin, velmax=computeVelMax, depth=321, thickmin=6, thickmax=21) * 1e6 #times 1e6 to make it m/s
    lyr_sm = smooth(lyr,window_len=19, window='bartlett') #smoothing
    dz=0.05 #meters
    total_time(lyr_sm,dz)
    
    tt = 400*1e-9 #field settings - 400ns
    if total_time(lyr_sm,dz) > tt:  #making sure the total time is the same as the field settings
        tt_list.append(total_time(lyr_sm,dz))
        lyr_sma = add_air(lyr_sm) #adds air layers

        arr.append(lyr_sma) 
        pbar.update(1)
pbar.close()
arr = np.array(arr)
tt_list = np.array(tt_list)


#Display
# ind =np.random.randint(n) 
# sample = arr[ind].T
# ax = plt.gca()
# ax.set_xticks([10,50,100,150,200,250])
# ax.set_xticklabels([0,2.5,5,7.5,10,12.5])
# ax.set_xlabel('Common Offset (m)')
# plt.plot(sample)



# %% Making 8k from 2D generator, taking 10 slice from 800 models

#Generate 800 models
n=800
pbar = tqdm(total=n, position=0, leave=True)    
arr_2d = []
while len(arr_2d) < n:
    model = generate_interface_gpr(num_vel=1, air=0, num_layers=(4,21), num_bricks_each_layer=(1,2),
                   layer_grid_fluc = 6, height=339, width=127,
                   x_var=(0,128), order=(2,6), vel_min=computeVelMin,vel_max=computeVelMax)
    model = model[0] #remove first axis
    model_sm = np.apply_along_axis(smooth,1,model,window_len=71, window='blackman')
    model_smb = fil.gaussian_filter(model_sm,sigma=0.5)
    # plt.imshow(model_smb)
    
    check_total_time = []
    for i in range(197):
        each_line = model_smb[:,i]
        tt = total_time(each_line,0.05)
        check_total_time.append(tt > 400*1e-9) 
    if sum(check_total_time) == 197:
        model_final = np.apply_along_axis(add_air,0,model_smb)
        arr_2d.append(model_final)
        pbar.update(1)
pbar.close()
arr_2d = np.array(arr_2d)

#display
# img = arr_2d[np.random.randint(n)]
# plt.imshow(img,cmap='jet',vmax=2e8)
# plt.colorbar()

#taking 10 slices from each of the 800 models - 8000 slices
arr_2d1d=[]
for i in range(n):
    each_model = arr_2d[i,:,:]
    ind = np.random.randint(0,197,size=10)
    slices = each_model[:,ind]
    for j in range(10):
        arr_2d1d.append(slices[:,j])
arr_2d1d = np.array(arr_2d1d) 

# #selecting random 6000 velocity profiles
# sample = arr_2d1d[np.random.randint(6000),:]

# display
# ind = np.random.randint(8000)
# sample= arr_2d1d[ind,:]
# plt.plot(sample)


# %% Combine

combined_arr = np.concatenate((arr,arr_2d1d))

# ind = np.random.randint(50000)
# sample = combined_arr[ind,:]
# plt.plot(sample)

ep_stack = 299792500**2 / combined_arr**2



# %% Convert into time domain

veltd_stack = []
for i in range(50000):
    each_vel=combined_arr[i,:]
    ori_tt = 2*np.cumsum(0.05/each_vel[0:len(each_vel)-1])
    ori_tt = np.insert(ori_tt,0,0)
    dest_tt = np.arange(0,400*1e-9,1e-9)
    vel_td = interp1d(ori_tt,each_vel,kind='linear',bounds_error=False,fill_value=np.nan)(dest_tt)
    veltd_stack.append(vel_td)
veltd_stack=np.array(veltd_stack).T

#display
# ind = np.random.randint(50000)
# print(ind)
# plt.plot(veltd_stack[:,ind])


#check minimum index that is not nan
# init = 500
# ls = []
# for i in range(50000):
#     each_im = veltd_stack[:,i]
#     boob = np.isnan(each_im)
#     last_val_ind = len(np.where(boob == False)[0]) - 1
#     ls.append(last_val_ind)
#     if last_val_ind < init:
#         init = last_val_ind
# print('Last index that is not nan value: ', init)


# %% Save

sio.savemat('Field/Data/ep.mat',{'ep_stack':ep_stack})
sio.savemat('Field/Data/veltd_raw.mat',{'veltd_stack':veltd_stack})



