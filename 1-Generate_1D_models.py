'''@author: Zi Xian Leong (zxleong@psu.edu) '''

#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os


import scipy.ndimage.filters as fil
from scipy.interpolate import interp1d
from utils.random_vel_generator import vel_generator
from tqdm import tqdm
from utils.gpr_generate_model import generate_interface_gpr
from utils.smooth import smooth
from scipy.signal import tukey

'''Generates a total of 10,000 random velocity profiles

    - 7,000 from totally random 1D list realization
    - 3,000 came from selecting 5 random 1D profile from 600 2D 'wavy' subsurface models 

'''


#%% Some Fucntions

def total_time(input_arr,diffdepth):
    """"Computes the time taken for wave to pass each layer """
    
    return 2*np.sum(diffdepth/input_arr)

def add_air(input_arr):
    '''Adds 10 layers of air'''
    c = 299792500;
    return np.insert(input_arr,0,(c,c,c,c,c,c,c,c,c,c))

# %% Making 7k from random generator

n=7000
pbar = tqdm(total=n, position=0, leave=True)    
arr = []
while len(arr) < n:
    vegas = np.random.randint(0,2)
    lyr = vel_generator(gamble=vegas, velmin=48, velmax=175, depth=111, thickmin=8, thickmax=15) * 1e6 #times 1e6 to make it m/s
    lyr_sm = smooth(lyr,window_len=9, window='bartlett') #smoothing
    dz=0.05 #depth domain, this means 0.05m
    tt = 1280*8e-11 #set minimum tt
    if total_time(lyr,dz) >= tt:  #making sure the total time is the same as the field settings
        lyr_sma = add_air(lyr_sm) #adds air layers

        arr.append(lyr_sma) 
        pbar.update(1)
pbar.close()
arr = np.array(arr)

sample = arr[np.random.randint(n)].T
plt.plot(sample)

total_time(sample,0.05)


# %% Making 2k from 2D generator, taking 5 slice from 600 models

#Generate 600 models
n=600
pbar = tqdm(total=n, position=0, leave=True)    
arr_2d = []
while len(arr_2d) < n:
    model = generate_interface_gpr(num_vel=1, air=0, num_layers=(4,12), num_bricks_each_layer=(1,2),
                   layer_grid_fluc = 4, height=119, width=127,
                   x_var=(0,128), order=(2,6))
    model = model[0] #remove first axis
    model_sm = np.apply_along_axis(smooth,1,model,window_len=71, window='blackman')
    model_smb = fil.gaussian_filter(model_sm,sigma=0.8)
    # plt.imshow(model_smb)
    
    check_total_time = []
    for i in range(197):
        each_line = model_smb[:,i]
        tt = total_time(each_line,0.05)
        check_total_time.append(tt >= 1280*8e-11)
    if sum(check_total_time) == 197:
        model_final = np.apply_along_axis(add_air,0,model_smb)
        arr_2d.append(model_final)
        pbar.update(1)
pbar.close()
arr_2d = np.array(arr_2d)

#display
img = arr_2d[np.random.randint(n)]
plt.imshow(img,cmap='jet',vmax=2e8)
plt.colorbar()

#taking 5 slices from each of the 600 models
arr_2d1d=[]
for i in range(n):
    each_model = arr_2d[i,:,:]
    ind = np.random.randint(0,197,size=5)
    slices = each_model[:,ind]
    for j in range(5):
        arr_2d1d.append(slices[:,j])
arr_2d1d = np.array(arr_2d1d) 

# sample = arr_2d1d[np.random.randint(3000),:]
# sample= arr_2d1d[1834,:]
# plt.plot(sample)

# %% Combine 7k and 3k models = 10k models

combined_arr = np.concatenate((arr,arr_2d1d))

# sample = combined_arr[np.random.randint(10000),:]
# plt.plot(sample)

#Compute dielectric permittivity
ep_stack = 299792500**2 / combined_arr**2

# %% Convert into time domain - this will be our labels (GPRNet Output)

veltd_stack = []
for i in range(10000):
    each_vel=combined_arr[i,:]
    ori_tt = 2*np.cumsum(0.05/each_vel[0:len(each_vel)-1])
    ori_tt = np.insert(ori_tt,0,0)
    dest_tt = np.arange(0,1680*8e-11,8e-11)
    vel_td = interp1d(ori_tt,each_vel,kind='linear',bounds_error=False,fill_value=np.nan)(dest_tt)
    veltd_stack.append(vel_td)
veltd_stack=np.array(veltd_stack).T

# plt.plot(veltd_stack[:,np.random.randint(10000)])


#check minimum index that is not nan
init = 1676
ls = []
for i in range(n):
    each_im = veltd_stack[:,i]
    boob = np.isnan(each_im)
    last_val_ind = len(np.where(boob == False)[0]) - 1
    ls.append(last_val_ind)
    if last_val_ind < init:
        init = last_val_ind
print('Last index that is not nan value: ', init)

#cut to 1280
veltd_stack_cut = veltd_stack[:1280,:]

# plt.plot(veltd_stack_cut[:,np.random.randint(10000)])



# %% Save

# sio.savemat('Synthetic/Data/1D/ep.mat',{'ep_stack':ep_stack})
# np.save('Synthetic/1D/veltd.npy',veltd_stack_cut)




