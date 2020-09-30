import numpy as np
import matplotlib.pyplot as plt
from random import uniform as rand
import scipy.io as sio
'''@author: Zi Xian Leong (zxleong@psu.edu) '''

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

from utils.random_vel_generator import vel_generator, vel_sig_generator
from utils.random_ep_generator import ep_generator


from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.smooth import smooth

from utils.gpr_generate_model import generate_interface_gpr_td, generate_interface_gpr
from scipy.interpolate import interp1d
import hdf5storage as hdf


'''Part2 - Process raw GPR data and Velocity to create training data for GPRNet'''


# %% Load raw gather

#This is in dt=1e-9/20

AllRawGathers = hdf.loadmat('Field/Data/AllRawGathers.mat')['AllRawGathers'] 
vel_td_stack = sio.loadmat('Field/Data/veltd_raw_corr.mat')['vel_td_stack']


#%% Downsample raw gathers so that it's the same dt as veltd

sim_dt = 1e-9/20
data_tt = np.arange(0,8000*sim_dt,sim_dt)
vel_tt = np.arange(0,400*1e-9,1e-9)

numberOfSamples = 50000


#Downsample
dsGathers = []
for i in range(numberOfSamples):
    tpTrace_i = (AllRawGathers[i,:])
    dsTrace_i = interp1d(data_tt,tpTrace_i,kind='linear',fill_value=np.nan)(vel_tt)
    dsGathers.append(dsTrace_i)
dsGathers = np.array(dsGathers)


# Apply taper
def taper(in_array): 
    window = tukey(400,0.3)
    window[100:len(window)]=1
    window = window**4
    # plt.plot(window)
    
    out = in_array * window
    return out

#Tapered gathers
dstpGathers = np.apply_along_axis(taper,1,dsGathers)


#%%

# plt.plot(AllRawGathers[3455,:])
# plt.plot(AllRawGathers[26567,:])


#%% Sample one trace and vel

# ind = np.random.randint(numberOfSamples)
# data = dstpGathers[ind,:]
# vel = vel_td_stack[ind,:]

# #display
# fig =plt.figure(figsize=(5,8))
# ax = fig.add_subplot(111)
# x = np.arange(400)
# ax.invert_yaxis()
# ax.plot(data,x,color='black')
# ax2 = ax.twiny()
# ax2.plot(vel/1e9,x,color='red')



# %% sample one trace

# ind = np.random.randint(20000)
# sampleTrace = AllRawGathers[ind,:]
# sampleVel = vel_td_stack[ind,:]

# # plt.plot(sampleTrace)
# plt.plot(taper(sampleTrace))

# #Downsample it to 200ns
# ori_tt = np.arange(0,2000*1e-10,1e-10)
# vel_tt = np.arange(0,200*1e-9,1e-9)
# dest_dt = 200e-9/256
# dest_tt = np.arange(0,256*dest_dt,dest_dt)
# dsTrace = interp1d(ori_tt,taper(sampleTrace),kind='linear',fill_value=np.nan)(dest_tt)
# dsVel = interp1d(vel_tt,sampleVel,kind='linear',fill_value='extrapolate')(dest_tt)

# %% Creating noise

'''
Add a range of noise ranging from 0.15sigma to 0.85sigmas, with coefficient of random uniform
'''


noiseData=[]
multiplier=8
for itr in range(multiplier):
    for i in range(numberOfSamples):
        sig_tr = np.std(dstpGathers[i,:])
        mean_tr = np.mean(dstpGathers[i,:])
        noiseCoeff = np.random.uniform(0.15,0.85)
        randnoise = np.random.normal(loc=mean_tr, scale=sig_tr*noiseCoeff,size=dstpGathers[i,:].shape)
        noiseData.append(dstpGathers[i,:]+randnoise)
noiseData=np.array(noiseData)



#display
# ind = np.random.randint(numberOfSamples*multiplier)
# fig =plt.figure(figsize=(5,8))
# ax = fig.add_subplot(111)
# x = np.arange(400)
# ax.invert_yaxis()
# ax.plot(noiseData[ind,:],x,color='black')
# ax2 = ax.twiny()
# ax2.plot(vel_td_stack[ind,:]/1e9,x,color='red')

    
#%% Augment data by adding time gain on tapered gathers

dt = np.arange(0,1e-9*400,1e-9)

#apply time gain on tapered gathers without noise
timegainedGathers_only = []
for itr in range(multiplier):
    for i in range(numberOfSamples):
        timegainCoeff = np.random.uniform(3,20)
        tfactor = np.exp(dt*1e6)**timegainCoeff
        timegainedGathers_only.append(dstpGathers[i,:]*tfactor)    
timegainedGathers_only = np.array(timegainedGathers_only)

#display
# ind = np.random.randint(numberOfSamples*multiplier)
# fig =plt.figure(figsize=(5,8))
# ax = fig.add_subplot(111)
# x = np.arange(256)
# ax.invert_yaxis()
# ax.plot(timegainedGathers_only[ind,:],x,color='black')
# ax2 = ax.twiny()
# ax2.plot(vel_td_stack[ind,:]/1e9,x,color='red')

# ind = np.random.randint(numberOfSamples)
# plt.plot(dstpGathers[ind,:])
# plt.plot(timegainedGathers_only[ind,:])


# %% Augment data by applying range of noise on time gained gathers


dt = np.arange(0,1e-9*400,1e-9)
# tfactor = np.exp(dt*1e7)**1.2

#apply time gain on tapered gathers with noise
timegainedGathers_withNoise = []
for itr in range(multiplier):
    for i in range(numberOfSamples):
        sig_tr = np.std(timegainedGathers_only[i,:])
        mean_tr = np.mean(timegainedGathers_only[i,:])
        noiseCoeff = np.random.uniform(0.15,0.85)
        randnoise = np.random.normal(loc=mean_tr, scale=sig_tr*noiseCoeff,size=timegainedGathers_only[i,:].shape)
        timegainedGathers_withNoise.append(timegainedGathers_only[i,:]+randnoise)    
timegainedGathers_withNoise = np.array(timegainedGathers_withNoise)

#display
# ind = np.random.randint(numberOfSamples*multiplier)
# fig =plt.figure(figsize=(5,8))
# ax = fig.add_subplot(111)
# x = np.arange(400)
# ax.invert_yaxis()
# ax.plot(timegainedGathers_withNoise[ind,:],x,color='black')
# ax2 = ax.twiny()
# ax2.plot(vel_td_stack[ind,:]/1e9,x,color='red')





#%% Concatenate augmented data

combinedGathers = np.concatenate((dstpGathers,noiseData,timegainedGathers_only,timegainedGathers_withNoise),axis=0)
combinedVel = np.concatenate((vel_td_stack,                                                     #Original
                              vel_td_stack,vel_td_stack,vel_td_stack,vel_td_stack,vel_td_stack,vel_td_stack, vel_td_stack, vel_td_stack, #noise only
                              vel_td_stack,vel_td_stack,vel_td_stack,vel_td_stack,vel_td_stack,vel_td_stack, vel_td_stack, vel_td_stack, #time gain only
                              vel_td_stack,vel_td_stack,vel_td_stack,vel_td_stack,vel_td_stack,vel_td_stack, vel_td_stack, vel_td_stack,), #time gain with noise
                             axis=0)

#add normalization
combinedGathers_norm = pp.normalize(combinedGathers,norm='max',axis=1)

# a = pp.normalize(noiseData,norm='max',axis=1)

# plt.plot(noiseData[141,:])

#display
ind = np.random.randint(numberOfSamples*25)
print(ind)
fig =plt.figure(figsize=(5,8))
ax = fig.add_subplot(111)
x = np.arange(400)
ax.invert_yaxis()
ax.plot(combinedGathers[ind,:],x,color='black')
ax2 = ax.twiny()
ax2.plot(combinedVel[ind,:],x,color='red')



# %% Find nan

#check minimum index that is not nan
init = 455
ls = []
for i in range(numberOfSamples*25):
    each_im = combinedGathers_norm[i,:]
    boob = np.isnan(each_im)
    last_val_ind = len(np.where(boob == False)[0]) - 1
    ls.append(last_val_ind)
    if last_val_ind < init:
        init = last_val_ind
print('Last index that is not nan value: ', init)


# %% Save


np.save('Field/Data/ForDL/GPRData.npy',combinedGathers_norm)
np.save('Field/Data/ForDL/Vel.npy',combinedVel)











