'''@author: Zi Xian Leong (zxleong@psu.edu) '''

import numpy as np
import matplotlib.pyplot as plt
from random import uniform as rand
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as pp

from numpy.matlib import repmat

from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import scipy.ndimage.filters as fil
# from skimage.metrics import structural_similarity as ssim
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from tqdm import tqdm
import scipy.stats
from scipy.signal import tukey

#%% 2D velocity model used is given:

#in spatial depth domain (m)
yTrue2D_vel_dd = np.load('Synthetic/Data/2D/yTrue2D_vel_dd.npy')

#convert to time depth (ns) for neural network training
''' Some info:
    yTrue2D_vel is in m, To convert to ns, use t = 2d/v .
    dt used is 8e-11 s
'''

yTrue2D_vel_td = []
for i in range(yTrue2D_vel_dd.shape[1]):
    each_vel=yTrue2D_vel_dd[:,i]
    ori_tt = 2*np.cumsum(0.05/each_vel[0:yTrue2D_vel_dd.shape[0]-1])
    ori_tt = np.insert(ori_tt,0,0)
    dest_tt = np.arange(0,1680*8e-11,8e-11)
    vel_td = interp1d(ori_tt,each_vel,kind='linear',bounds_error=False,fill_value=np.nan)(dest_tt)
    yTrue2D_vel_td.append(vel_td)
yTrue2D_vel_td=np.array(yTrue2D_vel_td).T



# ep (dielectric permittivity)
ep = 299792500**2 / yTrue2D_vel_dd **2


#save
#we only cut the original velocity model to 1280

np.save('Synthetic/Data/2D/yTrue2D_vel_td.npy',yTrue2D_vel_td[:1280,:])
sio.savemat('Synthetic/Data/2D/yTrue2D_ep.mat',{'ep':ep})


