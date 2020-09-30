'''@author: Zi Xian Leong (zxleong@psu.edu) '''

#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

from sklearn import preprocessing as pp
from sklearn.preprocessing import StandardScaler
import scipy.ndimage.filters as fil
from scipy.interpolate import interp1d
from utils.random_vel_generator import vel_generator
from tqdm import tqdm
from utils.gpr_generate_model import generate_interface_gpr
from utils.smooth import smooth
from scipy.signal import tukey

'''

Proccess the GPR traces that was created from FD_GPR_sim.m

'''

# %% Process gathers

fdrawgather_stack = sio.loadmat('Synthetic/Data/1D/fdrawgather.mat')['gather_stack']
vel_stack = np.load('Synthetic/Data/1D/veltd.npy').T

#Detect if there are any NaN values in the GPR traces. If there is a faulty one,
#delete that pair of data set.

# check nan indices
nan_index = []
all_check = np.isnan(fdrawgather_stack)
for i in range(10000):
    if sum(all_check[i,:]) > 0:
        nan_index.append(i)

#row number 2280 has nan values, let's delete them
fdrawgather_stack_cut = fdrawgather_stack.copy()
fdrawgather_stack_cut=np.delete(fdrawgather_stack_cut,2280,0) #delete 2280
vel_stack_cut = np.delete(vel_stack,2280,0)


#remove first arrivals
def taper(in_array):
    window = tukey(1280,0.4)
    window[600:len(window)]=1
    window = window**4
    out = in_array * window
    return out

tapered_stack = np.apply_along_axis(taper,1,fdrawgather_stack_cut)



# %% save

# np.save('Synthetic/Data/1D/xTrain_gathers.npy',tapered_stack)
# np.save('Synthetic/Data/1D/yTrain_vels.npy',vel_stack_cut)

