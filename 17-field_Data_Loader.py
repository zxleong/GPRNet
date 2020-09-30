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

# from skimage.metrics import structural_similarity as ssim
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter
# from mpl_toolkits.mplot3d import Axes3D  

from tqdm import tqdm
import scipy.stats
from scipy.signal import tukey

from DLcodes.GPRNet import GPRNet

'''Field Data Loader'''


#%% Load Training data

X_train_raw = np.load('Field/Data/ForDL/GPRData.npy')
y_train_raw = np.load('Field/Data/ForDL/Vel.npy')

#%% Prep data for GPRNet Training, 

X_train_raw = pp.normalize(X_train_raw,norm='max',axis=1)
X_train_raw = np.expand_dims(X_train_raw, axis=2)

y_train_raw = y_train_raw/1e9
y_train_raw = np.expand_dims(y_train_raw, axis=2)

#%%
print("""# Spliting training and testing """)

X_train_main, X_test, y_train_main, y_True = train_test_split(X_train_raw, y_train_raw, test_size=0.01, random_state=155)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_main, y_train_main, test_size=0.01, random_state=155)

#%% Save the training, testing, validation dataset for easier accessibilitiy

np.save('Field/Data/ForDL/field_X_train.npy',X_train)
np.save('Field/Data/ForDL/field_y_train.npy',y_train)
np.save('Field/Data/ForDL/field_y_valid.npy',y_valid)
np.save('Field/Data/ForDL/field_X_valid.npy',X_valid)
np.save('Field/Data/ForDL/field_X_test.npy',X_test)
np.save('Field/Data/ForDL/field_y_true.npy',y_True)

