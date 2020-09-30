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
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add, UpSampling1D, Flatten, Dense, Reshape, SpatialDropout1D, GaussianDropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv1D, ZeroPadding1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.callbacks import CSVLogger
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import scipy.ndimage.filters as fil
from keras import backend as K
# from skimage.metrics import structural_similarity as ssim
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter
# from mpl_toolkits.mplot3d import Axes3D  

from tqdm import tqdm
import scipy.stats
from scipy.signal import tukey

from DLcodes.GPRNet import GPRNet


#%% Split Data set for training, testing, validation

raw_xTrain = np.load('Synthetic/Data/1D/xTrain_gathers.npy')
raw_yTrain = np.load('Synthetic/Data/1D/yTrain_vels.npy')


#%% Apply MaxNormalization  on GPR trace

raw_xTrain = pp.normalize(raw_xTrain,norm='max',axis=1)

#%% Rescale Velocity so it is between 0 and 1

raw_yTrain = raw_yTrain / 1e9 


#%% use random_state=155 for reproducibility
X_train_main, X_test, y_train_main, y_True = train_test_split(raw_xTrain, raw_yTrain, test_size=0.01, random_state=155)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_main, y_train_main, test_size=0.01, random_state=155)


#%% Save the training, testing, validation dataset for easier accessibilitiy

np.save('Synthetic/Data/1D/ForDL/Synthetic_Xtrain_1d.npy',X_train)
np.save('Synthetic/Data/1D/ForDL/Synthetic_ytrain_1d.npy',y_train)
np.save('Synthetic/Data/1D/ForDL/Synthetic_yvalid_1d.npy',y_valid)
np.save('Synthetic/Data/1D/ForDL/Synthetic_xvalid_1d.npy',X_valid)
np.save('Synthetic/Data/1D/ForDL/Synthetic_Xtest_1d.npy',X_test)
np.save('Synthetic/Data/1D/ForDL/Synthetic_yTrue_1d.npy',y_True)

