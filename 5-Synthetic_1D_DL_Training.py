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


#%% Load Training data

X_train = np.load('Synthetic/Data/1D/ForDL/Synthetic_Xtrain_1d.npy')
X_valid = np.load('Synthetic/Data/1D/ForDL/Synthetic_yvalid_1d.npy')

y_train = np.load('Synthetic/Data/1D/ForDL/Synthetic_ytrain_1d.npy')
y_valid = np.load('Synthetic/Data/1D/ForDL/Synthetic_yvalid_1d.npy')

#%% Prep data for GPRNet Training, 
'''i.e. making sure X_train and y_train has dimensions of (x,y,1)'''


X_train = np.expand_dims(X_train, axis=2)
X_valid = np.expand_dims(X_valid, axis=2)
y_train = np.expand_dims(y_train, axis=2)
y_valid = np.expand_dims(y_valid, axis=2)



#%% R-squared metric function

def R2_score(v_true, v_pred):
    ssres = K.sum(K.square(v_true - v_pred))
    sstot = K.sum(K.square(v_true - K.mean(v_true)))
    return 1 - ssres / sstot

#%% Set Training Parameters
    

base_path = 'Synthetic/Weights/'
model_name = 'weight_GPRNet_n16k20'

cnn_model = GPRNet(im_width=1, im_height=1280, neurons=16, kern_sz = 20,enable_dropout=False) 

save_model_name = base_path + model_name +'.h5'
lr_val = 0.0001
c = optimizers.adam(lr = lr_val)
cnn_model.compile(optimizer=c, loss='mse', metrics=[R2_score])

early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_R2_score',
                                    save_best_only=True, verbose=1, mode='max')
reduce_lr = ReduceLROnPlateau(factor=0.5, monitor='val_loss',
                              patience=15, min_lr=0.000001, verbose=1)

csv_logger = CSVLogger(base_path +model_name +".csv", append=True)

#%% Train!

epochs = 10000
batch_size = 40


history = cnn_model.fit(X_train, y_train,
          validation_data=[X_valid,y_valid],
                  epochs=epochs,
                   batch_size=batch_size,
                   callbacks=[model_checkpoint,reduce_lr,csv_logger, early_stopping],
                   verbose=1)






