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


from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
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
import time
from DLcodes.GPRNet import GPRNet


#%% Load Training data



X_train = np.load('Field/Data/ForDL/field_X_train.npy')
y_train = np.load('Field/Data/ForDL/field_y_train.npy')
X_valid = np.load('Field/Data/ForDL/field_X_valid.npy')
y_valid = np.load('Field/Data/ForDL/field_y_valid.npy')



#%% R-squared metric function

def R2_score(v_true, v_pred):
    ssres = K.sum(K.square(v_true - v_pred))
    sstot = K.sum(K.square(v_true - K.mean(v_true)))
    return 1 - ssres / sstot

#%% Set Training Parameters

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

base_path = 'Field/Weights/'
model_name = 'weight_GPRNet_n32k10'
cnn_model = GPRNet(im_width=1, im_height=400, neurons=32, kern_sz = 10, )
save_model_name = base_path + model_name +'.h5'
lr_val = 0.0001

c = optimizers.adam(lr = lr_val)
cnn_model.compile(optimizer=c, loss='mse', metrics=[R2_score])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_R2_score',
                                    save_best_only=True, verbose=1, mode='max')
reduce_lr = ReduceLROnPlateau(factor=0.5, monitor='val_loss',
                              patience=15, min_lr=0.000001, verbose=1)

csv_logger = CSVLogger("Weights/" + model_name +".csv", append=True)



epochs = 10000
batch_size = 1000
print('''Training!!!! with batchsize = {}'''.format(batch_size))

#%% Train!

history = cnn_model.fit(X_train, y_train,
          validation_data=[X_valid,y_valid],
                  epochs=epochs,
                   batch_size=batch_size,
                   callbacks=[model_checkpoint,reduce_lr,csv_logger, early_stopping, time_callback],
                   verbose=1)


EpochTimes = time_callback.times
np.save('Field/Weights/EpochTimes_n32k10.npy',np.array(EpochTimes))








