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
from mpl_toolkits.mplot3d import Axes3D  

from tqdm import tqdm
import scipy.stats
from scipy.signal import tukey



def GPRNet(im_width=1, im_height=1200, neurons=8, kern_sz = 30, enable_dropout=False,dp_coeff=0.2):
    ''' Zero dropout is similar to the original version (without dropout)'''

    
    img_width = im_width
    img_height = im_height
    input_img = Input((img_height,img_width))
    
    if enable_dropout==True:
        dropout_params = np.ones(16)*dp_coeff
    else:
        dropout_params = np.zeros(16)

    #encoder
    conv1 = Conv1D(neurons, kernel_size=kern_sz, strides=1, activation='relu', padding='same', name='Conv1')(input_img)
    pool1 = MaxPooling1D(2, name='Pool1') (conv1) #640
    pool1 = Dropout(dropout_params[0], name='Dropout1')(pool1,training=True) #
    
    conv2 = Conv1D(neurons*2, kernel_size=kern_sz, strides=1, activation='relu', padding='same', name='Conv2')(pool1)
    pool2 = MaxPooling1D(2, name='Pool2') (conv2) #320
    pool2 = Dropout(dropout_params[1], name='Dropout2')(pool2,training=True) #
    
    conv3 = Conv1D(neurons*4, kernel_size=kern_sz, strides=1, activation='relu', padding='same', name='Conv3')(pool2)
    pool3 = MaxPooling1D(2, name='Pool3') (conv3) #160
    pool3 = Dropout(dropout_params[2], name='Dropout3')(pool3,training=True) #-
    
    conv4 = Conv1D(neurons*8, kernel_size=kern_sz, strides=1, activation='relu', padding='same', name='Conv4')(pool3)
    pool4 = MaxPooling1D(2, name='Pool4') (conv4) 
    pool4 = Dropout(dropout_params[3], name='Dropout4')(pool4,training=True)

    py1 = Conv1D(neurons*16, kernel_size=kern_sz, strides=1, activation='relu', padding='same', name='Py1')(pool4)
    py1 = Dropout(dropout_params[4], name='Dropout5')(py1,training=True)

    py2 = Conv1D(neurons*16, kernel_size=kern_sz, strides=1, dilation_rate = 6, activation='relu', padding='same', name='Py2')(pool4)
    py2 = Dropout(dropout_params[5], name='Dropout6')(py2,training=True)

    py3 = Conv1D(neurons*16, kernel_size=kern_sz, strides=1, dilation_rate = 12, activation='relu', padding='same', name='Py3')(pool4)
    py3 = Dropout(dropout_params[6], name='Dropout7')(py3,training=True)

    py4 = Conv1D(neurons*16, kernel_size=kern_sz, strides=1, dilation_rate = 18, activation='relu', padding='same', name='Py4')(pool4)
    py4 = Dropout(dropout_params[7], name='Dropout8')(py4,training=True)

    merge1 = concatenate([py1,py2,py3,py4,pool4],name='Merge1')
    merge1 = Dropout(dropout_params[8], name='Dropout9')(merge1,training=True)
    
    mgconv = Conv1D(neurons*16, kernel_size =3, strides=1, activation='relu', padding='same',name='MgConv')(merge1)
    upmgconv = UpSampling1D(4, name='UpMgConv') (mgconv) #UpMgConv
    upmgconv = Dropout(dropout_params[9],name='Dropout10')(upmgconv,training=True)

    #Decoder
    
    deconv1 = Conv1D(neurons*16, kernel_size=kern_sz, strides=1, activation='relu', padding='same',name='Deconv1')(pool4)
    up1 = UpSampling1D(2, name='Up1') (deconv1) #100->200
    up1 = Dropout(dropout_params[10], name='Dropout11')(up1,training=True)
    
    deconv2 = Conv1D(neurons*8, kernel_size=kern_sz, strides=1, activation='relu', padding='same', name='Deconv2')(up1)
    up2 = UpSampling1D(2, name='Up2') (deconv2) #200 -> 400
    up2 = Dropout(dropout_params[11], name='Dropout12')(up2,training=True)
    
    merge2 = concatenate([upmgconv, up2], name='Merge2') 
    merge2 = Dropout(dropout_params[12], name='Dropout13')(merge2,training=True)

    deconv3 = Conv1D(neurons*4, kernel_size=kern_sz, strides=1, activation='relu', padding='same', name='Deconv3')(merge2)
    up3 = UpSampling1D(2, name='Up3') (deconv3) #400 -> 800    
    up3 = Dropout(dropout_params[13], name='Dropout14')(up3,training=True)
    
    deconv4 = Conv1D(neurons*2, kernel_size=kern_sz, strides=1, activation='relu', padding='same', name='Deconv4')(up3)
    up4 = UpSampling1D(2, name='Up4') (deconv4) #800 -> 1600
    up4 = Dropout(dropout_params[14], name='Dropout15')(up4,training=True)
    
    deconv5 = Conv1D(1, kernel_size=kern_sz, strides=1, activation='relu',padding='same', name='Deconv5')(up4)
    deconv5 = GaussianDropout(dropout_params[15], name='Dropout16')(deconv5,training=True)

    model = Model(inputs=[input_img], outputs=[deconv5])
    
    return model
    