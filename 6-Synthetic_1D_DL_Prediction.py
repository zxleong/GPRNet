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
import tensorflow as tf


from keras import backend as K
# from skimage.metrics import structural_similarity as ssim
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter
# from mpl_toolkits.mplot3d import Axes3D  

from tqdm import tqdm
import scipy.stats
from scipy.signal import tukey

from DLcodes.GPRNet import GPRNet


#%% R-squared metric function

def R2_score(v_true, v_pred):
    ssres = K.sum(K.square(v_true - v_pred))
    sstot = K.sum(K.square(v_true - K.mean(v_true)))
    return 1 - ssres / sstot



#%% Load data

X_test = np.load('Synthetic/Data/1D/ForDL/Synthetic_Xtest_1d.npy')
y_true = np.load('Synthetic/Data/1D/ForDL/Synthetic_yTrue_1d.npy')
X_test = np.expand_dims(X_test, axis=2)


#%% Prediction
weights_path = 'Synthetic/Weights/weight_GPRNet_n16k20.h5'

cnn_model = GPRNet(im_width=1, im_height=1280, neurons=16, kern_sz = 20,enable_dropout=False) #use this
cnn_model.load_weights(weights_path)

ypred = np.squeeze(cnn_model.predict(X_test),axis=2)

#%% SAve Prediction

np.save('Synthetic/Data/1D/ForDL/Synthetic_ypred_1D.npy',ypred)



