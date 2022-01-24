from numpy.core.defchararray import lower
import tensorflow as tf
from loop import create_foldername
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from lda import train_lda, predict, eval_lda, eval_lda_ch
from sklearn.utils import shuffle
import sVAE_utils as dl
import process_data as prd
import copy as cp
from datetime import date
import time
from numpy.linalg import eig, inv
from types import SimpleNamespace 
import keras.backend as K

class Sess():
    def __init__(self,**settings):
        self.sub_type = settings.get('sub_type','AB')
        self.train_grp = settings.get('train_grp',2)
        self.train_scale = settings.get('train_scale',5)
        self.train = settings.get('train','fullallmix4')
        self.cv_type = settings.get('cv_type','manual')
        self.feat_type = settings.get('feat_type','feat')
        self.scaler_load = settings.get('scaler_load',True)

    def prep_train_data(self, raw, params, sub):
        x_train, _, x_valid, p_train, _, p_valid = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.cv_type,load=True,train_grp=self.train_grp)

        emg_scale = np.ones((np.size(x_train,1),1))
        for i in range(np.size(x_train,1)):
            emg_scale[i] = 5/np.max(np.abs(x_train[:,i,:]))
        x_train = x_train*emg_scale
        x_valid = x_valid*emg_scale

        x_train_noise, _, y_train_clean = prd.add_noise(x_train, p_train, sub, self.train, self.train_scale)
        x_valid_noise, _, y_valid_clean = prd.add_noise(x_valid, p_valid, sub, self.train, self.train_scale)

        # shuffle data to make even batches
        x_train_noise, y_train_clean = shuffle(x_train_noise, y_train_clean, random_state = 0)

        # Extract features
        scaler = MinMaxScaler(feature_range=(0,1))
        x_train_noise_cnn, scaler = prd.extract_scale(x_train_noise,scaler,self.scaler_load,ft=self.feat_type,emg_scale=emg_scale) 
        x_valid_noise_cnn, _ = prd.extract_scale(x_valid_noise,scaler,ft=self.feat_type,emg_scale=emg_scale)
        x_train_noise_cnn = x_train_noise_cnn.astype('float32')
        x_valid_noise_cnn = x_valid_noise_cnn.astype('float32')

        # reshape data for nonconvolutional network
        x_train_noise_mlp = x_train_noise_cnn.reshape(x_train_noise_cnn.shape[0],-1)
        x_valid_noise_mlp = x_valid_noise_cnn.reshape(x_valid_noise_cnn.shape[0],-1)

        # create batches
        trainmlp_ds = tf.data.Dataset.from_tensor_slices((x_train_noise_mlp, y_train_clean)).batch(128)
        testmlp_ds = tf.data.Dataset.from_tensor_slices((x_valid_noise_mlp, y_valid_clean)).batch(128)
        traincnn_ds = tf.data.Dataset.from_tensor_slices((x_train_noise_cnn, y_train_clean)).batch(128)
        testcnn_ds = tf.data.Dataset.from_tensor_slices((x_valid_noise_cnn, y_valid_clean)).batch(128)

        return trainmlp_ds, testmlp_ds, traincnn_ds, testcnn_ds, y_train_clean, y_valid_clean

