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
        self.sub = settings.get('sub',1)

        self.train_grp = settings.get('train_grp',2)
        self.train_scale = settings.get('train_scale',5)
        self.train = settings.get('train','fullallmix4')
        self.cv_type = settings.get('cv_type','manual')
        self.feat_type = settings.get('feat_type','feat')
        self.scaler_load = settings.get('scaler_load',True)
        self.epochs = settings.get('epochs',30)

        self.test_grp = settings.get('test_grp',4)
        self.test = settings.get('test','partposrealmixeven14')

        self.emg_scale = settings.get('emg_scale',1)
        self.scaler = settings.get('scaler',MinMaxScaler(feature_range=(0,1)))
    
    def update(self,**settings):
        for k in settings:
            setattr(self, k, settings[k])