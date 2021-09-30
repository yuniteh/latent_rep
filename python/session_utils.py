from numpy.core.defchararray import lower
import tensorflow as tf
import tensorflow.keras
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

class Session():
    def __init__(self,**settings):
        self.train_grp = settings.get('train_grp',2)
        self.dt = settings.get('dt',0)
        self.feat_type = settings.get('feat_type','feat')
        self.load = settings.get('load',True)
        self.noise = settings.get('noise',True)
        self.start_cv = settings.get('start_cv',1)
        self.max_cv = settings.get('max_cv',5)
        self.sparsity = settings.get('sparsity',True)
        self.batch_size = settings.get('batch_size',32)
        self.latent_dim = settings.get('latent_dim',10)
        self.epochs = settings.get('epochs',100)
        self.lr = settings.get('lr',0.001)
        self.train_scale = settings.get('train_scale',5)
        self.n_train = settings.get('n_train','gauss')
        self.n_test = settings.get('n_test','gauss')
        self.mod = settings.get('mod','all')
        self.gens = settings.get('gens',50)

    def create_foldername(self):
        # Set folder
        if self.dt == 0:
            today = date.today()
            self.dt = today.strftime("%m%d")
        foldername = 'models' + '_' + str(self.train_grp) + '_' + self.dt
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        return foldername

    def create_filename(self,foldername,cv,sub_type,sub):
        filename = foldername + '/' + sub_type + str(sub) + '_' + self.feat_type + '_dim_' + str(self.latent_dim) + '_ep_' + str(self.epochs) + '_bat_' + str(self.batch_size) + '_' + self.n_train + '_' + str(self.train_scale) + '_lr_' + str(int(self.lr*10000)) 
        if self.dt == 'cv':
            filename += '_cv_' + str(cv)
        if self.sparsity:
            filename += '_sparse'
        
        return filename

    