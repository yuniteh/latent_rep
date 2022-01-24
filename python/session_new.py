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

class Session():
    def __init__(self,**settings):
        self.sub_type = settings.get('sub_type','AB')
        self.train_grp = settings.get('train_grp',2)
        self.dt = settings.get('dt',0)
        self.mod_dt = settings.get('mod_dt','')
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
        self.train_load = settings.get('train_load',True)
        self.test_dt = settings.get('test_dt','')

    def create_foldername(self,ftype=''):
        if self.dt == 0:
            today = date.today()
            self.dt = today.strftime("%m%d")
        # Set folder
        if ftype == 'train':
            foldername = 'noisedata_' + self.dt + '_' + self.mod_dt + '_' + self.feat_type
        elif ftype == 'test':
            foldername = 'testdata_' + self.dt + '_' + self.mod_dt + '_' + self.test_dt + '_' + self.feat_type
        elif ftype =='results':
            foldername = 'results_' + str(self.train_grp) + '_' + self.dt + '_' + self.mod_dt + '_' + self.test_dt
        else:
            foldername = 'models_' + str(self.train_grp) + '_' + self.dt
            if self.mod_dt != 0:
                foldername += '_' + self.mod_dt
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        return foldername

    def create_filename(self,foldername,cv=0,sub=0,ftype='',test_scale=0):
        # train noise file
        if ftype == 'train':
            filename = foldername + '/' + self.sub_type + str(sub) + '_grp_' + str(self.train_grp) + '_' + str(self.n_train) + '_' + str(self.train_scale)
        elif ftype == 'test':
            filename = foldername + '/' + self.sub_type + str(sub) + '_grp_' + str(self.train_grp) + '_' + str(self.n_test) + '_' + str(test_scale)
        elif ftype =='results':
            filename = foldername + '/' + self.sub_type + '_' + self.feat_type + '_dim_' + str(self.latent_dim) + '_ep_' + str(self.epochs) + '_bat_' + str(self.batch_size) + '_' + self.n_train + '_' + str(self.train_scale) + '_lr_' + str(int(self.lr*10000)) 
        # model file
        else:
            filename = foldername + '/' + self.sub_type + str(sub) + '_' + self.feat_type + '_dim_' + str(self.latent_dim) + '_ep_' + str(self.epochs) + '_bat_' + str(self.batch_size) + '_' + self.n_train + '_' + str(self.train_scale) + '_lr_' + str(int(self.lr*10000)) 
        if self.dt == 'cv':
            filename += '_cv_' + str(cv)
        
        return filename

