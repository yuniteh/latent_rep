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
from numpy.linalg import eig, inv
from types import SimpleNamespace 
import keras.backend as K

class Session():
    def __init__(self,**settings):
        self.sub_type = settings.get('sub_type','AB')
        self.train_grp = settings.get('train_grp',2)
        self.dt = settings.get('dt',0)
        self.mod_dt = settings.get('mod_dt',0)
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
        self.gens = settings.get('gens',50)
        self.train_load = settings.get('train_load',True)

    def create_foldername(self):
        # Set folder
        if self.dt == 0:
            today = date.today()
            self.dt = today.strftime("%m%d")
        foldername = 'models' + '_' + str(self.train_grp) + '_' + self.dt
        if self.mod_dt != 0:
            foldername += '_' + self.mod_dt
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        return foldername

    def create_filename(self,foldername,cv=0,sub=0,results=False):
        if results:
            filename = foldername + '/' + self.sub_type + '_' + self.feat_type + '_dim_' + str(self.latent_dim) + '_ep_' + str(self.epochs) + '_' + self.n_train + '_' + str(self.train_scale) + '_' + self.n_test
        else:
            filename = foldername + '/' + self.sub_type + str(sub) + '_' + self.feat_type + '_dim_' + str(self.latent_dim) + '_ep_' + str(self.epochs) + '_bat_' + str(self.batch_size) + '_' + self.n_train + '_' + str(self.train_scale) + '_lr_' + str(int(self.lr*10000)) 
            if self.dt == 'cv':
                filename += '_cv_' + str(cv)

        if self.sparsity:
            filename += '_sparse'
        
        return filename

    def loop_cv(self, raw, params, sub=1, mod='all'):
        np.set_printoptions(precision=3,suppress=True)
        i_tot = 14
        filename = 0
        if self.dt == 'manual':
            self.start_cv = 1
            self.max_cv = 2

        # Set folder
        foldername = self.create_foldername()

        # initialize final accuracy arrays and potential outputs
        last_acc = np.full([self.max_cv-1,6], np.nan)
        last_val = np.full([self.max_cv-1,6], np.nan)
        gen_clf = np.nan

        # index training group and subject
        ind = (params[:,0] == sub) & (params[:,3] == self.train_grp)

        # Check if training data exists
        if np.sum(ind):
            # initialize x out matrix
            dec_cv = np.full([self.max_cv-1,self.gens*np.max(params[ind,4]),raw.shape[1],4,1], np.nan)
            
            if mod != 'none':
                if self.dt == 'cv':
                    x_full, x_valid, _, p_full, p_valid, _ = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt,train_grp=self.train_grp)
                else:
                    x_train, x_test, x_valid, p_train, p_test, p_valid = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt,load=False,train_grp=self.train_grp)

            # loop through cross validation
            for cv in range(self.start_cv,self.max_cv):
                filename = self.create_filename(foldername, cv, sub)

                print('Running sub ' + str(sub) + ', model ' + str(self.train_grp) + ', latent dim ' + str(self.latent_dim) + ', cv ' + str(cv))

                ## TRAIN ##
                # Load saved data
                if self.load:
                    self.load = True
                    with open(filename + '.p', 'rb') as f:
                        scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, \
                            ecnn_w, ecnn_enc_w, ecnn_clf_w, w_svae, c_svae, w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w_ecnn, c_ecnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise = pickle.load(f)   

                    with open(filename + '_hist.p', 'rb') as f:
                        svae_hist, sae_hist, cnn_hist, vcnn_hist, ecnn_hist = pickle.load(f)

                    try:
                        with open(filename + '_aug.p', 'rb') as f:
                            w_rec, c_rec, w_rec_al, c_rec_al, w_gen, c_gen, w_gen_al, c_gen_al = pickle.load(f)
                    except:
                        print('no augmented data file')

                    try:
                        with open(filename + '_red.p', 'rb') as f:
                            v_svae, v_sae, v_cnn, v_vcnn, v_ecnn, v, v_noise = pickle.load(f)
                    except:
                        print('no transformation matrix file')
                else:
                    scaler = MinMaxScaler(feature_range=(0,1))
                    load = False
                    
                # prepare training data if training models    
                if mod != 'none':
                    noisefolder = 'noisedata_' + self.dt
                    noisefile = noisefolder + '/' + self.sub_type + str(sub) + '_grp_' + str(self.train_grp) + '_' + str(self.n_train) + '_' + str(self.train_scale)
                    if not os.path.isdir(noisefolder):
                        os.mkdir(noisefolder) 
                    if self.dt == 'cv':
                        noisefile += '_cv_' + str(cv)
                    
                    if os.path.isfile(noisefile + '.p') and self.train_load:
                        print('loading data')
                        with open(noisefile + '.p','rb') as f:
                            scaler, x_train_noise_vae, x_train_clean_vae, x_valid_noise_vae, x_valid_clean_vae, y_train_clean, y_valid_clean, x_train_lda, y_train_lda, x_train_noise_lda, y_train_noise_lda = pickle.load(f)                       
                    else:
                        # Add noise to training data                        
                        if self.dt == 'cv':
                            x_valid, p_valid = x_full[p_full[:,6] == cv,...], p_full[p_full[:,6] == cv,...]
                            x_train, p_train = x_full[p_full[:,6] != cv,...], p_full[p_full[:,6] != cv,...]
                
                        x_train_noise, x_train_clean, y_train_clean = prd.add_noise(x_train, p_train, sub, self.n_train, self.train_scale)
                        x_valid_noise, x_valid_clean, y_valid_clean = prd.add_noise(x_valid, p_valid, sub, self.n_train, self.train_scale)
                        # if not adding noise, copy clean training data
                        if not self.noise:
                            x_train_noise = cp.deepcopy(x_train_clean)
                            x_valid_noise = cp.deepcopy(x_valid_clean)

                        # shuffle data to make even batches
                        x_train_noise, x_train_clean, y_train_clean = shuffle(x_train_noise, x_train_clean, y_train_clean, random_state = 0)

                        # Training data for LDA/QDA
                        y_train = p_train[:,4]
                        x_train_lda = prd.extract_feats(x_train)
                        y_train_lda = y_train[...,np.newaxis] - 1
                        x_train_noise_lda = prd.extract_feats(x_train_noise)
                        y_train_noise_lda = np.argmax(y_train_clean, axis=1)[...,np.newaxis]

                        # Shape data based on feature type
                        if self.feat_type == 'feat':
                            # extract and scale features from training and validation data
                            x_train_noise_vae, scaler = prd.extract_scale(x_train_noise,scaler,self.load) 
                            x_train_clean_vae, _ = prd.extract_scale(x_train_clean,scaler)
                            x_valid_noise_vae, _ = prd.extract_scale(x_valid_noise,scaler)
                            x_valid_clean_vae, _ = prd.extract_scale(x_valid_clean,scaler)

                        elif self.feat_type == 'raw': # not finalized
                            x_train_noise_vae = 0.5+cp.deepcopy(x_train_noise[:,:,::4,:])/10
                            x_train_clean_vae = 0.5+cp.deepcopy(x_train_clean[:,:,::4,:])/10

                            x_valid_noise_vae = 0.5+cp.deepcopy(x_valid_noise[:,:,::4,:])/10
                            x_valid_clean_vae = 0.5+cp.deepcopy(x_valid_clean[:,:,::4,:])/10

                        with open(noisefile + '.p', 'wb') as f:
                            pickle.dump([scaler, x_train_noise_vae, x_train_clean_vae, x_valid_noise_vae, x_valid_clean_vae, y_train_clean, y_valid_clean, x_train_lda, y_train_lda, x_train_noise_lda, y_train_noise_lda],f)

                    # reshape data for nonconvolutional network
                    x_train_noise_sae = x_train_noise_vae.reshape(x_train_noise_vae.shape[0],-1)
                    x_valid_noise_sae = x_valid_noise_vae.reshape(x_valid_noise_vae.shape[0],-1)

                    # TEMP - CNN extended
                    x_train_noise_ext = np.concatenate((x_train_noise_vae,x_train_noise_vae[:,:2,...]),axis=1)
                    x_valid_noise_ext = np.concatenate((x_valid_noise_vae,x_valid_noise_vae[:,:2,...]),axis=1)

                    # Build models
                    K.clear_session()
                    svae, svae_enc, svae_dec, svae_clf = dl.build_M2(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    sae, sae_enc, sae_clf = dl.build_sae(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    cnn, cnn_enc, cnn_clf = dl.build_cnn(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    vcnn, vcnn_enc, vcnn_clf = dl.build_vcnn(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    ecnn, ecnn_enc, ecnn_dec, ecnn_clf = dl.build_M2S2(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    
                # Train SVAE
                if mod == 'all' or any("svae" in s for s in mod):
                    # get number of batches
                    n_batches = len(x_train_noise_vae) // self.batch_size
                    # initialize history array
                    temp_epochs = 50
                    svae_hist = np.zeros((temp_epochs,14))
                    # loop through epochs
                    for ep in range(temp_epochs):
                        # set loss weight vector, not finalized
                        # if ep < self.epochs/2: 
                        #     weight = np.array([[0,2**((ep-(self.epochs/2))/2)] for _ in range(self.batch_size)])
                        # else:
                        #     weight = np.array([[1,.5] for _ in range(self.batch_size)])
                        
                        if ep < 15: #temp_epochs//3: 
                            # weight = np.array([[0,(2**(ep-(temp_epochs//3)))/10] for _ in range(self.batch_size)])
                            weight = np.array([[0,(2**(ep-(20)))/100] for _ in range(self.batch_size)])
                        # elif ep < 40:
                        #     weight = np.array([[1,.01] for _ in range(self.batch_size)])
                        # elif ep < 60:
                        #     weight = np.array([[0,(2**(ep-(60)))/100] for _ in range(self.batch_size)])
                        # elif ep < 80:
                        #     weight = np.array([[1,.01] for _ in range(self.batch_size)])
                        # elif ep < 100:
                        #     weight = np.array([[0,(2**(ep-(100)))/100] for _ in range(self.batch_size)])
                        else:
                            weight = np.array([[1,.01] for _ in range(self.batch_size)])
                        
                        # get batches for inputs
                        x_train_noise_ep = dl.get_batches(x_train_noise_vae, self.batch_size)
                        x_train_clean_ep = dl.get_batches(x_train_clean_vae, self.batch_size)
                        y_train_ep = dl.get_batches(y_train_clean, self.batch_size)
                        # loop through batches
                        for ii in range(n_batches):
                            x_train_noise_bat = next(x_train_noise_ep)
                            x_train_clean_bat = next(x_train_clean_ep)
                            y_train_bat = next(y_train_ep)

                            # train to reconstruct clean data
                            temp_out = svae.train_on_batch([x_train_noise_bat,y_train_bat,weight],[x_train_noise_bat,y_train_bat,x_train_clean_bat[:,0,0]])
                            ## train to reconstruct noisy data
                            # temp_out = svae.train_on_batch([x_train_noise_bat,weight],[x_train_noise_bat,y_train_bat,x_train_clean_bat[:,0,0]])
                        
                        ## currently unused - divide previous losses to create subsequent loss weights
                        # if temp_out[1]/temp_out[2] > 1:
                        #     rat = temp_out[1]/temp_out[2]
                        # else:
                        #     rat = 1
                        # weight = np.array([[rat, (temp_out[2])/temp_out[3]] for _ in range(self.batch_size)])
                        # weight = np.array([1 for _ in range(len(x_valid_noise_vae))])

                        # dummy loss weight for testing
                        test_weight = np.array([[1,1] for _ in range(len(x_valid_noise_vae))])

                        # save training metrics in history array
                        svae_hist[ep,:7] = temp_out

                        ## validation testing to reconstruct clean features
                        svae_hist[ep,7:] = svae.test_on_batch([x_valid_noise_vae,y_valid_clean,test_weight],[x_valid_noise_vae,y_valid_clean,x_valid_clean_vae[:,0,0]])

                        # validation testing to reconstruct noisy features
                        # svae_hist[ep,7:] = svae.test_on_batch([x_valid_noise_vae,test_weight],[x_valid_noise_vae,y_valid_clean,x_valid_clean_vae[:,0,0]])

                        # print training losses as we train
                        if ep == 0:
                            print(svae.metrics_names)
                        print(svae_hist[ep,7:])

                    # get weights
                    svae_w = svae.get_weights()
                    svae_enc_w = svae_enc.get_weights()
                    svae_dec_w = svae_dec.get_weights()
                    svae_clf_w = svae_clf.get_weights()

                # Fit NNs and get weights
                if mod == 'all' or any("var_c" in s for s in mod):
                    # vcnn_hist = vcnn.fit(x_train_noise_vae, y_train_clean,epochs=30,validation_data = [x_valid_noise_vae, y_valid_clean],batch_size=self.batch_size)                
                    # vcnn_w = vcnn.get_weights()
                    # vcnn_enc_w = vcnn_enc.get_weights()
                    # vcnn_clf_w = vcnn_clf.get_weights()
                    # vcnn_hist = vcnn_hist.history
                    temp_epochs = 50

                    # get number of batches
                    n_batches = len(x_train_noise_vae) // self.batch_size
                    # initialize history array
                    vcnn_hist = np.zeros((temp_epochs,4))
                    # loop through epochs
                    for ep in range(temp_epochs):
                        # set loss weight vector, not finalized
                        if ep < 15: #temp_epochs//3: 
                            # weight = np.array([[0,(2**(ep-(temp_epochs//3)))/10] for _ in range(self.batch_size)])
                            weight = np.array([[0,(2**(ep-(15)))/100] for _ in range(self.batch_size)])
                        # elif ep < 40:
                        #     weight = np.array([[1,.01] for _ in range(self.batch_size)])
                        # elif ep < 60:
                        #     weight = np.array([[0,(2**(ep-(60)))/100] for _ in range(self.batch_size)])
                        # elif ep < 80:
                        #     weight = np.array([[1,.01] for _ in range(self.batch_size)])
                        # elif ep < 100:
                        #     weight = np.array([[0,(2**(ep-(100)))/100] for _ in range(self.batch_size)])
                        else:
                            weight = np.array([[1,.01] for _ in range(self.batch_size)])
                        
                        # print(weight[0,1])
                        
                        # get batches for inputs
                        x_train_noise_ep = dl.get_batches(x_train_noise_vae, self.batch_size)
                        x_train_clean_ep = dl.get_batches(x_train_clean_vae, self.batch_size)
                        y_train_ep = dl.get_batches(y_train_clean, self.batch_size)
                        # loop through batches
                        for ii in range(n_batches):
                            x_train_noise_bat = next(x_train_noise_ep)
                            x_train_clean_bat = next(x_train_clean_ep)
                            y_train_bat = next(y_train_ep)

                            # train
                            temp_out = vcnn.train_on_batch([x_train_noise_bat,weight],y_train_bat)
                        
                        ## currently unused - divide previous losses to create subsequent loss weights
                        # if temp_out[1]/temp_out[2] > 1:
                        #     rat = temp_out[1]/temp_out[2]
                        # else:
                        #     rat = 1
                        # weight = np.array([[rat, (temp_out[2])/temp_out[3]] for _ in range(self.batch_size)])
                        # weight = np.array([1 for _ in range(len(x_valid_noise_vae))])

                        # dummy loss weight for testing
                        test_weight = np.array([[1,1] for _ in range(len(x_valid_noise_vae))])

                        # save training metrics in history array
                        vcnn_hist[ep,:2] = temp_out

                        ## validation testing
                        vcnn_hist[ep,2:] = vcnn.test_on_batch([x_valid_noise_vae,test_weight],y_valid_clean)

                        # print training losses as we train
                        if ep == 0:
                            print(vcnn.metrics_names)
                        print(vcnn_hist[ep,:])

                    # get weights
                    vcnn_w = vcnn.get_weights()
                    vcnn_enc_w = vcnn_enc.get_weights()
                    vcnn_clf_w = vcnn_clf.get_weights()

                if mod == 'all' or any("sae" in s for s in mod):
                    sae_hist = sae.fit(x_train_noise_sae, y_train_clean,epochs=30,validation_data = [x_valid_noise_sae, y_valid_clean],batch_size=self.batch_size)
                    sae_w = sae.get_weights()
                    sae_enc_w = sae_enc.get_weights()
                    sae_clf_w = sae_clf.get_weights()
                    sae_hist = sae_hist.history

                if mod == 'all' or any("cnn" in s for s in mod):
                    cnn_hist = cnn.fit(x_train_noise_vae, y_train_clean,epochs=30,validation_data = [x_valid_noise_vae, y_valid_clean],batch_size=self.batch_size)
                    cnn_w = cnn.get_weights()
                    cnn_enc_w = cnn_enc.get_weights()
                    cnn_clf_w = cnn_clf.get_weights()
                    cnn_hist = cnn_hist.history
                
                if mod == 'all' or any("ext_c" in s for s in mod):
                    # get number of batches
                    n_batches = len(x_train_noise_vae) // self.batch_size
                    # initialize history array
                    temp_epochs = 50
                    ecnn_hist = np.zeros((temp_epochs,10))
                    # loop through epochs
                    for ep in range(temp_epochs):
                        # set loss weight vector, not finalized
                        # if ep < self.epochs/2: 
                        #     weight = np.array([[0,2**((ep-(self.epochs/2))/2)] for _ in range(self.batch_size)])
                        # else:
                        #     weight = np.array([[1,.5] for _ in range(self.batch_size)])
                        
                        if ep < 15: #temp_epochs//3: 
                            # weight = np.array([[0,(2**(ep-(temp_epochs//3)))/10] for _ in range(self.batch_size)])
                            weight = np.array([[0,(2**(ep-(20)))/100] for _ in range(self.batch_size)])
                        # elif ep < 40:
                        #     weight = np.array([[1,.01] for _ in range(self.batch_size)])
                        # elif ep < 60:
                        #     weight = np.array([[0,(2**(ep-(60)))/100] for _ in range(self.batch_size)])
                        # elif ep < 80:
                        #     weight = np.array([[1,.01] for _ in range(self.batch_size)])
                        # elif ep < 100:
                        #     weight = np.array([[0,(2**(ep-(100)))/100] for _ in range(self.batch_size)])
                        else:
                            weight = np.array([[1,.01] for _ in range(self.batch_size)])
                        
                        # get batches for inputs
                        x_train_noise_ep = dl.get_batches(x_train_noise_vae, self.batch_size)
                        x_train_clean_ep = dl.get_batches(x_train_clean_vae, self.batch_size)
                        y_train_ep = dl.get_batches(y_train_clean, self.batch_size)
                        # loop through batches
                        for ii in range(n_batches):
                            x_train_noise_bat = next(x_train_noise_ep)
                            x_train_clean_bat = next(x_train_clean_ep)
                            y_train_bat = next(y_train_ep)

                            # train to reconstruct clean data
                            temp_out = ecnn.train_on_batch([x_train_noise_bat,y_train_bat,weight],[x_train_clean_bat,y_train_bat])
                            ## train to reconstruct noisy data
                            # temp_out = svae.train_on_batch([x_train_noise_bat,weight],[x_train_noise_bat,y_train_bat,x_train_clean_bat[:,0,0]])
                        
                        ## currently unused - divide previous losses to create subsequent loss weights
                        # if temp_out[1]/temp_out[2] > 1:
                        #     rat = temp_out[1]/temp_out[2]
                        # else:
                        #     rat = 1
                        # weight = np.array([[rat, (temp_out[2])/temp_out[3]] for _ in range(self.batch_size)])
                        # weight = np.array([1 for _ in range(len(x_valid_noise_vae))])

                        # dummy loss weight for testing
                        test_weight = np.array([[1,1] for _ in range(len(x_valid_noise_vae))])

                        # save training metrics in history array
                        ecnn_hist[ep,:5] = temp_out

                        ## validation testing to reconstruct clean features
                        ecnn_hist[ep,5:] = ecnn.test_on_batch([x_valid_noise_vae,y_valid_clean,test_weight],[x_valid_noise_vae,y_valid_clean])

                        # validation testing to reconstruct noisy features
                        # svae_hist[ep,7:] = svae.test_on_batch([x_valid_noise_vae,test_weight],[x_valid_noise_vae,y_valid_clean,x_valid_clean_vae[:,0,0]])

                        # print training losses as we train
                        if ep == 0:
                            print(ecnn.metrics_names)
                        print(ecnn_hist[ep,5:])

                    # get weights
                    ecnn_w = ecnn.get_weights()
                    ecnn_enc_w = ecnn_enc.get_weights()
                    ecnn_dec_w = ecnn_dec.get_weights()
                    ecnn_clf_w = ecnn_clf.get_weights()

                    # ecnn_hist = ecnn.fit(x_train_noise_ext, y_train_clean,epochs=30,validation_data = [x_valid_noise_ext, y_valid_clean],batch_size=self.batch_size)
                    # ecnn_w = ecnn.get_weights()
                    # ecnn_enc_w = ecnn_enc.get_weights()
                    # ecnn_clf_w = ecnn_clf.get_weights()
                    # ecnn_hist = ecnn_hist.history

                # Align training data for ENC-LDA
                if mod == 'all' or any("aligned" in s for s in mod):
                    # set weights from trained models
                    # svae_enc.set_weights(svae_enc_w)
                    sae_enc.set_weights(sae_enc_w)
                    cnn_enc.set_weights(cnn_enc_w)
                    vcnn_enc.set_weights(vcnn_enc_w)
                    ecnn_enc.set_weights(ecnn_enc_w)
                    svae_clf.set_weights(svae_clf_w)

                    # align input data
                    # _, _, x_train_svae = svae_enc.predict(x_train_noise_vae)
                    _, x_train_svae = svae_clf.predict(x_train_noise_vae)
                    x_train_sae = sae_enc.predict(x_train_noise_sae)
                    x_train_cnn = cnn_enc.predict(x_train_noise_vae)
                    _, _, x_train_vcnn = vcnn_enc.predict(x_train_noise_vae)
                    # x_train_ecnn = ecnn_enc.predict(x_train_noise_ext)
                    _,_,_, x_train_ecnn = ecnn_enc.predict(x_train_noise_vae)

                    # prepare class labels
                    y_train_aligned = np.argmax(y_train_clean, axis=1)[...,np.newaxis]

                    # Train ENC-LDA
                    w_svae, c_svae,_, _, v_svae = train_lda(x_train_svae,y_train_aligned)
                    w_sae, c_sae,_, _, v_sae = train_lda(x_train_sae,y_train_aligned)
                    w_cnn, c_cnn,_, _, v_cnn = train_lda(x_train_cnn,y_train_aligned)
                    w_vcnn, c_vcnn, _, _, v_vcnn = train_lda(x_train_vcnn,y_train_aligned)
                    w_ecnn, c_ecnn, _, _, v_ecnn = train_lda(x_train_ecnn,y_train_aligned)

                # Train LDA
                if mod == 'all' or any("lda" in s for s in mod):
                    w,c, mu, C, v = train_lda(x_train_lda,y_train_lda)
                    w_noise,c_noise, _, _, v_noise = train_lda(x_train_noise_lda,y_train_noise_lda)
                
                # Train QDA
                if mod == 'all' or any("qda" in s for s in mod):
                    # Train QDA
                    qda = QDA()
                    qda.fit(x_train_lda, np.squeeze(y_train_lda))
                    qda_noise = QDA()
                    qda_noise.fit(x_train_noise_lda, np.squeeze(y_train_noise_lda))
                
                # Reconstruct training data for data augmentation
                if mod == 'all' or any("recon" in s for s in mod):
                    # set weights from trained svae
                    svae.set_weights(svae_w)
                    svae_enc.set_weights(svae_enc_w)
                    svae_dec.set_weights(svae_dec_w)
                    svae_clf.set_weights(svae_clf_w)

                    # dummy test weights and reconstructing data from noisy input
                    test_weights = np.array([[1,1] for _ in range(len(x_train_noise_vae))])
                    dec_out, _, _ = svae.predict([x_train_noise_vae, test_weights])

                    # concatenate noisy and reconstructed data for augmented training data
                    x_train_aug = np.concatenate((x_train_noise_vae, dec_out))
                    y_train_all = np.argmax(np.tile(y_train_clean,(2,1)), axis=1)[...,np.newaxis]

                    # inverse transform augmented training data to train regular LDA
                    x_train_aug_lda = scaler.inverse_transform(x_train_aug.reshape(x_train_aug.shape[0]*x_train_aug.shape[1],-1)).reshape(x_train_aug.shape)
                    x_train_aug_lda = np.transpose(x_train_aug_lda,(0,2,1,3)).reshape(x_train_aug_lda.shape[0],-1)
                    w_rec,c_rec, _, _, _ = train_lda(x_train_aug_lda,y_train_all)

                    # align augmented training data
                    _, _, _, x_train_aug_align = svae_enc.predict(x_train_aug)

                    # Train ENC-LDA with augmented data
                    w_rec_al, c_rec_al,_, _, _ = train_lda(x_train_aug_align,y_train_all)

                if mod == 'all' or any("gen" in s for s in mod):
                    # set weights from trained svae
                    svae.set_weights(svae_w)
                    svae_enc.set_weights(svae_enc_w)
                    svae_dec.set_weights(svae_dec_w)
                    svae_clf.set_weights(svae_clf_w)

                    # for testing, set weights for CNN
                    cnn_enc.set_weights(cnn_enc_w)

                    # dummy test weights
                    test_weights = np.array([[1,1] for _ in range(len(x_train_noise_vae))])

                    # generated these classes, top for integer coding and bottom for one hot coding
                    gen_clf = np.argmax(np.tile(np.eye(y_train_clean.shape[-1]),(self.gens,1)),axis=1)
                    # gen_clf = np.tile(np.eye(y_train_clean.shape[-1]),(gens,1))

                    # sample from normal distribution, forward pass through decoder
                    latent_in = np.random.normal(0,1,size=(gen_clf.shape[0],self.latent_dim))
                    dec_out = svae_dec.predict([latent_in, gen_clf])
                    dec_cv[cv-1,...] = dec_out

                    # concatenate noisy and generated data for augmented training data
                    x_train_aug = np.concatenate((x_train_noise_vae, dec_out))
                    y_train_all = np.concatenate((np.argmax(y_train_clean, axis=1),gen_clf))[...,np.newaxis]

                    # inverse transform augmented training data to train regular LDA
                    x_train_aug_lda = scaler.inverse_transform(x_train_aug.reshape(x_train_aug.shape[0]*x_train_aug.shape[1],-1)).reshape(x_train_aug.shape)
                    x_train_aug_lda = np.transpose(x_train_aug_lda,(0,2,1,3)).reshape(x_train_aug_lda.shape[0],-1)
                    w_gen,c_gen, _, _, _ = train_lda(x_train_aug_lda,y_train_all)

                    ## align augmented data
                    # _, _, _, x_train_aug_align = svae_enc.predict(x_train_aug)

                    ## for testing, align using CNN
                    x_train_aug_align = cnn_enc.predict(x_train_aug)

                    # Train ENC-LDA with augmented data
                    w_gen_al, c_gen_al, _, _, _ = train_lda(x_train_aug_align,y_train_all)

                # Pickle variables
                if mod != 'none':
                    with open(filename + '.p', 'wb') as f:
                        pickle.dump([scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, \
                            ecnn_w, ecnn_enc_w, ecnn_clf_w, w_svae, c_svae, w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w_ecnn, c_ecnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise],f)

                    with open(filename + '_hist.p', 'wb') as f:
                        pickle.dump([svae_hist, sae_hist, cnn_hist, vcnn_hist, ecnn_hist],f)

                    if mod == 'all' or any("gen" in s for s in mod) or any ("recon" in s for s in mod):
                        with open(filename + '_aug.p', 'wb') as f:
                            pickle.dump([w_rec, c_rec, w_rec_al, c_rec_al, w_gen, c_gen, w_gen_al, c_gen_al],f)
                    
                    if mod == 'all' or any("aligned" in s for s in mod) or any("lda" in s for s in mod):
                        with open(filename + '_red.p', 'wb') as f:
                            pickle.dump([v_svae, v_sae, v_cnn, v_vcnn, v_ecnn, v, v_noise],f)

        return

    def loop_test(self, raw, params):
        # set number of models to test
        mod_tot = 15
        # set testing noise type
        noise_type = self.n_test[4:-1]

        # set number of tests for each noise type
        if noise_type == 'gauss' or noise_type == '60hz':
            test_tot = 5 # noise amplitude (1-5)
        elif noise_type == 'pos':
            test_tot = 4 # number of positions
        elif noise_type == 'flat':
            test_tot = 1

        # set number of cvs
        if self.dt == 'cv':
            cv_tot = 4
        else:
            cv_tot = 1
        self.max_cv = cv_tot + 1
        
        # Initialize accuracy arrays
        acc_all = np.full([np.max(params[:,0]), cv_tot, test_tot, mod_tot],np.nan)
        acc_clean = np.full([np.max(params[:,0]), cv_tot, test_tot, mod_tot],np.nan)
        acc_noise = np.full([np.max(params[:,0]), cv_tot, test_tot, mod_tot],np.nan)

        filename = 0

        # Set folder
        foldername = self.create_foldername()

        # loop through subjects
        for sub in range(1,np.max(params[:,0])+1):            
            # index based on training group and subject
            ind = (params[:,0] == sub) & (params[:,3] == self.train_grp)

            # Check if training data exists
            if np.sum(ind):
                # split data into training, testing, validation sets
                if self.dt == 'cv':
                    x_full, x_test, _, p_full, p_test, _ = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt,train_grp=self.train_grp)
                else:
                    x_train, x_test, x_valid, p_train, p_test, p_valid = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt,train_grp=self.train_grp)

                # loop through cvs
                for cv in range(self.start_cv,self.max_cv):
                    filename = self.create_filename(foldername, cv, sub)

                    if self.dt == 'cv':
                        x_valid, p_valid = x_full[p_full[:,6] == cv,...], p_full[p_full[:,6] == cv,...]
                        x_train, p_train = x_full[p_full[:,6] != cv,...], p_full[p_full[:,6] != cv,...]

                    print('Running sub ' + str(sub) + ', model ' + str(self.train_grp) + ', latent dim ' + str(self.latent_dim))
                    
                    # Load saved data
                    with open(filename + '.p', 'rb') as f:
                        scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, ecnn_w, ecnn_enc_w, ecnn_clf_w, w_svae, c_svae, w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w_ecnn, c_ecnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise = pickle.load(f)   

                    # with open(filename + '_aug.p', 'rb') as f:
                    #     w_rec, c_rec, w_rec_al, c_rec_al, w_gen, c_gen, w_gen_al, c_gen_al = pickle.load(f)

                    # Add noise to training data
                    y_shape = np.max(p_train[:,4])

                    # Build models and set weights
                    K.clear_session()
                    svae, svae_enc, svae_dec, svae_clf = dl.build_M2(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    sae, sae_enc, sae_clf = dl.build_sae(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    cnn, cnn_enc, cnn_clf = dl.build_cnn(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    vcnn, vcnn_enc, vcnn_clf = dl.build_vcnn_manual(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    ecnn, ecnn_enc, ecnn_dec, ecnn_clf = dl.build_M2S2(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)

                    svae.set_weights(svae_w)
                    svae_enc.set_weights(svae_enc_w)
                    svae_dec.set_weights(svae_dec_w)
                    svae_clf.set_weights(svae_clf_w)

                    sae.set_weights(sae_w)
                    sae_enc.set_weights(sae_enc_w)
                    sae_clf.set_weights(sae_clf_w)

                    cnn.set_weights(cnn_w)
                    cnn_enc.set_weights(cnn_enc_w)
                    cnn_clf.set_weights(cnn_clf_w)

                    vcnn.set_weights(vcnn_w)
                    vcnn_enc.set_weights(vcnn_enc_w)
                    vcnn_clf.set_weights(vcnn_clf_w)

                    ecnn.set_weights(ecnn_w)
                    ecnn_enc.set_weights(ecnn_enc_w)
                    ecnn_clf.set_weights(ecnn_clf_w)
                    
                    # set test on validation data for cv mode
                    if self.dt == 'cv':
                        x_test, p_test = x_valid, p_valid
                    if noise_type == 'pos':
                        test_grp = int(self.n_test[-1])
                        _, x_test, _, _, p_test, _ = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt,train_grp=test_grp)

                    clean_size = int(np.size(x_test,axis=0))
                    # loop through test levels
                    for test_scale in range(1,test_tot + 1):
                        noisefolder = 'testdata_' + self.dt
                        noisefile = noisefolder + '/' + self.sub_type + str(sub) + '_grp_' + str(self.train_grp) + '_' + str(self.n_test) + '_' + str(test_scale)
                        if not os.path.isdir(noisefolder):
                            os.mkdir(noisefolder) 
                        if self.dt == 'cv':
                            noisefile += '_cv_' + str(cv)
                        
                        if os.path.isfile(noisefile + '.p'):
                            print('loading data')
                            with open(noisefile + '.p','rb') as f:
                                x_test_vae, x_test_clean_vae, x_test_lda, y_test_clean = pickle.load(f) 
                            test_load = True
                            skip = False
                        else:                    
                            skip = False
                            test_load = False
                            # load test data for diff limb positions
                            if noise_type == 'pos':
                                pos_ind = p_test[:,-1] == test_scale
                                if pos_ind.any():
                                    x_test_noise = x_test[pos_ind,...]
                                    x_test_clean = x_test[pos_ind,...]
                                    y_test_clean = to_categorical(p_test[pos_ind,4]-1)
                                    clean_size = 0
                                elif x_test.size > 0:
                                    x_test_noise = x_test
                                    x_test_clean = x_test
                                    y_test_clean = to_categorical(p_test[:,4]-1)
                                    clean_size = 0
                                else: 
                                    skip = True                    
                            else:
                                # Add noise and index testing data
                                x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_test, p_test, sub, self.n_test, test_scale)
                                # copy clean data if not using noise
                                if not self.noise:
                                    x_test_noise = cp.deepcopy(x_test_clean)

                        if not skip:
                            if not test_load:
                                # extract and scale features
                                if self.feat_type == 'feat':
                                    x_test_vae, _ = prd.extract_scale(x_test_noise,scaler)
                                    x_test_clean_vae, _ = prd.extract_scale(x_test_clean,scaler)
                                # not finalized, scale raw data
                                elif self.feat_type == 'raw':
                                    x_test_vae = cp.deepcopy(x_test_noise[:,:,::2,:])/5
                                    x_test_clean_vae = cp.deepcopy(x_test_clean[:,:,::2,:])/5
                                
                                x_test_lda = prd.extract_feats(x_test_noise)
                                with open(noisefile + '.p','wb') as f:
                                    pickle.dump([x_test_vae, x_test_clean_vae, x_test_lda, y_test_clean],f) 
                            if noise_type == 'pos':
                                clean_size = 0

                            # Reshape for nonconvolutional SAE
                            x_test_dlsae = x_test_vae.reshape(x_test_vae.shape[0],-1)
                            x_test_clean_sae = x_test_clean_vae.reshape(x_test_clean_vae.shape[0],-1)

                            # TEMP - CNN extended
                            x_test_ext = np.concatenate((x_test_vae,x_test_vae[:,:2,...]),axis=1)

                            # Align test data for ENC-LDA
                            # _,_, x_test_svae = svae_enc.predict(x_test_vae)
                            _, x_test_svae = svae_clf.predict(x_test_vae)
                            x_test_sae = sae_enc.predict(x_test_dlsae)
                            x_test_cnn = cnn_enc.predict(x_test_vae)
                            _,_,_,x_test_ecnn = ecnn_enc.predict(x_test_vae)
                            _, _, x_test_vcnn = vcnn_enc.predict(x_test_vae)

                            y_test_aligned = np.argmax(y_test_clean, axis=1)[...,np.newaxis]

                            # Non NN methods
                            y_test_lda = np.argmax(y_test_clean, axis=1)[...,np.newaxis]

                            y_test_ch = y_test_lda[:y_test_lda.shape[0]//2,...]

                            # Compile models and test data into lists
                            dl_mods = 5
                            align_mods = 5
                            lda_mods = 2
                            qda_mods = 2
                            mods_all = [svae,sae,cnn,vcnn,ecnn,[w_svae,c_svae],[w_sae,c_sae],[w_cnn,c_cnn],[w_vcnn,c_vcnn],[w_ecnn,c_ecnn],[w,c],[w_noise,c_noise],qda,qda_noise,[mu, C, self.n_test]]
                            x_test_all = ['x_test_vae', 'x_test_dlsae', 'x_test_vae', 'x_test_vae', 'x_test_vae','x_test_svae', 'x_test_sae', 'x_test_cnn', 'x_test_vcnn', 'x_test_ecnn', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test']
                            y_test_all = np.append(np.append(np.append(np.full(dl_mods,'y_test_clean'), np.full(align_mods, 'y_test_aligned')), np.full(lda_mods+qda_mods, 'y_test_lda')),np.full(1,'y_test_ch'))
                            mods_type =  np.append(np.append(np.append(np.full(dl_mods,'dl'),np.full(align_mods+lda_mods,'lda')),np.full(qda_mods,'qda')), np.full(1,'lda_ch'))
                            
                            # need to figure out what n_test = 0 is
                            if self.n_test == 0:
                                max_i = len(mods_all) - 1
                            else:
                                max_i = len(mods_all)

                            for i in range(max_i):
                                acc_all[sub-1,cv-1,test_scale - 1,i], acc_noise[sub-1,cv-1,test_scale - 1,i], acc_clean[sub-1,cv-1,test_scale - 1,i] = self.eval_mod(eval(x_test_all[i]), eval(y_test_all[i]), clean_size, mods_all[i], mods_type[i])
                        else:
                            acc_all[sub-1,cv-1,test_scale - 1,:], acc_noise[sub-1,cv-1,test_scale - 1,:], acc_clean[sub-1,cv-1,test_scale - 1,:] = np.nan, np.nan, np.nan
                
                # Save subject specific cv results
                print('saving ' + filename + '_' + self.n_test + '_cvresults.p')
                with open(filename + '_' + self.n_test + '_cvresults.p', 'wb') as f:
                    pickle.dump([acc_all[sub-1,...], acc_noise[sub-1,...], acc_clean[sub-1,...]],f)
        
        # save results for all subjects, need to figure out n_test = 0
        if self.n_test != 0:
            resultsfile = foldername + '/' + self.sub_type + '_' + self.feat_type + '_dim_' + str(self.latent_dim) + '_ep_' + str(self.epochs) + '_' + self.n_train + '_' + str(self.train_scale) + '_' + self.n_test
            if self.sparsity:
                resultsfile = resultsfile + '_sparse'
        else:
            resultsfile = filename

        with open(resultsfile + '_results.p', 'wb') as f:
            pickle.dump([acc_all, acc_clean, acc_noise],f)

        out = {'acc_all':acc_all, 'acc_noise':acc_noise, 'acc_clean':acc_clean}
        return out

    def eval_mod(self, x_test, y_test, clean_size, mod, eval_type):
        if clean_size == 0:
            clean_size_2 = x_test.shape[0]
        else:
            clean_size_2 = clean_size

        if eval_type == 'dl':
            _, acc_all = dl.eval_vae(mod, x_test, y_test)
            _, acc_noise = dl.eval_vae(mod,x_test[clean_size:,...], y_test[clean_size:,:])
            _, acc_clean = dl.eval_vae(mod,x_test[:clean_size_2,...], y_test[:clean_size_2,:])
        elif eval_type == 'lda':
            acc_all = eval_lda(mod[0], mod[1], x_test, y_test)
            acc_noise = eval_lda(mod[0], mod[1], x_test[clean_size:,:], y_test[clean_size:,:])
            acc_clean = eval_lda(mod[0], mod[1], x_test[:clean_size_2,:], y_test[:clean_size_2,:])
        elif eval_type == 'qda':
            acc_all = mod.score(x_test,np.squeeze(y_test))
            acc_noise = mod.score(x_test[clean_size:,:],np.squeeze(y_test[clean_size:,:]))
            acc_clean = mod.score(x_test[:clean_size_2,:],np.squeeze(y_test[:clean_size_2,:]))
        elif eval_type == 'lda_ch':
            if clean_size == 0:
                acc_noise = 0
            else:
                acc_noise = eval_lda_ch(mod[0], mod[1], mod[2], x_test, y_test)
            acc_all = 0
            acc_clean = 0

        return acc_all, acc_noise, acc_clean
    
    def reduce_latent(self, raw, params, sub, test_scale=1):
        # Load training and testing data
        x_train, x_test, x_valid, p_train, p_test, p_valid = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt,train_grp=self.train_grp)
        # get folder and file names
        foldername = self.create_foldername()
        filename = self.create_filename(foldername, sub = sub)

        # Load saved data
        with open(filename + '.p', 'rb') as f:
            scaler, _, svae_enc_w, _, svae_clf_w, _, sae_enc_w, _, _, cnn_enc_w, _, _, vcnn_enc_w, _, \
                _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = pickle.load(f)
        with open(filename + '_red.p', 'rb') as f:
            v_svae, v_sae, v_cnn, v_vcnn, _, v, v_noise = pickle.load(f)

        noisefolder = 'testdata_' + self.dt

        noisefile = noisefolder + '/' + self.sub_type + str(sub) + '_grp_' + str(self.train_grp) + '_' + str(self.n_test) + '_' + str(test_scale)
        if os.path.isfile(noisefile + '.p'):
            print('loading test data')
            with open(noisefile + '.p','rb') as f:
                x_test_vae, x_test_clean_vae, x_test_lda, y_test_clean = pickle.load(f) 
        
        x_test_dlsae = x_test_vae.reshape(x_test_vae.shape[0],-1)
        
        K.clear_session()
        _, svae_enc, _, svae_clf = dl.build_M2(self.latent_dim, y_test_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
        _, sae_enc, _ = dl.build_sae(self.latent_dim, y_test_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
        _, cnn_enc, _ = dl.build_cnn(self.latent_dim, y_test_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
        _, vcnn_enc, _ = dl.build_vcnn_manual(self.latent_dim, y_test_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)

        # set weights from trained models
        svae_enc.set_weights(svae_enc_w)
        sae_enc.set_weights(sae_enc_w)
        cnn_enc.set_weights(cnn_enc_w)
        vcnn_enc.set_weights(vcnn_enc_w)
        svae_clf.set_weights(svae_clf_w)
        
        # align to latent space
        _, x_test_svae = svae_clf.predict(x_test_vae)
        x_test_sae = sae_enc.predict(x_test_dlsae)
        x_test_cnn = cnn_enc.predict(x_test_vae)
        _, _, x_test_vcnn = vcnn_enc.predict(x_test_vae)

        # reduce training data
        x_test_svae_red = np.matmul(x_test_svae,v_svae)
        x_test_sae_red = np.matmul(x_test_sae,v_sae)
        x_test_cnn_red = np.matmul(x_test_cnn,v_cnn)
        x_test_vcnn_red = np.matmul(x_test_vcnn,v_vcnn)

        x_test_lda_red = np.matmul(x_test_lda,v)
        x_test_noise_red = np.matmul(x_test_lda,v_noise)

        clean_size = int(np.size(x_test,axis=0))
        y_test_clean = np.argmax(y_test_clean, axis=1)[...,np.newaxis]

        # compile results
        out_dict = {'x_test_svae_red':x_test_svae_red,'x_test_sae_red':x_test_sae_red,'x_test_cnn_red':x_test_cnn_red,'x_test_vcnn_red':x_test_vcnn_red, 'x_test_lda_red':x_test_lda_red, 'x_test_noise_red':x_test_noise_red, 'y_test':y_test_clean,'clean_size':clean_size}

        return out_dict

