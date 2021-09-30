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
        self.sub_type = settings.get('sub_type','AB')
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

    def create_filename(self,foldername,cv,sub):
        filename = foldername + '/' + self.sub_type + str(sub) + '_' + self.feat_type + '_dim_' + str(self.latent_dim) + '_ep_' + str(self.epochs) + '_bat_' + str(self.batch_size) + '_' + self.n_train + '_' + str(self.train_scale) + '_lr_' + str(int(self.lr*10000)) 
        if self.dt == 'cv':
            filename += '_cv_' + str(cv)
        if self.sparsity:
            filename += '_sparse'
        
        return filename

    def loop_cv(self, raw, params, sub=1, mod='all'):
        np.set_printoptions(precision=3,suppress=True)
        i_tot = 13
        filename = 0
        if self.dt == 'manual':
            self.start_cv = 0
            self.max_cv = 1

        # Set folder
        foldername = self.create_foldername()

        # initialize final accuracy arrays and potential outputs
        last_acc = np.full([self.max_cv-1,4], np.nan)
        last_val = np.full([self.max_cv-1,4], np.nan)
        gen_clf = np.nan

        # index training group and subject
        ind = (params[:,0] == sub) & (params[:,3] == self.train_grp)

        # Check if training data exists
        if np.sum(ind):
            if mod != 'none':
                if self.dt == 'cv':
                    x_full, x_test, _, p_full, p_test, _ = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt)
                else:
                    x_train, x_test, x_valid, p_train, p_test, p_valid = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt)

            # loop through cross validation
            for cv in range(self.start_cv,self.max_cv):
                filename = self.create_filename(foldername, cv, sub)
                if self.dt == 'cv':
                    if mod != 'none':
                        x_valid, p_valid = x_full[p_full[:,6] == cv,...], p_full[p_full[:,6] == cv,...]
                        x_train, p_train = x_full[p_full[:,6] != cv,...], p_full[p_full[:,6] != cv,...]

                print('Running sub ' + str(sub) + ', model ' + str(self.train_grp) + ', latent dim ' + str(self.latent_dim) + ', cv ' + str(cv))

                ## TRAIN ##
                # Load saved data
                if self.load:
                    self.load = True
                    with open(filename + '.p', 'rb') as f:
                        scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, w_svae, c_svae, \
                            w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise = pickle.load(f)   

                    with open(filename + '_hist.p', 'rb') as f:
                        svae_hist, sae_hist, cnn_hist, vcnn_hist = pickle.load(f)

                    try:
                        with open(filename + '_aug.p', 'rb') as f:
                            w_rec, c_rec, w_rec_al, c_rec_al, w_gen, c_gen, w_gen_al, c_gen_al = pickle.load(f)
                    except:
                        print('no augmented data file')
                else:
                    scaler = MinMaxScaler(feature_range=(0,1))
                    load = False
                    
                # prepare training data if training models
                if mod != 'none':
                    # Add noise to training data
                    y_train = p_train[:,4]
                    x_train_noise, x_train_clean, y_train_clean = prd.add_noise(x_train, p_train, sub, self.n_train, self.train_scale)
                    x_valid_noise, x_valid_clean, y_valid_clean = prd.add_noise(x_valid, p_valid, sub, self.n_train, self.train_scale)
                    # if not adding noise, copy clean training data
                    if not self.noise:
                        x_train_noise = cp.deepcopy(x_train_clean)
                        x_valid_noise = cp.deepcopy(x_valid_clean)

                    # shuffle data to make even batches
                    x_train_noise, x_train_clean, y_train_clean = shuffle(x_train_noise, x_train_clean, y_train_clean, random_state = 0)

                    # Build models
                    svae, svae_enc, svae_dec, svae_clf = dl.build_svae_manual(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    sae, sae_enc, sae_clf = dl.build_sae(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    cnn, cnn_enc, cnn_clf = dl.build_cnn(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    vcnn, vcnn_enc, vcnn_clf = dl.build_vcnn(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)

                    # Training data for LDA/QDA
                    x_train_lda = prd.extract_feats(x_train)
                    y_train_lda = y_train[...,np.newaxis] - 1
                    x_train_lda2 = prd.extract_feats(x_train_noise)
                    y_train_lda2 = np.argmax(y_train_clean, axis=1)[...,np.newaxis]

                    # Shape data based on feature type
                    if self.feat_type == 'feat':
                        # extract features from training data
                        x_train_noise_temp = np.transpose(prd.extract_feats(x_train_noise).reshape((x_train_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                        x_train_clean_temp = np.transpose(prd.extract_feats(x_train_clean).reshape((x_train_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                        
                        # scale features, only fit new scaler if not loading from old model
                        if self.load:
                            x_train_noise_vae = scaler.transform(x_train_noise_temp.reshape(x_train_noise_temp.shape[0]*x_train_noise_temp.shape[1],-1)).reshape(x_train_noise_temp.shape)
                        else:
                            x_train_noise_vae = scaler.fit_transform(x_train_noise_temp.reshape(x_train_noise_temp.shape[0]*x_train_noise_temp.shape[1],-1)).reshape(x_train_noise_temp.shape)
                        
                        x_train_vae = scaler.transform(x_train_clean_temp.reshape(x_train_clean_temp.shape[0]*x_train_clean_temp.shape[1],-1)).reshape(x_train_clean_temp.shape)

                        # repeat for validation data
                        x_valid_noise_temp = np.transpose(prd.extract_feats(x_valid_noise).reshape((x_valid_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                        x_valid_clean_temp = np.transpose(prd.extract_feats(x_valid_clean).reshape((x_valid_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                        x_valid_noise_vae = scaler.transform(x_valid_noise_temp.reshape(x_valid_noise_temp.shape[0]*x_valid_noise_temp.shape[1],-1)).reshape(x_valid_noise_temp.shape)
                        x_valid_vae = scaler.transform(x_valid_clean_temp.reshape(x_valid_clean_temp.shape[0]*x_valid_clean_temp.shape[1],-1)).reshape(x_valid_clean_temp.shape)
                    elif self.feat_type == 'raw': # not finalized
                        x_train_noise_vae = scaler.fit_transform(x_train_noise_temp.reshape(x_train_noise_temp.shape[0]*x_train_noise_temp.shape[1],-1)).reshape(x_train_noise_temp.shape)
                        x_train_noise_vae = 0.5+cp.deepcopy(x_train_noise[:,:,::4,:])/10
                        x_train_vae = 0.5+cp.deepcopy(x_train_clean[:,:,::4,:])/10

                        x_valid_noise_vae = 0.5+cp.deepcopy(x_valid_noise[:,:,::4,:])/10
                        x_valid_vae = 0.5+cp.deepcopy(x_valid_clean[:,:,::4,:])/10

                    # reshape data for nonconvolutional network
                    x_train_noise_sae = x_train_noise_vae.reshape(x_train_noise_vae.shape[0],-1)
                    x_train_sae = x_train_vae.reshape(x_train_vae.shape[0],-1)
                    x_valid_noise_sae = x_valid_noise_vae.reshape(x_valid_noise_vae.shape[0],-1)
                    x_valid_sae = x_valid_vae.reshape(x_valid_vae.shape[0],-1)
                
                # Train SVAE
                if mod == 'all' or any("svae" in s for s in mod):
                    # get number of batches
                    n_batches = len(x_train_noise_vae) // self.batch_size
                    # initialize history array
                    svae_hist = np.zeros((self.epochs,14))
                    # loop through epochs
                    for ep in range(self.epochs):
                        # set loss weight vector, not finalized
                        if ep < self.epochs/2: 
                            weight = np.array([[0,2**((ep-(self.epochs/2))/2)] for _ in range(self.batch_size)])
                        else:
                            weight = np.array([[1,.5] for _ in range(self.batch_size)])
                        
                        # get batches for inputs
                        x_train_noise_ep = dl.get_batches(x_train_noise_vae, self.batch_size)
                        x_train_vae_ep = dl.get_batches(x_train_vae, self.batch_size)
                        y_train_ep = dl.get_batches(y_train_clean, self.batch_size)
                        # loop through batches
                        for ii in range(n_batches):
                            x_train_noise_bat = next(x_train_noise_ep)
                            x_train_vae_bat = next(x_train_vae_ep)
                            y_train_bat = next(y_train_ep)

                            # train to reconstruct clean data
                            temp_out = svae.train_on_batch([x_train_noise_bat,weight],[x_train_vae_bat,y_train_bat,x_train_vae_bat[:,0,0]])
                            ## train to reconstruct noisy data
                            # temp_out = svae.train_on_batch([x_train_noise_bat,weight],[x_train_noise_bat,y_train_bat,x_train_vae_bat[:,0,0]])
                        
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
                        # svae_hist[ep,7:] = svae.test_on_batch([x_valid_noise_vae,test_weight],[x_valid_vae,y_valid_clean,x_valid_vae[:,0,0]])

                        # validation testing to reconstruct noisy features
                        svae_hist[ep,7:] = svae.test_on_batch([x_valid_noise_vae,test_weight],[x_valid_noise_vae,y_valid_clean,x_valid_vae[:,0,0]])

                        # print training losses as we train
                        if ep == 0:
                            print(svae.metrics_names)
                        print(svae_hist[ep,:7])

                    # get weights
                    svae_w = svae.get_weights()
                    svae_enc_w = svae_enc.get_weights()
                    svae_dec_w = svae_dec.get_weights()
                    svae_clf_w = svae_clf.get_weights()

                # Fit NNs and get weights
                if mod == 'all' or any("vcnn" in s for s in mod):
                    vcnn_hist = vcnn.fit(x_train_noise_vae, y_train_clean,epochs=30,validation_data = [x_valid_noise_vae, y_valid_clean],batch_size=self.batch_size)                
                    vcnn_w = vcnn.get_weights()
                    vcnn_enc_w = vcnn_enc.get_weights()
                    vcnn_clf_w = vcnn_clf.get_weights()
                    vcnn_hist = vcnn_hist.history

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

                # Align training data for ENC-LDA
                if mod == 'all' or any("aligned" in s for s in mod):
                    # set weights from trained models
                    svae_enc.set_weights(svae_enc_w)
                    sae_enc.set_weights(sae_enc_w)
                    cnn_enc.set_weights(cnn_enc_w)
                    vcnn_enc.set_weights(vcnn_enc_w)

                    # align input data
                    _, _, _, x_train_svae = svae_enc.predict(x_train_noise_vae)
                    x_train_sae = sae_enc.predict(x_train_noise_sae)
                    x_train_cnn = cnn_enc.predict(x_train_noise_vae)
                    _, _, x_train_vcnn = vcnn_enc.predict(x_train_noise_vae)

                    # prepare class labels
                    y_train_aligned = np.argmax(y_train_clean, axis=1)[...,np.newaxis]

                    # Train ENC-LDA
                    w_svae, c_svae,_, _ = train_lda(x_train_svae,y_train_aligned)
                    w_sae, c_sae,_, _ = train_lda(x_train_sae,y_train_aligned)
                    w_cnn, c_cnn,_, _ = train_lda(x_train_cnn,y_train_aligned)
                    w_vcnn, c_vcnn, _, _ = train_lda(x_train_vcnn,y_train_aligned)

                # Train LDA
                if mod == 'all' or any("lda" in s for s in mod):
                    w,c, mu, C = train_lda(x_train_lda,y_train_lda)
                    w_noise,c_noise, _, _ = train_lda(x_train_lda2,y_train_lda2)
                
                # Train QDA
                if mod == 'all' or any("qda" in s for s in mod):
                    # Train QDA
                    qda = QDA()
                    qda.fit(x_train_lda, np.squeeze(y_train_lda))
                    qda_noise = QDA()
                    qda_noise.fit(x_train_lda2, np.squeeze(y_train_lda2))
                
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
                    w_rec,c_rec, _, _ = train_lda(x_train_aug_lda,y_train_all)

                    # align augmented training data
                    _, _, _, x_train_aug_align = svae_enc.predict(x_train_aug)

                    # Train ENC-LDA with augmented data
                    w_rec_al, c_rec_al,_, _ = train_lda(x_train_aug_align,y_train_all)

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

                    # concatenate noisy and generated data for augmented training data
                    x_train_aug = np.concatenate((x_train_noise_vae, dec_out))
                    y_train_all = np.concatenate((np.argmax(y_train_clean, axis=1),gen_clf))[...,np.newaxis]

                    # inverse transform augmented training data to train regular LDA
                    x_train_aug_lda = scaler.inverse_transform(x_train_aug.reshape(x_train_aug.shape[0]*x_train_aug.shape[1],-1)).reshape(x_train_aug.shape)
                    x_train_aug_lda = np.transpose(x_train_aug_lda,(0,2,1,3)).reshape(x_train_aug_lda.shape[0],-1)
                    w_gen,c_gen, _, _ = train_lda(x_train_aug_lda,y_train_all)

                    ## align augmented data
                    # _, _, _, x_train_aug_align = svae_enc.predict(x_train_aug)

                    ## for testing, align using CNN
                    x_train_aug_align = cnn_enc.predict(x_train_aug)

                    # Train ENC-LDA with augmented data
                    w_gen_al, c_gen_al,_, _ = train_lda(x_train_aug_align,y_train_all)

                # Pickle variables
                if mod != 'none':
                    with open(filename + '.p', 'wb') as f:
                        pickle.dump([scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, \
                            w_svae, c_svae, w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise],f)
                    with open(filename + '_hist.p', 'wb') as f:
                        pickle.dump([svae_hist, sae_hist, cnn_hist, vcnn_hist],f)

                    if mod == 'all' or any("gen" in s for s in mod) or any ("recon" in s for s in mod):
                        with open(filename + '_aug.p', 'wb') as f:
                            pickle.dump([w_rec, c_rec, w_rec_al, c_rec_al, w_gen, c_gen, w_gen_al, c_gen_al],f)
                        print('saving aug')

                # allocate last accuracies
                last_acc[cv-1,:] = np.array([svae_hist[-1,5], sae_hist['accuracy'][-1], cnn_hist['accuracy'][-1], vcnn_hist['accuracy'][-1]])
                last_val[cv-1,:] = np.array([svae_hist[-1,12], sae_hist['val_accuracy'][-1], cnn_hist['val_accuracy'][-1], vcnn_hist['val_accuracy'][-1]])
                    
        return last_acc, last_val, filename, x_train_noise_vae, x_train_vae, y_train_clean, scaler, gen_clf, dec_out