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

        # initialize x out matrix
        dec_cv = np.full([self.max_cv-1,self.gens*np.max(params[ind,4]),raw.shape[1],4,1], np.nan)

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
                    dec_cv[cv-1,...] = dec_out

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

        hist_dict = {'last_acc':last_acc,'last_val':last_val}
        in_dict = {'x_noisy':x_train_noise_vae,'x_clean':x_train_vae,'y_in':y_train_clean,'scaler':scaler}        
        out = dict(hist_dict,**in_dict)
        
        if mod == 'all' or any("align" in s for s in mod):
            align_dict = {'x_svae_al':x_train_svae, 'x_sae_al':x_train_sae, 'x_cnn_al':x_train_cnn, 'x_vcnn_al':x_train_vcnn}
            out.update(align_dict)

        if mod =='all' or any("gen" in s for s in mod) or any("recon" in s for s in mod):
            out_dict = {'gen_clf':gen_clf,'x_out':dec_cv} 
            out.update(out_dict)
        
        return out

    def loop_test(self, raw, params):
        # set number of models to test
        mod_tot = 14
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
        max_cv = cv_tot + 1
        
        # Initialize accuracy arrays
        acc_all = np.full([np.max(params[:,0])+1, cv_tot, test_tot, mod_tot],np.nan)
        acc_clean = np.full([np.max(params[:,0])+1, cv_tot, test_tot, mod_tot],np.nan)
        acc_noise = np.full([np.max(params[:,0])+1, cv_tot, test_tot, mod_tot],np.nan)

        filename = 0

        # Set folder
        foldername = self.create_foldername()

        # loop through subjects
        for sub in range(1,2):#6):#np.max(params[:,0])+1):            
            # index based on training group and subject
            ind = (params[:,0] == sub) & (params[:,3] == self.train_grp)

            # Check if training data exists
            if np.sum(ind):
                # split data into training, testing, validation sets
                if self.dt == 'cv':
                    x_full, x_test, _, p_full, p_test, _ = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt)
                else:
                    x_train, x_test, x_valid, p_train, p_test, p_valid = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt)

                # loop through cvs
                for cv in range(self.start_cv,self.max_cv):
                    filename = self.create_filename(foldername, cv, sub)

                    if self.dt == 'cv':
                        x_valid, p_valid = x_full[p_full[:,6] == cv,...], p_full[p_full[:,6] == cv,...]
                        x_train, p_train = x_full[p_full[:,6] != cv,...], p_full[p_full[:,6] != cv,...]

                    print('Running sub ' + str(sub) + ', model ' + str(self.train_grp) + ', latent dim ' + str(self.latent_dim))
                    
                    # Load saved data
                    with open(filename + '.p', 'rb') as f:
                        scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, w_svae, c_svae, \
                            w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise = pickle.load(f)   

                    with open(filename + '_aug.p', 'rb') as f:
                        w_rec, c_rec, w_rec_al, c_rec_al, w_gen, c_gen, w_gen_al, c_gen_al = pickle.load(f)

                    # Add noise to training data
                    y_shape = np.max(p_train[:,4])

                    # Build models and set weights
                    svae, svae_enc, svae_dec, svae_clf = dl.build_svae_manual(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    sae, sae_enc, sae_clf = dl.build_sae(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    cnn, cnn_enc, cnn_clf = dl.build_cnn(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    vcnn, vcnn_enc, vcnn_clf = dl.build_vcnn(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)

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
                    
                    # set test on validation data for cv mode
                    if self.dt == 'cv':
                        x_test, p_test = x_valid, p_valid

                    # loop through test levels
                    for test_scale in range(1,test_tot + 1):
                        skip = False
                        # load test data for diff limb positions
                        if noise_type == 'pos':
                            test_grp = int(self.n_test[-1])
                            _, x_test, _, _, p_test, _ = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt,train_grp=test_grp)
                            pos_ind = p_test[:,-1] == self.test_scale
                            if pos_ind.any():
                                x_test_noise = x_test[pos_ind,...]
                                x_test_clean = x_test[pos_ind,...]
                                y_test_clean = to_categorical(p_test[pos_ind,4]-1)
                                clean_size = 0
                                skip = False
                            elif x_test.size > 0:
                                x_test_noise = x_test
                                x_test_clean = x_test
                                y_test_clean = to_categorical(p_test[:,4]-1)
                                clean_size = 0
                                skip = False
                            else: 
                                skip = True                    
                        else:
                            # Add noise and index testing data
                            x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_test, p_test, sub, self.n_test, test_scale)
                            clean_size = int(np.size(x_test,axis=0))
                            # copy clean data if not using noise
                            if not self.noise:
                                x_test_noise = cp.deepcopy(x_test_clean)

                        if not skip:
                            # extract and scale features
                            if self.feat_type == 'feat':
                                x_test_noise_temp = np.transpose(prd.extract_feats(x_test_noise).reshape((x_test_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                                x_test_clean_temp = np.transpose(prd.extract_feats(x_test_clean).reshape((x_test_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                                
                                x_test_vae = scaler.transform(x_test_noise_temp.reshape(x_test_noise_temp.shape[0]*x_test_noise_temp.shape[1],-1)).reshape(x_test_noise_temp.shape)
                                x_test_clean_vae = scaler.transform(x_test_clean_temp.reshape(x_test_clean_temp.shape[0]*x_test_clean_temp.shape[1],-1)).reshape(x_test_clean_temp.shape)
                            # not finalized, scale raw data
                            elif self.feat_type == 'raw':
                                x_test_vae = cp.deepcopy(x_test_noise[:,:,::2,:])/5
                                x_test_clean_vae = cp.deepcopy(x_test_clean[:,:,::2,:])/5

                            # Reshape for nonconvolutional SAE
                            x_test_dlsae = x_test_vae.reshape(x_test_vae.shape[0],-1)
                            x_test_clean_sae = x_test_clean_vae.reshape(x_test_clean_vae.shape[0],-1)

                            # Align test data for ENC-LDA
                            _,_, _, x_test_svae = svae_enc.predict(x_test_vae)
                            x_test_sae = sae_enc.predict(x_test_dlsae)
                            x_test_cnn = cnn_enc.predict(x_test_vae)
                            _, _, x_test_vcnn = vcnn_enc.predict(x_test_vae)

                            y_test_aligned = np.argmax(y_test_clean, axis=1)[...,np.newaxis]

                            # Non NN methods
                            x_test_lda = prd.extract_feats(x_test_noise)
                            y_test_lda = np.argmax(y_test_clean, axis=1)[...,np.newaxis]

                            y_test_ch = y_test_lda[:y_test_lda.shape[0]//2,...]

                            # Compile models and test data into lists
                            dl_mods = 4
                            align_mods = 5
                            lda_mods = 2
                            qda_mods = 2
                            mods_all = [svae,sae,cnn,vcnn,[w_svae,c_svae],[w_sae,c_sae],[w_cnn,c_cnn],[w_vcnn,c_vcnn],[w_gen_al,c_gen_al],[w,c],[w_noise,c_noise],qda,qda_noise,[mu, C, self.n_test]]
                            x_test_all = ['x_test_vae', 'x_test_dlsae', 'x_test_vae', 'x_test_vae', 'x_test_svae', 'x_test_sae', 'x_test_cnn', 'x_test_vcnn', 'x_test_cnn', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test']
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
                with open(filename + '_cvresults.p', 'wb') as f:
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