import numpy as np
import os
import pickle

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from latent.ml.lda import train_lda, predict, eval_lda, eval_lda_ch
from sklearn.utils import shuffle
import latent.ml.dl_functional as dl
import latent.utils.data_utils as prd
import copy as cp
from datetime import date
import time
import tensorflow.keras.backend as K
import timeit
from latent.ml.dl_subclass import MLP, CNN

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
        self.gens = settings.get('gens',50)
        self.train_load = settings.get('train_load',True)
        self.test_dt = settings.get('test_dt','')

    def create_foldername(self,ftype=''):
        if self.dt == 0:
            today = date.today()
            self.dt = today.strftime("%m%d")
        # Set folder
        if ftype == 'trainnoise':
            foldername = 'noisedata_' + self.dt + '_' + self.mod_dt
            if self.feat_type == 'tdar':
                foldername += '_tdar'
        elif ftype == 'testnoise':
            foldername = 'testdata_' + self.dt + '_' + self.mod_dt + '_' + self.test_dt
            if self.feat_type == 'tdar':
                foldername += '_tdar'
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
        if ftype == 'trainnoise':
            filename = foldername + '/' + self.sub_type + str(sub) + '_grp_' + str(self.train_grp) + '_' + str(self.n_train) + '_' + str(self.train_scale)
        elif ftype == 'testnoise':
            filename = foldername + '/' + self.sub_type + str(sub) + '_grp_' + str(self.train_grp) + '_' + str(self.n_test) + '_' + str(test_scale)
        elif ftype =='allresults':
            filename = foldername + '/' + self.sub_type + '_' + self.feat_type + '_dim_' + str(self.latent_dim) + '_ep_' + str(self.epochs) + '_bat_' + str(self.batch_size) + '_' + self.n_train + '_' + str(self.train_scale) + '_lr_' + str(int(self.lr*10000)) 
        # model file
        else:
            # filename = foldername + '/' + self.sub_type + str(sub) + '_' + self.feat_type + '_dim_' + str(self.latent_dim) + '_ep_' + str(self.epochs) + '_bat_' + str(self.batch_size) + '_' + self.n_train + '_' + str(self.train_scale) + '_lr_' + str(int(self.lr*10000)) 
            filename = foldername + '/' + self.sub_type + str(sub) + '_' + 'feat' + '_dim_' + str(self.latent_dim) + '_ep_' + str(self.epochs) + '_bat_' + str(self.batch_size) + '_' + self.n_train + '_' + str(self.train_scale) + '_lr_' + str(int(self.lr*10000))
        if self.dt == 'cv':
            filename += '_cv_' + str(cv)
        if self.sparsity and ftype == '':
            filename += '_sparse'
        
        return filename

    def loop_cv(self, raw, params, sub=1, mod='all'):
        np.set_printoptions(precision=3,suppress=True)
        out_dict = {}
        i_tot = 14
        filename = 0
        if self.dt != 'cv':
            self.start_cv = 1
            self.max_cv = 2
        
        if self.feat_type == 'feat':
            num_feats = 4
        elif self.feat_type == 'tdar':
            num_feats = 10

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
                    x_train, x_test, x_valid, p_train, p_test, p_valid = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt,load=True,train_grp=self.train_grp)

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
                            ecnn_w, ecnn_enc_w, ecnn_clf_w, w_svae, c_svae, w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w_ecnn, c_ecnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise, emg_scale = pickle.load(f) 
                        # scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, \
                        # ecnn_w, ecnn_enc_w, ecnn_clf_w, w_svae, c_svae, w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w_ecnn, c_ecnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise= pickle.load(f)
                        # emg_scale = 1     

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
                    noisefolder = self.create_foldername(ftype='trainnoise')
                    noisefile = self.create_filename(noisefolder, cv, sub, ftype='trainnoise')
                    
                    if os.path.isfile(noisefile + '.p') and self.train_load:
                        print('loading data')
                        with open(noisefile + '.p','rb') as f:
                            scaler, x_train_noise_vae, x_train_clean_vae, x_valid_noise_vae, x_valid_clean_vae, y_train_clean, y_valid_clean, x_train_lda, y_train_lda, x_train_noise_lda, y_train_noise_lda = pickle.load(f)

                        emg_scale = np.ones((np.size(x_train,1),1))
                        if 'emgscale' in self.mod_dt:
                            for i in range(np.size(x_train,1)):
                                emg_scale[i] = 5/np.max(np.abs(x_train[:,i,:]))

                    else:
                        # Add noise to training data                        
                        if self.dt == 'cv':
                            x_valid, p_valid = x_full[p_full[:,6] == cv,...], p_full[p_full[:,6] == cv,...]
                            x_train, p_train = x_full[p_full[:,6] != cv,...], p_full[p_full[:,6] != cv,...]

                        emg_scale = np.ones((np.size(x_train,1),1))

                        if 'emgscale' in self.mod_dt:
                            for i in range(np.size(x_train,1)):
                                emg_scale[i] = 5/np.max(np.abs(x_train[:,i,:]))

                        x_train = x_train*emg_scale
                        x_valid = x_valid*emg_scale

                        noise_start = time.time()
                        x_train_noise, x_train_clean, y_train_clean = prd.add_noise(x_train, p_train, sub, self.n_train, self.train_scale)
                        noise_time = time.time() - noise_start 
                        print('add noise time: ' + str(noise_time))
                        out_dict['noise_time'] = noise_time

                        x_valid_noise, x_valid_clean, y_valid_clean = prd.add_noise(x_valid, p_valid, sub, self.n_train, self.train_scale)
                        # if not adding noise, copy clean training data
                        if not self.noise:
                            x_train_noise = cp.deepcopy(x_train_clean)
                            x_valid_noise = cp.deepcopy(x_valid_clean)

                        # shuffle data to make even batches
                        x_train_noise, x_train_clean, y_train_clean = shuffle(x_train_noise, x_train_clean, y_train_clean, random_state = 0)

                        if 'lim' in self.mod_dt:
                            x_train_noise[x_train_noise > 5] = 5
                            x_train_noise[x_train_noise < -5] = -5
                            x_train_clean[x_train_clean > 5] = 5
                            x_train_clean[x_train_clean < -5] = -5

                        # Training data for LDA/QDA
                        y_train = p_train[:,4]
                        x_train_lda = prd.extract_feats(x_train,ft=self.feat_type,emg_scale=emg_scale)
                        y_train_lda = y_train[...,np.newaxis] - 1
                        x_train_noise_lda = prd.extract_feats(x_train_noise,ft=self.feat_type,emg_scale=emg_scale)
                        y_train_noise_lda = np.argmax(y_train_clean, axis=1)[...,np.newaxis]

                        # Shape data based on feature type
                        if self.feat_type == 'feat' or self.feat_type == 'tdar':
                            # extract and scale features from training and validation data
                            x_train_noise_vae, scaler = prd.extract_scale(x_train_noise,scaler,self.load,ft=self.feat_type,emg_scale=emg_scale) 
                            x_train_clean_vae, _ = prd.extract_scale(x_train_clean,scaler,ft=self.feat_type,emg_scale=emg_scale)
                            x_valid_noise_vae, _ = prd.extract_scale(x_valid_noise,scaler,ft=self.feat_type,emg_scale=emg_scale)
                            x_valid_clean_vae, _ = prd.extract_scale(x_valid_clean,scaler,ft=self.feat_type,emg_scale=emg_scale)

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
                    # svae, svae_enc, svae_dec, svae_clf = dl.build_M2(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    sae, sae_enc, sae_clf = dl.build_sae(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    cnn, cnn_enc, cnn_clf = dl.build_cnn(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    # vcnn, vcnn_enc, vcnn_clf = dl.build_vcnn(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    # ecnn, ecnn_enc, ecnn_dec, ecnn_clf = dl.build_M2S2(self.latent_dim, y_train_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    
                # Train SVAE
                # if mod == 'all' or any("svae" in s for s in mod):
                #     # get number of batches
                #     n_batches = len(x_train_noise_vae) // self.batch_size
                #     # initialize history array
                #     temp_epochs = 30
                #     svae_hist = np.zeros((temp_epochs,14))
                #     # loop through epochs
                #     for ep in range(temp_epochs):
                #         # set loss weight vector, not finalized                        
                #         if ep < 15: #temp_epochs//3: 
                #             weight = np.array([[0,(2**(ep-(20)))/100] for _ in range(self.batch_size)])
                #         else:
                #             weight = np.array([[1,.01] for _ in range(self.batch_size)])
                        
                #         # get batches for inputs
                #         x_train_noise_ep = dl.get_batches(x_train_noise_vae, self.batch_size)
                #         x_train_clean_ep = dl.get_batches(x_train_clean_vae, self.batch_size)
                #         y_train_ep = dl.get_batches(y_train_clean, self.batch_size)
                #         # loop through batches
                #         for ii in range(n_batches):
                #             x_train_noise_bat = next(x_train_noise_ep)
                #             x_train_clean_bat = next(x_train_clean_ep)
                #             y_train_bat = next(y_train_ep)

                #             # train to reconstruct clean data
                #             temp_out = svae.train_on_batch([x_train_noise_bat,y_train_bat,weight],[x_train_noise_bat,y_train_bat,x_train_clean_bat[:,0,0]])

                #         # dummy loss weight for testing
                #         test_weight = np.array([[1,1] for _ in range(len(x_valid_noise_vae))])

                #         # save training metrics in history array
                #         svae_hist[ep,:7] = temp_out

                #         ## validation testing to reconstruct clean features
                #         svae_hist[ep,7:] = svae.test_on_batch([x_valid_noise_vae,y_valid_clean,test_weight],[x_valid_noise_vae,y_valid_clean,x_valid_clean_vae[:,0,0]])

                #         # print training losses as we train
                #         if ep == 0:
                #             print(svae.metrics_names)
                #         print(svae_hist[ep,7:])

                #     # get weights
                #     svae_w = svae.get_weights()
                #     svae_enc_w = svae_enc.get_weights()
                #     svae_dec_w = svae_dec.get_weights()
                #     svae_clf_w = svae_clf.get_weights()

                # # Fit NNs and get weights
                # if mod == 'all' or any("var_c" in s for s in mod):
                #     vcnn_hist = vcnn.fit(x_train_noise_vae, y_train_clean,epochs=30,validation_data = [x_valid_noise_vae, y_valid_clean],batch_size=self.batch_size)                
                #     vcnn_w = vcnn.get_weights()
                #     vcnn_enc_w = vcnn_enc.get_weights()
                #     vcnn_clf_w = vcnn_clf.get_weights()
                #     vcnn_hist = vcnn_hist.history

                #     # get weights
                #     vcnn_w = vcnn.get_weights()
                #     vcnn_enc_w = vcnn_enc.get_weights()
                #     vcnn_clf_w = vcnn_clf.get_weights()

                if mod == 'all' or any("sae" in s for s in mod):
                    sae_start = time.time()
                    sae_hist = sae.fit(x_train_noise_sae, y_train_clean,epochs=30,validation_data = [x_valid_noise_sae, y_valid_clean],batch_size=self.batch_size,verbose=0,use_multiprocessing=True, workers=12)
                    sae_time = time.time() - sae_start
                    print('sae time: ' + str(sae_time))
                    out_dict['sae_time'] = sae_time

                    sae_w = sae.get_weights()
                    sae_enc_w = sae_enc.get_weights()
                    sae_clf_w = sae_clf.get_weights()
                    sae_hist = sae_hist.history

                if mod == 'all' or any("cnn" in s for s in mod):
                    cnn_start = time.time()
                    cnn_hist = cnn.fit(x_train_noise_vae, y_train_clean,epochs=30,validation_data = [x_valid_noise_vae, y_valid_clean],batch_size=self.batch_size,verbose=0,use_multiprocessing=True, workers=12)
                    cnn_time = time.time() - cnn_start
                    print('cnn time: ' + str(cnn_time))
                    out_dict['cnn_time'] = cnn_time

                    cnn_w = cnn.get_weights()
                    cnn_enc_w = cnn_enc.get_weights()
                    cnn_clf_w = cnn_clf.get_weights()
                    cnn_hist = cnn_hist.history
                
                # if mod == 'all' or any("ext_c" in s for s in mod):
                #     ecnn_hist = ecnn.fit([x_train_noise_vae,y_train_clean], [x_train_noise_vae,y_train_clean],epochs=30,validation_data = [[x_valid_noise_vae, y_valid_clean],[x_valid_noise_vae,y_valid_clean]],batch_size=self.batch_size)
                #     ecnn_w = ecnn.get_weights()
                #     ecnn_enc_w = ecnn_enc.get_weights()
                #     ecnn_clf_w = ecnn_clf.get_weights()
                #     ecnn_hist = ecnn_hist.history

                #     # get weights
                #     ecnn_w = ecnn.get_weights()
                #     ecnn_enc_w = ecnn_enc.get_weights()
                #     ecnn_dec_w = ecnn_dec.get_weights()
                #     ecnn_clf_w = ecnn_clf.get_weights()

                # Align training data for ENC-LDA
                if mod == 'all' or any("aligned" in s for s in mod):
                    # set weights from trained models
                    # svae_enc.set_weights(svae_enc_w)
                    sae_enc.set_weights(sae_enc_w)
                    cnn_enc.set_weights(cnn_enc_w)
                    # vcnn_enc.set_weights(vcnn_enc_w)
                    # ecnn_enc.set_weights(ecnn_enc_w)
                    # svae_clf.set_weights(svae_clf_w)

                    # align input data
                    # _, x_train_svae = svae_clf.predict(x_train_noise_vae)
                    # _, _, x_train_vcnn = vcnn_enc.predict(x_train_noise_vae)
                    # _,_,_, x_train_ecnn = ecnn_enc.predict(x_train_noise_vae)

                    # prepare class labels
                    y_train_aligned = np.argmax(y_train_clean, axis=1)[...,np.newaxis]

                    # Train ENC-LDA
                    # w_svae, c_svae,_, _, v_svae = train_lda(x_train_svae,y_train_aligned)

                    sae_a_start = time.time()
                    x_train_sae = sae_enc.predict(x_train_noise_sae)
                    w_sae, c_sae,_, _, v_sae = train_lda(x_train_sae,y_train_aligned)
                    sae_a_time = time.time() - sae_a_start
                    print('aligned sae time: ' + str(sae_a_time))
                    out_dict['sae_a_time'] = sae_a_time

                    cnn_a_start = time.time()
                    x_train_cnn = cnn_enc.predict(x_train_noise_vae)
                    w_cnn, c_cnn,_, _, v_cnn = train_lda(x_train_cnn,y_train_aligned)
                    cnn_a_time = time.time() - cnn_a_start
                    print('aligned cnn time: ' + str(cnn_a_time))
                    out_dict['cnn_a_time'] = cnn_a_time

                    # w_vcnn, c_vcnn, _, _, v_vcnn = train_lda(x_train_vcnn,y_train_aligned)
                    # w_ecnn, c_ecnn, _, _, v_ecnn = train_lda(x_train_ecnn,y_train_aligned)

                # Train LDA
                if mod == 'all' or any("lda" in s for s in mod):
                    lda_start = time.time()
                    w,c, mu, C, v = train_lda(x_train_lda,y_train_lda)
                    lda_time = time.time() - lda_start
                    print('lda time: ' + str(lda_time))
                    out_dict['lda_time'] = lda_time

                    aug_lda_start = time.time()
                    w_noise,c_noise, _, _, v_noise = train_lda(x_train_noise_lda,y_train_noise_lda)
                    aug_lda_time = time.time() - aug_lda_start
                    print('aug lda time: ' + str(aug_lda_time))
                    out_dict['aug lda time'] = aug_lda_time
                
                # # Train QDA
                # if mod == 'all' or any("qda" in s for s in mod):
                #     # Train QDA
                #     qda = QDA()
                #     qda.fit(x_train_lda, np.squeeze(y_train_lda))
                #     qda_noise = QDA()
                #     qda_noise.fit(x_train_noise_lda, np.squeeze(y_train_noise_lda))
                
                # # Reconstruct training data for data augmentation
                # if any("recon" in s for s in mod):
                #     # set weights from trained svae
                #     svae.set_weights(svae_w)
                #     svae_enc.set_weights(svae_enc_w)
                #     svae_dec.set_weights(svae_dec_w)
                #     svae_clf.set_weights(svae_clf_w)

                #     # dummy test weights and reconstructing data from noisy input
                #     test_weights = np.array([[1,1] for _ in range(len(x_train_noise_vae))])
                #     dec_out, _, _ = svae.predict([x_train_noise_vae, test_weights])

                #     # concatenate noisy and reconstructed data for augmented training data
                #     x_train_aug = np.concatenate((x_train_noise_vae, dec_out))
                #     y_train_all = np.argmax(np.tile(y_train_clean,(2,1)), axis=1)[...,np.newaxis]

                #     # inverse transform augmented training data to train regular LDA
                #     x_train_aug_lda = scaler.inverse_transform(x_train_aug.reshape(x_train_aug.shape[0]*x_train_aug.shape[1],-1)).reshape(x_train_aug.shape)
                #     x_train_aug_lda = np.transpose(x_train_aug_lda,(0,2,1,3)).reshape(x_train_aug_lda.shape[0],-1)
                #     w_rec,c_rec, _, _, _ = train_lda(x_train_aug_lda,y_train_all)

                #     # align augmented training data
                #     _, _, _, x_train_aug_align = svae_enc.predict(x_train_aug)

                #     # Train ENC-LDA with augmented data
                #     w_rec_al, c_rec_al,_, _, _ = train_lda(x_train_aug_align,y_train_all)

                # if any("gen" in s for s in mod):
                #     # set weights from trained svae
                #     svae.set_weights(svae_w)
                #     svae_enc.set_weights(svae_enc_w)
                #     svae_dec.set_weights(svae_dec_w)
                #     svae_clf.set_weights(svae_clf_w)

                #     # for testing, set weights for CNN
                #     cnn_enc.set_weights(cnn_enc_w)

                #     # dummy test weights
                #     test_weights = np.array([[1,1] for _ in range(len(x_train_noise_vae))])

                #     # generated these classes, top for integer coding and bottom for one hot coding
                #     gen_clf = np.argmax(np.tile(np.eye(y_train_clean.shape[-1]),(self.gens,1)),axis=1)
                #     # gen_clf = np.tile(np.eye(y_train_clean.shape[-1]),(gens,1))

                #     # sample from normal distribution, forward pass through decoder
                #     latent_in = np.random.normal(0,1,size=(gen_clf.shape[0],self.latent_dim))
                #     dec_out = svae_dec.predict([latent_in, gen_clf])
                #     dec_cv[cv-1,...] = dec_out

                #     # concatenate noisy and generated data for augmented training data
                #     x_train_aug = np.concatenate((x_train_noise_vae, dec_out))
                #     y_train_all = np.concatenate((np.argmax(y_train_clean, axis=1),gen_clf))[...,np.newaxis]

                #     # inverse transform augmented training data to train regular LDA
                #     x_train_aug_lda = scaler.inverse_transform(x_train_aug.reshape(x_train_aug.shape[0]*x_train_aug.shape[1],-1)).reshape(x_train_aug.shape)
                #     x_train_aug_lda = np.transpose(x_train_aug_lda,(0,2,1,3)).reshape(x_train_aug_lda.shape[0],-1)
                #     w_gen,c_gen, _, _, _ = train_lda(x_train_aug_lda,y_train_all)

                #     ## align augmented data
                #     # _, _, _, x_train_aug_align = svae_enc.predict(x_train_aug)

                #     ## for testing, align using CNN
                #     x_train_aug_align = cnn_enc.predict(x_train_aug)

                #     # Train ENC-LDA with augmented data
                #     w_gen_al, c_gen_al, _, _, _ = train_lda(x_train_aug_align,y_train_all)

                # Pickle variables
                if 0:#mod != 'none':
                    with open(filename + '.p', 'wb') as f:
                        pickle.dump([scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, \
                            ecnn_w, ecnn_enc_w, ecnn_clf_w, w_svae, c_svae, w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w_ecnn, c_ecnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise, emg_scale],f)

                    with open(filename + '_hist.p', 'wb') as f:
                        pickle.dump([svae_hist, sae_hist, cnn_hist, vcnn_hist, ecnn_hist],f)

                    if any("gen" in s for s in mod) or any ("recon" in s for s in mod):
                        with open(filename + '_aug.p', 'wb') as f:
                            pickle.dump([w_rec, c_rec, w_rec_al, c_rec_al, w_gen, c_gen, w_gen_al, c_gen_al],f)
                    
                    if mod == 'all' or any("aligned" in s for s in mod) or any("lda" in s for s in mod):
                        with open(filename + '_red.p', 'wb') as f:
                            pickle.dump([v_svae, v_sae, v_cnn, v_vcnn, v_ecnn, v, v_noise],f)

        return out_dict

    def loop_test(self, raw, params):
        sae_time = [None] * np.max(params[:,0])
        cnn_time = [None] * np.max(params[:,0])
        lda_time = [None] * np.max(params[:,0])
        aug_time = [None] * np.max(params[:,0])
        
        # set number of models to test
        mod_tot = 15
        # set testing noise type
        noise_type = self.n_test[4:-1]
        print(noise_type)
        # set number of tests for each noise types
        if noise_type == 'pos':
            test_tot = 4 # number of positions
        elif 'flat' in noise_type or 'mix' in noise_type or 'real' in noise_type:
            test_tot = 1
        else:
            test_tot = 5 # noise amplitude (1-5)
        
        # load real_noise
        if 'real' in noise_type:
            with open('real_noise/all_real_noise.p', 'rb') as f:
                real_noise_temp, _ = pickle.load(f)

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
        resultsfolder = self.create_foldername(ftype='results')

        if self.feat_type == 'feat':
            num_feats = 4
        elif self.feat_type == 'tdar':
            num_feats = 10

        # loop through subjects
        for sub in range(2,np.max(params[:,0])+1):            
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
                        scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, ecnn_w, ecnn_enc_w, ecnn_clf_w, w_svae, c_svae, w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w_ecnn, c_ecnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise,emg_scale = pickle.load(f)   
                        # scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, ecnn_w, ecnn_enc_w, ecnn_clf_w, w_svae, c_svae, w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w_ecnn, c_ecnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise = pickle.load(f)  
                        # emg_scale = 1 
                    with open('subclass/models/' + str(self.sub_type) + str(sub) + '_' + str(self.feat_type) + '.p','rb') as f:
                        sae_w, _, cnn_w, w_sae, c_sae, _, _, w_cnn, c_cnn, w, c, w_noise, c_noise, emg_scale, scaler, mu, C = pickle.load(f)
                    # Add noise to training data
                    y_shape = np.max(p_train[:,4])
                    # Build models and set weights
                    K.clear_session()
                    svae, svae_enc, svae_dec, svae_clf = dl.build_M2(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    sae, sae_enc, sae_clf = dl.build_sae(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    cnn, cnn_enc, cnn_clf = dl.build_cnn(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    vcnn, vcnn_enc, vcnn_clf = dl.build_vcnn(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    ecnn, ecnn_enc, ecnn_dec, ecnn_clf = dl.build_M2S2(self.latent_dim, y_shape, input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
                    n_dof = y_shape#len(np.unique(p_train[:,4]))
                    sae = MLP(n_class=n_dof)
                    # mlp_beta = MLPbeta(n_class=n_dof)
                    cnn = CNN(n_class=n_dof)
                    # svae.set_weights(svae_w)
                    # svae_enc.set_weights(svae_enc_w)
                    # svae_dec.set_weights(svae_dec_w)
                    # svae_clf.set_weights(svae_clf_w)

                    # sae.set_weights(sae_w)
                    # # sae_enc.set_weights(sae_enc_w)
                    # # sae_clf.set_weights(sae_clf_w)

                    # cnn.set_weights(cnn_w)
                    # cnn_enc.set_weights(cnn_enc_w)
                    # cnn_clf.set_weights(cnn_clf_w)

                    # vcnn.set_weights(vcnn_w)
                    # vcnn_enc.set_weights(vcnn_enc_w)
                    # vcnn_clf.set_weights(vcnn_clf_w)

                    # ecnn.set_weights(ecnn_w)
                    # ecnn_enc.set_weights(ecnn_enc_w)
                    # ecnn_clf.set_weights(ecnn_clf_w)
                    
                    # set test on validation data for cv mode
                    skip = False
                    if self.dt == 'cv':
                        x_test, p_test = x_valid, p_valid
                    if noise_type[:3] == 'pos':
                        test_grp = int(self.n_test[-1])
                        _, x_test, _, _, p_test, _ = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt,train_grp=test_grp)
                        if x_test.size == 0:
                            skip = True
                    if noise_type == 'pos':
                        clean_size = 0
                    else:
                        clean_size = int(np.size(x_test,axis=0))
                    
                    # print(emg_scale)
                    x_test_old = cp.deepcopy(x_test)
                    if 'emgscale' in self.mod_dt:
                        x_test = x_test*emg_scale
                    x_test_scaled = cp.deepcopy(x_test)

                    # loop through test levels
                    for test_scale in range(1,test_tot + 1):
                        noisefolder = self.create_foldername(ftype='testnoise')
                        noisefile = self.create_filename(noisefolder,cv, sub, ftype='testnoise', test_scale=test_scale)
                        
                        if not skip:
                            if os.path.isfile(noisefile + '.p'):
                                print('loading data')
                                with open(noisefile + '.p','rb') as f:
                                    x_test_vae, x_test_clean_vae, x_test_lda, y_test_clean = pickle.load(f) 
                                test_load = True
                            else:                    
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
                                    if 'real' in noise_type:
                                        if 'noisescale' in self.test_dt:
                                            print('scaling noise')
                                            # def add_noise(raw, params, n_type='flat', scale=5, real_noise=0,emg_scale=[1,1,1,1,1,1]):
                                            # print(emg_scale)
                                            x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_test, p_test, self.n_test, test_scale, real_noise=real_noise_temp, emg_scale = emg_scale)
                                            # x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_test, p_test, sub, self.n_test, test_scale, real_noise=real_noise_temp, emg_scale = emg_scale)
                                        else:
                                            x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_test, p_test, sub, self.n_test, test_scale, real_noise=real_noise_temp)
                                    else:
                                        x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_test, p_test, sub, self.n_test, test_scale)
                                    # copy clean data if not using noise
                                    if not self.noise:
                                        x_test_noise = cp.deepcopy(x_test_clean)

                        if not skip:
                            if not test_load:
                                if 'lim' in self.mod_dt:
                                    x_test_noise[x_test_noise > 5] = 5
                                    x_test_noise[x_test_noise < -5] = -5
                                    x_test_clean[x_test_clean > 5] = 5
                                    x_test_clean[x_test_clean < -5] = -5
                                # extract and scale features
                                if self.feat_type == 'feat' or self.feat_type == 'tdar':
                                    x_test_vae, _ = prd.extract_scale(x_test_noise,scaler,ft=self.feat_type,emg_scale=emg_scale)
                                    x_test_clean_vae, _ = prd.extract_scale(x_test_clean,scaler,ft=self.feat_type,emg_scale=emg_scale)
                                # not finalized, scale raw data
                                elif self.feat_type == 'raw':
                                    x_test_vae = cp.deepcopy(x_test_noise[:,:,::2,:])/5
                                    x_test_clean_vae = cp.deepcopy(x_test_clean[:,:,::2,:])/5
                                
                                x_test_lda = prd.extract_feats(x_test_noise,ft=self.feat_type,emg_scale=emg_scale)
                                with open(noisefile + '.p','wb') as f:
                                    pickle.dump([x_test_vae, x_test_clean_vae, x_test_lda, y_test_clean],f) 
                            x_temp = np.transpose(x_test_lda.reshape((x_test_lda.shape[0],num_feats,-1)),(0,2,1))[...,np.newaxis]
                            x_test_vae = scaler.transform(x_temp.reshape(x_temp.shape[0]*x_temp.shape[1],-1)).reshape(x_temp.shape)

                            # Reshape for nonconvolutional SAE
                            x_test_dlsae = x_test_vae.reshape(x_test_vae.shape[0],-1)
                            x_test_clean_sae = x_test_clean_vae.reshape(x_test_clean_vae.shape[0],-1)

                            # TEMP - CNN extended
                            x_test_ext = np.concatenate((x_test_vae,x_test_vae[:,:2,...]),axis=1)

                            sae(x_test_dlsae[:1,...].astype('float32'))
                            cnn(x_test_vae[:1,...].astype('float32'))
                            sae.set_weights(sae_w)
                            # sae_enc.set_weights(sae_enc_w)
                            # sae_clf.set_weights(sae_clf_w)

                            cnn.set_weights(cnn_w)

                            # Align test data for ENC-LDA
                            # _, x_test_svae = svae_clf.predict(x_test_vae)
                            x_test_sae = sae.enc(x_test_dlsae.astype('float32')).numpy()
                            x_test_cnn = cnn.enc(x_test_vae.astype('float32')).numpy()
                            # x_test_sae = sae_enc.predict(x_test_dlsae)
                            # x_test_cnn = cnn_enc.predict(x_test_vae)
                            # _,_,_,x_test_ecnn = ecnn_enc.predict(x_test_vae)
                            # _, _, x_test_vcnn = vcnn_enc.predict(x_test_vae)

                            y_test_aligned = np.argmax(y_test_clean, axis=1)[...,np.newaxis]

                            # Non NN methods
                            y_test_lda = np.argmax(y_test_clean, axis=1)[...,np.newaxis]

                            y_test_ch = y_test_lda[:y_test_lda.shape[0]//2,...]
                            
                            # Compile models and test data into lists
                            dl_mods = 5
                            align_mods = 5
                            lda_mods = 2
                            qda_mods = 2
                            # mods_all = [svae,sae,cnn,vcnn,ecnn,[w_svae,c_svae],[w_sae,c_sae],[w_cnn,c_cnn],[w_vcnn,c_vcnn],[w_ecnn,c_ecnn],[w,c],[w_noise,c_noise],qda,qda_noise,[mu, C, self.n_test,emg_scale]]
                            mods_all = [cnn,sae,cnn,cnn,cnn,[w_cnn,c_cnn],[w_sae,c_sae],[w_cnn,c_cnn],[w_cnn,c_cnn],[w_cnn,c_cnn],[w,c],[w_noise,c_noise],[w,c],[w,c],[mu, C, self.n_test, emg_scale]]
                            # x_test_all = ['x_test_vae', 'x_test_dlsae', 'x_test_vae', 'x_test_vae', 'x_test_vae','x_test_svae', 'x_test_sae', 'x_test_cnn', 'x_test_vcnn', 'x_test_ecnn', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test']
                            x_test_all = ['x_test_vae', 'x_test_dlsae', 'x_test_vae', 'x_test_vae', 'x_test_vae','x_test_cnn', 'x_test_sae', 'x_test_cnn', 'x_test_cnn', 'x_test_cnn', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test']
                            y_test_all = np.append(np.append(np.append(np.full(dl_mods,'y_test_clean'), np.full(align_mods, 'y_test_aligned')), np.full(lda_mods+qda_mods, 'y_test_lda')),np.full(1,'y_test_ch'))
                            mods_type =  np.append(np.append(np.append(np.full(dl_mods,'dl'),np.full(align_mods+lda_mods,'lda')),np.full(qda_mods,'lda')), np.full(1,'lda_ch'))

                            ## for timing
                            sae_start = time.time()
                            x_temp_sae = sae_enc(x_test_dlsae[[0],...].astype('float32')).numpy()
                            print(x_temp_sae)
                            # temp_out = predict(x_temp_sae, w_sae, c_sae)
                            sae_time[sub-1] = time.time() - sae_start

                            cnn_start = time.time()
                            x_temp_cnn = cnn_enc(x_test_vae[[0],...].astype('float32')).numpy()
                            # temp_out = predict(x_temp_cnn, w_cnn, c_cnn)
                            cnn_time[sub-1] = time.time() - cnn_start

                            # lda_start = time.time()
                            # temp_out = predict(x_test_lda[[0],...], w, c)
                            lda_temp = timeit.timeit(lambda: predict(x_test_lda[[0],...], w, c),number= 100)
                            print(lda_temp)
                            # print(temp_time)
                            # print(lda_start)
                            # print(time.time())
                            lda_time[sub-1] = lda_temp

                            aug_start = time.time()
                            temp_out = predict(x_test_lda[[0],...], w_noise, c_noise)
                            aug_time[sub-1] = time.time() - aug_start
                            
                            # need to figure out what n_test = 0 is
                            if self.n_test == 0:
                                max_i = len(mods_all) - 1
                            else:
                                max_i = len(mods_all)
                            for i in range(max_i):
                                acc_all[sub-1,cv-1,test_scale - 1,i], acc_noise[sub-1,cv-1,test_scale - 1,i], acc_clean[sub-1,cv-1,test_scale - 1,i] = self.eval_mod(eval(x_test_all[i]), eval(y_test_all[i]), clean_size, mods_all[i], mods_type[i])
                            print(acc_all[sub-1,cv-1,test_scale - 1,:])
                        else:
                            acc_all[sub-1,cv-1,test_scale - 1,:], acc_noise[sub-1,cv-1,test_scale - 1,:], acc_clean[sub-1,cv-1,test_scale - 1,:] = np.nan, np.nan, np.nan
                
                # Save subject specific cv results
                subresults = self.create_filename(resultsfolder, cv, sub)
                # print(np.squeeze(acc_noise[sub-1,...]))
                # print(np.squeeze(acc_clean[sub-1,...]))
                print('saving sub results: ' + subresults + '_' + self.n_test + '_subresults.p')
                with open(subresults + '_' + self.n_test + '_subresults.p', 'wb') as f:
                    pickle.dump([acc_all[sub-1,...], acc_noise[sub-1,...], acc_clean[sub-1,...]],f)
        
        # save results for all subjects, need to figure out n_test = 0
        allresults = self.create_filename(resultsfolder, cv, sub, ftype='allresults')
        # print('saving all results: ' + allresults + '_' + self.n_test + '_results.p')
        # with open(allresults + '_' + self.n_test + '_results.p', 'wb') as f:
        #     pickle.dump([acc_all, acc_clean, acc_noise],f)

        # out = {'acc_all':acc_all, 'acc_noise':acc_noise, 'acc_clean':acc_clean}
        # return out , x_test_noise, x_test_clean, y_test_clean
        out = {'sae_time': sae_time, 'cnn_time':cnn_time, 'lda_time': lda_time, 'aug_time': aug_time}

        return out

    def eval_mod(self, x_test, y_test, clean_size, mod, eval_type):
        if clean_size == 0:
            clean_size_2 = x_test.shape[0]
        else:
            clean_size_2 = clean_size

        if eval_type == 'dl':
            _, acc_all = dl.eval_vae(mod, x_test.astype('float32'), y_test.astype('float32'))
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
                acc_noise = eval_lda_ch(mod[0], mod[1], mod[2], x_test, y_test, ft=self.feat_type,emg_scale=mod[3])
            acc_all = 0
            acc_clean = 0

        return acc_all, acc_noise, acc_clean
    
    def reduce_latent(self, raw, params, sub, test_scale=1, traintest='testnoise'):
        num_feat = 4
        # x_train, _, _, _, _, _ = prd.train_data_split(raw,params,sub,self.sub_type,dt=self.dt,load=True,train_grp=self.train_grp)
        # get folder and file names
        foldername = self.create_foldername()
        filename = self.create_filename(foldername, sub = sub)
        print(filename)
        # Load saved data
        with open(filename + '.p', 'rb') as f:
            scaler, _, svae_enc_w, _, svae_clf_w, _, sae_enc_w, _, _, cnn_enc_w, _, _, vcnn_enc_w, _, \
                _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,_ = pickle.load(f)
        with open(filename + '_red.p', 'rb') as f:
            v_svae, v_sae, v_cnn, v_vcnn, _, v, v_noise = pickle.load(f)

        noisefolder = self.create_foldername(ftype=traintest)
        if traintest == 'testnoise':
            noisefolder = 'testdata/' + noisefolder
        else:
            noisefolder = 'noisedata/' + noisefolder
        noisefile = self.create_filename(noisefolder,sub=sub,ftype=traintest,test_scale=1)
        
        print(noisefile)
        if os.path.isfile(noisefile + '.p'):
            print('loading data')
            if traintest == 'testnoise':
                with open(noisefile + '.p','rb') as f:
                    x_test_vae, x_test_clean_vae, x_test_lda, y_test_clean = pickle.load(f) 
                x_test_noise_lda = cp.deepcopy(x_test_lda)
                y_test_noise = cp.deepcopy(y_test_clean)
            else:
                with open(noisefile + '.p','rb') as f:
                    _, x_test_vae, _, _, _, y_train_clean, _, x_test_lda, y_test_clean, x_test_noise_lda, y_test_noise = pickle.load(f)

        x_temp = np.transpose(x_test_noise_lda.reshape((x_test_noise_lda.shape[0],num_feat,-1)),(0,2,1))[...,np.newaxis]
        x_test_vae = scaler.transform(x_temp.reshape(x_temp.shape[0]*x_temp.shape[1],-1)).reshape(x_temp.shape)

        x_test_dlsae = x_test_vae.reshape(x_test_vae.shape[0],-1)
        
        K.clear_session()
        _, sae_enc, _ = dl.build_sae(self.latent_dim, y_test_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)
        _, cnn_enc, _ = dl.build_cnn(self.latent_dim, y_test_clean.shape[1], input_type=self.feat_type, sparse=self.sparsity,lr=self.lr)

        # set weights from trained models
        sae_enc.set_weights(sae_enc_w)
        cnn_enc.set_weights(cnn_enc_w)
        
        # align to latent space
        x_test_sae = sae_enc.predict(x_test_dlsae)
        x_test_cnn = cnn_enc.predict(x_test_vae)

        # reduce training data
        x_test_sae_red = np.matmul(x_test_sae,v_sae)
        x_test_cnn_red = np.matmul(x_test_cnn,v_cnn)

        x_test_lda_red = np.matmul(x_test_lda,v)
        x_test_noise_red = np.matmul(x_test_noise_lda,v_noise)        
        
        if traintest == 'testnoise':
            clean_size = int(x_test_vae.shape[0]/2)  
            y_test_clean = np.argmax(y_test_clean, axis=1)[...,np.newaxis]
            y_test_noise = np.argmax(y_test_noise, axis=1)[...,np.newaxis]  

            # compile results
            out_dict = {'x_test_sae_red':x_test_sae_red,'x_test_cnn_red':x_test_cnn_red, 'x_test_lda_red':x_test_lda_red, 'x_test_noise_red':x_test_noise_red, 'y_test':y_test_clean,'y_test_noise':y_test_clean,'clean_size':clean_size, 'clean_size_lda': clean_size}
        else:

            out_dict =  {'x_test_sae_red':x_test_sae_red,'x_test_cnn_red':x_test_cnn_red, 'x_test_lda_red':x_test_lda_red, 'x_test_noise_red':x_test_noise_red, 'y_test':y_test_clean, 'y_test_noise':y_test_noise,'clean_size_lda':x_test_lda.shape[0],'clean_size':x_test_lda.shape[0]}

        return out_dict

