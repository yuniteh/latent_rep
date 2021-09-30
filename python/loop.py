
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

def loop_cv(raw, params, sub_type, sub = 1, train_grp = 2, dt=0, sparsity=True, load=True, batch_size=32, latent_dim=4, epochs=30,train_scale=5, n_train='gauss',feat_type='feat', noise=True, start_cv = 1, max_cv = 5,lr=0.001, mod = 'all',gens=50):
    np.set_printoptions(precision=3,suppress=True)
    i_tot = 13
    filename = 0
    if dt == 'manual':
        start_cv = 0
        max_cv = 1

    # Set folder
    if dt == 0:
        today = date.today()
        dt = today.strftime("%m%d")
    foldername = 'models' + '_' + str(train_grp) + '_' + dt
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # initialize final accuracy arrays and potential outputs
    last_acc = np.full([max_cv-1,4], np.nan)
    last_val = np.full([max_cv-1,4], np.nan)
    gen_clf = np.nan

    # index training group and subject
    ind = (params[:,0] == sub) & (params[:,3] == train_grp)

    # Check if training data exists
    if np.sum(ind):
        if mod != 'none':
            if dt == 'cv':
                x_full, x_test, _, p_full, p_test, _ = prd.train_data_split(raw,params,sub,sub_type,dt=dt)
            else:
                x_train, x_test, x_valid, p_train, p_test, p_valid = prd.train_data_split(raw,params,sub,sub_type,dt=dt)

        # loop through cross validation
        for cv in range(start_cv,max_cv):
            filename = foldername + '/' + sub_type + str(sub) + '_' + feat_type + '_dim_' + str(latent_dim) + '_ep_' + str(epochs) + '_bat_' + str(batch_size) + '_' + n_train + '_' + str(train_scale) + '_lr_' + str(int(lr*10000)) 
            if dt == 'cv':
                if mod != 'none':
                    x_valid, p_valid = x_full[p_full[:,6] == cv,...], p_full[p_full[:,6] == cv,...]
                    x_train, p_train = x_full[p_full[:,6] != cv,...], p_full[p_full[:,6] != cv,...]
                filename += '_cv_' + str(cv)

            print('Running sub ' + str(sub) + ', model ' + str(train_grp) + ', latent dim ' + str(latent_dim) + ', cv ' + str(cv))
            if sparsity:
                filename += '_sparse'

            ## TRAIN ##
            # Load saved data
            if load:
                load = True
                with open(filename + '.p', 'rb') as f:
                    scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, w_svae, c_svae, \
                        w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise = pickle.load(f)   

                with open(filename + '_hist.p', 'rb') as f:
                    svae_hist, sae_hist, cnn_hist, vcnn_hist = pickle.load(f)

                try:
                    with open(filename + '_aug.p', 'wb') as f:
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
                x_train_noise, x_train_clean, y_train_clean = prd.add_noise(x_train, p_train, sub, n_train, train_scale)
                x_valid_noise, x_valid_clean, y_valid_clean = prd.add_noise(x_valid, p_valid, sub, n_train, train_scale)
                # if not adding noise, copy clean training data
                if not noise:
                    x_train_noise = cp.deepcopy(x_train_clean)
                    x_valid_noise = cp.deepcopy(x_valid_clean)

                # shuffle data to make even batches
                x_train_noise, x_train_clean, y_train_clean = shuffle(x_train_noise, x_train_clean, y_train_clean, random_state = 0)

                # Build models
                svae, svae_enc, svae_dec, svae_clf = dl.build_svae_manual(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity,lr=lr)
                sae, sae_enc, sae_clf = dl.build_sae(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity,lr=lr)
                cnn, cnn_enc, cnn_clf = dl.build_cnn(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity,lr=lr)
                vcnn, vcnn_enc, vcnn_clf = dl.build_vcnn(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity,lr=lr)

                # Training data for LDA/QDA
                x_train_lda = prd.extract_feats(x_train)
                y_train_lda = y_train[...,np.newaxis] - 1
                x_train_lda2 = prd.extract_feats(x_train_noise)
                y_train_lda2 = np.argmax(y_train_clean, axis=1)[...,np.newaxis]

                # Shape data based on feature type
                if feat_type == 'feat':
                    # extract features from training data
                    x_train_noise_temp = np.transpose(prd.extract_feats(x_train_noise).reshape((x_train_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_train_clean_temp = np.transpose(prd.extract_feats(x_train_clean).reshape((x_train_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    
                    # scale features, only fit new scaler if not loading from old model
                    if load:
                        x_train_noise_vae = scaler.transform(x_train_noise_temp.reshape(x_train_noise_temp.shape[0]*x_train_noise_temp.shape[1],-1)).reshape(x_train_noise_temp.shape)
                    else:
                        x_train_noise_vae = scaler.fit_transform(x_train_noise_temp.reshape(x_train_noise_temp.shape[0]*x_train_noise_temp.shape[1],-1)).reshape(x_train_noise_temp.shape)
                    
                    x_train_vae = scaler.transform(x_train_clean_temp.reshape(x_train_clean_temp.shape[0]*x_train_clean_temp.shape[1],-1)).reshape(x_train_clean_temp.shape)

                    # repeat for validation data
                    x_valid_noise_temp = np.transpose(prd.extract_feats(x_valid_noise).reshape((x_valid_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_valid_clean_temp = np.transpose(prd.extract_feats(x_valid_clean).reshape((x_valid_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_valid_noise_vae = scaler.transform(x_valid_noise_temp.reshape(x_valid_noise_temp.shape[0]*x_valid_noise_temp.shape[1],-1)).reshape(x_valid_noise_temp.shape)
                    x_valid_vae = scaler.transform(x_valid_clean_temp.reshape(x_valid_clean_temp.shape[0]*x_valid_clean_temp.shape[1],-1)).reshape(x_valid_clean_temp.shape)
                elif feat_type == 'raw': # not finalized
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
                n_batches = len(x_train_noise_vae) // batch_size
                # initialize history array
                svae_hist = np.zeros((epochs,14))
                # loop through epochs
                for ep in range(epochs):
                    # set loss weight vector, not finalized
                    if ep < epochs/2: 
                        weight = np.array([[0,2**((ep-(epochs/2))/2)] for _ in range(batch_size)])
                    else:
                        weight = np.array([[1,.5] for _ in range(batch_size)])
                    
                    # get batches for inputs
                    x_train_noise_ep = dl.get_batches(x_train_noise_vae, batch_size)
                    x_train_vae_ep = dl.get_batches(x_train_vae, batch_size)
                    y_train_ep = dl.get_batches(y_train_clean, batch_size)
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
                    # weight = np.array([[rat, (temp_out[2])/temp_out[3]] for _ in range(batch_size)])
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
                vcnn_hist = vcnn.fit(x_train_noise_vae, y_train_clean,epochs=30,validation_data = [x_valid_noise_vae, y_valid_clean],batch_size=batch_size)                
                vcnn_w = vcnn.get_weights()
                vcnn_enc_w = vcnn_enc.get_weights()
                vcnn_clf_w = vcnn_clf.get_weights()
                vcnn_hist = vcnn_hist.history

            if mod == 'all' or any("sae" in s for s in mod):
                sae_hist = sae.fit(x_train_noise_sae, y_train_clean,epochs=30,validation_data = [x_valid_noise_sae, y_valid_clean],batch_size=batch_size)
                sae_w = sae.get_weights()
                sae_enc_w = sae_enc.get_weights()
                sae_clf_w = sae_clf.get_weights()
                sae_hist = sae_hist.history

            if mod == 'all' or any("cnn" in s for s in mod):
                cnn_hist = cnn.fit(x_train_noise_vae, y_train_clean,epochs=30,validation_data = [x_valid_noise_vae, y_valid_clean],batch_size=batch_size)
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
                gen_clf = np.argmax(np.tile(np.eye(y_train_clean.shape[-1]),(gens,1)),axis=1)
                # gen_clf = np.tile(np.eye(y_train_clean.shape[-1]),(gens,1))

                # sample from normal distribution, forward pass through decoder
                latent_in = np.random.normal(0,1,size=(gen_clf.shape[0],latent_dim))
                dec_out = svae_dec.predict([latent_in, gen_clf])

                # concatenate noisy and generated data for augmented training data
                x_train_aug = np.concatenate((x_train_noise_vae, dec_out))
                y_train_all = np.concatenate((np.argmax(y_train_clean, axis=1),gen_clf))[...,np.newaxis]

                # inverse transform augmented training data to train regular LDA
                x_train_aug_lda = scaler.inverse_transform(x_train_aug.reshape(x_train_aug.shape[0]*x_train_aug.shape[1],-1)).reshape(x_train_aug.shape)
                x_train_aug_lda = np.transpose(x_train_aug_lda,(0,2,1,3)).reshape(x_train_aug_lda.shape[0],-1)
                w_gen,c_gen, _, _ = train_lda(x_train_aug_lda,y_train_all)

                # align augmented data
                _, _, _, x_train_aug_align = svae_enc.predict(x_train_aug)

                ## for testing, align using CNN
                # x_train_aug_align = cnn_enc.predict(x_train_aug)

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

def loop_test(raw, params, sub_type, train_grp = 2, dt=0, sparsity=True, load=True, batch_size=32, latent_dim=10, epochs=100,train_scale=5, n_train='gauss', n_test='gauss',feat_type='feat', noise=True, start_cv = 1, max_cv = 5,lr=0.001):
    # set number of models to test
    mod_tot = 14

    # set testing noise type
    noise_type = n_test[4:-1]

    # set number of tests for each noise type
    if noise_type == 'gauss' or noise_type == '60hz':
        test_tot = 5 # noise amplitude (1-5)
    elif noise_type == 'pos':
        test_tot = 4 # number of positions
    elif noise_type == 'flat':
        test_tot = 1

    # set number of cvs
    if dt == 'cv':
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
    if dt == 0:
        today = date.today()
        dt = today.strftime("%m%d")
    foldername = 'models' + '_' + str(train_grp) + '_' + dt
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # loop through subjects
    for sub in range(1,2):#6):#np.max(params[:,0])+1):            
        # index based on training group and subject
        ind = (params[:,0] == sub) & (params[:,3] == train_grp)

        # Check if training data exists
        if np.sum(ind):

            # split data into training, testing, validation sets
            if dt == 'cv':
                x_full, x_test, _, p_full, p_test, _ = prd.train_data_split(raw,params,sub,sub_type,dt=dt)
            else:
                x_train, x_test, x_valid, p_train, p_test, p_valid = prd.train_data_split(raw,params,sub,sub_type,dt=dt)

            # loop through cvs
            for cv in range(start_cv,max_cv):
                filename = foldername + '/' + sub_type + str(sub) + '_' + feat_type + '_dim_' + str(latent_dim) + '_ep_' + str(epochs) + '_bat_' + str(batch_size) + '_' + n_train + '_' + str(train_scale) + '_lr_' + str(int(lr*10000)) 
                if dt == 'cv':
                    x_valid, p_valid = x_full[p_full[:,6] == cv,...], p_full[p_full[:,6] == cv,...]
                    x_train, p_train = x_full[p_full[:,6] != cv,...], p_full[p_full[:,6] != cv,...]
                    filename += '_cv_' + str(cv)

                print('Running sub ' + str(sub) + ', model ' + str(train_grp) + ', latent dim ' + str(latent_dim))
                if sparsity:
                    filename += '_sparse'
                
                # Load saved data
                with open(filename + '.p', 'rb') as f:
                    scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, w_svae, c_svae, \
                        w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise = pickle.load(f)   

                with open(filename + '_aug.p', 'rb') as f:
                    w_rec, c_rec, w_rec_al, c_rec_al, w_gen, c_gen, w_gen_al, c_gen_al = pickle.load(f)

                # Add noise to training data
                y_shape = np.max(p_train[:,4])

                # Build models and set weights
                svae, svae_enc, svae_dec, svae_clf = dl.build_svae_manual(latent_dim, y_shape, input_type=feat_type, sparse=sparsity,lr=lr)
                sae, sae_enc, sae_clf = dl.build_sae(latent_dim, y_shape, input_type=feat_type, sparse=sparsity,lr=lr)
                cnn, cnn_enc, cnn_clf = dl.build_cnn(latent_dim, y_shape, input_type=feat_type, sparse=sparsity,lr=lr)
                vcnn, vcnn_enc, vcnn_clf = dl.build_vcnn(latent_dim, y_shape, input_type=feat_type, sparse=sparsity,lr=lr)

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
                if dt == 'cv':
                    x_test, p_test = x_valid, p_valid

                # loop through test levels
                for test_scale in range(1,test_tot + 1):
                    skip = False
                    # load test data for diff limb positions
                    if noise_type == 'pos':
                        test_grp = int(n_test[-1])
                        _, x_test, _, _, p_test, _ = prd.train_data_split(raw,params,sub,sub_type,dt=dt,train_grp=test_grp)
                        pos_ind = p_test[:,-1] == test_scale
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
                        x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_test, p_test, sub, n_test, test_scale)
                        clean_size = int(np.size(x_test,axis=0))
                        # copy clean data if not using noise
                        if not noise:
                            x_test_noise = cp.deepcopy(x_test_clean)

                    if not skip:
                        # extract and scale features
                        if feat_type == 'feat':
                            x_test_noise_temp = np.transpose(prd.extract_feats(x_test_noise).reshape((x_test_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                            x_test_clean_temp = np.transpose(prd.extract_feats(x_test_clean).reshape((x_test_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                            
                            x_test_vae = scaler.transform(x_test_noise_temp.reshape(x_test_noise_temp.shape[0]*x_test_noise_temp.shape[1],-1)).reshape(x_test_noise_temp.shape)
                            x_test_clean_vae = scaler.transform(x_test_clean_temp.reshape(x_test_clean_temp.shape[0]*x_test_clean_temp.shape[1],-1)).reshape(x_test_clean_temp.shape)
                        # not finalized, scale raw data
                        elif feat_type == 'raw':
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
                        mods_all = [svae,sae,cnn,vcnn,[w_svae,c_svae],[w_sae,c_sae],[w_cnn,c_cnn],[w_vcnn,c_vcnn],[w_gen_al,c_gen_al],[w,c],[w_noise,c_noise],qda,qda_noise,[mu, C, n_test]]
                        x_test_all = ['x_test_vae', 'x_test_dlsae', 'x_test_vae', 'x_test_vae', 'x_test_svae', 'x_test_sae', 'x_test_cnn', 'x_test_vcnn', 'x_test_cnn', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test']
                        y_test_all = np.append(np.append(np.append(np.full(dl_mods,'y_test_clean'), np.full(align_mods, 'y_test_aligned')), np.full(lda_mods+qda_mods, 'y_test_lda')),np.full(1,'y_test_ch'))
                        mods_type =  np.append(np.append(np.append(np.full(dl_mods,'dl'),np.full(align_mods+lda_mods,'lda')),np.full(qda_mods,'qda')), np.full(1,'lda_ch'))
                        
                        # need to figure out what n_test = 0 is
                        if n_test == 0:
                            max_i = len(mods_all) - 1
                        else:
                            max_i = len(mods_all)

                        for i in range(max_i):
                            acc_all[sub-1,cv-1,test_scale - 1,i], acc_noise[sub-1,cv-1,test_scale - 1,i], acc_clean[sub-1,cv-1,test_scale - 1,i] = eval_noise_clean(eval(x_test_all[i]), eval(y_test_all[i]), clean_size, mod=mods_all[i], eval_type=mods_type[i])
                    else:
                        acc_all[sub-1,cv-1,test_scale - 1,:], acc_noise[sub-1,cv-1,test_scale - 1,:], acc_clean[sub-1,cv-1,test_scale - 1,:] = np.nan, np.nan, np.nan
            
            # Save subject specific cv results
            with open(filename + '_cvresults.p', 'wb') as f:
                pickle.dump([acc_all[sub-1,...], acc_noise[sub-1,...], acc_clean[sub-1,...]],f)
    
    # save results for all subjects, need to figure out n_test = 0
    if n_test != 0:
        resultsfile = foldername + '/' + sub_type + '_' + feat_type + '_dim_' + str(latent_dim) + '_ep_' + str(epochs) + '_' + n_train + '_' + str(train_scale) + '_' + n_test
        if sparsity:
            resultsfile = resultsfile + '_sparse'
    else:
        resultsfile = filename

    with open(resultsfile + '_results.p', 'wb') as f:
        pickle.dump([acc_all, acc_clean, acc_noise],f)

    return acc_all, acc_noise, acc_clean, filename

def loop_recon(raw, params, sub_type, train_grp = 2, dt=0, sparsity=True, load=True, batch_size=128, latent_dim=4, epochs=30,train_scale=5, n_train='gauss', n_test='gauss',feat_type='feat', noise=True, start_cv = 1, max_cv = 5,lr=0.001):
    i_tot = 13
    noise_type = n_test[4:-1]

    if noise_type == 'gauss' or noise_type == '60hz':
        test_tot = 5
    elif noise_type == 'pos':
        test_tot = 4
    elif noise_type == 'flat':
        test_tot = 1

    if not noise:
        test_tot = 1

    # Initialize accuracy arrays
    if dt == 'cv':
        cv_tot = 4
    else:
        cv_tot = 1
    
    max_cv = cv_tot + 1
    
    acc_all = np.full([np.max(params[:,0])+1, cv_tot, test_tot, i_tot],np.nan)
    acc_clean = np.full([np.max(params[:,0])+1, cv_tot, test_tot, i_tot],np.nan)
    acc_noise = np.full([np.max(params[:,0])+1, cv_tot, test_tot, i_tot],np.nan)

    filename = 0

    # Set folder
    if dt == 0:
        today = date.today()
        dt = today.strftime("%m%d")
    foldername = 'models' + '_' + str(train_grp) + '_' + dt
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for sub in range(1,2):#np.max(params[:,0])+1):            
        ind = (params[:,0] == sub) & (params[:,3] == train_grp)

        # Check if training data exists
        if np.sum(ind):
            if dt == 'cv':
                x_full, x_test, _, p_full, p_test, _ = prd.train_data_split(raw,params,sub,sub_type,dt=dt)
            else:
                x_train, x_test, x_valid, p_train, p_test, p_valid = prd.train_data_split(raw,params,sub,sub_type,dt=dt)

            for cv in range(1,2):#start_cv,max_cv):
                filename = foldername + '/' + sub_type + str(sub) + '_' + feat_type + '_dim_' + str(latent_dim) + '_ep_' + str(epochs) + '_bat_' + str(batch_size) + '_' + n_train + '_' + str(train_scale) + '_lr_' + str(int(lr*10000)) 
                if dt == 'cv':
                    x_valid, p_valid = x_full[p_full[:,6] == cv,...], p_full[p_full[:,6] == cv,...]
                    x_train, p_train = x_full[p_full[:,6] != cv,...], p_full[p_full[:,6] != cv,...]
                    filename += '_cv_' + str(cv)

                print('Running sub ' + str(sub) + ', model ' + str(train_grp) + ', latent dim ' + str(latent_dim))
                if sparsity:
                    filename += '_sparse'
                
                # Load saved data
                with open(filename + '.p', 'rb') as f:
                    scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, w_svae, c_svae, \
                        w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise, mu, C, qda, qda_noise = pickle.load(f)

                # Add noise to training data
                y_shape = np.max(p_train[:,4])

                # Build VAE
                svae, svae_enc, svae_dec, svae_clf = dl.build_svae_manual(latent_dim, y_shape, input_type=feat_type, sparse=sparsity,lr=lr)

                svae.set_weights(svae_w)
                svae_enc.set_weights(svae_enc_w)
                svae_dec.set_weights(svae_dec_w)
                svae_clf.set_weights(svae_clf_w)

                if dt == 'cv':
                    x_test, p_test = x_valid, p_valid

                for test_scale in range(1,test_tot + 1):
                    skip = False
                    # load test data for diff limb positions
                    if noise_type == 'pos':
                        test_grp = int(n_test[-1])
                        _, x_test, _, _, p_test, _ = prd.train_data_split(raw,params,sub,sub_type,dt=dt,train_grp=test_grp)
                        pos_ind = p_test[:,-1] == test_scale
                        if pos_ind.any():
                            x_test_noise = x_test[pos_ind,...]
                            x_test_clean = x_test[pos_ind,...]
                            y_test_clean = to_categorical(p_test[pos_ind,4]-1)
                            clean_size = 0
                            skip = False
                        else:
                            skip = True                    
                    else:
                        # Add noise and index EMG data
                        x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_test, p_test, sub, n_test, test_scale)
                        clean_size = int(np.size(x_test,axis=0))
                        if not noise:
                            x_test_noise = cp.deepcopy(x_test_clean)

                    if not skip:
                        # Extract features
                        if feat_type == 'feat':
                            x_test_noise_temp = np.transpose(prd.extract_feats(x_test_noise).reshape((x_test_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                            x_test_clean_temp = np.transpose(prd.extract_feats(x_test_clean).reshape((x_test_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                            
                            x_test_vae = scaler.transform(x_test_noise_temp.reshape(x_test_noise_temp.shape[0]*x_test_noise_temp.shape[1],-1)).reshape(x_test_noise_temp.shape)
                            x_test_clean_vae = scaler.transform(x_test_clean_temp.reshape(x_test_clean_temp.shape[0]*x_test_clean_temp.shape[1],-1)).reshape(x_test_clean_temp.shape)
                        elif feat_type == 'raw':
                            x_test_vae = cp.deepcopy(x_test_noise[:,:,::2,:])/5
                            x_test_clean_vae = cp.deepcopy(x_test_clean[:,:,::2,:])/5

                        # Align test data for ENC-LDA
                        _,_, _, x_test_svae = svae_enc.predict(x_test_vae)
                        test_weights = np.array([[1,1] for _ in range(len(x_test_vae))])
                        dec_out, clf_out, enc_out = svae.predict([x_test_vae, test_weights])
                    else:
                        x = 0
    return dec_out, clf_out, enc_out, x_test_clean, x_test_vae, y_test_clean, filename

def loop_noise_old(raw, params, sub_type, train_grp = 2, dt=0, sparsity=True, load=True, batch_size=32, latent_dim=4, epochs=30,train_scale=5, n_train='gauss', n_test='gauss',feat_type='feat', noise=True, start_cv = 1, max_cv = 5, suf =''):
    i_tot = 13
    if n_test == 0:
        noise_type = 'none'
    else:
        noise_type = n_test[4:-1]

    if noise_type == 'gauss' or noise_type == '60hz' or noise_type == 'none':
        test_tot = 5
    elif noise_type == 'pos':
        test_tot = 4
    else:
        test_tot = 4

    cv_tot = 4

    acc_all = np.full([np.max(params[:,0])+1, cv_tot, test_tot, i_tot],np.nan)
    acc_clean = np.full([np.max(params[:,0])+1, cv_tot, test_tot, i_tot],np.nan)
    acc_noise = np.full([np.max(params[:,0])+1, cv_tot, test_tot, i_tot],np.nan)
    filename = 0

    # Set folder
    if dt == 0:
        today = date.today()
        dt = today.strftime("%m%d")
    foldername = 'models' + '_' + str(train_grp) + '_' + dt
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for sub in range(1,np.max(params[:,0])+1):            
        ind = (params[:,0] == sub) & (params[:,3] == train_grp)

        # Check if training data exists
        if np.sum(ind):
            if dt == 'cv':
                x_full, x_test, _, p_full, p_test, _ = prd.train_data_split(raw,params,sub,sub_type,dt=dt)
            else:
                x_train, x_test, x_valid, p_train, p_test, p_valid = prd.train_data_split(raw,params,sub,sub_type,dt=dt)

            for cv in range(start_cv,max_cv):
                filename = foldername + '/' + sub_type + str(sub) + '_' + feat_type + '_dim_' + str(latent_dim) + '_ep_' + str(epochs) + '_bat_' + str(batch_size) + '_' + n_train + '_' + str(train_scale)
                if dt == 'cv':
                    x_valid, p_valid = x_full[p_full[:,6] == cv,...], p_full[p_full[:,6] == cv,...]
                    x_train, p_train = x_full[p_full[:,6] != cv,...], p_full[p_full[:,6] != cv,...]
                    filename += '_cv_' + str(cv)

                print('Running sub ' + str(sub) + ', model ' + str(train_grp) + ', latent dim ' + str(latent_dim))
                if sparsity:
                    filename += '_sparse'
                
                filename += suf

                # Load saved data
                if load:
                    load = True
                    with open(filename + '.p', 'rb') as f:
                        scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, w_svae, c_svae, \
                            w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise, mu, C = pickle.load(f)   
                else:
                    scaler = MinMaxScaler(feature_range=(-1,1))
                    load = False
                # else:
                y_train = p_train[:,4]
                
                x_train_noise, x_train_clean, y_train_clean = prd.add_noise(x_train, p_train, sub, n_train, train_scale)
                x_valid_noise, x_valid_clean, y_valid_clean = prd.add_noise(x_valid, p_valid, sub, n_train, train_scale)
                if not noise:
                    x_train_noise = cp.deepcopy(x_train_clean)
                    x_valid_noise = cp.deepcopy(x_valid_clean)

                x_train_noise, x_train_clean, y_train_clean = shuffle(x_train_noise, x_train_clean, y_train_clean, random_state = 0)

                # Build VAE
                svae, svae_enc, svae_dec, svae_clf = dl.build_svae(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity)
                sae, sae_enc, sae_clf = dl.build_sae(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity)
                cnn, cnn_enc, cnn_clf = dl.build_cnn(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity)
                vcnn, vcnn_enc, vcnn_clf = dl.build_vcnn(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity)

                # Training data for LDA/QDA
                x_train_lda = prd.extract_feats(x_train)
                y_train_lda = y_train[...,np.newaxis] - 1
                x_train_lda2 = prd.extract_feats(x_train_noise)
                y_train_lda2 = np.argmax(y_train_clean, axis=1)[...,np.newaxis]

                # Train QDA
                qda = QDA()
                qda.fit(x_train_lda, np.squeeze(y_train_lda))
                qda_noise = QDA()
                qda_noise.fit(x_train_lda2, np.squeeze(y_train_lda2))

                if not load:
                    if feat_type == 'feat':
                        x_train_noise_temp = np.transpose(prd.extract_feats(x_train_noise).reshape((x_train_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                        x_train_clean_temp = np.transpose(prd.extract_feats(x_train_clean).reshape((x_train_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                        x_train_noise_vae = scaler.fit_transform(x_train_noise_temp.reshape(x_train_noise_temp.shape[0]*x_train_noise_temp.shape[1],-1)).reshape(x_train_noise_temp.shape)
                        
                        x_train_vae = scaler.transform(x_train_clean_temp.reshape(x_train_clean_temp.shape[0]*x_train_clean_temp.shape[1],-1)).reshape(x_train_clean_temp.shape)

                        x_valid_noise_temp = np.transpose(prd.extract_feats(x_valid_noise).reshape((x_valid_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                        x_valid_clean_temp = np.transpose(prd.extract_feats(x_valid_clean).reshape((x_valid_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                        x_valid_noise_vae = scaler.transform(x_valid_noise_temp.reshape(x_valid_noise_temp.shape[0]*x_valid_noise_temp.shape[1],-1)).reshape(x_valid_noise_temp.shape)
                        
                        x_valid_vae = scaler.transform(x_valid_clean_temp.reshape(x_valid_clean_temp.shape[0]*x_valid_clean_temp.shape[1],-1)).reshape(x_valid_clean_temp.shape)
                    elif feat_type == 'raw':
                        x_train_noise_vae = cp.deepcopy(x_train_noise[:,:,::2,:])/5
                        x_train_vae = cp.deepcopy(x_train_clean[:,:,::2,:])/5

                        x_valid_noise_vae = cp.deepcopy(x_valid_noise[:,:,::2,:])/5
                        x_valid_vae = cp.deepcopy(x_valid_clean[:,:,::2,:])/5

                    x_train_noise_sae = x_train_noise_vae.reshape(x_train_noise_vae.shape[0],-1)
                    x_train_sae = x_train_vae.reshape(x_train_vae.shape[0],-1)
                    x_valid_noise_sae = x_valid_noise_vae.reshape(x_valid_noise_vae.shape[0],-1)
                    x_valid_sae = x_valid_vae.reshape(x_valid_vae.shape[0],-1)

                    # Fit NNs and get weights
                    svae_hist = svae.fit(x_train_noise_vae, [x_train_vae,y_train_clean],epochs=epochs,validation_data = [x_valid_noise_vae,[x_valid_vae, y_valid_clean]],batch_size=batch_size)
                    svae_w = svae.get_weights()
                    svae_enc_w = svae_enc.get_weights()
                    svae_dec_w = svae_dec.get_weights()
                    svae_clf_w = svae_clf.get_weights()
                    
                    sae_hist = sae.fit(x_train_noise_sae, y_train_clean,epochs=epochs,validation_data = [x_valid_noise_sae, y_valid_clean],batch_size=batch_size)
                    sae_w = sae.get_weights()
                    sae_enc_w = sae_enc.get_weights()
                    sae_clf_w = sae_clf.get_weights()

                    cnn_hist = cnn.fit(x_train_noise_vae, y_train_clean,epochs=epochs,validation_data = [x_valid_noise_vae, y_valid_clean],batch_size=batch_size)
                    cnn_w = cnn.get_weights()
                    cnn_enc_w = cnn_enc.get_weights()
                    cnn_clf_w = cnn_clf.get_weights()
                    
                    vcnn_hist = vcnn.fit(x_train_noise_vae, y_train_clean,epochs=epochs,validation_data = [x_valid_noise_vae, y_valid_clean],batch_size=batch_size)
                    vcnn_w = vcnn.get_weights()
                    vcnn_enc_w = vcnn_enc.get_weights()
                    vcnn_clf_w = vcnn_clf.get_weights()

                    # Align training data for ENC-LDA
                    _, _, x_train_svae = svae_enc.predict(x_train_noise_vae)
                    x_train_sae = sae_enc.predict(x_train_noise_sae)
                    x_train_cnn = cnn_enc.predict(x_train_noise_vae)
                    _, _, x_train_vcnn = vcnn_enc.predict(x_train_noise_vae)

                    y_train_aligned = np.argmax(y_train_clean, axis=1)[...,np.newaxis]

                    # Train ENC-LDA
                    w_svae, c_svae,_, _ = train_lda(x_train_svae,y_train_aligned)
                    w_sae, c_sae,_, _ = train_lda(x_train_sae,y_train_aligned)
                    w_cnn, c_cnn,_, _ = train_lda(x_train_cnn,y_train_aligned)
                    w_vcnn, c_vcnn, _, _ = train_lda(x_train_vcnn,y_train_aligned)

                    # Train LDA
                    w,c, mu, C = train_lda(x_train_lda,y_train_lda)
                    w_noise,c_noise, _, _ = train_lda(x_train_lda2,y_train_lda2)

                    # Pickle variables
                    with open(filename + '.p', 'wb') as f:
                        pickle.dump([scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, \
                            w_svae, c_svae, w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise, mu, C],f)
                    
                    with open(filename + '_hist.p', 'wb') as f:
                        pickle.dump([svae_hist.history, sae_hist.history, cnn_hist.history, vcnn_hist.history],f)
                else:
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

                for test_scale in range(1,test_tot + 1):
                    if n_test == 0:
                        acc_ind = cv - 1
                    else:
                        acc_ind = test_scale - 1
                    skip = False
                    # load test data for diff limb positions
                    if noise_type == 'pos':
                        test_grp = int(n_test[-1])
                        _, x_test, _, _, p_test, _ = prd.train_data_split(raw,params,sub,sub_type,dt=dt,train_grp=test_grp)
                        pos_ind = p_test[:,-1] == test_scale
                        if pos_ind.any():
                            x_test_noise = x_test[pos_ind,...]
                            x_test_clean = x_test[pos_ind,...]
                            y_test_clean = to_categorical(p_test[pos_ind,4]-1)
                            clean_size = 0
                            skip = False
                        else:
                            skip = True                    
                    else:
                        # Add noise and index EMG data
                        if noise_type == 'none':
                            x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_valid, p_valid, sub, n_test, test_scale)
                            clean_size = int(np.size(x_valid,axis=0))
                        else:
                            x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_test, p_test, sub, n_test, test_scale)
                            clean_size = int(np.size(x_test,axis=0))
                        if not noise:
                            x_test_noise = cp.deepcopy(x_test_clean)

                    if not skip:
                        # Extract features
                        if feat_type == 'feat':
                            x_test_noise_temp = np.transpose(prd.extract_feats(x_test_noise).reshape((x_test_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                            x_test_clean_temp = np.transpose(prd.extract_feats(x_test_clean).reshape((x_test_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                            
                            x_test_vae = scaler.transform(x_test_noise_temp.reshape(x_test_noise_temp.shape[0]*x_test_noise_temp.shape[1],-1)).reshape(x_test_noise_temp.shape)
                            x_test_clean_vae = scaler.transform(x_test_clean_temp.reshape(x_test_clean_temp.shape[0]*x_test_clean_temp.shape[1],-1)).reshape(x_test_clean_temp.shape)
                        
                        elif feat_type == 'raw':
                            x_test_vae = cp.deepcopy(x_test_noise[:,:,::2,:])/5
                            x_test_clean_vae = cp.deepcopy(x_test_clean[:,:,::2,:])/5

                        # Reshape for nonconvolutional SAE
                        x_test_dlsae = x_test_vae.reshape(x_test_vae.shape[0],-1)
                        x_test_clean_sae = x_test_clean_vae.reshape(x_test_clean_vae.shape[0],-1)

                        # Align test data for ENC-LDA
                        _,_, x_test_svae = svae_enc.predict(x_test_vae)
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
                        align_mods = 4
                        lda_mods = 2
                        qda_mods = 2
                        mods_all = [svae,sae,cnn,vcnn,[w_svae,c_svae],[w_sae,c_sae],[w_cnn,c_cnn],[w_vcnn,c_vcnn],[w,c],[w_noise,c_noise],qda,qda_noise,[mu, C, n_test]]
                        x_test_all = ['x_test_vae', 'x_test_dlsae', 'x_test_vae', 'x_test_vae', 'x_test_svae', 'x_test_sae', 'x_test_cnn', 'x_test_vcnn', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test']
                        y_test_all = np.append(np.append(np.append(np.full(dl_mods,'y_test_clean'), np.full(align_mods, 'y_test_aligned')), np.full(lda_mods+qda_mods, 'y_test_lda')),np.full(1,'y_test_ch'))
                        mods_type =  np.append(np.append(np.append(np.full(dl_mods,'dl'),np.full(align_mods+lda_mods,'lda')),np.full(qda_mods,'qda')), np.full(1,'lda_ch'))
                        if n_test == 0:
                            max_i = len(mods_all) - 1
                        else:
                            max_i = len(mods_all)

                        for i in range(max_i):
                            acc_all[sub-1,acc_ind,i], acc_noise[sub-1,acc_ind,i], acc_clean[sub-1,acc_ind,i] = eval_noise_clean(eval(x_test_all[i]), eval(y_test_all[i]), clean_size, mod=mods_all[i], eval_type=mods_type[i])
                    else:
                        acc_all[sub-1,acc_ind,:], acc_noise[sub-1,acc_ind,:], acc_clean[sub-1,acc_ind,:] = np.nan, np.nan, np.nan
            
            # Save sub cv results
            with open(filename + '_cvresults.p', 'wb') as f:
                pickle.dump([acc_all[sub-1,...], acc_noise[sub-1,...], acc_clean[sub-1,...]],f)
    
    if n_test != 0:
        resultsfile = foldername + '/' + sub_type + '_' + feat_type + '_dim_' + str(latent_dim) + '_ep_' + str(epochs) + '_' + n_train + '_' + str(train_scale) + '_' + n_test
        if sparsity:
            resultsfile = resultsfile + '_sparse'
    else:
        resultsfile = filename

    with open(resultsfile + '_results.p', 'wb') as f:
        pickle.dump([acc_all, acc_clean, acc_noise],f)

    return acc_all, acc_noise, acc_clean, filename

def run_loop(raw, params, sub_type, nn='svae', load=True, batch_size=128, latent_dim=3, epochs=30,train_scale=5, test_scale=5, n_train='gauss', n_test='gauss',feat_type='feat'): 

    # initialize variable to collect accuracies
    acc_all = np.zeros([np.max(params[:,0]),4])
    acc_clean = np.zeros([np.max(params[:,0]),4])
    acc_noise = np.zeros([np.max(params[:,0]),4])
    if n_train == 'flat':
        train_scale = 0
    if n_test == 'flat':
        test_scale = 0
    enc_w = 0
    dec_w = 0
    clf_w = 0

    # Loop through subjects
    for sub in range(1,np.max(params[:,0])+1):
        # Loop through training groups
        for train_grp in range(2,3):#np.max(params[:,3])+1):
            ind = (params[:,0] == sub) & (params[:,3] == train_grp)

            # Check if training data exists
            if np.sum(ind):
                print('Running sub ' + str(sub) + ', model ' + str(train_grp))
                # Set folder and file names
                foldername = 'models_' + str(train_grp)
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                filename = foldername + '/' + nn + '_' + feat_type + '_dim_' + str(latent_dim) + '_ep_' + str(epochs) + '_' + n_train + '_' + str(train_scale) + '_' + sub_type

                # Load saved data
                if load:
                    with open(filename + str(sub) + '.p', 'rb') as f:
                        scaler, vae_w, enc_w, dec_w, clf_w, w, c, w_noise, c_noise, w_aligned, c_aligned, x_train, x_test, p_train, p_test = pickle.load(f)
                else:
                    # Split training and testing data
                    x_train, x_test, p_train, p_test = train_test_split(raw[ind,:,:], params[ind,:], test_size = 0.33, stratify=params[ind,4])
                    # # Initialize scaler
                    scaler = MinMaxScaler(feature_range=(-1,1))
                
                # # Get ground truth
                y_train = p_train[:,4]
                y_test = p_test[:,4]

                # # Add noise and index EMG data
                x_train_noise, x_train_clean, y_train_clean = prd.add_noise(x_train, p_train, sub, n_train, train_scale)
                x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_test, p_test, sub, n_test, test_scale)
                clean_size = int(np.size(x_test_clean,axis=0)/(np.size(x_test_clean,axis=1)+1))
                if n_train == 'none':
                    x_train_noise = cp.deepcopy(x_train_clean)
                    x_test_noise = cp.deepcopy(x_test_clean) 

                # Extract and scale features
                if feat_type == 'feat':
                    x_train_noise_temp = np.transpose(prd.extract_feats(x_train_noise).reshape((x_train_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_test_noise_temp = np.transpose(prd.extract_feats(x_test_noise).reshape((x_test_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_train_clean_temp = np.transpose(prd.extract_feats(x_train_clean).reshape((x_train_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_test_clean_temp = np.transpose(prd.extract_feats(x_test_clean).reshape((x_test_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    if load:
                        x_train_noise_vae = scaler.transform(x_train_noise_temp.reshape(x_train_noise_temp.shape[0]*x_train_noise_temp.shape[1],-1)).reshape(x_train_noise_temp.shape)
                    else:
                        x_train_noise_vae = scaler.fit_transform(x_train_noise_temp.reshape(x_train_noise_temp.shape[0]*x_train_noise_temp.shape[1],-1)).reshape(x_train_noise_temp.shape)
                    
                    x_train_vae = scaler.transform(x_train_clean_temp.reshape(x_train_clean_temp.shape[0]*x_train_clean_temp.shape[1],-1)).reshape(x_train_clean_temp.shape)
                    x_test_vae = scaler.transform(x_test_noise_temp.reshape(x_test_noise_temp.shape[0]*x_test_noise_temp.shape[1],-1)).reshape(x_test_noise_temp.shape)
                    x_test_clean_vae = scaler.transform(x_test_clean_temp.reshape(x_test_clean_temp.shape[0]*x_test_clean_temp.shape[1],-1)).reshape(x_test_clean_temp.shape)
                elif feat_type == 'raw':
                    x_train_noise_vae = cp.deepcopy(x_train_noise)/5
                    x_test_vae = cp.deepcopy(x_test_noise)/5
                    x_train_vae = cp.deepcopy(x_train_clean)/5
                    x_test_clean_vae = cp.deepcopy(x_test_clean)/5
        
                # Build VAE
                if nn == 'svae':
                    vae, encoder, decoder,clf = dl.build_svae(latent_dim, y_train_clean.shape[1], input_type=feat_type)
                    y_fit = [x_train_vae,y_train_clean]
                elif nn == 'vae':
                    vae, encoder, decoder = dl.build_vae(latent_dim, input_type=feat_type)
                    y_fit = x_train_vae
                elif nn == 'sae':
                    vae, encoder, clf = dl.build_sae(latent_dim, y_train_clean.shape[1], input_type=feat_type)
                    y_fit = y_train_clean

                # Fit sVAE and get weights
                if not load:
                    vae.fit(x_train_noise_vae, y_fit,epochs=epochs,batch_size=batch_size)
                    vae_w = vae.get_weights()
                    enc_w = encoder.get_weights()
                    if nn != 'sae':
                        dec_w = decoder.get_weights()
                    if nn != 'vae':
                        clf_w = clf.get_weights()

                # Load and set weights
                if load:
                    vae.set_weights(vae_w)
                    encoder.set_weights(enc_w)
                    if nn != 'sae':
                        decoder.set_weights(dec_w)
                    if nn != 'vae':
                        clf.set_weights(clf_w)

                # Test full VAE
                y_pred, acc_all[sub-1,0] = dl.eval_vae(vae, x_test_vae, y_test_clean)
                _, acc_noise[sub-1,0] = dl.eval_vae(vae,x_test_vae[clean_size:,:,:,:], y_test_clean[clean_size:,:])
                _, acc_clean[sub-1,0] = dl.eval_vae(vae,x_test_vae[:clean_size,:,:,:], y_test_clean[:clean_size,:])

                # Test encoder-LDA combo
                _, _, x_train_aligned = encoder.predict(x_train_noise_vae)
                _,_, x_test_aligned = encoder.predict(x_test_vae)
                y_train_aligned = np.argmax(y_train_clean, axis=1)[...,np.newaxis]
                y_test_aligned = np.argmax(y_test_clean, axis=1)[...,np.newaxis]
                w_aligned, c_aligned = train_lda(x_train_aligned,y_train_aligned)
                acc_all[sub-1,1] = eval_lda(w_aligned, c_aligned, x_test_aligned, y_test_aligned)
                acc_noise[sub-1,1] = eval_lda(w_aligned, c_aligned, x_test_aligned[clean_size:,:], y_test_aligned[clean_size:,:])
                acc_clean[sub-1,1] = eval_lda(w_aligned, c_aligned, x_test_aligned[:clean_size,:], y_test_aligned[:clean_size,:])

                # Baseline LDA
                x_train_lda = prd.extract_feats(x_train)
                x_test_lda = prd.extract_feats(x_test_noise)
                y_train_lda = y_train[...,np.newaxis] - 1
                y_test_lda = np.argmax(y_test_clean, axis=1)[...,np.newaxis]
                w,c = train_lda(x_train_lda,y_train_lda)
                acc_all[sub-1,2] = eval_lda(w, c, x_test_lda, y_test_lda)
                acc_noise[sub-1,2] = eval_lda(w, c, x_test_lda[clean_size:,:], y_test_lda[clean_size:,:])
                acc_clean[sub-1,2] = eval_lda(w, c, x_test_lda[:clean_size,:], y_test_lda[:clean_size,:])

                # LDA trained with corrupted data
                x_train_lda2 = prd.extract_feats(x_train_noise)
                y_train_lda2 = np.argmax(y_train_clean, axis=1)[...,np.newaxis]
                w_noise,c_noise = train_lda(x_train_lda2,y_train_lda2)
                acc_all[sub-1,3] = eval_lda(w_noise, c_noise, x_test_lda, y_test_lda)
                acc_noise[sub-1,3] = eval_lda(w_noise, c_noise, x_test_lda[clean_size:,:], y_test_lda[clean_size:,:])
                acc_clean[sub-1,3] = eval_lda(w_noise, c_noise, x_test_lda[:clean_size,:], y_test_lda[:clean_size,:])

                # Pickle variables
                with open(filename + str(sub) + '.p', 'wb') as f:
                    pickle.dump([scaler, vae_w, enc_w, dec_w, clf_w, w, c, w_noise, c_noise, w_aligned, c_aligned, x_train, x_test, p_train, p_test],f)

    return acc_all, acc_noise, acc_clean, filename

def load_models(filename, latent_dim, x_train, x_train_noise, y_train, y_train_clean, feat_type, sparsity, lr):
    # Load saved data
    with open(filename + '.p', 'rb') as f:
        scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_dec_w, vcnn_clf_w, w_svae, c_svae, \
            w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise, mu, C = pickle.load(f)   
    
    # Build VAE
    svae, svae_enc, svae_dec, svae_clf = dl.build_svae_manual(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity, lr=lr)
    sae, sae_enc, sae_clf = dl.build_sae(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity, lr=lr)
    cnn, cnn_enc, cnn_clf = dl.build_cnn(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity, lr=lr)
    vcnn, vcnn_enc, vcnn_dec, vcnn_clf = dl.build_svae(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity, lr=lr)

    # Training data for LDA/QDA
    x_train_lda = prd.extract_feats(x_train)
    y_train_lda = y_train[...,np.newaxis] - 1
    x_train_lda2 = prd.extract_feats(x_train_noise)
    y_train_lda2 = np.argmax(y_train_clean, axis=1)[...,np.newaxis]

    # Train QDA
    qda = QDA()
    qda.fit(x_train_lda, np.squeeze(y_train_lda))
    qda_noise = QDA()
    qda_noise.fit(x_train_lda2, np.squeeze(y_train_lda2))

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
    vcnn_dec.set_weights(vcnn_dec_w)
    vcnn_clf.set_weights(vcnn_clf_w)

    return

def compile_acc(acc_all, acc_noise, acc_clean, results_file, test_scale):
    acc_all = acc_all[~np.all(acc_all == 0, axis=1)]
    acc_clean = acc_clean[~np.all(acc_clean == 0, axis=1)]
    acc_noise = acc_noise[~np.all(acc_noise == 0, axis=1)]

    ave_all = np.mean(acc_all,axis=0)
    ave_clean = np.mean(acc_clean,axis=0)
    ave_noise = np.mean(acc_noise,axis=0)

    # Pickle variables
    with open(results_file + '_' + str(test_scale)  + '_results.p', 'wb') as f:
        pickle.dump([acc_all, acc_clean, acc_noise, ave_all, ave_clean, ave_noise],f)

    return acc_all, acc_clean, acc_noise, ave_all, ave_clean, ave_noise

def loop_sub(raw, params, sub_type, train_grp = 2, dt=0, sparsity=True, load=True, batch_size=128, latent_dim=4, epochs=30,train_scale=5, test_scale=5, n_train='gauss', n_test='gauss',feat_type='feat', noise=True):
    i_tot = 13
    acc_all = np.full([np.max(params[:,0])+1, i_tot],np.nan)
    acc_clean = np.full([np.max(params[:,0])+1, i_tot],np.nan)
    acc_noise = np.full([np.max(params[:,0])+1, i_tot],np.nan)
    filename = 0

    # Set folder
    if dt == 0:
        today = date.today()
        dt = today.strftime("%m%d")
    foldername = 'models' + '_' + str(train_grp) + '_' + dt
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # Loop through subjects
    for sub in range(1,np.max(params[:,0])+1):            
        ind = (params[:,0] == sub) & (params[:,3] == train_grp)

        # Check if training data exists
        if np.sum(ind):
            x_train, x_test, x_valid, p_train, p_test, p_valid = prd.train_data_split(raw,params,sub,sub_type,dt=dt)
            scaler = MinMaxScaler(feature_range=(-1,1))
            print('Running sub ' + str(sub) + ', model ' + str(train_grp) + ', latent dim ' + str(latent_dim))
            filename = foldername + '/' + sub_type + str(sub) + '_' + feat_type + '_dim_' + str(latent_dim) + '_ep_' + str(epochs) + '_' + n_train + '_' + str(train_scale)
            if sparsity:
                filename = filename + '_sparse'
            # Load saved data
            if load:
            # if sub < 13:
                load = True
                with open(filename + '.p', 'rb') as f:
                    scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, w_svae, c_svae, \
                        w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise, mu, C = pickle.load(f)   
            else:
                load = False

            # Get ground truth
            y_train = p_train[:,4]
            y_test = p_test[:,4]

            # Add noise and index EMG data
            x_train_noise, x_train_clean, y_train_clean = prd.add_noise(x_train, p_train, sub, n_train, train_scale)
            x_valid_noise, x_valid_clean, y_valid_clean = prd.add_noise(x_valid, p_valid, sub, n_train, train_scale)
            x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_test, p_test, sub, n_test, test_scale)
            clean_size = int(np.size(x_test,axis=0))
            if not noise:
                x_train_noise = cp.deepcopy(x_train_clean)
                x_valid_noise = cp.deepcopy(x_valid_clean)
                x_test_noise = cp.deepcopy(x_test_clean)

            # Build VAE
            svae, svae_enc, svae_dec, svae_clf = dl.build_svae(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity)
            sae, sae_enc, sae_clf = dl.build_sae(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity)
            cnn, cnn_enc, cnn_clf = dl.build_cnn(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity)
            vcnn, vcnn_enc, vcnn_clf = dl.build_vcnn(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity)

            # Training data for LDA/QDA
            x_train_lda = prd.extract_feats(x_train)
            y_train_lda = y_train[...,np.newaxis] - 1
            x_train_lda2 = prd.extract_feats(x_train_noise)
            y_train_lda2 = np.argmax(y_train_clean, axis=1)[...,np.newaxis]

            # Train QDA
            qda = QDA()
            qda.fit(x_train_lda, np.squeeze(y_train_lda))
            qda_noise = QDA()
            qda_noise.fit(x_train_lda2, np.squeeze(y_train_lda2))

            if not load:
                if feat_type == 'feat':
                    x_train_noise_temp = np.transpose(prd.extract_feats(x_train_noise).reshape((x_train_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_train_clean_temp = np.transpose(prd.extract_feats(x_train_clean).reshape((x_train_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_train_noise_vae = scaler.fit_transform(x_train_noise_temp.reshape(x_train_noise_temp.shape[0]*x_train_noise_temp.shape[1],-1)).reshape(x_train_noise_temp.shape)
                    
                    x_train_vae = scaler.transform(x_train_clean_temp.reshape(x_train_clean_temp.shape[0]*x_train_clean_temp.shape[1],-1)).reshape(x_train_clean_temp.shape)
                    x_train_noise_sae = x_train_noise_vae.reshape(x_train_noise_vae.shape[0],-1)
                    x_train_sae = x_train_vae.reshape(x_train_vae.shape[0],-1)

                    x_valid_noise_temp = np.transpose(prd.extract_feats(x_valid_noise).reshape((x_valid_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_valid_clean_temp = np.transpose(prd.extract_feats(x_valid_clean).reshape((x_valid_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_valid_noise_vae = scaler.transform(x_valid_noise_temp.reshape(x_valid_noise_temp.shape[0]*x_valid_noise_temp.shape[1],-1)).reshape(x_valid_noise_temp.shape)
                    
                    x_valid_vae = scaler.transform(x_valid_clean_temp.reshape(x_valid_clean_temp.shape[0]*x_valid_clean_temp.shape[1],-1)).reshape(x_valid_clean_temp.shape)
                    x_valid_noise_sae = x_valid_noise_vae.reshape(x_valid_noise_vae.shape[0],-1)
                    x_valid_sae = x_valid_vae.reshape(x_valid_vae.shape[0],-1)
                elif feat_type == 'raw':
                    x_train_noise_temp = cp.deepcopy(x_train_noise)/5
                    x_train_clean_temp = cp.deepcopy(x_train_clean)/5

                # Fit NNs and get weights
                svae.fit(x_train_noise_vae, [x_train_vae,y_train_clean],epochs=epochs,validation_data=[x_valid_noise_vae,[x_valid_vae, y_valid_clean]],batch_size=batch_size)
                svae_w = svae.get_weights()
                svae_enc_w = svae_enc.get_weights()
                svae_dec_w = svae_dec.get_weights()
                svae_clf_w = svae_clf.get_weights()
                
                sae.fit(x_train_noise_sae, y_train_clean,epochs=epochs,validation_data=[x_valid_noise_sae,y_valid_clean],batch_size=batch_size)
                sae_w = sae.get_weights()
                sae_enc_w = sae_enc.get_weights()
                sae_clf_w = sae_clf.get_weights()

                cnn.fit(x_train_noise_vae, y_train_clean,epochs=epochs,validation_data=[x_valid_noise_vae,y_valid_clean],batch_size=batch_size)
                cnn_w = cnn.get_weights()
                cnn_enc_w = cnn_enc.get_weights()
                cnn_clf_w = cnn_clf.get_weights()

                vcnn.fit(x_train_noise_vae, y_train_clean,epochs=epochs,validation_data=[x_valid_noise_vae,y_valid_clean],batch_size=batch_size)
                vcnn_w = vcnn.get_weights()
                vcnn_enc_w = vcnn_enc.get_weights()
                vcnn_clf_w = vcnn_clf.get_weights()

                # Align training data for ENC-LDA
                _, _, x_train_svae = svae_enc.predict(x_train_noise_vae)
                x_train_sae = sae_enc.predict(x_train_noise_sae)
                x_train_cnn = cnn_enc.predict(x_train_noise_vae)
                _, _, x_train_vcnn = vcnn_enc.predict(x_train_noise_vae)

                y_train_aligned = np.argmax(y_train_clean, axis=1)[...,np.newaxis]

                # Train ENC-LDA
                w_svae, c_svae, _, _ = train_lda(x_train_svae,y_train_aligned)
                w_sae, c_sae, _, _ = train_lda(x_train_sae,y_train_aligned)
                w_cnn, c_cnn, _, _ = train_lda(x_train_cnn,y_train_aligned)
                w_vcnn, c_vcnn, _, _ = train_lda(x_train_vcnn,y_train_aligned)

                # Train LDA
                w,c, mu, C = train_lda(x_train_lda,y_train_lda)
                w_noise,c_noise, _, _ = train_lda(x_train_lda2,y_train_lda2)

                # Pickle variables
                with open(filename + '.p', 'wb') as f:
                    pickle.dump([scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, \
                        w_svae, c_svae, w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise, mu, C],f)
            else:
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

            # Extract features
            if feat_type == 'feat':
                x_test_noise_temp = np.transpose(prd.extract_feats(x_test_noise).reshape((x_test_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                x_test_clean_temp = np.transpose(prd.extract_feats(x_test_clean).reshape((x_test_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                
                x_test_vae = scaler.transform(x_test_noise_temp.reshape(x_test_noise_temp.shape[0]*x_test_noise_temp.shape[1],-1)).reshape(x_test_noise_temp.shape)
                x_test_clean_vae = scaler.transform(x_test_clean_temp.reshape(x_test_clean_temp.shape[0]*x_test_clean_temp.shape[1],-1)).reshape(x_test_clean_temp.shape)
            
            elif feat_type == 'raw':
                x_test_vae = cp.deepcopy(x_test_noise)/5
                x_test_clean_vae = cp.deepcopy(x_test_clean)/5

            # Reshape for nonconvolutional SAE
            x_test_dlsae = x_test_vae.reshape(x_test_vae.shape[0],-1)
            x_test_clean_sae = x_test_clean_vae.reshape(x_test_clean_vae.shape[0],-1)

            # Align test data for ENC-LDA
            _,_, x_test_svae = svae_enc.predict(x_test_vae)
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
            align_mods = 4
            lda_mods = 2
            qda_mods = 2
            mods_all = [svae,sae,cnn,vcnn,[w_svae,c_svae],[w_sae,c_sae],[w_cnn,c_cnn],[w_vcnn,c_vcnn],[w,c],[w_noise,c_noise],qda,qda_noise,[mu, C, n_test]]
            x_test_all = ['x_test_vae', 'x_test_dlsae', 'x_test_vae', 'x_test_vae', 'x_test_svae', 'x_test_sae', 'x_test_cnn', 'x_test_vcnn', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test_lda', 'x_test']
            y_test_all = np.append(np.append(np.append(np.full(dl_mods,'y_test_clean'), np.full(align_mods, 'y_test_aligned')), np.full(lda_mods+qda_mods, 'y_test_lda')),np.full(1,'y_test_ch'))
            mods_type =  np.append(np.append(np.append(np.full(dl_mods,'dl'),np.full(align_mods+lda_mods,'lda')),np.full(qda_mods,'qda')), np.full(1,'lda_ch'))

            for i in range(0,len(mods_all)):
                acc_all[sub-1,i], acc_noise[sub-1,i], acc_clean[sub-1,i] = eval_noise_clean(eval(x_test_all[i]), eval(y_test_all[i]), clean_size, mod=mods_all[i], eval_type=mods_type[i])

    resultsfile = foldername + '/' + sub_type + '_' + feat_type + '_dim_' + str(latent_dim) + '_ep_' + str(epochs) + '_' + n_train + '_' + str(train_scale) + '_' + n_test + '_' + str(test_scale)
    if sparsity:
        resultsfile = resultsfile + '_sparse'
    with open(resultsfile + '_results.p', 'wb') as f:
        pickle.dump([acc_all, acc_clean, acc_noise],f)

    return acc_all, acc_noise, acc_clean, filename

def loop_alldim(raw, params, sub_type, train_grp = 2, dt=0, sparsity=True, load=True, batch_size=128, latent_dim=3, epochs=30,train_scale=5, test_scale=5, n_train='gauss', n_test='gauss',feat_type='feat', noise=True):
    i_tot = 12
    lat_tot = 8
    sub_all = np.zeros([np.max(params[:,0])+1, lat_tot, i_tot])
    sub_clean = np.zeros([np.max(params[:,0])+1, lat_tot, i_tot])
    sub_noise = np.zeros([np.max(params[:,0])+1, lat_tot, i_tot])
    filename = 0

    # Set folder
    if dt == 0:
        today = date.today()
        dt = today.strftime("%m%d")
    foldername = 'models' + '_' + str(train_grp) + '_' + dt
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for sub in range(1,np.max(params[:,0])+1):            
        acc_all = np.zeros([lat_tot,i_tot])
        acc_clean = np.zeros([lat_tot,i_tot])
        acc_noise = np.zeros([lat_tot,i_tot])
        ind = (params[:,0] == sub) & (params[:,3] == train_grp)

        # Check if training data exists
        if np.sum(ind):
            x_train, x_test, p_train, p_test = prd.train_data_split(raw,params,sub,sub_type,dt=dt)
            for latent_dim in range(1,9):
                latent_i = latent_dim - 1
                scaler = MinMaxScaler(feature_range=(-1,1))
                print('Running sub ' + str(sub) + ', model ' + str(train_grp) + ', latent dim ' + str(latent_dim))
                filename = foldername + '/' + sub_type + str(sub) + '_' + feat_type + '_dim_' + str(latent_dim) + '_ep_' + str(epochs) + '_' + n_train + '_' + str(train_scale)
                if sparsity:
                    filename = filename + '_sparse'
                # if os.path.isfile(filename):
                #     load = 'False'
                # else:
                #     load = 'True'
                # Load saved data
                if load:
                # if latent_dim < 8:
                    load = True
                    with open(filename + '.p', 'rb') as f:
                        scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, w_svae, c_svae, \
                            w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise = pickle.load(f)   
                else:
                    load = False
                # else:

                # Get ground truth
                y_train = p_train[:,4]
                y_test = p_test[:,4]

                # Add noise and index EMG data
                x_train_noise, x_train_clean, y_train_clean = prd.add_noise(x_train, p_train, sub, n_train, train_scale)
                x_test_noise, x_test_clean, y_test_clean = prd.add_noise(x_test, p_test, sub, n_test, test_scale)
                clean_size = int(np.size(x_test_clean,axis=0)/(np.size(x_test_clean,axis=1)+1))
                if not noise:
                    x_train_noise = cp.deepcopy(x_train_clean)
                    x_test_noise = cp.deepcopy(x_test_clean)

                # Extract features
                if feat_type == 'feat':
                    x_train_noise_temp = np.transpose(prd.extract_feats(x_train_noise).reshape((x_train_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_test_noise_temp = np.transpose(prd.extract_feats(x_test_noise).reshape((x_test_noise.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_train_clean_temp = np.transpose(prd.extract_feats(x_train_clean).reshape((x_train_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    x_test_clean_temp = np.transpose(prd.extract_feats(x_test_clean).reshape((x_test_clean.shape[0],4,-1)),(0,2,1))[...,np.newaxis]
                    if load:
                        x_train_noise_vae = scaler.transform(x_train_noise_temp.reshape(x_train_noise_temp.shape[0]*x_train_noise_temp.shape[1],-1)).reshape(x_train_noise_temp.shape)
                    else:
                        x_train_noise_vae = scaler.fit_transform(x_train_noise_temp.reshape(x_train_noise_temp.shape[0]*x_train_noise_temp.shape[1],-1)).reshape(x_train_noise_temp.shape)
                    
                    x_train_vae = scaler.transform(x_train_clean_temp.reshape(x_train_clean_temp.shape[0]*x_train_clean_temp.shape[1],-1)).reshape(x_train_clean_temp.shape)
                    x_test_vae = scaler.transform(x_test_noise_temp.reshape(x_test_noise_temp.shape[0]*x_test_noise_temp.shape[1],-1)).reshape(x_test_noise_temp.shape)
                    x_test_clean_vae = scaler.transform(x_test_clean_temp.reshape(x_test_clean_temp.shape[0]*x_test_clean_temp.shape[1],-1)).reshape(x_test_clean_temp.shape)
                    
                    # Reshape for nonconvolutional SAE
                    x_train_noise_sae = x_train_noise_vae.reshape(x_train_noise_vae.shape[0],-1)
                    x_train_sae = x_train_vae.reshape(x_train_vae.shape[0],-1)
                    x_test_sae = x_test_vae.reshape(x_test_vae.shape[0],-1)
                    x_test_clean_sae = x_test_clean_vae.reshape(x_test_clean_vae.shape[0],-1)
                elif feat_type == 'raw':
                    x_train_noise_temp = cp.deepcopy(x_train_noise)/5
                    x_test_noise_temp = cp.deepcopy(x_test_noise)/5
                    x_train_clean_temp = cp.deepcopy(x_train_clean)/5
                    x_test_clean_temp = cp.deepcopy(x_test_clean)/5

                # Build VAE
                svae, svae_enc, svae_dec, svae_clf = dl.build_svae(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity)
                sae, sae_enc, sae_clf = dl.build_sae(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity)
                cnn, cnn_enc, cnn_clf = dl.build_cnn(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity)
                vcnn, vcnn_enc, vcnn_clf = dl.build_vcnn(latent_dim, y_train_clean.shape[1], input_type=feat_type, sparse=sparsity)

                # Fit sVAE and get weights
                if not load:
                    svae.fit(x_train_noise_vae, [x_train_vae,y_train_clean],epochs=epochs,batch_size=batch_size)
                    svae_w = svae.get_weights()
                    svae_enc_w = svae_enc.get_weights()
                    svae_dec_w = svae_dec.get_weights()
                    svae_clf_w = svae_clf.get_weights()
                    
                    sae.fit(x_train_noise_sae, y_train_clean,epochs=epochs,batch_size=batch_size)
                    sae_w = sae.get_weights()
                    sae_enc_w = sae_enc.get_weights()
                    sae_clf_w = sae_clf.get_weights()

                    cnn.fit(x_train_noise_vae, y_train_clean,epochs=epochs,batch_size=batch_size)
                    cnn_w = cnn.get_weights()
                    cnn_enc_w = cnn_enc.get_weights()
                    cnn_clf_w = cnn_clf.get_weights()

                    vcnn.fit(x_train_noise_vae, y_train_clean,epochs=epochs,batch_size=batch_size)
                    vcnn_w = vcnn.get_weights()
                    vcnn_enc_w = vcnn_enc.get_weights()
                    vcnn_clf_w = vcnn_clf.get_weights()

                # Load and set weights
                if load:
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
                    # vcnn.fit(x_train_noise_vae, y_train_clean,epochs=epochs,batch_size=batch_size)
                    # vcnn_w = vcnn.get_weights()
                    # vcnn_enc_w = vcnn_enc.get_weights()
                    # vcnn_clf_w = vcnn_clf.get_weights()

                i = 0
                # Test full VAE
                y_pred, acc_all[latent_i,i] = dl.eval_vae(svae, x_test_vae, y_test_clean)
                _, acc_noise[latent_i,i] = dl.eval_vae(svae,x_test_vae[clean_size:,...], y_test_clean[clean_size:,:])
                _, acc_clean[latent_i,i] = dl.eval_vae(svae,x_test_vae[:clean_size,...], y_test_clean[:clean_size,:])
                i += 1

                y_pred, acc_all[latent_i,i] = dl.eval_vae(sae, x_test_sae, y_test_clean)
                _, acc_noise[latent_i,i] = dl.eval_vae(sae,x_test_sae[clean_size:,...], y_test_clean[clean_size:,:])
                _, acc_clean[latent_i,i] = dl.eval_vae(sae,x_test_sae[:clean_size,...], y_test_clean[:clean_size,:])
                i += 1

                y_pred, acc_all[latent_i,i] = dl.eval_vae(cnn, x_test_vae, y_test_clean)
                _, acc_noise[latent_i,i] = dl.eval_vae(cnn,x_test_vae[clean_size:,...], y_test_clean[clean_size:,:])
                _, acc_clean[latent_i,i] = dl.eval_vae(cnn,x_test_vae[:clean_size,...], y_test_clean[:clean_size,:])
                i += 1

                y_pred, acc_all[latent_i,i] = dl.eval_vae(vcnn, x_test_vae, y_test_clean)
                _, acc_noise[latent_i,i] = dl.eval_vae(vcnn,x_test_vae[clean_size:,...], y_test_clean[clean_size:,:])
                _, acc_clean[latent_i,i] = dl.eval_vae(vcnn,x_test_vae[:clean_size,...], y_test_clean[:clean_size,:])
                i += 1

                # Test encoder-LDA combo
                _, _, x_train_svae = svae_enc.predict(x_train_noise_vae)
                _,_, x_test_svae = svae_enc.predict(x_test_vae)
                x_train_sae = sae_enc.predict(x_train_noise_sae)
                x_test_sae = sae_enc.predict(x_test_sae)
                x_train_cnn = cnn_enc.predict(x_train_noise_vae)
                x_test_cnn = cnn_enc.predict(x_test_vae)
                _, _, x_train_vcnn = vcnn_enc.predict(x_train_noise_vae)
                _, _, x_test_vcnn = vcnn_enc.predict(x_test_vae)

                y_train_aligned = np.argmax(y_train_clean, axis=1)[...,np.newaxis]
                y_test_aligned = np.argmax(y_test_clean, axis=1)[...,np.newaxis]
                w_svae, c_svae = train_lda(x_train_svae,y_train_aligned)
                acc_all[latent_i,i] = eval_lda(w_svae, c_svae, x_test_svae, y_test_aligned)
                acc_noise[latent_i,i] = eval_lda(w_svae, c_svae, x_test_svae[clean_size:,:], y_test_aligned[clean_size:,:])
                acc_clean[latent_i,i] = eval_lda(w_svae, c_svae, x_test_svae[:clean_size,:], y_test_aligned[:clean_size,:])
                i += 1

                w_sae, c_sae = train_lda(x_train_sae,y_train_aligned)
                acc_all[latent_i,i] = eval_lda(w_sae, c_sae, x_test_sae, y_test_aligned)
                acc_noise[latent_i,i] = eval_lda(w_sae, c_sae, x_test_sae[clean_size:,:], y_test_aligned[clean_size:,:])
                acc_clean[latent_i,i] = eval_lda(w_sae, c_sae, x_test_sae[:clean_size,:], y_test_aligned[:clean_size,:])
                i += 1

                w_cnn, c_cnn = train_lda(x_train_cnn,y_train_aligned)
                acc_all[latent_i,i] = eval_lda(w_cnn, c_cnn, x_test_cnn, y_test_aligned)
                acc_noise[latent_i,i] = eval_lda(w_cnn, c_cnn, x_test_cnn[clean_size:,:], y_test_aligned[clean_size:,:])
                acc_clean[latent_i,i] = eval_lda(w_cnn, c_cnn, x_test_cnn[:clean_size,:], y_test_aligned[:clean_size,:])
                i += 1

                w_vcnn, c_vcnn = train_lda(x_train_vcnn,y_train_aligned)
                acc_all[latent_i,i] = eval_lda(w_vcnn, c_vcnn, x_test_vcnn, y_test_aligned)
                acc_noise[latent_i,i] = eval_lda(w_vcnn, c_vcnn, x_test_vcnn[clean_size:,:], y_test_aligned[clean_size:,:])
                acc_clean[latent_i,i] = eval_lda(w_vcnn, c_vcnn, x_test_vcnn[:clean_size,:], y_test_aligned[:clean_size,:])
                i += 1

                # Baseline LDA
                x_train_lda = prd.extract_feats(x_train)
                x_test_lda = prd.extract_feats(x_test_noise)
                y_train_lda = y_train[...,np.newaxis] - 1
                y_test_lda = np.argmax(y_test_clean, axis=1)[...,np.newaxis]
                w,c = train_lda(x_train_lda,y_train_lda)
                acc_all[latent_i,i] = eval_lda(w, c, x_test_lda, y_test_lda)
                acc_noise[latent_i,i] = eval_lda(w, c, x_test_lda[clean_size:,:], y_test_lda[clean_size:,:])
                acc_clean[latent_i,i] = eval_lda(w, c, x_test_lda[:clean_size,:], y_test_lda[:clean_size,:])
                i += 1

                # LDA trained with corrupted data
                x_train_lda2 = prd.extract_feats(x_train_noise)
                y_train_lda2 = np.argmax(y_train_clean, axis=1)[...,np.newaxis]
                w_noise,c_noise = train_lda(x_train_lda2,y_train_lda2)
                acc_all[latent_i,i] = eval_lda(w_noise, c_noise, x_test_lda, y_test_lda)
                acc_noise[latent_i,i] = eval_lda(w_noise, c_noise, x_test_lda[clean_size:,:], y_test_lda[clean_size:,:])
                acc_clean[latent_i,i] = eval_lda(w_noise, c_noise, x_test_lda[:clean_size,:], y_test_lda[:clean_size,:])
                i += 1

                # QDA trained with clean data
                qda = QDA()
                qda.fit(x_train_lda, np.squeeze(y_train_lda))
                acc_all[latent_i,i] = qda.score(x_test_lda,np.squeeze(y_test_lda))
                acc_noise[latent_i,i] = qda.score(x_test_lda[clean_size:,:],np.squeeze(y_test_lda[clean_size:,:]))
                acc_clean[latent_i,i] = qda.score(x_test_lda[:clean_size,:],np.squeeze(y_test_lda[:clean_size,:]))
                i += 1

                # QDA trained with corrupted data
                qda_noise = QDA()
                qda_noise.fit(x_train_lda2, np.squeeze(y_train_lda2))
                acc_all[latent_i,i] = qda_noise.score(x_test_lda,np.squeeze(y_test_lda))
                acc_noise[latent_i,i] = qda_noise.score(x_test_lda[clean_size:,:],np.squeeze(y_test_lda[clean_size:,:]))
                acc_clean[latent_i,i] = qda_noise.score(x_test_lda[:clean_size,:],np.squeeze(y_test_lda[:clean_size,:]))

                # Pickle variables
                with open(filename + '.p', 'wb') as f:
                    pickle.dump([scaler, svae_w, svae_enc_w, svae_dec_w, svae_clf_w, sae_w, sae_enc_w, sae_clf_w, cnn_w, cnn_enc_w, cnn_clf_w, vcnn_w, vcnn_enc_w, vcnn_clf_w, \
                        w_svae, c_svae, w_sae, c_sae, w_cnn, c_cnn, w_vcnn, c_vcnn, w, c, w_noise, c_noise],f)
            resultsfile = foldername + '/' + sub_type + str(sub) + '_' + feat_type + '_ep_' + str(epochs) + '_' + n_train + '_' + str(train_scale) + '_' + n_test + '_' + str(test_scale)
            if sparsity:
                resultsfile = resultsfile + '_sparse'
            with open(resultsfile + '_results.p', 'wb') as f:
                pickle.dump([acc_all, acc_clean, acc_noise],f)

            sub_all[sub-1,:,:] = acc_all
            sub_noise[sub-1,:,:] = acc_noise
            sub_clean[sub-1,:,:] = acc_clean

    return sub_all, sub_noise, sub_clean, filename

def eval_noise_clean(x_test, y_test, clean_size, mod=0, eval_type='dl'):

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

def ave_results(params, sub_type, train_grp=2, dt=0, feat_type='feat',epochs=30,n_train='gaussflat',train_scale=3,n_test='gauss',test_scale=1,sparsity=True, mod_tot=12, dim_tot=8, latent_dim=4,loop_i='noise'):
    if dt == 0:
        today = date.today()
        dt = today.strftime("%m%d")
    foldername = 'models' + '_' + str(train_grp) + '_' + dt

    if n_train == 'flat':
        train_scale = 0
    
    if loop_i == 'noise':
        ax = 0
        resultsfile = foldername + '/' + sub_type + '_' + feat_type + '_dim_' + str(latent_dim) + '_ep_' + str(epochs) + '_' + n_train + '_' + str(train_scale) + '_' + n_test
        if sparsity:
             resultsfile += '_sparse'
        with open(resultsfile + '_results.p', 'rb') as f:
            sub_all, sub_clean, sub_noise = pickle.load(f)
    else:
        if dim_tot < 0:
            i_start = 1
            i_tot = np.abs(dim_tot)+1
            sub_all = np.full([i_tot-1,np.max(params[:,0])+1, mod_tot],np.nan)
            sub_clean = np.full([i_tot-1,np.max(params[:,0])+1, mod_tot],np.nan)
            sub_noise = np.full([i_tot-1,np.max(params[:,0])+1, mod_tot],np.nan)
            ax = 1
        elif dim_tot == 0:
            i_tot = 1
            i_start = 0
            sub_all = np.full([1,np.max(params[:,0])+1, mod_tot],np.nan)
            sub_clean = np.full([1,np.max(params[:,0])+1, mod_tot],np.nan)
            sub_noise = np.full([1,np.max(params[:,0])+1, mod_tot],np.nan)
            ax = 1
        else:
            i_start = 1
            sub_all = np.full([np.max(params[:,0])+1, dim_tot, mod_tot],np.nan)
            sub_clean = np.full([np.max(params[:,0])+1, dim_tot, mod_tot],np.nan)
            sub_noise = np.full([np.max(params[:,0])+1, dim_tot, mod_tot],np.nan)
            i_tot = np.max(params[:,0])+1
            ax = 0
        for i in range(i_start,i_tot): 
            if dim_tot <= 0:
                filename = foldername + '/' + sub_type + '_' + feat_type + '_dim_' + str(latent_dim) + '_ep_' + str(epochs) + '_' + n_train + '_' + str(train_scale) + '_' + n_test + '_' + str(i)
            else:
                filename = foldername + '/' + sub_type + str(i) + '_' + feat_type + '_ep_' + str(epochs) + '_' + n_train + '_' + str(train_scale) + '_' + n_test + '_' + str(test_scale)
            if sparsity:
                filename = filename + '_sparse'     
            results_file = filename + '_results.p'
            if os.path.isfile(results_file):
                with open(results_file, 'rb') as f:
                    acc_all, acc_clean, acc_noise = pickle.load(f)
                sub_all[i-i_start,...] = acc_all
                sub_noise[i-i_start,...] = acc_noise
                sub_clean[i-i_start,...] = acc_clean

    ave_all = np.nanmean(sub_all,axis=ax)
    ave_noise = np.nanmean(sub_noise,axis=ax)
    ave_clean = np.nanmean(sub_clean,axis=ax)

    if ave_all.ndim > 2:
        ave_all = np.nanmean(ave_all,axis=ax)
        ave_noise = np.nanmean(ave_noise,axis=ax)
        ave_clean = np.nanmean(ave_clean,axis=ax)
    return sub_all, sub_noise, sub_clean, ave_all, ave_noise, ave_clean
    