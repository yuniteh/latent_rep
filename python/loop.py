
import tensorflow as tf
import tensorflow.keras
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lda import train_lda, predict, eval_lda
import sVAE_utils as svae
import process_data as prd
import copy as cp

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
                    vae, encoder, decoder,clf = svae.build_svae(latent_dim, y_train_clean.shape[1], input_type=feat_type)
                    y_fit = [x_train_vae,y_train_clean]
                elif nn == 'vae':
                    vae, encoder, decoder = svae.build_vae(latent_dim, input_type=feat_type)
                    y_fit = x_train_vae
                elif nn == 'sae':
                    vae, encoder, clf = svae.build_sae(latent_dim, y_train_clean.shape[1], input_type=feat_type)
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
                y_pred, acc_all[sub-1,0] = svae.eval_vae(vae, x_test_vae, y_test_clean)
                _, acc_noise[sub-1,0] = svae.eval_vae(vae,x_test_vae[clean_size:,:,:,:], y_test_clean[clean_size:,:])
                _, acc_clean[sub-1,0] = svae.eval_vae(vae,x_test_vae[:clean_size,:,:,:], y_test_clean[:clean_size,:])

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
