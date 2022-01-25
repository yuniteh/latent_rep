import numpy as np
import scipy.io 
import pandas as pd
import copy as cp
import pickle
import os
from datetime import date
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from itertools import combinations, product
import time
import json
import pickle
import tensorflow as tf
from scipy.fftpack import fft, ifft


def load_raw(filename):
    struct = scipy.io.loadmat(filename)
    daq = struct['data'][0,0]['daq']['DAQ_DATA'][0,0]
    pvd = struct['data'][0,0]['pvd']

    return daq,pvd

# params structure: [subject ID, Iter, Index, Training group, DOF, Pos]
def load_train_data(filename):
    struct = scipy.io.loadmat(filename)
    feat = struct['feat']
    params = struct['params']
    daq = struct['daq']

    return feat,params,daq

def process_daq(daq,params,win=200,ch=6):
    trial_data = np.zeros((win,6,params.shape[0]))
    for trial in range(0,params.shape[0]-1):
        sub = params[trial,0]
        grp = params[trial,1]
        ind = params[trial,2]
        trial_data[:,:,trial] = daq[sub-1,0][0,grp-1][ind-1:ind+win-1,:]
    
    return trial_data

def process_df(params):
    df = pd.DataFrame(data=params,columns=['sub','trial','ind','group','class','pos'])
    df = df.set_index('sub')
    
    return df

def load_noise_data(filename):
    struct = scipy.io.loadmat(filename)
    raw_noise = struct['raw_win']

    return raw_noise

def process_save_noise(foldername):
    files = os.listdir(foldername)
    raw_all =  np.full([len(files),1000,200],np.nan)
    for i in range(len(files)):
        raw_noise = load_noise_data(foldername + '/' + files[i])
        print(files[i])
        raw_all[i,...] = raw_noise[:1000,:]

    with open(foldername + '/all_real_noise.p', 'wb') as f:
        pickle.dump([raw_all, files],f)
    
    return

def truncate(data):
    data[data > 5] = 5
    data[data < -5] = -5
    return data

def prep_train_data(d, raw, params):
    x_train, _, x_valid, p_train, _, p_valid = train_data_split(raw,params,d.sub,d.sub_type,dt=d.cv_type,load=True,train_grp=d.train_grp)

    emg_scale = np.ones((np.size(x_train,1),1))
    for i in range(np.size(x_train,1)):
        emg_scale[i] = 5/np.max(np.abs(x_train[:,i,:]))
    x_train = x_train*emg_scale
    x_valid = x_valid*emg_scale

    x_train_noise, _, y_train_clean = add_noise(x_train, p_train, d.sub, d.train, d.train_scale)
    x_valid_noise, _, y_valid_clean = add_noise(x_valid, p_valid, d.sub, d.train, d.train_scale)

    # shuffle data to make even batches
    x_train_noise, y_train_clean = shuffle(x_train_noise, y_train_clean, random_state = 0)

    # Extract features
    scaler = MinMaxScaler(feature_range=(0,1))
    x_train_noise_cnn, scaler = extract_scale(x_train_noise,scaler,d.scaler_load,ft=d.feat_type,emg_scale=emg_scale) 
    x_valid_noise_cnn, _ = extract_scale(x_valid_noise,scaler,ft=d.feat_type,emg_scale=emg_scale)
    x_train_noise_cnn = x_train_noise_cnn.astype('float32')
    x_valid_noise_cnn = x_valid_noise_cnn.astype('float32')

    # reshape data for nonconvolutional network
    x_train_noise_mlp = x_train_noise_cnn.reshape(x_train_noise_cnn.shape[0],-1)
    x_valid_noise_mlp = x_valid_noise_cnn.reshape(x_valid_noise_cnn.shape[0],-1)

    # create batches
    trainmlp = tf.data.Dataset.from_tensor_slices((x_train_noise_mlp, y_train_clean)).batch(128)
    validmlp = tf.data.Dataset.from_tensor_slices((x_valid_noise_mlp, y_valid_clean)).batch(128)
    traincnn = tf.data.Dataset.from_tensor_slices((x_train_noise_cnn, y_train_clean)).batch(128)
    validcnn = tf.data.Dataset.from_tensor_slices((x_valid_noise_cnn, y_valid_clean)).batch(128)

    d.emg_scale = emg_scale
    d.scaler = scaler

    return trainmlp, validmlp, traincnn, validcnn, y_train_clean, y_valid_clean, x_train_noise_mlp, x_train_noise_cnn

def prep_test_data(d,raw,params,real_noise_temp):
    _, x_test, _, _, p_test, _ = train_data_split(raw,params,d.sub,d.sub_type,dt=d.cv_type,train_grp=d.test_grp)
    clean_size = int(np.size(x_test,axis=0))
    x_test = x_test*d.emg_scale

    x_test_noise, _, y_test_clean = add_noise(x_test, p_test, d.sub, d.test, 1, real_noise=real_noise_temp, emg_scale = d.emg_scale)
    x_test_cnn, _ = extract_scale(x_test_noise,d.scaler,ft=d.feat_type,emg_scale=d.emg_scale)
    x_test_cnn = x_test_cnn.astype('float32')
    x_test_mlp = x_test_cnn.reshape(x_test_cnn.shape[0],-1)

    return x_test_cnn, x_test_mlp, y_test_clean, clean_size

def sub_train_test(feat,params,sub,train_grp,test_grp):
    # Index EMG data
    x_train, y_train = sub_split(feat, params, sub, train_grp)
    x_test, y_test = sub_split(feat, params, sub, test_grp)

    return x_train, y_train, x_test, y_test

def sub_split(feat, params, sub, grp):
    ind = (params[:,0] == sub) & (params[:,3] == grp)
    
    if np.sum(ind):
        if feat.ndim == 3:
            x, y = shuffle(feat[ind,:,:],to_categorical(params[ind,-2]-1))
            # Add dimension to x data to fit CNN architecture
            x = x[...,np.newaxis]
        else:
            x, y = shuffle(feat[ind,:],params[ind,-2]-1)
            y = y[...,np.newaxis]
    else:
        x = 0
        y = 0
    return x,y

def sub_split_stat(feat, params, sub, grp):
    ind = (params[:,0] == sub) & (params[:,3] == grp) & (params[:,5] == 1)

    if np.sum(ind):
        if feat.ndim == 3:
            x, y = shuffle(feat[ind,:,:],to_categorical(params[ind,-2]-1))
            # Add dimension to x data to fit CNN architecture
            x = x[...,np.newaxis]
        else:
            x, y = shuffle(feat[ind,:],params[ind,-2]-1)
            y = y[...,np.newaxis]
    else:
        x = 0
        y = 0
    return x,y

def sub_split_loo(feat, params, sub, grp):
    ind = (params[:,0] == sub) & (params[:,3] == 4)
    
    if np.sum(ind):
        if feat.ndim == 3:
            x, y = shuffle(feat[ind,:,:],to_categorical(params[ind,-2]-1))
            # Add dimension to x data to fit CNN architecture
            x = x[...,np.newaxis]
        else:
            x, y = shuffle(feat[ind,:],params[ind,-2]-1)
            y = y[...,np.newaxis]
    else:
        x = 0
        y = 0
    return x,y

def train_data_split(raw, params, sub, sub_type, dt=0, train_grp=2, load=True, test_i = 5, valid_i = 5):
    if dt == 0:
        today = date.today()
        dt = today.strftime("%m%d")
    # foldername = 'models' + '_' + str(train_grp) + '_' + dt
    foldername = 'traindata_' + dt
    filename = foldername + '/' + sub_type + str(sub) + '_traindata_' + str(train_grp)  + '.p'
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    
    if load:
        if os.path.isfile(filename):
            print('Loading training data: ' + filename)
            with open(filename,'rb') as f:
                x_train, x_test, x_valid, p_train, p_test, p_valid = pickle.load(f)
        else:
            load=False
    if not load:
        ind = (params[:,0] == sub) & (params[:,3] == train_grp)
        if train_grp > 2:
            x_train, x_valid, p_train, p_valid = 0,0,0,0
            x_test, p_test = raw[ind,:,:], params[ind,:] 
        else:
            if dt == 'cv':
                train_ind = ind & (params[:,6] != test_i)
                test_ind = ind & (params[:,6] == test_i)
                x_train, p_train = raw[train_ind,:,:], params[train_ind,:]
                x_valid, p_valid = raw[train_ind,:,:], params[train_ind,:]
                x_test, p_test = raw[test_ind,:,:], params[test_ind,:]
            elif dt == 'manual':
                train_ind = ind & (params[:,6] != test_i)
                valid_ind = ind & (params[:,6] == valid_i)
                test_ind = ind & (params[:,6] == test_i)
                x_train, p_train = raw[train_ind,:,:], params[train_ind,:]
                x_valid, p_valid = raw[valid_ind,:,:], params[valid_ind,:]
                x_test, p_test = raw[test_ind,:,:], params[test_ind,:]
            elif dt == 'all':
                valid_ind = ind & (params[:,6] == valid_i)
                x_train, p_train = raw[ind,:,:], params[ind,:]
                x_valid, p_valid = raw[valid_ind,:,:], params[valid_ind,:]
                x_test, p_test = raw[valid_ind,:,:], params[valid_ind,:]
            else:
                # Split training and testing data
                x_temp, x_test, p_temp, p_test = train_test_split(raw[ind,:,:], params[ind,:], test_size = 0.2, stratify=params[ind,4], shuffle=True)
                x_train, x_valid, p_train, p_valid = train_test_split(x_temp, p_temp, test_size = 0.33, stratify=p_temp[:,4], shuffle=True)
        
        with open(filename, 'wb') as f:
            pickle.dump([x_train, x_test, x_valid, p_train, p_test, p_valid],f)

    if train_grp < 3:
        x_train, p_train = shuffle(x_train, p_train, random_state = 0)
        x_valid, p_valid = shuffle(x_valid, p_valid, random_state = 0)
    x_test, p_test = shuffle(x_test, p_test, random_state = 0)
    return x_train, x_test, x_valid, p_train, p_test, p_valid

def norm_sub(feat, params):
    for sub in range(1,np.max(params[:,0])):
        ind = params[:,0] == sub
        scaler = MinMaxScaler(feature_range=(-1,1))
        feat_out = feat[ind,:,:]
        feat[ind,:,:] = scaler.fit_transform(feat_out.reshape(feat_out.shape[0],-1)).reshape(feat_out.shape)
    
    return feat

def remove_ch(raw, params, sub, n_type='flat', scale=5):
    # Index subject and training group
    max_ch = raw.shape[1] + 1
    num_ch = int(n_type[-1]) + 1
    out = cp.deepcopy(raw)
    full_type = n_type[0:4]
    noise_type = n_type[4:-1]

    # tile data once for each channel
    if full_type == 'full':
        start_ch = 1
        sub_params = np.tile(params,(num_ch,1))
        orig = np.tile(raw,(num_ch,1,1))
    # tile data twice, once for clean and once for noise
    elif full_type == 'part':
        start_ch = num_ch - 1
        sub_params = np.tile(params,(2,1))
        orig = np.tile(raw,(2,1,1))

    # loop through channel noise
    for num_noise in range(start_ch,num_ch):
        ch_all = list(combinations(range(0,6),num_noise))
        temp = cp.deepcopy(raw)
        if noise_type == 'gaussflat':
            ch_split = temp.shape[0]//(3*len(ch_all))
        else:
            ch_split = temp.shape[0]//len(ch_all)
        for ch in range(0,len(ch_all)):
            for i in ch_all[ch]:
                # if full_type == 'part':
                #     temp = cp.deepcopy(raw)
                # temp[:,i,:] += np.random.normal(0,scale,temp.shape[2])
                if noise_type == 'gaussflat':
                    # if ch == 0:
                    #     print('gaussflat:' + str(ch_split))
                    temp[3*ch*ch_split:(3*ch+1)*ch_split,i,:] = np.nan
                    temp[(3*ch+1)*ch_split:(3*ch+2)*ch_split,i,:] = np.nan
                    temp[(3*ch+2)*ch_split:(3*ch+3)*ch_split,i,:] = np.nan
                elif noise_type == 'gauss':
                    # if ch == 0:
                    #     print('gauss:' + str(ch_split))
                    temp[ch*ch_split:(ch+1)*ch_split,i,:] = np.nan
                    # temp[:,i,:] += np.random.normal(0,scale,temp.shape[2])
                elif noise_type == 'flat':
                    # if ch == 0:
                    #     print('flat:' + str(ch_split))
                    temp[ch*ch_split:(ch+1)*ch_split,i,:] = np.nan
                    # temp[:,i,:] = 0
                # if full_type == 'part':
                #     out = np.concatenate((out,temp))
        # if full_type == 'full':
        out = np.concatenate((out,temp))
        

    noisy, clean, y = out, orig, to_categorical(sub_params[:,-2]-1)

    clean = clean[...,np.newaxis]
    noisy = noisy[...,np.newaxis]
    return noisy,clean,y

def add_noise_all(x_train,x_test,p_train,p_test, sub, sub_type, dt=0, train_grp=2, load=True, cv=1,n_type='fullgaussflat4',scale=5):
    if dt == 0:
        today = date.today()
        dt = today.strftime("%m%d")

    noisefolder = 'noisedata_' + dt
    noisefile = noisefolder + '/' + sub_type + str(sub) + '_grp_' + str(train_grp) + '_' + str(n_type) + '_' + str(scale)
    if not os.path.isdir(noisefolder):
        os.mkdir(noisefolder) 
    if dt == 'cv':
        noisefile += '_cv_' + str(cv)

    if load:
        if os.path.isfile(noisefile + '.p'):
            with open(noisefile + '.p','rb') as f:
                x_train_noise, x_train_clean, y_train_clean, x_test_noise, x_test_clean, y_test_clean = pickle.load(f)
        else:
            load = False
            print('here')
    
    if not load:
        print('hi')
        if dt == 'cv':
            x_full = cp.deepcopy(x_train)
            p_full = cp.deepcopy(p_train)
            x_test, p_test = x_full[p_full[:,6] == cv,...], p_full[p_full[:,6] == cv,...]
            x_train, p_train = x_full[p_full[:,6] != cv,...], p_full[p_full[:,6] != cv,...]
            
        x_train_noise, x_train_clean, y_train_clean = add_noise(x_train, p_train, sub, n_type, scale)
        x_test_noise, x_test_clean, y_test_clean = add_noise(x_test, p_test, sub, n_type, scale)

        with open(noisefile + '.p','wb') as f:
            pickle.dump([x_train_noise, x_train_clean, y_train_clean, x_test_noise, x_test_clean, y_test_clean],f)
    
    return x_train_noise, x_train_clean, y_train_clean, x_test_noise, x_test_clean, y_test_clean

def add_noise(raw, params, sub, n_type='flat', scale=5, real_noise=0,emg_scale=[1,1,1,1,1,1]):
    # Index subject and training group
    max_ch = raw.shape[1] + 1
    num_ch = int(n_type[-1]) + 1
    full_type = n_type[0:4]
    noise_type = n_type[4:-1]

    if noise_type[:3] == 'pos':
        num_ch = int(noise_type[-1]) + 1
        noise_type = noise_type[3:-1]
    
    if full_type == 'full':
        if noise_type == 'gaussflat60hz' or noise_type == 'allmix':
            split = 6
        elif noise_type == 'gauss60hz' or noise_type == '60hzall' or noise_type == 'gaussall' or noise_type == 'testall':
            split = 5
        else:
            split = 3
    else:
        split = 1

    rep = 1

    # tile data once for each channel
    if full_type == 'full':
        if noise_type != '60hzall' or noise_type != 'gaussall':
            rep = 2
        elif noise_type == 'testall':
            rep = 3
        start_ch = 1
        sub_params = np.tile(params,(rep*(num_ch-1)+1,1))
        orig = np.tile(raw,(rep*(num_ch-1)+1,1,1))
    # tile data twice, once for clean and once for noise
    elif full_type == 'part':
        start_ch = num_ch - 1
        sub_params = np.tile(params,(2,1))
        orig = np.tile(raw,(2,1,1))
        
    out = np.array([]).reshape(0,6,200)
    x = np.linspace(0,0.2,200)
    if noise_type == 'realmix':
        real_noise = np.delete(real_noise,(2),axis=0)
        # real_noise = np.delete(real_noise,(1),axis=0)
    elif noise_type == 'realmixnew' or noise_type == 'realmixeven':
        real_noise = np.delete(real_noise,(3),axis=0)
        # real_noise = np.delete(real_noise,(1),axis=0)
        real_type = real_noise.shape[0]

    # repeat twice if adding gauss and flat
    for rep_i in range(rep):   
        # loop through channel noise
        for num_noise in range(start_ch,num_ch):
            # find all combinations of noisy channels
            ch_all = list(combinations(range(0,6),num_noise))
            temp = cp.deepcopy(raw)
            ch_split = temp.shape[0]//(split*len(ch_all))
            
            # loop through all channel combinations
            for ch in range(0,len(ch_all)):
                if noise_type == 'mix' or noise_type == 'allmix':
                    ch_noise = np.random.randint(3,size=(ch_split,num_noise))
                    ch_level = np.random.randint(5,size=(ch_split,num_noise))

                    if num_noise > 1:
                        for i in range(ch_split):
                            while np.array([x == ch_noise[i,0] for x in ch_noise[i,:]]).all() and np.array([x == ch_level[i,0] for x in ch_level[i,:]]).all():
                                ch_noise[i,:] = np.random.randint(3,size = num_noise)
                                ch_level[i,:] = np.random.randint(5,size = num_noise)
                elif noise_type[:4] == 'real':
                    ch_noise = np.random.randint(1000,size=(ch_split,num_noise))
                    ch_level = np.random.randint(real_type,size=(ch_split,num_noise))
                    if noise_type == 'realmix':
                        if num_noise > 1:
                            for i in range(ch_split):
                                while np.array([x == ch_level[i,0] for x in ch_level[i,:]]).all():
                                    ch_level[i,:] = np.random.randint(real_type,size = num_noise)
                    elif noise_type == 'realmixeven':
                        noise_combo = np.array([x for x in product(np.arange(real_type),repeat=num_noise)])
                        rep_noise = ch_split//noise_combo.shape[0]
                        noise_all = np.tile(noise_combo,(rep_noise,1))
                        noise_extra = np.random.randint(real_type,size=(ch_split%noise_combo.shape[0],num_noise))
                        noise_all = np.concatenate((noise_all,noise_extra))
                    else:
                        ch_level = np.random.randint(real_type,size=(ch_split,num_noise))

                ch_ind = 0
                for i in ch_all[ch]:
                    if noise_type == '60hzall':
                        for scale_i in range(5):
                            temp[(5*ch+scale_i)*ch_split:(5*ch+scale_i+1)*ch_split,i,:] += (scale_i+1)*np.sin(2*np.pi*60*x)
                    elif noise_type == 'gaussall':
                        for scale_i in range(5):
                            temp[(5*ch+scale_i)*ch_split:(5*ch+scale_i+1)*ch_split,i,:] += np.random.normal(0,scale_i+1,temp.shape[2])
                    elif noise_type == 'gaussflat':
                        if rep_i == 0:
                            temp[3*ch*ch_split:(3*ch+1)*ch_split,i,:] = 0
                            temp[(3*ch+1)*ch_split:(3*ch+2)*ch_split,i,:] += np.random.normal(0,1,temp.shape[2])
                            temp[(3*ch+2)*ch_split:(3*ch+3)*ch_split,i,:] += np.random.normal(0,2,temp.shape[2])
                        else:
                            temp[3*ch*ch_split:(3*ch+1)*ch_split,i,:] += np.random.normal(0,3,temp.shape[2])
                            temp[(3*ch+1)*ch_split:(3*ch+2)*ch_split,i,:] += np.random.normal(0,4,temp.shape[2])
                            temp[(3*ch+2)*ch_split:(3*ch+3)*ch_split,i,:] += np.random.normal(0,5,temp.shape[2])
                    elif noise_type == 'flat60hz':
                        if rep_i == 0:
                            temp[3*ch*ch_split:(3*ch+1)*ch_split,i,:] = 0
                            temp[(3*ch+1)*ch_split:(3*ch+2)*ch_split,i,:] += np.sin(2*np.pi*60*x)
                            temp[(3*ch+2)*ch_split:(3*ch+3)*ch_split,i,:] += 2*np.sin(2*np.pi*60*x)
                        else:
                            temp[3*ch*ch_split:(3*ch+1)*ch_split,i,:] += 3*np.sin(2*np.pi*60*x)
                            temp[(3*ch+1)*ch_split:(3*ch+2)*ch_split,i,:] += 4*np.sin(2*np.pi*60*x)
                            temp[(3*ch+2)*ch_split:(3*ch+3)*ch_split,i,:] += 5*np.sin(2*np.pi*60*x)
                    elif noise_type == 'gauss60hz':
                        if rep_i == 0:
                            for scale_i in range(5):
                                temp[(5*ch+scale_i)*ch_split:(5*ch+scale_i+1)*ch_split,i,:] += (scale_i+1)*np.sin(2*np.pi*60*x)
                        else:        
                            for scale_i in range(5):
                                temp[(5*ch+scale_i)*ch_split:(5*ch+scale_i+1)*ch_split,i,:] += np.random.normal(0,scale_i+1,temp.shape[2])
                    elif noise_type == 'gaussflat60hz':
                        if rep_i == 0:
                            temp[6*ch*ch_split:(6*ch+2)*ch_split,i,:] = 0
                            temp[(6*ch+2)*ch_split:(6*ch+3)*ch_split,i,:] += np.sin(2*np.pi*60*x)
                            temp[(6*ch+3)*ch_split:(6*ch+4)*ch_split,i,:] += 2*np.sin(2*np.pi*60*x)
                            temp[(6*ch+4)*ch_split:(6*ch+5)*ch_split,i,:] += 3*np.sin(2*np.pi*60*x) 
                            temp[(6*ch+5)*ch_split:(6*ch+6)*ch_split,i,:] += 4*np.sin(2*np.pi*60*x)
                        else:        
                            temp[(6*ch)*ch_split:(6*ch+1)*ch_split,i,:] += 5*np.sin(2*np.pi*60*x)
                            temp[(6*ch+1)*ch_split:(6*ch+2)*ch_split,i,:] += np.random.normal(0,1,temp.shape[2])
                            temp[(6*ch+2)*ch_split:(6*ch+3)*ch_split,i,:] += np.random.normal(0,2,temp.shape[2])
                            temp[(6*ch+3)*ch_split:(6*ch+4)*ch_split,i,:] += np.random.normal(0,3,temp.shape[2])
                            temp[(6*ch+4)*ch_split:(6*ch+5)*ch_split,i,:] += np.random.normal(0,4,temp.shape[2])
                            temp[(6*ch+5)*ch_split:(6*ch+6)*ch_split,i,:] += np.random.normal(0,5,temp.shape[2])                    
                    elif noise_type == 'allmix':
                        if rep_i == 0:
                            temp[6*ch*ch_split:(6*ch+1)*ch_split,i,:] = 0
                            temp[(6*ch+2)*ch_split:(6*ch+3)*ch_split,i,:] += np.sin(2*np.pi*60*x)
                            temp[(6*ch+3)*ch_split:(6*ch+4)*ch_split,i,:] += 2*np.sin(2*np.pi*60*x)
                            temp[(6*ch+4)*ch_split:(6*ch+5)*ch_split,i,:] += 3*np.sin(2*np.pi*60*x) 
                            temp[(6*ch+5)*ch_split:(6*ch+6)*ch_split,i,:] += 4*np.sin(2*np.pi*60*x)
                            temp_split = temp[(6*ch+1)*ch_split:(6*ch+2)*ch_split,i,:]
                            for temp_iter in range(ch_split):
                                if ch_noise[temp_iter,ch_ind] == 0:
                                    temp_split[temp_iter,...] = 0
                                elif ch_noise[temp_iter,ch_ind] == 1:
                                    temp_split[temp_iter,...] += np.random.normal(0,ch_level[temp_iter,ch_ind]+1,temp.shape[2])
                                else:
                                    temp_split[temp_iter,...] += (ch_level[temp_iter,ch_ind]+1)*np.sin(2*np.pi*60*x)
                            temp[(6*ch+1)*ch_split:(6*ch+2)*ch_split,i,:] = cp.deepcopy(temp_split)
                        else:        
                            temp[(6*ch)*ch_split:(6*ch+1)*ch_split,i,:] += 5*np.sin(2*np.pi*60*x)
                            temp[(6*ch+1)*ch_split:(6*ch+2)*ch_split,i,:] += np.random.normal(0,1,temp.shape[2])
                            temp[(6*ch+2)*ch_split:(6*ch+3)*ch_split,i,:] += np.random.normal(0,2,temp.shape[2])
                            temp[(6*ch+3)*ch_split:(6*ch+4)*ch_split,i,:] += np.random.normal(0,3,temp.shape[2])
                            temp[(6*ch+4)*ch_split:(6*ch+5)*ch_split,i,:] += np.random.normal(0,4,temp.shape[2])
                            temp[(6*ch+5)*ch_split:(6*ch+6)*ch_split,i,:] += np.random.normal(0,5,temp.shape[2])
                    elif noise_type == 'testall':
                        if rep_i == 0:
                            for scale_i in range(5):
                                temp[(5*ch+scale_i)*ch_split:(5*ch+scale_i+1)*ch_split,i,:] += (scale_i+1)*np.sin(2*np.pi*60*x)
                        elif rep_i == 1:        
                            for scale_i in range(5):
                                temp[(5*ch+scale_i)*ch_split:(5*ch+scale_i+1)*ch_split,i,:] += np.random.normal(0,scale_i+1,temp.shape[2])
                        else:
                            temp[5*ch*ch_split:(5*ch+2)*ch_split,i,:] = 0
                            temp_split = temp[(5*ch+2)*ch_split:(5*ch+5)*ch_split,i,:]
                            for temp_iter in range(ch_split):
                                if ch_noise[temp_iter,ch_ind] == 0:
                                    temp_split[temp_iter,...] = 0
                                elif ch_noise[temp_iter,ch_ind] == 1:
                                    temp_split[temp_iter,...] += np.random.normal(0,ch_level[temp_iter,ch_ind]+1,temp.shape[2])
                                else:
                                    temp_split[temp_iter,...] += (ch_level[temp_iter,ch_ind]+1)*np.sin(2*np.pi*60*x)
                            temp[(5*ch+2)*ch_split:(5*ch+5)*ch_split,i,:] = cp.deepcopy(temp_split)
                    elif noise_type == 'flat':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] = 0
                    elif noise_type == 'gauss':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += np.random.normal(0,scale,temp.shape[2])
                    elif noise_type == '60hz':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += scale*np.sin(2*np.pi*60*x)
                    elif noise_type == 'mix':
                        temp_split = temp[ch*ch_split:(ch+1)*ch_split,i,:]
                        for temp_iter in range(ch_split):
                            if ch_noise[temp_iter,ch_ind] == 0:
                                temp_split[temp_iter,...] = 0
                            elif ch_noise[temp_iter,ch_ind] == 1:
                                temp_split[temp_iter,...] += np.random.normal(0,ch_level[temp_iter,ch_ind]+1,temp.shape[2])
                            else:
                                temp_split[temp_iter,...] += (ch_level[temp_iter,ch_ind]+1)*np.sin(2*np.pi*60*x)
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] = cp.deepcopy(temp_split)
                    elif noise_type == 'realcontact':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += real_noise[2,ch_noise[:,ch_ind],:] * emg_scale[i]
                    elif noise_type == 'realcontactbig':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += real_noise[3,ch_noise[:,ch_ind],:] * emg_scale[i]
                    elif noise_type == 'realbreak':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += real_noise[0,ch_noise[:,0],:] * emg_scale[i]
                    elif noise_type == 'realbreaknm':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += real_noise[1,ch_noise[:,0],:] * emg_scale[i]
                    elif noise_type == 'realmove':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += real_noise[-1,ch_noise[:,ch_ind],:] * emg_scale[i]
                    elif noise_type == 'realmixeven':
                        noise = real_noise[noise_all[:,ch_ind],ch_noise[:,ch_ind],:] * emg_scale[i]
                        noise[noise > 5] = 5
                        noise[noise < -5] = -5
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += noise
                    elif noise_type[:7] == 'realmix':
                        temp[ch*ch_split:(ch+1)*ch_split,i,:] += real_noise[ch_level[:,ch_ind],ch_noise[:,ch_ind],:] * emg_scale[i]
                    
                    ch_ind += 1 

            out = np.concatenate((out,temp))
    
    out = np.concatenate((raw, out))

    noisy, clean, y = out, orig, to_categorical(sub_params[:,4]-1)

    clean = clean[...,np.newaxis]
    noisy = noisy[...,np.newaxis]

    noisy = truncate(noisy)
    clean = truncate(clean)
    return noisy,clean,y

def add_noise_old(raw, params, sub, n_type='flat', scale=5):
    # Index subject and training group
    max_ch = raw.shape[1] + 1
    # if n_type[-5:] == 'skip2':
    #     ch_rot = np.concatenate(range(0,max_ch-1), range(0,max_ch-1))
    if n_type[0:4] == 'full':
        num_ch = int(n_type[-1]) + 1
        sub_params = np.tile(params,(num_ch,1))
        orig = np.tile(raw,(num_ch,1,1))
        out = cp.deepcopy(raw)

        # double channel noise
        for num_noise in range(1,num_ch):
            ch_all = list(combinations(range(0,6),num_noise))
            temp = cp.deepcopy(raw)
            ch_split = temp.shape[0]//(3*len(ch_all))
            for ch in range(0,len(ch_all)):
                for i in ch_all[ch]:
                    # temp[:,i,:] += np.random.normal(0,scale,temp.shape[2])
                    temp[3*ch*ch_split:(3*ch+1)*ch_split,i,:] += np.random.normal(0,scale,temp.shape[2])
                    temp[(3*ch+1)*ch_split:(3*ch+2)*ch_split,i,:] += np.random.normal(0,scale/5,temp.shape[2])
                    temp[(3*ch+2)*ch_split:(3*ch+3)*ch_split,i,:] = 0
            out = np.concatenate((out,temp))
    else:
        if n_type[-1].isnumeric():
            ch_rot = np.concatenate((range(0,max_ch-1), range(0,max_ch-1)))
            if n_type[-5:-1] == 'skip':
                max_i = int(n_type[-1])
            else:
                max_i = 1
            if n_type[-1] == '3':
                max_ch = 4
        sub_params = np.tile(params,(max_ch,1))
        orig = np.tile(raw,(max_ch,1,1))
        out = cp.deepcopy(raw)
        
        for ch in range(0,max_ch-1):
            temp = cp.deepcopy(raw)
            if n_type == 'gaussflat':
                temp[:temp.shape[0]//3,ch,:] += np.random.normal(0,scale,temp.shape[2])
                temp[temp.shape[0]//3:2*temp.shape[0]//3,ch,:] += np.random.normal(0,scale/2,temp.shape[2])
                temp[2*temp.shape[0]//3:,ch,:] = 0 
            elif n_type == 'gaussflatup2':
                temp[:temp.shape[0]//6,ch,:] += np.random.normal(0,scale,temp.shape[2])
                temp[temp.shape[0]//6:2*temp.shape[0]//6,ch,:] += np.random.normal(0,scale/2,temp.shape[2])
                temp[2*temp.shape[0]//6:3*temp.shape[0]//6,ch,:] = 0     
                for i_ch in range(0,max_i+1,max_i):
                    temp[3*temp.shape[0]//6:4*temp.shape[0]//6,ch_rot[ch+i_ch],:] += np.random.normal(0,scale,temp.shape[2])
                    temp[4*temp.shape[0]//6:5*temp.shape[0]//6,ch_rot[ch+i_ch],:] += np.random.normal(0,scale/2,temp.shape[2])
                    temp[5*temp.shape[0]//6:,ch_rot[ch+i_ch],:] = 0     
            elif n_type == 'gaussflatup12':
                temp[:temp.shape[0]//6,ch,:] += np.random.normal(0,scale,temp.shape[2])
                temp[temp.shape[0]//6:2*temp.shape[0]//6,ch,:] += np.random.normal(0,scale/5,temp.shape[2])
                temp[2*temp.shape[0]//6:3*temp.shape[0]//6,ch,:] = 0     
                for i_ch in range(0,max_i+1,max_i):
                    temp[3*temp.shape[0]//6:4*temp.shape[0]//6,ch_rot[ch+i_ch],:] += np.random.normal(0,scale,temp.shape[2])
                    temp[4*temp.shape[0]//6:5*temp.shape[0]//6,ch_rot[ch+i_ch],:] += np.random.normal(0,scale/5,temp.shape[2])
                    temp[5*temp.shape[0]//6:,ch_rot[ch+i_ch],:] = 0
            elif n_type == 'gaussflatskip2':
                temp[:temp.shape[0]//9,ch,:] += np.random.normal(0,scale,temp.shape[2])
                temp[temp.shape[0]//9:2*temp.shape[0]//9,ch,:] += np.random.normal(0,scale/5,temp.shape[2])
                temp[2*temp.shape[0]//9:3*temp.shape[0]//9,ch,:] = 0     
                for i_ch in range(0,2):
                    temp[3*temp.shape[0]//9:4*temp.shape[0]//9,ch_rot[ch+i_ch],:] += np.random.normal(0,scale,temp.shape[2])
                    temp[4*temp.shape[0]//9:5*temp.shape[0]//9,ch_rot[ch+i_ch],:] += np.random.normal(0,scale/5,temp.shape[2])
                    temp[5*temp.shape[0]//9:6*temp.shape[0]//9,ch_rot[ch+i_ch],:] = 0
                for i_ch in range(0,max_i+1,max_i):
                    temp[6*temp.shape[0]//9:7*temp.shape[0]//9,ch_rot[ch+i_ch],:] += np.random.normal(0,scale,temp.shape[2])
                    temp[7*temp.shape[0]//9:8*temp.shape[0]//9,ch_rot[ch+i_ch],:] += np.random.normal(0,scale/5,temp.shape[2])
                    temp[8*temp.shape[0]//9:,ch_rot[ch+i_ch],:] = 0    
            elif n_type == 'gauss':
                temp[:,ch,:] += np.random.normal(0,scale,temp.shape[2])
            elif n_type == 'flat':
                temp[:,ch,:] = 0 
            elif n_type == 'gauss2' or n_type == 'gaussskip2' or n_type == 'gaussskip3':
                for i_ch in range(0,max_i+1,max_i):
                    temp[:,ch_rot[ch+i_ch],:] += np.random.normal(0,scale,temp.shape[2])          
            elif n_type == 'flat2' or n_type == 'flatskip2' or n_type == 'gaussskip3':
                for i_ch in range(0,max_i+1,max_i):
                    temp[:,ch_rot[ch+i_ch],:] = 0        

            out = np.concatenate((out,temp))

    noisy, clean, y = out, orig, to_categorical(sub_params[:,-2]-1)

    # Add dimension to x data to fit CNN architecture
    clean = clean[...,np.newaxis]
    noisy = noisy[...,np.newaxis]
    return noisy,clean,y

def extract_feats_fast(raw):
    if raw.shape[-1] == 1:
        raw = np.squeeze(raw)
    N=raw.shape[2]
    samp = raw.shape[0]
    mav=np.sum(np.absolute(raw),axis=2)/N

    th = 0.01
    zc= np.zeros([samp,raw.shape[1]])
    for i in range(0,N-1):
        temp = 1*((raw[:,:,i]*raw[:,:,i+1]<0) & (np.absolute(raw[:,:,i]-raw[:,:,i+1])>th))
        if temp.ndim >= 3:
            temp = np.squeeze(temp)
        zc += temp

    ssc= np.zeros([samp, raw.shape[1]])
    for i in range(1,N-1):
        temp = (((raw[:,:,i]>raw[:,:,i-1]) & (raw[:,:,i]>raw[:,:,i+1])) | ((raw[:,:,i]<raw[:,:,i-1]) & (raw[:,:,i]<raw[:,:,i+1])))  & ((np.absolute(raw[:,:,i]-raw[:,:,i+1])>th) & (np.absolute(raw[:,:,i]-raw[:,:,i-1])>th))
        ssc += temp

    wl=0
    for i in range(1,N):
        wl+=np.absolute(raw[:,:,i]-raw[:,:,i-1])

    feat_out = np.concatenate([mav,zc,ssc,wl],-1)
    return feat_out

def extract_feats(raw,th=0.01,ft='feat',order=6,emg_scale=[1,1,1,1,1]):
    if raw.shape[-1] == 1:
        raw = np.squeeze(raw)
    N=raw.shape[2]
    samp = raw.shape[0]
    # th_array = np.multiply(emg_scale,th)
    # th = cp.deepcopy(th_array)
    z_th = 0.025
    s_th = 0.015

    mav=np.sum(np.absolute(raw),axis=2)/N

    if ft != 'mav':
        last = np.roll(raw, 1, axis=2)
        next = np.roll(raw, -1, axis=2)

        # zero crossings
        zero_change = (next[...,:-1]*raw[...,:-1] < 0) & (np.absolute(next[...,:-1]-raw[...,:-1])>(emg_scale*z_th))
        zc = np.sum(zero_change, axis=2)

        # slope sign change
        next_s = next[...,1:-1] - raw[...,1:-1]
        last_s = raw[...,1:-1] - last[...,1:-1]
        sign_change = ((next_s > 0) & (last_s < 0)) | ((next_s < 0) & (last_s > 0))
        th_check = (np.absolute(next_s) >(emg_scale*s_th)) & (np.absolute(last_s) > (emg_scale*s_th))
        ssc = np.sum(sign_change & th_check, axis=2)

        # waveform length
        wl = np.sum(np.absolute(next[...,:-1] - raw[...,:-1]), axis=2)

        # feat_out = 0
        feat_out = np.concatenate([mav,zc,ssc,wl],-1)
        
        if ft == 'tdar':
            AR = np.zeros((samp,raw.shape[1],order))
            for ch in range(raw.shape[1]):
                AR[:,ch,:] = np.squeeze(matAR_ch(raw[:,ch,:],order))
            reg_out = np.real(AR.transpose(0,2,1)).reshape((samp,-1))
            feat_out = np.hstack([feat_out,reg_out])
    else:
        feat_out = mav
    return feat_out

def extract_scale(x,scaler,load=True, ft='feat',emg_scale=[1,1,1,1,1,1]):
    # extract features 
    if ft == 'feat':
        num_feat = 4
    elif ft == 'tdar':
        num_feat = 10
    elif ft == 'mav':
        num_feat = 1
    x_temp = np.transpose(extract_feats(x,ft=ft,emg_scale=emg_scale).reshape((x.shape[0],num_feat,-1)),(0,2,1))[...,np.newaxis]
    
    # scale features
    if load:
        x_vae = scaler.transform(x_temp.reshape(x_temp.shape[0]*x_temp.shape[1],-1)).reshape(x_temp.shape)
    else:
        x_vae = scaler.fit_transform(x_temp.reshape(x_temp.shape[0]*x_temp.shape[1],-1)).reshape(x_temp.shape)
    
    return x_vae, scaler

def matAR(data,order):
    datalen = len(data)
    AR = np.zeros((order+1,1))
    K = np.zeros((order+1,1))

    R0 = 0.0
    ix = 0
    iy = 0
    for k in range(datalen):
        R0 += data[ix] * data[iy]
        ix += 1
        iy += 1

    R = np.zeros((order,1))

    for i in range(order):
        if 1 > (datalen - (1 + i)):
            i0 = 0
        else:
            i0 = datalen - (1 + i)

        if ((1 + i) + 1) > datalen:
            i1 = 1
        else:
            i1 = (1 + i) + 1
        
        q = 0
        if i0 >= 1:
            ix = 0
            iy = 0
            for k in range(1,i0+1):
                q += data[ix] * data[(i1 + iy)-1]
                ix += 1
                iy += 1
        
        R[i] = q

    AR[1] = -R[0]/R0
    K[0] = AR[1]
    q = R[0]
    temp = np.zeros((order,1))

    for i in range(order-1):
        R0 += q * K[i]
        q = 0
        for k in range(i+1):
            b = AR[(((1+i)+2) - (1 + k)) - 1]
            q += R[k] * b
        
        q += R[((1+i)+1)-1]
        K[1+i] = -q/R0
        for k in range(i+1):
            b = AR[(((1+i)+2)-(1+k)) - 1]
            temp[k] = K[((1+i)+1)-1] * b
        
        for k in range(((1+i)+1) - 1):
            AR[k+1] += temp[k]
        
        AR[(1+i)+1] = K[1+i]
    
    AR = np.nan_to_num(AR)
    return AR[1:]

def matAR_ch(data,order):
    datalen = data.shape[-1]
    num_ch = data.shape[0]
    AR = np.zeros((order+1,num_ch))
    K = np.zeros((order+1,num_ch))

    R0 = np.zeros((num_ch,))
    ix = 0
    iy = 0
    for k in range(datalen):
        R0 += np.multiply(data[:,ix],data[:,iy])
        ix += 1
        iy += 1

    R = np.zeros((order,num_ch))

    for i in range(order):
        if 1 > (datalen - (1 + i)):
            i0 = 0
        else:
            i0 = datalen - (1 + i)

        if ((1 + i) + 1) > datalen:
            i1 = 1
        else:
            i1 = (1 + i) + 1
        
        q = np.zeros((num_ch,))
        if i0 >= 1:
            ix = 0
            iy = 0
            for k in range(1,i0+1):
                q += np.multiply(data[:,ix],data[:,(i1 + iy)-1])
                ix += 1
                iy += 1
        
        R[i,:] = q

    AR[1,:] = np.divide(-R[0,:],R0)
    K[0,:] = AR[1,:]
    q = np.full((num_ch,),R[0,:])
    temp = np.zeros((order,num_ch))

    for i in range(order-1):
        R0 += np.multiply(q,K[i,:])
        q = np.zeros((num_ch,))
        for k in range(i+1):
            b = AR[(((1+i)+2) - (1 + k)) - 1,:]
            q += np.multiply(R[k,:], b)

        q += R[((1+i)+1)-1,:]
        K[1+i,:] = np.divide(-q,R0)
        for k in range(i+1):
            b = AR[(((1+i)+2)-(1+k)) - 1,:]
            temp[k,:] = np.multiply(K[((1+i)+1)-1,:],b)
        
        for k in range(((1+i)+1) - 1):
            AR[k+1,:] += temp[k,:]
        
        AR[(1+i)+1,:] = K[1+i,:]
    
    AR = np.nan_to_num(AR).T
    return AR[:,1:]