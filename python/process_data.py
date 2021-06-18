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
from itertools import combinations

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

def train_data_split(raw, params, sub, sub_type, dt=0, train_grp=2, load=True):
    if dt == 0:
        today = date.today()
        dt = today.strftime("%m%d")
    # foldername = 'models' + '_' + str(train_grp) + '_' + dt
    foldername = 'traindata'
    filename = foldername + '/' + sub_type + str(sub) + '_traindata.p'
    if load:
        if os.path.isfile(filename):
            with open(filename,'rb') as f:
                x_train, x_test, p_train, p_test = pickle.load(f)
                x_train, p_train = shuffle(x_train, p_train, random_state = 0)
                x_test, p_test = shuffle(x_test, p_test, random_state = 0)
        else:
            load=False
    if not load:
        ind = (params[:,0] == sub) & (params[:,3] == train_grp)
        # Split training and testing data
        x_train, x_test, p_train, p_test = train_test_split(raw[ind,:,:], params[ind,:], test_size = 0.33, stratify=params[ind,4], shuffle=True)
        with open(filename, 'wb') as f:
            pickle.dump([x_train, x_test, p_train, p_test],f)
        x_train, p_train = shuffle(x_train, p_train, random_state = 0)
        x_test, p_test = shuffle(x_test, p_test, random_state = 0)
    return x_train, x_test, p_train, p_test

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

def add_noise(raw, params, sub, n_type='flat', scale=5):
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
                    temp[3*ch*ch_split:(3*ch+1)*ch_split,i,:] += np.random.normal(0,scale,temp.shape[2])
                    temp[(3*ch+1)*ch_split:(3*ch+2)*ch_split,i,:] += np.random.normal(0,scale/5,temp.shape[2])
                    temp[(3*ch+2)*ch_split:(3*ch+3)*ch_split,i,:] = 0
                elif noise_type == 'gauss':
                    # if ch == 0:
                    #     print('gauss:' + str(ch_split))
                    temp[ch*ch_split:(ch+1)*ch_split,i,:] += np.random.normal(0,scale,temp.shape[2])
                    # temp[:,i,:] += np.random.normal(0,scale,temp.shape[2])
                elif noise_type == 'flat':
                    # if ch == 0:
                    #     print('flat:' + str(ch_split))
                    temp[ch*ch_split:(ch+1)*ch_split,i,:] = 0
                    # temp[:,i,:] = 0
                # if full_type == 'part':
                #     out = np.concatenate((out,temp))
        # if full_type == 'full':
        out = np.concatenate((out,temp))
        

    noisy, clean, y = out, orig, to_categorical(sub_params[:,-2]-1)

    clean = clean[...,np.newaxis]
    noisy = noisy[...,np.newaxis]
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

        # single channel noise
        # temp = cp.deepcopy(raw)
        # ch_split = temp.shape[0]//(3*(max_ch-1))
        # for ch in range(0,max_ch-1):
        #     temp[3*ch*ch_split:(3*ch+1)*ch_split,ch,:] += np.random.normal(0,scale,temp.shape[2])
        #     temp[(3*ch+1)*ch_split:(3*ch+2)*ch_split,ch,:] += np.random.normal(0,scale/5,temp.shape[2])
        #     temp[(3*ch+2)*ch_split:(3*ch+3)*ch_split,ch,:] = 0 
        # out = np.concatenate((out,temp))

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
                # temp[temp.shape[0]//2:,ch,:] = 0 
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
    # x, x2, y = out,orig,to_categorical(sub_params[:,-2]-1)
    # Add dimension to x data to fit CNN architecture
    # x = x[...,np.newaxis]
    # x2 = x2[...,np.newaxis]
    clean = clean[...,np.newaxis]
    noisy = noisy[...,np.newaxis]
    return noisy,clean,y

def extract_feats(raw):
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
    for i in range(2,N):
        wl+=np.absolute(raw[:,:,i]-raw[:,:,i-1])

    feat_out = np.concatenate([mav,zc,ssc,wl],-1)
    return feat_out