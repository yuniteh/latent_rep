import numpy as np
import scipy.io 
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

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

def norm_sub(feat, params):
    for sub in range(1,np.max(params[:,0])):
        ind = params[:,0] == sub
        scaler = MinMaxScaler(feature_range=(-1,1))
        feat_out = feat[ind,:,:]
        feat[ind,:,:] = scaler.fit_transform(feat_out.reshape(feat_out.shape[0],-1)).reshape(feat_out.shape)
    
    return feat

def add_noise(raw, params, sub):
    # Index subject and training group
    ind = (params[:,0] == sub) & (params[:,3] == 2)
    num_ch = raw.shape[1]
    sub_params = np.tile(params[ind,:],(num_ch+1,1))
    orig = np.tile(raw[ind,:,:],(num_ch+1,1,1))
    out = raw[ind,:,:]

    for ch in range(0,num_ch):
        temp = raw[ind,:,:]
        temp[:,ch,:] = 0
        out = np.concatenate((out,temp))

    x, x2, y = out,orig,to_categorical(sub_params[:,-2]-1)
    # Add dimension to x data to fit CNN architecture
    x = x[...,np.newaxis]
    x2 = x2[...,np.newaxis]
    return x,x2,y

def extract_feats(raw):
    raw = np.squeeze(raw)
    N=raw.shape[2]
    samp = raw.shape[0]
    mav=np.sum(np.absolute(raw),axis=2)/N

    th = 0.01
    zc= np.zeros([samp,raw.shape[1]])
    for i in range(0,N-1):
        temp = np.squeeze(1*((raw[:,:,i]*raw[:,:,i+1]<0) & (np.absolute(raw[:,:,i]-raw[:,:,i+1])>th)))
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