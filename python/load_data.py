import numpy as np
import scipy.io 
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def load_raw(filename):
    struct = scipy.io.loadmat(filename)
    daq = struct['data'][0,0]['daq']['DAQ_DATA'][0,0]
    pvd = struct['data'][0,0]['pvd']

    return daq,pvd

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
    
    if feat.ndim == 3:
        x, y = shuffle(feat[ind,:,:],to_categorical(params[ind,-2]-1))
        # Add dimension to x data to fit CNN architecture
        x = x[...,np.newaxis]
    else:
        x, y = shuffle(feat[ind,:],params[ind,-2]-1)
        y = y[...,np.newaxis]
    
    return x,y