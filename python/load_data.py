import numpy as np
import scipy.io 
import pandas as pd

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