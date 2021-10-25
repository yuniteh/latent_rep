import numpy as np
import scipy.io 
import pandas as pd
from itertools import combinations
import process_data as prd
from numpy.linalg import eig, inv


# train and predict for data: (samples,feat), label: (samples, 1)
def eval_lda(w, c, x_test, y_test):
    out = predict(x_test, w, c)
    acc = np.sum(out.reshape(y_test.shape) == y_test)/y_test.shape[0]
    return acc

def eval_lda_ch(mu_class, C, n_type, x, y):
    # Index subject and training group
    num_ch = int(n_type[-1]) + 1
    full_type = n_type[0:4]
    noise_type = n_type[4:-1]

    if noise_type[:3] == 'pos':
        num_ch = int(noise_type[-1]) + 1
        noise_type = noise_type[3:-1]

    # sub_params = np.tile(params,(rep*(num_ch-1)+1,1))
    # tile data once for each channel
    if full_type == 'full':
        start_ch = num_ch - 1
    # tile data twice, once for clean and once for noise
    elif full_type == 'part':
        start_ch = num_ch - 1
    
    acc = np.zeros(num_ch-start_ch)
    # loop through channel noise
    for num_noise in range(start_ch,num_ch):
        ch_all = list(combinations(range(0,6),num_noise))
        ch_split = x.shape[0]//len(ch_all)
        acc_ch = np.zeros(len(ch_all))
        for ch in range(0,len(ch_all)):
            temp = x[ch*ch_split:(ch+1)*ch_split,...]
            y_test = y[ch*ch_split:(ch+1)*ch_split,...]
            mask = np.ones(temp.shape[1],dtype=bool)
            for i in ch_all[ch]:
                mask[i] = 0
            maskmu = np.tile(mask,4)
            test_data = prd.extract_feats(temp[:,mask,:])
            C_temp = C[maskmu,:]
            C_in = C_temp[:,maskmu]
            w_temp, c_temp = train_lda(test_data,y_test,mu_bool = True, mu_class = mu_class[:,maskmu], C = C_in)
            acc_ch[ch] = eval_lda(w_temp, c_temp, test_data, y_test)

        acc[num_noise-start_ch] = np.mean(acc_ch)
    return acc

# train LDA classifier for data: (samples,feat), label: (samples, 1)
def train_lda(data,label,mu_bool=False, mu_class = 0, C = 0):
    m = data.shape[1]
    u_class = np.unique(label)
    n_class = u_class.shape[0]

    if not mu_bool:
        mu = np.mean(data,axis=0,keepdims = True)
        C = np.zeros([m,m])
        mu_class = np.zeros([n_class,m])
        Sb = np.zeros([mu.shape[1],mu.shape[1]])
        Sw = np.zeros([mu.shape[1],mu.shape[1]])

        for i in range(0,n_class):
            ind = label == u_class[i]
            mu_class[i,:] = np.mean(data[ind[:,0],:],axis=0,keepdims=True)
            C += np.cov(data[ind[:,0],:].T)
            Sb += ind.shape[0] * np.dot((mu_class[np.newaxis,i,:] - mu).T,(mu_class[np.newaxis,i,:] - mu)) 

            Sw_temp = np.zeros([mu.shape[1],mu.shape[1]])
            for row in data[ind[:,0],:]:
                Sw_temp += np.dot((row[:,np.newaxis] - mu_class[i,:,np.newaxis]), (row[:,np.newaxis] - mu_class[i,:,np.newaxis]).T)
            Sw += Sw_temp
        C /= n_class
        u,v = eig(inv(Sw).dot(Sb))    
        v = v[:,np.flip(np.argsort(np.abs(u)))]
        v = v[:,:6].real

    prior = 1/n_class

    w = np.zeros([n_class, m])
    c = np.zeros([n_class, 1])

    for i in range(0, n_class):
        w[i,:] = np.dot(mu_class[np.newaxis,i,:],np.linalg.pinv(C))
        c[i,:] = np.dot(-.5 * np.dot(mu_class[np.newaxis,i,:], np.linalg.pinv(C)),mu_class[np.newaxis,i,:].T) + np.log(prior)    

    if not mu_bool:
        return w, c, mu_class, C, v
    else:
        return w, c

# train LDA classifier for data: (feat, samples)
def train_lda2(data,label):
    m = data.shape[0]
    u_class = np.unique(label)
    n_class = u_class.shape[0]

    mu = np.mean(data,axis=1,keepdims = True)
    C = np.zeros([m,m])
    mu_class = np.zeros([m,n_class])
    Sb = np.zeros([mu.shape[0],mu.shape[0]])

    for i in range(0,n_class):
        ind = label == u_class[i]
        mu_class[:,i,np.newaxis] = np.mean(data[:,ind[0,:]],axis=1,keepdims=True)
        Sb += ind.shape[1] * np.dot((mu_class[:,i,np.newaxis] - mu).T,(mu_class[:,i,np.newaxis] - mu))
        C += np.cov(data[:,ind[0,:]])

    C /= n_class
    prior = 1/n_class

    w = np.zeros([m,n_class])
    c = np.zeros([1,n_class])

    for i in range(0, n_class):
        w[:,i,np.newaxis] = np.dot(np.linalg.pinv(C),mu_class[:,i,np.newaxis])
        c[:,i,np.newaxis] = np.dot(mu_class[:,i,np.newaxis].T,-.5 * np.dot(np.linalg.pinv(C), mu_class[:,i,np.newaxis])) + np.log(prior)

    return w,c.T

def predict(data,w,c):
    f = np.dot(w,data.T) + c
    out = np.argmax(f, axis=0)
    return out

def predict2(data,w,c):
    f = np.dot(w.T,data) + c
    out = np.argmax(f, axis=0)
    return out