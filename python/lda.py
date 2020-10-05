import numpy as np
import scipy.io 
import pandas as pd

# train and predict for data: (samples,feat), label: (samples, 1)
def eval_lda(w, c, x_test, y_test):
    out = predict(x_test, w, c)
    acc = np.sum(out.reshape(y_test.shape) == y_test)/y_test.shape[0]
    return acc

# train LDA classifier for data: (samples,feat), label: (samples, 1)
def train_lda(data,label):
    m = data.shape[1]
    u_class = np.unique(label)
    n_class = u_class.shape[0]

    mu = np.mean(data,axis=0,keepdims = True)
    C = np.zeros([m,m])
    mu_class = np.zeros([n_class,m])
    Sb = np.zeros([mu.shape[1],mu.shape[1]])

    for i in range(0,n_class):
        ind = label == u_class[i]
        mu_class[i,:] = np.mean(data[ind[:,0],:],axis=0,keepdims=True)
        Sb += ind.shape[0] * np.dot((mu_class[np.newaxis,i,:] - mu).T,(mu_class[np.newaxis,i,:] - mu))
        C += np.cov(data[ind[:,0],:].T)

    C /= n_class
    prior = 1/n_class

    w = np.zeros([n_class, m])
    c = np.zeros([n_class, 1])

    for i in range(0, n_class):
        w[i,:] = np.dot(mu_class[np.newaxis,i,:],np.linalg.pinv(C))
        c[i,:] = np.dot(-.5 * np.dot(mu_class[np.newaxis,i,:], np.linalg.pinv(C)),mu_class[np.newaxis,i,:].T) + np.log(prior)
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