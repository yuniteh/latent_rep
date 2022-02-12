import numpy as np
import time

def dense(x_in, w, fxn = 'relu'):
    out = (x_in @ w[0]) + w[1]
    if fxn == 'relu':
        out = relu(out)
    elif fxn == 'softmax':
        out = softmax(out)
    return out

def bn(x_in, w):
    out = ((w[0] * (x_in - w[2])) / np.sqrt(w[3] + 0.001)) + w[1]
    return out

def nn_pass(x, w, arch):
    i = 0 
    if 'prop' not in arch:
        prop = 0
    for l in arch:
        if l == 'bn':
            w_layer = w[i:i+4]
            x = bn(x, w_layer)
            i += 4
        elif 'flat' in l:
            x = np.reshape(x,(x.shape[0],-1))
        else:
            w_layer = w[i:i+2]
            i += 2
            if 'conv' in l:
                x = conv(x, w_layer)
            elif 'prop' in l:
                x = dense(prev_x, w_layer)
            else:
                if 'softmax' in l:
                    prev_x = x
                x = dense(x, w_layer, fxn = l)
        
    return x, prop

def conv(x_in, w, stride=1, k = (3,3), fxn = 'relu'):
    out = np.zeros((x_in.shape[0], 1+(x_in.shape[1]-k[0]+2)//stride, 1+(x_in.shape[2]-k[1]+2)//stride, w[0].shape[-1]))
    for f in range(w[0].shape[-1]):
        padded = np.pad(x_in,pad_width = ((0,0),(1,1),(1,1),(0,0)))

        i = 0
        for row in range(out.shape[1]):
            j = 0
            for col in range(out.shape[2]):
                out[:,row,col,f] = np.sum(np.sum(np.sum(padded[:,i:i+k[0],j:j+k[1],:]*w[0][...,f],axis=-1),axis=-1),axis=-1,keepdims=False) + w[1][f]
                j += stride
            i += stride
        
    if fxn == 'relu':
        out = relu(out)
    
    return out

def softmax(x):
    out = np.exp(x) / np.sum(np.exp(x), axis=1)[...,np.newaxis]
    return out

def relu(x):
    out = x * (x > 0)
    return out