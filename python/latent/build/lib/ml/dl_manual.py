import numpy as np

def dense(x_in, w, fxn = 'relu'):
    out = (x_in @ w[0]) + w[1]
    if fxn == 'relu':
        out *= (out > 0)
    elif fxn == 'softmax':
        out = np.exp(out) / np.sum(np.exp(out), axis=1)[...,np.newaxis]
    return out

def bn(x_in, w):
    out = ((w[0] * (x_in - w[2])) / np.sqrt(w[3] + 0.001)) + w[1]
    return out

def nn_pass(x, w, arch):
    i = 0 
    for l in arch:
        if l == 'bn':
            w_layer = w[i:i+4]
            x = bn(x, w_layer)
            i += 4
        else:
            w_layer = w[i:i+2]
            x = dense(x, w_layer, fxn = l)
            i += 2
    return x