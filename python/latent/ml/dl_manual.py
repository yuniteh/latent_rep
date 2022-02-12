import numpy as np

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

def conv(x_in, w, fxn = 'relu'):
    out = np.zeros((x_in.shape[0], x_in.shape[1], x_in.shape[2], w[0].shape[-1]))
    for f in range(w[0].shape[-1]):
        padded = np.zeros((x_in.shape[0],x_in.shape[1]+2,x_in.shape[2]+1,x_in.shape[3]))
        padded[:,1:-1,:-1,:] = x_in
        f_out = np.zeros(x_in.shape)

        for i in range(padded.shape[1]-2):
            for j in range(padded.shape[2]-1):
                temp = padded[:,i:i+3,j:j+2,:]
                f_out[:,i,j,:] = np.sum(np.sum(temp*w[0][...,f],axis=1),axis=1) + w[1][f]
        
        if fxn == 'relu':
            f_out = relu(f_out)
        
        out[...,[f]] = f_out
    
    return out

def softmax(x):
    out = np.exp(x) / np.sum(np.exp(x), axis=1)[...,np.newaxis]
    return out

def relu(x):
    out = x * (x > 0)
    return out