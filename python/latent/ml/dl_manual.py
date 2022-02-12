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

def conv(x_in, w, stride, k = (3,3), fxn = 'relu'):
    out = np.zeros((x_in.shape[0], 1+(x_in.shape[1]-k[0]+2)//stride, 1+(x_in.shape[2]-k[1]+2)//stride, w[0].shape[-1]))
    # print(out.shape)
    for f in range(w[0].shape[-1]):
        padded = np.pad(x_in,pad_width = ((0,0),(1,1),(1,1),(0,0)))
        # f_out = np.zeros((x_in.shape[0], x_in.shape[1]//stride, x_in.shape[2]//stride,w[0].shape[-2]))
        # padded = x_in
        f_out = np.zeros((x_in.shape[0], 1+(x_in.shape[1]-k[0]+2)//stride, 1+(x_in.shape[2]-k[1]+2)//stride,1))

        # k = 0
        # for filt in range(w[0].shape[-2]):
        i = 0
        for row in range(out.shape[1]):
            j = 1
            for col in range(out.shape[2]):
                # print(i)
                # print(j)
                temp = padded[:,i:i+k[0],j:j+k[1],:]

                out[:,row,col,f] = np.sum(np.sum(np.sum(temp*w[0][...,f],axis=-1),axis=-1),axis=-1,keepdims=False) + w[1][f]
                j += stride
            i += stride
            # if filt == 0:
            #     print(f_out[0,:,:,0])
        
        # f_out = np.sum(f_out,axis=-1,keepdims=True)
        # if fxn == 'relu':
        #     f_out = relu(f_out)
        
        out = relu(out)
    
    return out

def softmax(x):
    out = np.exp(x) / np.sum(np.exp(x), axis=1)[...,np.newaxis]
    return out

def relu(x):
    out = x * (x > 0)
    return out