import pickle
import numpy as np
from matplotlib import pyplot as plt
import session

def plot_latent_dim(params,sess):
    all_acc = np.full([np.max(params[:,0]),10,4,6],np.nan)
    all_val = np.full([np.max(params[:,0]),10,4,6],np.nan)  
    
    foldername = sess.create_foldername()

    # Loop through subs
    for sub_i in range(1,2):#np.max(params[:,0])+1):
        # Loop through dimensions
        for lat in range(1,11):
            sess.latent_dim = lat
            # Loop through cross validation sets
            for cv in range(1,5):
                filename = sess.create_filename(foldername,cv,sub_i)
                with open(filename + '_hist.p', 'rb') as f:
                    svae_hist, sae_hist, cnn_hist, vcnn_hist, ecnn_hist = pickle.load(f)
                all_acc[sub_i-1,lat-1,cv-1,:] = np.array([svae_hist[-1,1],svae_hist[-1,5], sae_hist['accuracy'][-1], cnn_hist['accuracy'][-1], vcnn_hist['accuracy'][-1], ecnn_hist['accuracy'][-1]])
                all_val[sub_i-1,lat-1,cv-1,:] = np.array([svae_hist[-1,8],svae_hist[-1,12], sae_hist['val_accuracy'][-1], cnn_hist['val_accuracy'][-1], vcnn_hist['val_accuracy'][-1], ecnn_hist['val_accuracy'][-1]])

    mean_acc = np.nanmean(np.nanmean(all_acc,axis=2),axis=0)
    mean_val = np.nanmean(np.nanmean(all_val,axis=2),axis=0)

    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(1,6):
        ax.plot(mean_val[:,i],'-o')
    ax.set_ylabel('Validation Accuracy')
    fig.text(0.5, 0.04, 'Latent Dimension', ha='center')
    ax.set_ylim(0,1)
    ax.set_xticks(range(0,5))
    ax.set_xticklabels(['1','2','3','4','5'])
    ax.legend(['VCAE','NN','CNN','VCNN','ECNN'])

    return all_acc, all_val

def plot_latent_rep(x_red, y):
    # plot reduced dimensions in 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.tight_layout
    col = ['k','b','r','g','c','y','m']
    y = y.astype(int)
    # Loop through classes
    for cl in np.unique(y):
        ind = np.squeeze(y) == cl
        ax.plot3D(x_red[ind,0], x_red[ind,1], x_red[ind,2],'.', c=col[cl])
    
    return 

def plot_noise_ch(params, sess):
    if 'pos' in sess.ntest:
        i_start = 3
        i_end = 6
    elif 'gauss' in sess.ntest or 'flat' in sess.ntest:
        i_start = 1
        i_end = 5

    acc_all = np.full([np.max(params[:,0]), 3, i_end-i_start, 15],np.nan)
    acc_clean = np.full([np.max(params[:,0]), 3, i_end-i_start, 15],np.nan)
    acc_noise = np.full([np.max(params[:,0]), 3, i_end-i_start, 15],np.nan)

    ind = 0
    for i in range(i_start,i_end):
        sess.n_test = 'partpos' + str(i)
        foldername = sess.create_foldername()
        filename = sess.create_filename(foldername,results=True)
        with open(filename + '_results.p', 'rb') as f:
            temp_all, temp_clean, temp_noise = pickle.load(f)
        acc_all[:,ind,:,:], acc_clean[:,ind,:,:], acc_noise[:,ind,:,:] = np.squeeze(temp_all),np.squeeze(temp_clean),np.squeeze(temp_noise)
        ind += 1

    # average across channels
    ch_noise= np.nanmean(acc_noise,axis=2)
    ch_clean = np.nanmean(acc_clean,axis=2)

    # average across subjects
    ave_noise = np.nanmean(ch_noise,axis=0)
    ave_clean = np.nanmean(ch_clean,axis=0)

    # Plot accuracy vs. # noisy electrodes
    fig,ax = plt.subplots(1,3)
    for i in range(0,4):
        ax[0].plot(ave_noise[:,i],'-o')
    for i in range(5,9):    
        ax[1].plot(ave_noise[:,i],'-o')
    for i in [10,11,14]:
        ax[2].plot(ave_noise[:,i],'-o')    
    ax[0].set_ylabel('Accuracy')
    fig.text(0.5, 0, 'Number of Noisy Electrodes', ha='center')
    ax[0].legend(['VCAE','NN','CNN','VCNN'])
    ax[1].legend(['VCAE-LDA','NN-LDA','CNN-LDA','VCNN-LDA'])
    ax[2].legend(['LDA','LDA-corrupt','ch'])
    ax[1].set_yticks([])
    ax[2].set_yticks([])
    for i in range(0,3):
        ax[i].set_ylim(0,1)
        ax[i].set_xticks(range(0,4))
        ax[i].set_xticklabels(['1','2','3','4'])

    fig.set_tight_layout(True)

    return 

def plot_summary(acc_clean,acc_gauss,acc_60hz, acc_flat):
    lda_ind = 10
    cor_ind = 11
    sae_ind = 1
    cnn_ind = 2
    vcnn_ind = 3
    all_ind = [11, 1, 2, 3]
    labels = ['LDA-corrupt', 'SAE', 'CNN', 'VCNN']

    lda_gauss = np.tile(acc_gauss[...,lda_ind,np.newaxis],(1,1,4))
    lda_60hz = np.tile(acc_60hz[...,lda_ind,np.newaxis],(1,1,4))
    lda_flat = np.tile(acc_flat[...,lda_ind,np.newaxis],(1,1,4))
    lda_clean = np.tile(acc_clean[...,lda_ind,np.newaxis],(1,1,4))

    all_gauss_diff = acc_gauss[...,all_ind] - lda_gauss
    all_60hz_diff = acc_60hz[...,all_ind] - lda_60hz
    all_flat_diff = acc_flat[...,all_ind] - lda_flat
    all_clean_diff = acc_clean[...,all_ind] - lda_clean

    # averaged over noise levels, then noise channels, then subject
    ave_diff_gauss = np.nanmean(np.nanmean(all_gauss_diff,axis=1),axis=0)
    ave_diff_60hz = np.nanmean(np.nanmean(all_60hz_diff,axis=1),axis=0)
    ave_diff_flat = np.nanmean(np.nanmean(all_flat_diff,axis=1),axis=0)
    ave_diff_clean = np.nanmean(np.nanmean(all_clean_diff,axis=1),axis=0)

    # separated by channels
    diff_gauss = np.nanmean(all_gauss_diff,axis=0)
    diff_clean = np.nanmean(all_clean_diff,axis=0)

    fig, ax = plt.subplots()

    x = np.arange(4)  # the label locations
    width = 0.175 
    rects1 = ax.bar(x - 3*width/2, 100*ave_diff_clean, width, label='clean')
    rects2 = ax.bar(x - width/2, 100*ave_diff_gauss, width, label='gauss')
    rects3 = ax.bar(x + width/2, 100*ave_diff_flat, width, label='flat')
    rects4 = ax.bar(x + 3*width/2, 100*ave_diff_60hz, width, label='60hz')

    ax.set_ylabel('Difference from Baseline LDA (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(linewidth=.5,color='k')
    ax.set_ylim([-20,60])
    ax.legend()

    return 
