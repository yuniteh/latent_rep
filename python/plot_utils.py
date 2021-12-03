import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib as mpl


def plot_all_noise(real_noise):
    out = np.squeeze(np.zeros((200*1000//8,4)))
    t = np.linspace(0,200//8,200*1000//8)
    fig, ax = plt.subplots(4,1)
    ch_i = 0
    for ch in [0,1,2,4]:
        it = 0
        for i in range(0,1000,8):
            out[it*200:it*200+200,ch_i] = real_noise[ch,i,:]
            it += 1
        ax[ch_i].plot(t,out[:,ch_i],linewidth=.5,color='k')
        ax[ch_i].set_ylim([-5,5])
        ax[ch_i].set_xlim([10,15])
        if ch_i < 3:
            ax[ch_i].set_xticks([])
        else:    
            ax[ch_i].set_xticklabels(['0','1','2','3','4','5'])
        # ax[ch].tight_layout()
        ch_i += 1

def plot_noisy(noisy_in, clean_in, y, cl=4, iter=0, gs=0):
    c = np.flip(plt.cm.Blues(np.linspace(0.1,0.9,7)),axis=0)
    c2 = np.flip(plt.cm.Purples(np.linspace(0.1,0.9,7)),axis=0)
    c3 = np.flip(plt.cm.Reds(np.linspace(0.1,0.9,7)),axis=0)

    c = np.asarray(sns.color_palette("Set2"))
    c1 = np.flip(np.asarray(sns.light_palette(c[1])),axis=0)
    c2 = np.flip(np.asarray(sns.light_palette(c[2])),axis=0)
    # print(c)
    # c = c(0.5)
    # print(c)
    ind = np.argmax(y,axis=1)
    ind_p = np.squeeze(np.asarray(np.where(ind == cl)))

    xnoise = noisy_in[ind_p,:,:,0]
    xclean = clean_in[ind_p,:,:,0]
    noise = xnoise-xclean
    noise[noise > 5] = 5
    noise[noise < -5] = -5
    noiseind = np.argmax(np.max(noise,axis=2),axis=1)
    noiseind[np.max(np.max(noise,axis=2),axis=1) == 0] = -1

    std_all = np.max(xclean[:400,...],axis=2)
    std_ch = std_all[np.arange(std_all.shape[0]),noiseind[400:]]
    n = np.argmax(std_ch)+400

    if iter == 0:
        ax = plt.subplot(gs[3:5,0])
        ax.plot(np.squeeze(xclean[n,noiseind[n],:]),color=c[0])
        ax.set_ylim(-5.2,5.2)
        ax.set_xlim(0,200)
        ax.set_yticklabels('')
        ax.set_xlabel('Time')
    
    ax2 = plt.subplot(gs[2*iter:2+2*iter,1])
    ax2.plot(np.squeeze(noise[n,noiseind[n],:]),color=c1[iter])
    ax2.set_yticklabels('')
    ax2.set_xlim(0,200)
    
    ax3 = plt.subplot(gs[2*iter:2+2*iter,2])
    ax3.plot(np.squeeze(xnoise[n,noiseind[n],:]),color=c2[iter])
    ax3.set_yticklabels('')
    ax3.set_xlim(0,200)

    # ax.set_xticklabels('')
    ax2.set_ylim(-5.5,5.5)
    ax3.set_ylim(-5.5,5.5)

    if iter == 3:
        ax2.set_xlabel('Time')
        ax3.set_xlabel('Time')
    else:
        ax2.set_xticklabels('')
        ax3.set_xticklabels('')
        ax2.xaxis.set_ticks_position('none') 
        ax3.xaxis.set_ticks_position('none')

    ax2.yaxis.set_ticks_position('none') 
    ax3.yaxis.set_ticks_position('none')
        
    # fig,ax = plt.subplots(3,1)
    # ax[0].plot(np.squeeze(xclean[n,noiseind[n],:]))
    # ax[1].plot(np.squeeze(xnoise[n,noiseind[n],:]))
    # ax[2].plot(np.squeeze(noise[n,noiseind[n],:]))

    # for i in range(3):
    #     ax[i].set_ylim(-5.2,5.2)
    #     ax[i].set_xticklabels('')

def plot_latent_dim(params,sess):
    all_acc = np.full([np.max(params[:,0]),10,4,6],np.nan)
    all_val = np.full([np.max(params[:,0]),10,4,6],np.nan)  
    
    foldername = sess.create_foldername()

    # Loop through subs
    for sub_i in range(2,3):#np.max(params[:,0])+1):
        # Loop through dimensions
        for lat in range(1,11):
            sess.latent_dim = lat
            # Loop through cross validation sets
            for cv in range(1,5):
                filename = sess.create_filename(foldername,cv,sub_i)
                with open(filename + '_hist.p', 'rb') as f:
                    svae_hist, sae_hist, cnn_hist, vcnn_hist, ecnn_hist = pickle.load(f)
                all_acc[sub_i-1,lat-1,cv-1,:] = np.array([svae_hist[-1,1],svae_hist[-1,5], sae_hist['accuracy'][-1], cnn_hist['accuracy'][-1], vcnn_hist[-1,1], ecnn_hist[-1,4]])
                all_val[sub_i-1,lat-1,cv-1,:] = np.array([svae_hist[-1,8],svae_hist[-1,12], sae_hist['val_accuracy'][-1], cnn_hist['val_accuracy'][-1], vcnn_hist[-1,3], ecnn_hist[-1,9]])

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

    return svae_hist, cnn_hist, vcnn_hist, ecnn_hist

def create_dist(mu,std,mult):
    # Make data
    subdev = 20
    phi, theta = np.mgrid[0.0:np.pi:complex(0,subdev), 0.0:2.0 * np.pi:complex(0,subdev)]
    x = mult * std[0] * np.sin(phi) * np.cos(theta) + mu[0]
    y = mult * std[1] * np.sin(phi) * np.sin(theta) + mu[1]
    z = mult * std[2] * np.cos(phi) + mu[2]

    return x,y,z

def create_ellipse(mu1,mu2,std1,std2):
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    
    a = 1*std1 * np.cos(theta) + mu1
    b = 1*std2 * np.sin(theta) + mu2
    return a,b

def plot_latent_rep(x_red, class_in, fig,loc=0,downsamp=1,dim=3,lims=((-6,6),(-6,6),(-6,6)),lim_max=0,lim_min=0,std_lim=True,mult=3):
    # plot reduced dimensions in 3D
    # fig = plt.figure()
    if dim == 3:
        ax = fig.add_subplot(2,2,loc,projection='3d')
    else:
        ax = fig.add_subplot(2,2,loc)
    col = ['b','g','orange','r','pink','k','m']
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', "RdPu"]
    col = np.asarray(sns.color_palette("Accent"))
    col[[3,4],:] = col[[4,3],:]
    class_in = class_in.astype(int)
    x_red = x_red[:,:3]

    if isinstance(lim_max,int):
        lim_max = np.ones((3,))*-10000
        lim_min = np.ones((3,))*10000
    # Loop through classes
    for cl in np.unique(class_in):
        ind = np.squeeze(class_in) == cl
        x_ind = x_red[ind,:]
        mu = np.mean(x_ind[:,:3], axis=0)
        std = np.std(x_ind[:,:3], axis=0)
        x, y, z = create_dist(mu,std,mult)
                
        cur_min = mu-mult*std
        cur_max = mu+mult*std

        lim_max[lim_max < cur_max] = cur_max[lim_max < cur_max]
        lim_min[lim_min > cur_min] = cur_min[lim_min > cur_min]

        if dim == 3:
            x_ind[(x_ind[...,:3] < cur_min)|(x_ind[...,:3] > cur_max)] = np.nan
            if std_lim:
                ax.plot_surface(x,y,z, cmap=cmaps[-cl], alpha=0.4, linewidth=2)
            else:
                ax.plot3D(x_ind[0:-1:downsamp,0], x_ind[0:-1:downsamp,1], x_ind[0:-1:downsamp,2],'.', c=col[cl]*.9,ms=3)
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_zticks([])

            # a,b = create_ellipse(mu[0],mu[1],std[0],std[1])
            # ax.plot3D(a,b,np.ones(a.shape)*mu[2], c='k',alpha=1,linewidth=1)

            # a,b = create_ellipse(mu[1],mu[2],std[1],std[2])
            # ax.plot3D(np.ones(a.shape)*mu[0],a,b, c='k',alpha=1,linewidth=1)

            # a,b = create_ellipse(mu[0],mu[2],std[0],std[2])
            # ax.plot3D(a,np.ones(a.shape)*mu[1],b, c='k',alpha=1,linewidth=1)
        else:
            if std_lim:
                ellipse = Ellipse((mu[0],mu[1]),mult*std[0],mult*std[1],facecolor=col[cl],alpha=.5)
                ax.add_patch(ellipse)
            else:
                ax.plot(x_ind[0:-1:downsamp,0], x_ind[0:-1:downsamp,1],'.', c=col[cl],ms=3)

    if 1:# std_lim:
        ax.set_xlim((lim_min[0],lim_max[0]))
        ax.set_ylim((lim_min[1],lim_max[1]))
        if dim == 3:
            ax.set_zlim((lim_min[2],lim_max[2]))
            # ax.set_zticklabels([])
            ax.zaxis.set_ticks_position('none')  

    else:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        if dim == 3:
            ax.set_zlim(lims[2])
            # ax.set_zticklabels([])
            ax.zaxis.set_ticks_position('none')  

    ax.xaxis.set_rotate_label(True)
    ax.yaxis.set_rotate_label(True)
    ax.zaxis.set_rotate_label(True)

    ax.set_xlabel('LDA1',labelpad=.001,linespacing=1)
    ax.set_ylabel('LDA2',labelpad=.001,linespacing=1)
    ax.set_zlabel('LDA3',labelpad=.001,linespacing=1)

    # ax.xaxis.set(label='none')       
    # ax.yaxis.set_ticks_position('none')   
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


    ax.dist = 9
    
    return lim_min, lim_max

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

    plot_electrode_results(ave_noise,ave_clean)
    return 

def plot_electrode_results(acc_noise,acc_clean,ntrain='',ntest='',subtype='AB',gs=0):
    line_col = sns.color_palette("Paired")
    eb_col = sns.color_palette("Paired")
    # line_col[2:4] = eb_col[4:6]
    # line_col[4:6] = eb_col[2:4]
    line_col[8] = ((0.6,0.6,0.6))
    line_col[9] = ((0,0,0))

    n_subs = np.sum(~np.isnan(acc_clean[:,0,0]))

    acc_clean[:,0,-1] = acc_clean[:,0,10]
    acc_noise = np.hstack([acc_clean[:,0,:][:,np.newaxis,:],acc_noise])
    ave_noise = np.nanmean(100*acc_noise,axis=0)

    all_std = np.nanstd(100*acc_noise,axis=0)/n_subs

    x = [-.02, 1, 2, 3, 4.02]
    # Plot accuracy vs. # noisy electrodes
    c_i = 0
    if gs == 0:
        fig,ax = plt.subplots()
    else:
        ax = plt.subplot(gs)

    for i in [6,7,11,14,10]:#,9]:    
        ax.fill_between(np.arange(5),ave_noise[:,i]+all_std[:,i],ave_noise[:,i]-all_std[:,i],color=line_col[c_i],alpha=.5,ec=None)
        ax.plot(ave_noise[:,i],'-o',color=line_col[c_i+1],ms=4)
        c_i+=2

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Number of Noisy Electrodes')
    # fig.text(0.5, 0, 'Number of Noisy Electrodes', ha='center')
    # ax.legend(['sae-lda','cnn-lda','LDA','LDA-corrupt','LDA-ch'])
    ax.set_axisbelow(True)
    ax.yaxis.grid(1,color='lightgrey',linewidth=.5)
    ax.set_ylim(20,85)
    ax.set_xlim([-.1,4.1])
    ax.set_xticks(range(0,5))
    ax.set_xticklabels(['0','1','2','3','4'])
    # ax.set_title(subtype + ', Train: ' + ntrain + ', test: ' + ntest)
    return

def plot_summary(acc_clean, acc_mix,gs=0):
    n_subs = np.sum(~np.isnan(acc_clean[:,0,0]))

    line_col = sns.color_palette("Paired")
    line_col[8] = ((0.6,0.6,0.6))
    line_col[9] = ((0,0,0))

    lda_ind = 10
    all_ind = [11, 6, 7, 14]
    labels = ['LDA-corrupt', 'SAE-LDA', 'CNN-LDA', 'LDA-ch']
    acc_clean[...,-1] = acc_clean[...,10]

    lda_mix = np.tile(acc_mix[...,lda_ind,np.newaxis],(1,1,len(all_ind)))
    lda_clean = np.tile(acc_clean[...,lda_ind,np.newaxis],(1,1,len(all_ind)))

    all_mix_diff = acc_mix[...,all_ind] - lda_mix
    all_clean_diff = acc_clean[...,all_ind] - lda_clean

    # separated by channels
    diff_mix = np.nanmean(all_mix_diff,axis=0)
    diff_clean = np.nanmean(all_clean_diff,axis=0)

    std_clean = 100*np.nanstd(all_clean_diff,axis=0)/n_subs
    std_mix = 100*np.nanstd(all_mix_diff,axis=0)/n_subs

    if gs == 0:
        fig,ax = plt.subplots()
    else:
        ax = plt.subplot(gs)
    c = ['r','tab:purple','tab:blue', 'm']
    c = line_col[0::2]
    c2 = line_col[0::2]
    c[0],c[1],c[2] = c[2],c[0],c[1]

    x = np.arange(len(all_ind))  # the label locations
    width = 0.15
    ax.bar(x - 2*width, 100*diff_clean[0,:], width, color = c, edgecolor='w',yerr=std_clean[0,:],linewidth=.5)
    ax.bar(x - width, 100*diff_mix[0,:], width, color = c, edgecolor='w',yerr=std_mix[0,:],linewidth=.5)
    ax.bar(x, 100*diff_mix[1,:], width, color = c, edgecolor='w',yerr=std_mix[1,:],linewidth=.5)
    ax.bar(x + width, 100*diff_mix[2,:], width, color = c, edgecolor='w',yerr=std_mix[2,:],linewidth=.5)
    ax.bar(x + 2*width, 100*diff_mix[3,:], width, color = c, edgecolor='w',yerr=std_mix[3,:],linewidth=.5)

    xticks = [-2*width, -width,0,width,2*width]

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Difference from Baseline LDA (%)')
    ax.set_xlabel('Number of Noisy Electrodes',loc='left')
    ax.set_xticks(xticks)
    ax.set_xticklabels(['0','1','2','3','4'])
    ax.axhline(linewidth=.5,color='k')
    ax.set_ylim([-25,50])
    ax.set_axisbelow(True)
    ax.yaxis.grid(1,color='lightgrey',linewidth=.5)

    return 

def plot_electrode_old(ave_noise,ave_clean,ntrain='',ntest='',subtype='AB'):
    ave_clean[0,-1] = ave_clean[0,10]
    ave_noise = np.vstack([ave_clean[0,:],ave_noise])
    all_temp = np.tile(ave_noise[:,10][...,np.newaxis],(1,15))
    ave_noise_diff = np.divide((1-ave_noise) - (1-ave_noise[0,:]), 1)#(1-ave_noise[0,:]))
    ave_clean_diff = np.divide((1-ave_noise) - (1-ave_clean[0,10]), 1)#(1-ave_clean[0,10]))
    ave_all_diff = np.divide((1-ave_noise) - (1-all_temp), 1)#(1-ave_clean[0,10]))

    # Plot accuracy vs. # noisy electrodes
    fig,ax = plt.subplots()
    c = ['k','r','m']
    c_tab = ['tab:purple','tab:blue', 'tab:orange', 'tab:green','tab:red']
    c_i = 0
    for i in [6,7]:#,9]:    
        ax.plot(100*ave_noise[:,i],'-o',color=c_tab[c_i])
        c_i+=1
    c_i = 0
    for i in [10,11,14]:
        ax.plot(100*ave_noise[:,i],'--o',color=c[c_i])
        c_i+=1    

    ax.set_ylabel('Accuracy (%)')
    fig.text(0.5, 0, 'Number of Noisy Electrodes', ha='center')
    ax.legend(['sae-lda','cnn-lda','LDA','LDA-corrupt','LDA-ch'])
    ax.set_ylim(0,80)
    ax.set_xticks(range(0,5))
    ax.set_xticklabels(['0','1','2','3','4'])
    ax.set_title(subtype + ', Train: ' + ntrain + ', test: ' + ntest)

    fig.set_tight_layout(True)

    # Plot accuracy vs. # noisy electrodes
    fig,ax = plt.subplots()
    c = ['k','r','m']
    c_i = 0
    for i in [1,2,4]:    
        ax.plot(100*ave_noise_diff[:,i],':o',color=c_tab[c_i])
        c_i+=1
    c_i = 0
    for i in [6,7,9]:    
        ax.plot(100*ave_noise_diff[:,i],'-o',color=c_tab[c_i])
        c_i+=1
    c_i = 0
    for i in [10,11,14]:
        ax.plot(100*ave_noise_diff[:,i],'--o',color=c[c_i])
        c_i+=1    

    ax.set_ylabel('Difference in Error Rate (%)')
    fig.text(0.5, 0, 'Number of Noisy Electrodes', ha='center')
    ax.legend(['sae','cnn','svae','sae-lda','cnn-lda','svae-lda','LDA','LDA-corrupt','LDA-ch'])
    ax.set_xticks(range(0,5))
    ax.set_xticklabels(['0','1','2','3','4'])
    ax.set_title(subtype + ', Train: ' + ntrain + ', test: ' + ntest)

    fig.set_tight_layout(True)

    # Plot accuracy vs. # noisy electrodes
    fig,ax = plt.subplots()
    c = ['k','r','m']
    c_i = 0
    for i in [1,2,4]:    
        ax.plot(100*ave_clean_diff[:,i],':o',color=c_tab[c_i])
        c_i+=1
    c_i = 0
    for i in [6,7,9]:    
        ax.plot(100*ave_clean_diff[:,i],'-o',color=c_tab[c_i])
        c_i+=1
    c_i = 0
    for i in [10,11,14]:
        ax.plot(100*ave_clean_diff[:,i],'--o',color=c[c_i])
        c_i+=1    

    ax.set_ylabel('Difference in Error Rate (%)')
    fig.text(0.5, 0, 'Number of Noisy Electrodes', ha='center')
    ax.legend(['sae','cnn','svae','sae-lda','cnn-lda','svae-lda','LDA','LDA-corrupt','LDA-ch'])
    ax.set_xticks(range(0,5))
    ax.set_xticklabels(['0','1','2','3','4'])
    ax.set_title(subtype + ', Train: ' + ntrain + ', test: ' + ntest)

    fig.set_tight_layout(True)

    # Plot accuracy vs. # noisy electrodes
    fig,ax = plt.subplots()
    c = ['k','r','m']
    c_i = 0
    for i in [1,2]:    
        ax.plot(100*ave_all_diff[:,i],':o',color=c_tab[c_i])
        c_i+=1
    c_i = 0
    for i in [6,7]:    
        ax.plot(100*ave_all_diff[:,i],'-o',color=c_tab[c_i])
        c_i+=1
    c_i = 0
    for i in [10,11,14]:
        ax.plot(100*ave_all_diff[:,i],'--o',color=c[c_i])
        c_i+=1    

    ax.set_ylabel('Difference in Error Rate (%)')
    fig.text(0.5, 0, 'Number of Noisy Electrodes', ha='center')
    ax.legend(['sae','cnn','sae-lda','cnn-lda','LDA','LDA-corrupt','LDA-ch'])
    ax.set_xticks(range(0,5))
    ax.set_xticklabels(['0','1','2','3','4'])
    ax.set_title(subtype + ', Train: ' + ntrain + ', test: ' + ntest)

    fig.set_tight_layout(True)

    return

def plot_pos_results(ave_pos):
    # Plot accuracy vs. position
    fig,ax = plt.subplots(3,3)
    for r in range(0,3):
        for i in [1,2,4]:
            ax[r,0].plot(ave_pos[r,:,i],'-o')
        for i in [6,7,9]:    
            ax[r,1].plot(ave_pos[r,:,i],'-o')            
        for i in [10,11,14]:
            ax[r,1].plot(ave_pos[r,:,i],'-o')    
        ax[r,1].set_yticks([])
        ax[r,2].set_yticks([])
        for i in range(0,3):
            ax[r,i].set_ylim(0,1)
            ax[r,i].set_xticks([])
            ax[2,i].set_xticks(range(0,4))
            ax[2,i].set_xticklabels(['1','2','3','4'])
    ax[1,0].set_ylabel('Accuracy')
    fig.text(0.5, 0, 'Limb Position', ha='center')
    ax[0,0].legend(['sae','cnn','svae','ecnn'])
    ax[0,1].legend(['sae-lda','cnn-lda','svae-lda','LDA','LDA-corrupt'])
    ax[0,2].legend(['LDA','LDA-corrupt'])

    fig.set_tight_layout(True)

    return

def plot_summary_old(acc_clean,acc_gauss,acc_60hz, acc_flat):
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
