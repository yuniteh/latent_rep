########################################################################################################################
# FILE            : adaptiveVR_VirtualCoach.py
# VERSION         : 9.0.0
# FUNCTION        : Adapt upper limb data using an LDA classifier. To be used with Unity project.
# DEPENDENCIES    : None
# SLAVE STEP      : Replace Classify
#_author__ = 'lhargrove & rwoodward & yteh'
########################################################################################################################

# Import all the required modules. These are helper functions that will allow us to get variables from CAPS PC
import os

from numpy import extract
import pcepy.pce as pce
import pcepy.feat as feat
import numpy as np
import copy as cp
import time

# Class dictionary
classmap = [1,10,11,12,13,16,19]
# Specify where the saved data is stored.
datafolder = 'DATA'
datadir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', datafolder))
# Number of modes/classes.
numModes = int(len(classmap))
# Number of EMG channels.
numEMG = int(len(pce.get_var('DAQ_CHAN').to_np_array()[0]))
# Feature value ('47'; time domain and autoregression)
featVal = 15
# Number of features. (10 for '47')
featNum = 4
# Matrix size.
matSize = numEMG * featNum
# Threshold multiplier
thresX = 1.1
# Sample threshold
samp_thres = 100
# Voltage range of EMG signal (typically +/- 5V)
voltRange = 5
# True: enhanced proportional control is used, otherwise incumbent.
useEnhanced = True
# True: use CAPS MAV method, otherwise use self-calculated method.
CAPSMAV = False
# True: ramp enabled, otherwise ramp disabled.
rampEnabled = True
# Ramp time (in ms)
rampTime = 500
# Define the starting ramp numerators and denominators.
ramp_numerator = np.zeros((1, numModes), dtype=float, order='F')
ramp_denominator = np.ones((1, numModes), dtype=float, order='F') * (rampTime / pce.get_var('DAQ_FRINC'))
# DAQ UINT ZERO
DAQ_conv = (2**16-1)/2
try:
    # Neural network architectures
    mlp_temp = pce.get_var('ARCH')
    mlp_arch = np.array(mlp_temp.split('/'))
    emg_scale = pce.get_var('EMG_SCALE').to_np_array()
    x_min = np.tile(pce.get_var('X_MIN').to_np_array(),(numEMG,1)).T
    x_max = np.tile(pce.get_var('X_MAX').to_np_array(),(numEMG,1)).T
    
    w_all = []
    for l in mlp_arch:
        if 'CONV' in l:
            sh = pce.get_var(l + '_shape').to_np_array()
            temp = pce.get_var(l).to_np_array()
            w_all.append(temp[:-1,:].reshape(sh))
            w_all.append(temp[-1,:])
        else:
            w_all.append(pce.get_var(l).to_np_array())
    nn = True
    # NN forward pass
    class_out = pce.get_var('CLAS_OUT').to_np_array()
except:
    print('missing trained params')
    nn = False

pce.set_var('CTRL',2)

def dispose():
    pass

############################################# MAIN FUNCTION LOOP #######################################################
def run():
    # Don't do anything if PCE is training.
    if pce.get_var('TRAIN_STATUS') != 1:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PROCESS DATA
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ctrl = pce.get_var('CTRL')

        if ctrl == 2 and nn:
            # Get raw DAQ data for the .
            raw_DAQ = np.array(pce.get_var('DAQ_DATA').to_np_array()[0:numEMG,:], order='F')
            scaled_raw = emg_scale * (raw_DAQ.astype('float') - DAQ_conv) + DAQ_conv ## might be problem
            feat_scaled = feat.extract(featVal, scaled_raw.astype('uint16')) ## size = 1x24 (numfeat)
            feat_out = minmax(feat_scaled)

            nn_out, prop_out = nn_pass(feat_out, mlp_arch)
            class_out[0,0] = classmap[np.argmax(nn_out)]
            # print(class_out[0,0])
            pce.set_var('CLAS_OUT', class_out.astype(float, order='F'))
            pce.set_var('PROP_OUT', prop_out.astype(float, order='F'))
            
#######################################################################################################################
# Function    : initialiseVariables(args)
# args        : None.
# Description : This function is used to initialise variables when starting for the first time, or when resetting.
#######################################################################################################################
def initialiseVariables():
    pce.set_var('TRAIN_FLAG', -1)
    pce.set_var('SEND_PD', 0)
    pce.set_var('CLASS_EST', -1)
    pce.set_var('THRESH_VAL', 0)
    pce.set_var('NEW_CLASS', 0)
    pce.set_var('CLASS_ACTIVE', 0)    
    pce.set_var('ADAPT_ON', 0)
    pce.set_var('ADAPT_GT', -1)
    pce.set_var('DNT_ON', 0)
    pce.set_var('TARGET_DOF', 0)
    pce.set_var('TARGET_ARM', 0)
    pce.set_var('TRIAL_FLAG', 0)
    pce.set_var('ARM_FLAG', 1)
    pce.set_var('SAVE', 0)
    pce.set_var('COLLECTING',0)
    pce.set_var('OUT_MAP', np.zeros((1, numModes), dtype=float, order='F'))
    pce.set_var('WG_DATA', np.zeros((matSize, numModes), dtype=float, order='F'))
    pce.set_var('CG_DATA', np.zeros((1, numModes), dtype=float, order='F'))
    pce.set_var('N_C', np.zeros((1, numModes), dtype=float, order='F'))
    pce.set_var('N_R', np.zeros((1, numModes), dtype=float, order='F'))
    pce.set_var('N_T', np.zeros((1, numModes), dtype=float, order='F'))
    pce.set_var('S_CONTROL', np.zeros((numEMG, numModes), dtype=float, order='F'))
    pce.set_var('PROP_CONTROL', np.zeros((1, numModes), dtype=float, order='F'))
    for i in range(0, numModes):
        pce.set_var('COV' + str(i), np.zeros((matSize, matSize), dtype=float, order='F'))
        pce.set_var('MN' + str(i), np.zeros((1, matSize), dtype=float, order='F'))
        pce.set_var('CLASS_MAV' + str(i), 0)

#######################################################################################################################
# Function    : classPreparer(args)
# args        : flag, transmitted value for identification: feat_data, the current feature data: 
#             : chan_mav, current MAV for all EMG channels: update_var, argument to update: adapt, 0/1 adapt off/on: 
#             : dnt, 0/1 do not train off/on.
# Description : This function is used to remove redundant repetition of code between no-movement and all other classes.
#######################################################################################################################
def classPreparer(flag, feat_data, chan_mav, update_var, adapt, dnt):
    # Only build up weights if dnt is turned off.
    if dnt == 0:
        pce.set_var('COLLECTING',1)
        # cov and mean are LDA variables.
        cov_C = pce.get_var('COV' + str(flag)).to_np_array()
        mean_C = pce.get_var('MN' + str(flag)).to_np_array()
        # N_C: Total number of windows used for training. This will increment to Inf.
        N_C = pce.get_var('N_C').to_np_array()
        # Get enhanced technique variables.
        s_control = pce.get_var('S_CONTROL').to_np_array()

        # Update the running average of training windows.
        update_val = updateAverage(pce.get_var(update_var), np.average(chan_mav), N_C[0, flag])
        pce.set_var(update_var, update_val)

        # Update the cov and mean for LDA classification.
        (mean_C, cov_C, N_C[0, flag]) = updateMeanAndCov(mean_C, cov_C, N_C[0, flag], feat_data)

        # Determie enhanced proportional control.
        # Loop through all EMG channels.
        for i in range(0, numEMG):
            # Summate the current average EMG channel MAV with s_control.
            # Each class will have its own addition of EMG MAV windows.
            s_control[i, flag] += chan_mav[i, 0]

        # Update cov, mean, and total
        pce.set_var('COV' + str(flag), cov_C)
        pce.set_var('MN' + str(flag), mean_C)
        pce.set_var('N_C', N_C)
        # Update proportional control variables
        pce.set_var('S_CONTROL', s_control)

    # Only perform the following section if running regular training (not adaptation).
    if adapt == 0:
        # N_T: Number of windows used for training on the current repetition. 
        N_T = pce.get_var('N_T').to_np_array()
        # Once the tmp training counter reaches the threshold, stop collecting data for training.
        if N_T[0, flag] == (samp_thres - 1):
            # Again, only update if dnt is turned off.
            if dnt == 0:
                # N_R: Number of training repetitions.
                N_R = pce.get_var('N_R').to_np_array()
                # Increment the repetition variable to indicate a new training session has been completed.
                N_R[0, flag] += 1        
                pce.set_var('N_R', N_R)
                # Set new_class to 1. This will indicate that a new training session is ready to be trained.
                pce.set_var('NEW_CLASS', 1)
            # Toggle the class_activate variable to 0.
            pce.set_var('CLASS_ACTIVE', 0)
            # Set the train_flag back to its standby value of -1.
            pce.set_var('TRAIN_FLAG', -1)
            pce.set_var('COLLECTING',0)
        else:
            # Increment and set the temp training counter separately.
            N_T[0, flag] = N_T[0, flag] + 1
            pce.set_var('N_T', N_T)

#######################################################################################################################
# Function    : updateWgAndCg(args)
# args        : wg_data, adapted wg weights: cg_data, adapted cg weights: classList, list of classes trained.
# Description : This function iteratively updates wg and cg matrices.
#######################################################################################################################            
def updateWgAndCg(wg_data, cg_data, classList):
    tmp_wg = pce.get_var('WG_DATA').to_np_array()
    tmp_cg = pce.get_var('CG_DATA').to_np_array()
    for idx, i in enumerate(classList):
        tmp_wg[:, classList[idx]] = wg_data[:, idx]
        tmp_cg[0, classList[idx]] = cg_data[0, idx]
    pce.set_var('WG_DATA', tmp_wg)
    pce.set_var('CG_DATA', tmp_cg)

#######################################################################################################################
# Function    : updateMeanAndCov(args)
# args        : mean_mat, the previous mean: cov_mat: the previous covariance: N: the number of points, cur_feat: the current feature vector
# Description : This function iteratively updates means and covariance matrix based on a new feature point.
#######################################################################################################################
def updateMeanAndCov(mean_mat, cov_mat, N, cur_feat):
    ALPHA = N / (N + 1)
    zero_mean_feats_old = cur_feat - mean_mat                                    # De-mean based on old mean value
    mean_feats = ALPHA * mean_mat + (1 - ALPHA) * cur_feat                       # Update the mean vector
    zero_mean_feats_new = cur_feat - mean_feats                                  # De-mean based on the updated mean value
    point_cov = np.dot(zero_mean_feats_old.transpose(), zero_mean_feats_new)
    point_cov = np.array(point_cov, np.float64, order='F')
    mean_feats = np.array(mean_feats, np.float64, order='F')
    cov_updated = ALPHA * cov_mat + (1 - ALPHA) * point_cov                      # Update the covariance
    N = N + 1

    return (mean_feats, cov_updated, N)

#######################################################################################################################
# Function    : updateMean(args)
# args        : mean_mat, the previous mean: N: the number of points, cur_feat: the current feature vector
# Description : This function iteratively updates means based on a new feature point.
#######################################################################################################################
def updateMean(mean_mat, N, cur_feat):
    ALPHA = N/(N+1)
    mean_feats = ALPHA * mean_mat + (1 - ALPHA) * cur_feat                       # Update the mean vector
    mean_feats = np.array(mean_feats, np.float64,order='F')
    N = N + 1
    
    return (mean_feats, N)

def updateAverage(prev_val, avg_chan_mav, N):
    ALPHA = N / (N + 1)
    new_val = ALPHA * prev_val + (1 - ALPHA) * avg_chan_mav
    
    return new_val

#######################################################################################################################
# Function    : makeLDAClassifier(args)
# args        : class_list, the list of class labels in the classifier
# Description : Will compute the LDA weights and biases.
#######################################################################################################################
def makeLDAClassifier(class_list):
    for i in class_list:
        if i == 0:                                                              # Build pooled covariance, assumes that no-movment is always involved
            pooled_cov = pce.get_var('COV' + str(i)).to_np_array();
        else:
            tmpVal = pce.get_var('COV' + str(i)).to_np_array();
            pooled_cov += tmpVal

    num_classes = np.shape(class_list)
    pooled_cov = pooled_cov / num_classes[0]
    inv_pooled_cov = np.linalg.inv(pooled_cov)                                  # Find the pooled inverse covariance matrix
    inv_pooled_cov = np.array(inv_pooled_cov, np.float64, order='F')
    pce.set_var('INVPOOL', inv_pooled_cov)

    for i in class_list:
        mVal = pce.get_var('MN' + str(i)).to_np_array();
        tmpWg = np.dot(inv_pooled_cov, mVal.T)
        tmpCg = -0.5 * (mVal.dot(inv_pooled_cov).dot(mVal.T))

        if i == 0:
            Wg = tmpWg;
            Cg = tmpCg;
        else:
            Wg = np.concatenate((Wg,tmpWg), axis=1)
            Cg = np.concatenate((Cg,tmpCg), axis=1)

    Wg = np.array(Wg, np.float64, order='F')
    Cg = np.array(Cg, np.float64, order='F')

    return (Wg, Cg)

#######################################################################################################################
# Function    : saveWeights(args)
# args        : name: the name of the file.
# Description : Save weights to csv file.
#######################################################################################################################
def saveWeights(dir, name):
    np.savetxt(dir + '/CSV/' + name + '_wg.csv', pce.get_var('WG_DATA').to_np_array(), fmt='%.20f', delimiter=',')
    np.savetxt(dir + '/CSV/' + name + '_cg.csv', pce.get_var('CG_DATA').to_np_array(), fmt='%.20f', delimiter=',')

#######################################################################################################################
# Function    : clearWeights(args)
# args        : None.
# Description : clear csv weight files.
#######################################################################################################################
def clearWeights():
    np.savetxt(datadir + '/wg.csv', [], fmt='%.20f', delimiter=',')
    np.savetxt(datadir + '/cg.csv', [], fmt='%.20f', delimiter=',')

#######################################################################################################################
# Functions    : dense, bn, nn_pass(args)
# args        : x_in: input, w: layer weights, fxn: activation fxn, arch: string of layer type or act fxn
# Description : forward pass through trained nn
#######################################################################################################################
def dense(x_in, w, fxn = 'RELU'):
    out = np.dot(x_in,w[:-1,:]) + w[-1,:]
    if 'RELU' in fxn:
        out = relu(out)
    elif 'SOFTMAX' in fxn:
        out = softmax(out)
    return out

def bn(x_in, w):
    out = ((w[0,:] * (x_in - w[2,:])) / np.sqrt(w[3,:] + 0.001)) + w[1,:]
    return out

def conv(x_in, w, w2, stride=1, k = (3,3), fxn = 'relu'):
    out = np.zeros((x_in.shape[0], 1+(x_in.shape[1]-k[0]+2)//stride, 1+(x_in.shape[2]-k[1]+2)//stride, w[0].shape[-1]))
    for f in range(w[0].shape[-1]):
        padded = np.pad(x_in,pad_width = ((0,0),(1,1),(1,1),(0,0)))

        i = 0
        for row in range(out.shape[1]):
            j = 0
            for col in range(out.shape[2]):
                out[:,row,col,f] = np.sum(np.sum(np.sum(padded[:,i:i+k[0],j:j+k[1],:]*w[0][...,f],axis=-1),axis=-1),axis=-1,keepdims=False) + w2[f]
                j += stride
            i += stride
        
    if fxn == 'relu':
        out = relu(out)
    
    return out

def nn_pass(x, arch):
    if 'PROP' not in arch:
        prop = 0
    i = 0
    for l in arch:
        w = w_all[i]
        if 'BN' in l:
            x = bn(x, w)
        elif 'CONV' in l:
            w2 = w_all[i+1]
            x = conv(x, w, w2)
            i += 1
        elif 'FLAT' in l:
            x = np.reshape(x,(x.shape[0],-1))
        elif 'PROP' in l:
            prop = dense(prev_x, w, fxn = 'RELU')
        else:
            if 'SOFTMAX' in l:
                prev_x = cp.deepcopy(x)
            x = dense(x, w, fxn = l)
        i += 1
            
    return x, prop

def minmax(x):
    return (x - x_min) / (x_max - x_min)

def softmax(x):
    out = np.exp(x) / np.sum(np.exp(x), axis=1)[...,np.newaxis]
    return out

def relu(x):
    out = x * (x > 0)
    return out