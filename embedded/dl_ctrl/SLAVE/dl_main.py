########################################################################################################################
# FILE            : adaptiveVR_VirtualCoach.py
# VERSION         : 9.0.0
# FUNCTION        : Adapt upper limb data using an LDA classifier. To be used with Unity project.
# DEPENDENCIES    : None
# SLAVE STEP      : Replace Classify
__author__ = 'lhargrove & rwoodward & yteh'
########################################################################################################################

# Import all the required modules. These are helper functions that will allow us to get variables from CAPS PC
import os
import csv
import pcepy.pce as pce
import pcepy.feat as feat
import numpy as np

# Class dictionary
classmap = {0: 'NO MOVEMENT',
            1: 'HAND OPEN',
            2: 'HAND CLOSE',
            3: 'WRIST PRO.',
            4: 'WRIST SUP.',
            5: 'WRIST FLEX.',
            6: 'WRIST EXT.',
            7: 'WRIST ADD.',
            8: 'WRIST ABD.',
            9: 'ELBOW FLEX.',
            10: 'ELBOW EXT.'}
# Specify where the saved data is stored.
datafolder = 'DATA'
datadir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', datafolder))
# Number of modes/classes.
numModes = int(len(classmap))
# Number of EMG channels.
numEMG = int(len(pce.get_var('DAQ_CHAN').to_np_array()[0]))
# Feature value ('47'; time domain and autoregression)
featVal = 47
# Number of features. (10 for '47')
featNum = 10
# Matrix size.
matSize = numEMG * featNum
# Threshold multiplier
thresX = 1.1
# Sample threshold
samp_thres = 100
# Voltage range of EMG signal (typically +/- 5V)
voltRange = 5
# Signal boost, gain, and threshold variables (threshold between 1 and 100)
sig_boost = 2
sig_gain = 1
sig_thres = 1
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

def dispose():
    pass

############################################# MAIN FUNCTION LOOP #######################################################
def run():
    # Try/catch to see if data already exists.
    try:
        pce.get_var('TRAIN_FLAG')
    except:
        # Initialise all variables.
        initialiseVariables()

    # Don't do anything if PCE is training.
    if pce.get_var('TRAIN_STATUS') != 1:
    
        # Get global variables.
        # global adaptNMCounter                                                 #yt
        # Define local variables.
        flag = int(pce.get_var('TRAIN_FLAG'))
        dnt_on = int(pce.get_var('DNT_ON'))
        N_R = pce.get_var('N_R').to_np_array()
        class_est = 0
        armflag = int(pce.get_var('ARM_FLAG'))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PROCESS DATA
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get raw DAQ data for the .
        raw_DAQ = np.array(pce.get_var('DAQ_DATA').to_np_array()[0:numEMG,:], order='F')
        # Get converted DAQ data between +/- voltRange.
        raw_conv = (raw_DAQ.astype(float) / (np.power(2, 16) - 1)) * (voltRange * 2) - voltRange
        
        # Get channel MAV.
        if CAPSMAV:
            chan_mav = pce.get_var('CHAN_MAV').to_np_array()[0:numEMG]
        else:
            # Get the absolute value of the data window.
            raw_abs = np.abs(raw_conv)
            # Get the average of the window.
            chan_mav = np.transpose([np.average(raw_abs, axis=1)])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # CLASSIFICATION
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract features from raw data
        feat_data = feat.extract(featVal, raw_DAQ)

        if pce.get_var('SAVE') == 1:
            dir_full = pce.get_var('DAQ_OUT_FNAME')
            dir = dir_full.split('/')
            daqname = dir[-1]
            dir = daqname.split('.')
            name = dir[0]
            ind = dir_full.rfind('/')
            ind = dir_full[0:ind].rfind('/')
            print(dir_full[0:ind])
            
            saveWeights(dir_full[0:ind],name)
            pce.set_var('SAVE', 0)

        # IF FLAG == RESET
        if (flag == 999):
            print('RESET')
            # All variables will be initialised back to 0 (or their initial values).
            initialiseVariables()
                            
        # IF FLAG == INITIAL CLASS TRAINER ACTIVATED
        elif (pce.get_var('CLASS_ACTIVE') == 0) & ((flag >= 0) & (flag < 99)):
            N_T = pce.get_var('N_T').to_np_array()
            # Reset the temp training counter for the specific class.
            N_T[0, flag] = 0
            pce.set_var('N_T', N_T)
            # Toggle the class_active variable to 1.
            pce.set_var('CLASS_ACTIVE', 1)            

        # IF FLAG == NO MOVEMENT
        elif (flag == 0):
            print 'here'
            if (armflag != 0):
                print('COLLECTING ' + classmap[flag])
                # Prepare data for the 'no movement' class. Create means and covariance matricies.
                classPreparer(flag, feat_data, chan_mav, 'THRESH_VAL', 0, dnt_on)
            else:
                pce.set_var('COLLECTING',0)

        # IF FLAG == ANY OTHER CLASS
        elif (99 > flag >= 1) & (N_R[0, 0] >= 1): 
            # Compare the current channel MAV against the no movement threshold, if it exceeds then continue.
            if (armflag != 0) & (np.average(chan_mav) > (thresX * pce.get_var('THRESH_VAL'))):
                print('COLLECTING ' + classmap[flag])
                # Prepare data for any movement other than 'no movement'. Create means and covariance matricies.
                classPreparer(flag, feat_data, chan_mav, ('CLASS_MAV' + str(flag)), 0, dnt_on)
            else:
                pce.set_var('COLLECTING',0)

        # CLASSIFY AND FORWARD PASS
        # To classify the flag must be 0, 'no motion' data must be collected (N_R[0,0] >= 1).
        elif (flag == -1) & (N_R[0, 0] >= 1.0):
            out_map = pce.get_var('OUT_MAP').to_np_array()
            # Check that there is a new class to train.
            if pce.get_var('NEW_CLASS') == 1:
                print('TRAINING')
                # Create vector with just the values of classes trained (for remapping purposes).
                classList = np.nonzero(N_R)[1]
                # Update out_map.
                out_map[0,0:len(classList)] = classList
                pce.set_var('OUT_MAP', out_map)
                # If channel data is poor, the LDA will fail to classify and will throw a singular matrix error. Catch this error.
                try:
                    # Train using an LDA classifier.
                    (wg_data, cg_data) = makeLDAClassifier(classList)   
                    # Add weights to WG and CG arrays and set to PCE.
                    updateWgAndCg(wg_data, cg_data, classList)
                    # Toggle new_class parameter.
                    pce.set_var('NEW_CLASS', 0)
                except: 
                    print('ERROR: Bad pooled covariance data resulting in singular matrix.')
            # Get weights. Remove non-trained columns.
            wg_data = pce.get_var('WG_DATA').to_np_array()[:, out_map.tolist()[0]]
            cg_data = pce.get_var('CG_DATA').to_np_array()[0, out_map.tolist()[0]]
            # Forward pass to get estimate.
            lda_out = (np.dot(feat_data, wg_data) + cg_data)
            # Take argmax and remap for class value (i.e. 0 for 'no movement')
            class_est =  float(out_map[0, (np.argmax(lda_out))])
            # Set estimate to PCE.
            pce.set_var('CLASS_EST', class_est)
            # Print message with class estimation
            print('FORWARD - ' + str(class_est))
                    
        # DO NOTHING
        # In the event that no classes have been trained, print a message to the PCE log.
        # This statement is purely for debugging/logging purposes. It can be removed if necessary.
        else:
            print('NO ACTION')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PROPORTIONAL CONTROL
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if useEnhanced:
            # Enhanced Technique.
            # Get total number of windows per class.
            N_C = pce.get_var('N_C').to_np_array()
            # Get summation of class/channel MAV.
            s_control = pce.get_var('S_CONTROL').to_np_array()
            # Determine average.
            s_controlAvg = s_control * (1 / N_C)
            # Calculate C from the equation.
            c_control = np.sum(np.square(s_controlAvg), axis=0)
            # Calculate proportional control.
            X = np.square((1 / c_control) * (np.sum(s_controlAvg * chan_mav, axis=0)))
        else:
            # Incumbent Technique.
            X = np.ones((1, numModes)) * np.average(chan_mav)
        # Pre-filled matrix may contain NaN's and Inf's. If they exist make them zeros.
        for i in range(0, X.shape[0]):
            if (np.isnan(X[i]) or np.isinf(X[i])):
                X[i] = 0.0
                
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # RAMPING
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if rampEnabled and (rampTime != 0):
            # Step 1: bias each class numerator as not selected by subtracting 2 from numerator
            for i in range(0, numModes):
                ramp_numerator[:, i] -= 2
            # Step 2: for any active DOF, reverse bias by adding 2, then add 1 more to decrease attenuation for that single class (total add = 3)
            for i in range(0, numModes):
                # Apply to active class as long as it's not 'no motion'.
                if ((i == class_est) and (class_est != 0)):
                    # Increment
                    ramp_numerator[:, i] += 3
                    # We never want to amplify, so make sure to cap the numerator.
                    if (ramp_numerator[:, i] > ramp_denominator[:, i]):
                        ramp_numerator[:, i] = ramp_denominator[:, i]
                    # Scale signal
                    X[i] = X[i] * ramp_numerator[:, i] / ramp_denominator[:, i]
                # Limit so that any non-active class numerator never falls below zero.
                if (ramp_numerator[:, i] < 0):
                    ramp_numerator[:, i] = 0.0

        # The original equation is 'O_j = B_j[G_j(X) - T_j]', where: 
        # O_j is output speed, B_j is the boost, G_j is the gain, T_j is the threshold, and X is the channel MAV.
        # For PR the threshold and gain are typically 1, while the boost is 2.
        #prop_control = sig_boost * ((sig_gain * X) - ((voltRange / 100.0) * sig_thres))
        prop_control = X
        # If the value of prop_control is below 0, it could be because of the threshold equation. If so, make the values 0.
        for i in range(0, prop_control.shape[0]):
            if (prop_control[i] < 0):
                prop_control[i] = 0.0
        # Set proportional control value to PCE variable.
        pce.set_var('PROP_CONTROL', np.array(prop_control, dtype=float, order='F'))        
    else:
        print('TRAINING HERE...')
#######################################################################################################################
# Function    : initialiseVariables(args)
# args        : None.
# Description : This function is used to initialise variables when starting for the first time, or when resetting.
#######################################################################################################################
def initialiseVariables():
    pce.set_var('TRAIN_FLAG', -1)
#    pce.set_var('PD_FLAG', -1)
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
