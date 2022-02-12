import pcepy.pce as pce
import numpy as np
import pcepy.feat as feat

# Class dictionary
classmap = [1,10,11,12,13,16,19]
# DAQ UINT ZERO
DAQ_conv = (2**16-1)/2
# Number of EMG channels.
numEMG = int(len(pce.get_var('DAQ_CHAN').to_np_array()[0]))
# EMG scale
emg_scale = pce.get_var('EMG_SCALE').to_np_array()

def dispose():
    pass

############################################# MAIN FUNCTION LOOP #######################################################
def run():
    ctrl = pce.get_var('CTRL')
    
    if ctrl == 2:
        raw_DAQ = np.array(pce.get_var('DAQ_DATA').to_np_array()[0:numEMG,:], order='F')
        pce.set_var('DAQ_DATA1', raw_DAQ.astype('uint16'))
        scaled_raw = emg_scale * (raw_DAQ.astype('float') - DAQ_conv) + DAQ_conv ## might be problem
        
        # raw_DAQ[:] = 0
        pce.set_var('DAQ_DATA3', scaled_raw.astype('uint16'))
        pce.set_var('DAQ_DATA', scaled_raw.astype('uint16'))

        # feat_scaled = feat.extract(15, scaled_raw.astype('uint16')) ## size = 1x24 (numfeat)
        # pce.set_var('FEAT_DATA3',feat_scaled.astype(float, order='F'))
        
