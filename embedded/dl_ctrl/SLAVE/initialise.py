# README:
# This user/mode relies on saving variables to PVD so that they're retrievable on power cycles.
# On a first boot of the user/mode you may see an error stating 'Error while request to save PCE variables:
# Error returned from PCE Variable file open routine'. This error is produced because the PCEVarList.cfg file
# located in /config/mode/caps/*MODENAME*/ is trying to load variables (e.g. ADAPT_FLAG) which don't currently 
# exist, and therefore cannot be loaded. To resolve this issue, run this script to initialise the variables so
# CAPS can find them. You should only need to run this once. If you're seeing the error often, contact CBM. 
#   1: Open PuTTY and navigate to the mode's SLAVE FOLDER:
#       a) cd /
#       b) cd /config/mode/caps/*MODENAME*/SLAVE/ 
#   2: Type the following:
#       a) python initialise.py
#   3: A message should state that initialise is running, and then a second message stating that it has completed.
#   4: Close PuTTY and power cycle the device. Null variables will now exist in the system and CAPS should operate normally.

import pcepy.pce as pce
import numpy as np

# Number of modes/classes.
numModes = 11
# Number of EMG channels.
numEMG = int(len(pce.get_var('DAQ_CHAN').to_np_array()[0]))

# Number of features. (10 for '47')
featNum = 10
# Matrix size.
matSize = numEMG * featNum

print('RUNNING INITIALISE...')

pce.set_var('TRAIN_FLAG', -1)
pce.set_var('PD_FLAG', -1)
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
pce.set_var('ARM_FLAG', 0)
pce.set_var('SAVE', 0)
pce.set_var('COLLECTING',0)
pce.set_var('POS', np.zeros((5,3), dtype=float, order='F'))
pce.set_var('ROT', np.zeros((5,3), dtype=float, order='F'))
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
    
print('INITIALISE COMPLETE')