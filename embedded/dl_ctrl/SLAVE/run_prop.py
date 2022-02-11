import pcepy.pce as pce
import pcepy.feat as feat
import numpy as np

# Class dictionary
classmap = [1,10,11,12,13,16,19]

def dispose():
    pass

############################################# MAIN FUNCTION LOOP #######################################################
def run():

    mv = pce.get_var('MV_CLAS_OUT').to_np_array()[0,0]
    prop = pce.get_var('PROP_OUT').to_np_array()

    ctrl = pce.get_var('CTRL')

    if ctrl == 2:
        prop_mv = prop[0,classmap == mv]
        pce.set_var('NN_MAV', prop_mv)
