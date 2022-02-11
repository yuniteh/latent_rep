import pcepy.pce as pce

# Class dictionary
classmap = [1,10,11,12,13,16,19]
prop_mv = pce.get_var('CLASFR_MAV').to_np_array()

def dispose():
    pass

############################################# MAIN FUNCTION LOOP #######################################################
def run():
    ctrl = pce.get_var('CTRL')
    
    if ctrl == 2:
        mv = pce.get_var('MV_CLAS_OUT').to_np_array()[0,0]
        prop = pce.get_var('PROP_OUT').to_np_array()
        prop_mv[0,0] = prop[0,classmap == mv]
        pce.set_var('CLASFR_MAV', prop_mv.astype('float',order = 'F'))
        
