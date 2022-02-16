import pcepy.pce as pce
import copy as cp
import numpy as np

# Class dictionary
classmap = [1,10,11,12,13,16,19]
prop_mv = pce.get_var('CLASFR_MAV').to_np_array()
sim = np.zeros((2,))
pce.set_var('COUNTER',0)
pce.set_var('SIM_COUNT', np.zeros((10,2),dtype=float,order='F'))

def dispose():
    pass

############################################# MAIN FUNCTION LOOP #######################################################
def run():
    ctrl = pce.get_var('CTRL')
    mv = pce.get_var('MV_CLAS_OUT').to_np_array()[0,0]
    sim_ctrl = pce.get_var('SIM_CTRL')
    if ctrl == 2:
        prop = pce.get_var('PROP_OUT').to_np_array()
        if mv < 0:
            mv = 1
        prop_mv[0,0] = prop[0,classmap == mv]
        pce.set_var('CLASFR_MAV', prop_mv.astype('float',order = 'F'))

        if sim_ctrl == 1:
            prop_temp = cp.deepcopy(prop)
            sim[0] = np.argmax(classmap == mv)
            if sim[0]%2 == 0:
                if sim[0] != 0:
                    prop_temp[0,sim[0]-1] = -1
            elif sim[0] < prop_temp.shape[1]-1:
                prop_temp[0,sim[0]+1] = -1
            if sim[0] != 0:
                prop_temp[0,sim[0]] = -1
            sim[1] = np.argmax(prop_temp)

            counter = pce.get_var('COUNTER')
            sim_count = pce.get_var('SIM_COUNT').to_np_array()
            sim_count = np.roll(sim_count,-1,axis=0)
            sim_count[counter,0] = sim[0]
            sim_count[counter,1] = sim[1]
            values,counts = unique1d(sim_count[:])
            values[values < 0] = 0
            counts[values==sim[0]] = -1
            if counts.argmax() > 0 and sim[0] > 0:
                sim[1] = values[counts.argmax()]
            else:
                sim[1] = 0

            if counter < 9:
                counter += 1
            pce.set_var('SIM_COUNT', sim_count.astype('float',order='F'))
            pce.set_var('SIM_OUT', sim.astype('float',order='F'))
            pce.set_var('COUNTER',counter)

        print('nn: ' + str(sim[0]) + ', ' + str(sim[1]) + ', p: ' + "{:.2f}".format(prop[0,sim[0]]) + ', ' + "{:.2f}".format(prop[0,sim[1]]))
    else:
        print('lda: ' + str(mv))


def unique1d(ar, return_counts=True):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = np.asanyarray(ar).flatten()

    ar.sort()
    aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    if aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and np.isnan(aux[-1]):
        if aux.dtype.kind == "c":  # for complex all NaNs are considered equivalent
            aux_firstnan = np.searchsorted(np.isnan(aux), True, side='left')
        else:
            aux_firstnan = np.searchsorted(aux, aux[-1], side='left')
        if aux_firstnan > 0:
            mask[1:aux_firstnan] = (
                aux[1:aux_firstnan] != aux[:aux_firstnan - 1])
        mask[aux_firstnan] = True
        mask[aux_firstnan + 1:] = False
    else:
        mask[1:] = aux[1:] != aux[:-1]

    ret = (aux[mask],)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret