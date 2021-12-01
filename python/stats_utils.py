import numpy as np
from matplotlib import pyplot as plt

def plot_fit(coef,T=1,Telec=1):
    ## plot best fit line
    x = np.linspace(0,5)
    c = ['k','r','m']
    c_tab = ['tab:purple','tab:blue', 'tab:orange', 'tab:purple','tab:blue', 'tab:orange','k','r','m']
    c_i = 0
    for n in [1,2,4,6,7,9,10,11,14]:
        y = coef['Intercept']+ T[n-1] + coef['elec']*x + Telec[n-1]*x
        plt.plot(x,y,color=c_tab[c_i])
        c_i+=1