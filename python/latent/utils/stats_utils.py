import numpy as np
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import pandas as pd

def create_dataframe(acc_clean,acc_noise):
    acc_clean[...,-1] = acc_clean[...,10]
    acc_clean = acc_clean[~np.isnan(acc_clean[:,0,0,0]),...]
    acc_noise = acc_noise[~np.isnan(acc_noise[:,0,0,0]),...]

    data = np.squeeze(np.hstack((acc_clean[:,[0],:,:], acc_noise))).reshape([-1])
    mask = np.ones((data.shape))
    mask[np.isnan(data)] = 0
    data = data[mask.astype(bool)]

    sub = 1
    sub_array = np.zeros(data.shape)
    temp_elec = np.zeros((acc_clean.shape[-1]*5,))
    elec = 0
    for i in range(0,acc_clean.shape[-1]*5,acc_clean.shape[-1]):
        temp_elec[i:i+acc_clean.shape[-1]] = elec
        elec+=1
    elec_array = np.tile(temp_elec,(acc_clean.shape[0],))
    for i in range(0,data.shape[0],acc_clean.shape[-1]*5):
        sub_array[i:i+acc_clean.shape[-1]*5,] = sub
        sub+=1
    mod_array = np.tile(np.arange(acc_clean.shape[-1]),(acc_clean.shape[0]*5,))
    data = 100*(1-data)
    data = np.stack((data,sub_array,elec_array,mod_array))
    df = pd.DataFrame(data.T,columns=['acc','sub','elec','mod'])

    df['elec2'] = df['elec']**2
    df['elec3'] = df['elec']**3
    df['elec4'] = df['elec']**4

    return df

def get_mods(df):
    if 0: # old models
        out_df = df[(df['mod']!=0) & (df['mod']!=3)& (df['mod']!=5)& (df['mod']!=8)& (df['mod']!=12)& (df['mod']!=13)]
    else:
        out_df = df[(df['mod']==7) | (df['mod']==14)| (df['mod']==11)| (df['mod']==10)| (df['mod']==6)]

    return out_df

def run_fit(df,ctrl=10):
    md = smf.mixedlm("acc ~ C(mod,Treatment(" + str(ctrl) + ")) + C(mod,Treatment(" + str(ctrl) + "))*elec", df,groups=df["sub"])

    all_md = {}

    mdf = md.fit()
    print(mdf.summary())
    all_md['main'] = mdf

    for i in np.unique(df['elec']):
        print(i)
        new_df = df[df['elec'] == i]
        md2 = smf.mixedlm("acc ~ C(mod,Treatment(" + str(ctrl) + "))", new_df, groups=new_df["sub"])
        mdf2 = md2.fit()
        print(mdf2.summary())
        all_md[str(i)] = mdf2

    return all_md

def get_coefs(mdf):
    coef = mdf.params
    i = 0
    T = np.zeros(14,)
    Telec = np.zeros(14,)
    Telec2 = np.zeros(14,)
    Telec3 = np.zeros(14,)
    for ind in coef.index:
        for iter in range(14):
            if iter > 8:
                st = 7
            else:
                st = 6
            str_i = 'T.' + str(iter+1) + '.0]'
            if ind[-st:] == str_i:
                T[iter] = coef[i]
            str_i = 'T.' + str(iter+1) + '.0]:elec'
            if ind[-(st+5):] == str_i:
                Telec[iter] = coef[i]
            str_i = 'T.' + str(iter+1) + '.0]:elec2'
            if ind[-(st+6):] == str_i:
                Telec2[iter] = coef[i]
            str_i = 'T.' + str(iter+1) + '.0]:elec3'
            if ind[-(st+6):] == str_i:
                Telec3[iter] = coef[i]
        i +=1
    
    return T, Telec, Telec2, Telec3

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
    
    return
