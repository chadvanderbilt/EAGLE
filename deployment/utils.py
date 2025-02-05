import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns
sns.set_style('whitegrid')


def npv_score(true, pred):
    tn, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()
    if tn+fn > 0:
        npv = tn / (tn+fn)
    else:
        npv = np.nan
    return npv


def ppv_score(true, pred):
    tn, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()
    if tp+fp > 0:
        ppv = tp / (tp+fp)
    else:
        ppv = np.nan
    return ppv


def se_score(true, pred):
    tn, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()
    if tp+fn > 0:
        se = tp / (tp+fn)
    else:
        se = np.nan
    return se


def sp_score(true, pred):
    tn, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()
    if tn+fp > 0:
        sp = tn / (tn+fp)
    else:
        sp = np.nan
    return sp


def get_metrics_assisted(df, th0, th1, target_col='target', rapid_col='rapid', eagle_col='score'):
    '''
    df - pandas dataframe. Must have columns:
       * target_col: ground truth from NGS (0/1)
       * rapid_col: rapid test results (0/1)
       * eagle_col: slide level score from EAGLE (0-1 float)
    th0 - sensitivity/npv threshold list
    th1 - specificity/ppv threshold list
    target_col - target column name
    rapid_col - rapid test column name
    eagle_col - EAGLE score column name
    '''
    assisted = df[rapid_col].copy()
    assisted.loc[df[eagle_col]<th0] = 0
    assisted.loc[df[eagle_col]>th1] = 1
    tn, fp, fn, tp = metrics.confusion_matrix(df[target_col], assisted).ravel()
    if tn+fn > 0:
        npv = tn / (tn+fn)
    else:
        npv = np.nan
    if tp+fp > 0:
        ppv = tp / (tp+fp)
    else:
        ppv = np.nan
    if tp+fn > 0:
        se = tp / (tp+fn)
    else:
        se = np.nan
    if tn+fp > 0:
        sp = tn / (tn+fp)
    else:
        sp = np.nan
    return tn, fp, fn, tp, ppv, npv, se, sp, (df[eagle_col]<th0).sum()+(df[eagle_col]>th1).sum()


def get_performance_assisted(df, th0=None, th1=None, target_col='target', rapid_col='rapid', eagle_col='score'):
    '''
    df - pandas dataframe. Must have columns:
       * target_col: ground truth from NGS (0/1)
       * rapid_col: rapid test results (0/1)
       * eagle_col: slide level score from EAGLE (0-1 float)
    th0 - sensitivity/npv threshold list
    th1 - specificity/ppv threshold list
    target_col - target column name
    rapid_col - rapid test column name
    eagle_col - EAGLE score column name
    '''
    keys = ['tn', 'fp', 'fn', 'tp', 'ppv', 'npv', 'se', 'sp', 'reduction']
    perf = {x:[] for x in keys}
    for t0, t1 in zip(th0,th1):
        tn, fp, fn, tp, ppv, npv, se, sp, red = get_metrics_assisted(df, t0, t1, target_col=target_col, rapid_col=rapid_col, eagle_col=eagle_col)
        for k, v in zip(keys, [tn, fp, fn, tp, ppv, npv, se, sp, red/len(df)]):
            perf[k].append(v)
    
    tmp = pd.DataFrame(perf)
    tmp['th0'] = th0
    tmp['th1'] = th1
    return tmp


def get_performance_assisted_bootstrapped(df, th0=None, th1=None, n=1000, target_col='target', rapid_col='rapid', eagle_col='score'):
    '''
    df - pandas dataframe. Must have columns:
       * target_col: ground truth from NGS (0/1)
       * rapid_col: rapid test results (0/1)
       * eagle_col: slide level score from EAGLE (0-1 float)
    th0 - sensitivity/npv threshold list
    th1 - specificity/ppv threshold list
    n - bootstrap iterations
    target_col - target column name
    rapid_col - rapid test column name
    eagle_col - EAGLE score column name
    '''
    curves = []
    keys = ['ppv', 'npv', 'se', 'sp', 'reduction']
    for _ in range(n):
        tmp = df.sample(frac=1, replace=True).copy()
        tmp = get_performance_assisted(tmp, th0=th0, th1=th1, target_col=target_col, rapid_col=rapid_col, eagle_col=eagle_col)
        curves.append(tmp)
    curves = pd.concat(curves).reset_index(drop=True)
    out = []
    for k in keys:
        out.append(curves.groupby(['th0','th1']).agg(a=(k, lambda x:np.quantile(x, 0.025)), b=(k, lambda x:np.quantile(x, 0.975))).rename(columns={'a':f'{k}_lo','b':f'{k}_hi'}))
    
    out = pd.concat(out, axis=1)
    return out.reset_index()


def simulation(df, se_max=0.5, sp_min=0.5, N=101, target_col='target', rapid_col='rapid', eagle_col='score'):
    '''
    df - pandas dataframe. Must have columns:
       * target_col: ground truth from NGS (0/1)
       * rapid_col: rapid test results (0/1)
       * eagle_col: slide level score from EAGLE (0-1 float)
    se_max - max sensitivity threshold to restrict analysis
    sp_min - min specificity threshold to restric analysis
    N - steps in each threshold dimension
    target_col - target column name
    rapid_col - rapid test column name
    eagle_col - EAGLE score column name
    '''
    npv = np.zeros((N, N))
    ppv = np.zeros((N, N))
    bac = np.zeros((N, N))
    red = np.zeros((N, N))
    sen = np.zeros((N, N))
    spe = np.zeros((N, N))
    se_arr = np.linspace(0,se_max,N)
    sp_arr = np.linspace(sp_min,1,N)
    for i, se_th in enumerate(se_arr):
        for j, sp_th in enumerate(sp_arr):
            assisted = df[rapid_col].copy()
            assisted.loc[df[eagle_col]<se_th] = 0
            assisted.loc[df[eagle_col]>sp_th] = 1
            npv[i, j] = npv_score(df[target_col], assisted)
            ppv[i, j] = ppv_score(df[target_col], assisted)
            bac[i, j] = metrics.balanced_accuracy_score(df[target_col], assisted)
            sen[i, j] = se_score(df[target_col], assisted)
            spe[i, j] = sp_score(df[target_col], assisted)
            red[i, j] = (df[eagle_col]<se_th).sum() + (df[eagle_col]>sp_th).sum()
    
    red = red / len(df)
    X, Y = np.meshgrid(sp_arr,se_arr)
    return X, Y, npv, ppv, sen, spe, bac, red


def plot_simulation(df, metric='reduction', rapid_performance=None, rapid_confint=None, se_max=0.5, sp_min=0.5, N=201, target_col='target', rapid_col='rapid', eagle_col='score'):
    '''
    df - pandas dataframe. Must have columns:
       * target_col: ground truth from NGS (0/1)
       * rapid_col: rapid test results (0/1)
       * eagle_col: slide level score from EAGLE (0-1 float)
    metric - what metric to plot: ['reduction', 'npv', 'ppv']
    rapid_performance - tuple (npv, ppv) of rapid test performance. Optional
    rapid_confint - tuple (npv_inf, npv_sup, ppv_inf, ppv_sup) of rapid test performance. Optional
    se_max - max sensitivity threshold to restrict analysis
    sp_min - min specificity threshold to restric analysis
    N - steps in each threshold dimension
    target_col - target column name
    rapid_col - rapid test column name
    eagle_col - EAGLE score column name
    '''
    cmap = plt.cm.Greys
    colnpv = 'orchid'
    colppv = 'aquamarine'
    # Simulation
    X, Y, npv, ppv, sen, spe, bac, red = simulation(df, se_max=se_max, sp_min=sp_min, N=N, target_col=target_col, rapid_col=rapid_col, eagle_col=eagle_col)
    # Plot
    f, ax = plt.subplots(1,1)
    if metric == 'reduction':
        hmap = ax.imshow(red, extent=(sp_min, 1, se_max, 0), interpolation='none', cmap=cmap, vmin=0.1, vmax=1)
    elif metric == 'npv':
        hmap = ax.imshow(npv, extent=(sp_min, 1, se_max, 0), interpolation='none', cmap=cmap, vmin=0.1, vmax=1)
    elif metric == 'ppv':
        hmap = ax.imshow(ppv, extent=(sp_min, 1, se_max, 0), interpolation='none', cmap=cmap, vmin=0.1, vmax=1)

    ax.set_xlabel('EAGLE PPV Threshold')
    ax.set_ylabel('EAGLE NPV Threshold')
    cbar = f.colorbar(hmap, ax=ax, shrink=0.6, location='bottom')
    cbar.set_label(metric)
    
    # Contours if present
    if rapid_performance is not None:
        CSnpv_med = ax.contour(X, Y, npv, levels=[rapid_performance[0]], colors=colnpv)
        CSppv_med = ax.contour(X, Y, ppv, levels=[rapid_performance[1]], colors=colppv)
    if rapid_confint is not None:
        CSnpv_conf = ax.contour(X, Y, npv, levels=[rapid_confint[0], rapid_confint[1]], colors=colnpv, linestyles='dashed')
        CSppv_conf = ax.contour(X, Y, ppv, levels=[rapid_confint[2], rapid_confint[3]], colors=colppv, linestyles='dashed')
    
    f.tight_layout()
    f.show()

'''
# Simulation: full window
X0, Y0, npv0, ppv0, sen0, spe0, bac0, red0 = simulation(df, se_max=0.5, sp_min=0.5, N=201)

# Simulation: small window
#X1, Y1, npv1, ppv1, sen1, spe1, bac1, red1 = simulation(df, se_max=0.1, sp_min=0.9, N=201)
X1, Y1, npv1, ppv1, sen1, spe1, bac1, red1 = simulation(df, se_max=0.05, sp_min=0.95, N=201)


## IRT
path3 = get_performance_assisted(irt, th0=th0, th1=th1)
path3boot = get_performance_assisted_bootstrapped(irt, th0=th0, th1=th1, n=1000)
'''
