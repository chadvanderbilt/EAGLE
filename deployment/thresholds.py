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

def get_ppv_npv(df, th, target_name='target', score_name='score'):
    tn, fp, fn, tp = metrics.confusion_matrix(df[target_name], (df[score_name]>=th).astype(int)).ravel()
    if tn+fn > 0:
        npv = tn / (tn+fn)
    else:
        npv = np.nan
    if tp+fp > 0:
        ppv = tp / (tp+fp)
    else:
        ppv = np.nan
    return tn, fp, fn, tp, ppv, npv, (df[score_name]<th).sum(), (df[score_name]>=th).sum()

def get_metrics_assisted(df, th0, th1, target_name='target', score_name='score'):
    assisted = df.idylla.copy()
    assisted.loc[df.score<th0] = 0
    assisted.loc[df.score>th1] = 1
    tn, fp, fn, tp = metrics.confusion_matrix(df[target_name], assisted).ravel()
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
    return tn, fp, fn, tp, ppv, npv, se, sp, (df[score_name]<th0).sum()+(df[score_name]>th1).sum()

def get_metrics(df, th, target_name='target', score_name='score'):
    tn, fp, fn, tp = metrics.confusion_matrix(df[target_name], (df[score_name]>=th).astype(int)).ravel()
    try:
        se = tp / (tp+fn)
    except:
        se = np.nan
    try:
        sp = tn / (tn+fp)
    except:
        sp = np.nan
    try:
        npv = tn / (tn+fn)
    except:
        npv = np.nan
    try:
        ppv = tp / (tp+fp)
    except:
        ppv = np.nan
    return tn, fp, fn, tp, se, sp, ppv, npv, (df[score_name]<th).sum(), (df[score_name]>=th).sum()

def get_performance2(df):
    keys = ['tn', 'fp', 'fn', 'tp', 'se', 'sp', 'ppv', 'npv', 'below', 'above']
    perf = {x:[] for x in keys}
    thresholds = [-0.1] + np.linspace(0, 1, 10001).tolist() + [1.1]
    for th in thresholds:
        tn, fp, fn, tp, se, sp, ppv, npv, below, above = get_metrics(df, th)
        for k, v in zip(keys, [tn, fp, fn, tp, se, sp, ppv, npv, below, above]):
            perf[k].append(v)
    tmp = pd.DataFrame(perf)
    tmp['threshold'] = thresholds
    tmp['threshold'] = tmp.threshold.clip(0,1)
    return tmp

def get_performance(df):
    keys = ['tn', 'fp', 'fn', 'tp', 'ppv', 'npv', 'below', 'above']
    perf = {x:[] for x in keys}
    fpr, tpr, thresholds = metrics.roc_curve(df['target'], df['score'])
    sp = 1-fpr
    for th in thresholds:
        tn, fp, fn, tp, ppv, npv, below, above = get_ppv_npv(df, th)
        for k, v in zip(keys, [tn, fp, fn, tp, ppv, npv, below, above]):
            perf[k].append(v)
    
    tmp = pd.concat([
        pd.DataFrame({'threshold': thresholds, 'sensitivity':tpr, 'specificity':sp}),
        pd.DataFrame(perf)
    ], axis=1)
    tmp['threshold'] = tmp.threshold.clip(0,1)
    return tmp

def get_performance_assisted(df, th0=None, th1=None):
    keys = ['tn', 'fp', 'fn', 'tp', 'ppv', 'npv', 'se', 'sp', 'reduction']
    perf = {x:[] for x in keys}
    for t0, t1 in zip(th0,th1):
        tn, fp, fn, tp, ppv, npv, se, sp, red = get_metrics_assisted(df, t0, t1)
        for k, v in zip(keys, [tn, fp, fn, tp, ppv, npv, se, sp, red/len(df)]):
            perf[k].append(v)
    
    tmp = pd.DataFrame(perf)
    tmp['th0'] = th0
    tmp['th1'] = th1
    return tmp

def get_performance_assisted_bootstrapped(df, th0=None, th1=None, n=1000):
    curves = []
    keys = ['ppv', 'npv', 'se', 'sp', 'reduction']
    for _ in range(n):
        tmp = df.sample(frac=1, replace=True).copy()
        tmp = get_performance_assisted(tmp, th0=th0, th1=th1)
        curves.append(tmp)
    curves = pd.concat(curves).reset_index(drop=True)
    out = []
    for k in keys:
        out.append(curves.groupby(['th0','th1']).agg(a=(k, lambda x:np.quantile(x, 0.025)), b=(k, lambda x:np.quantile(x, 0.975))).rename(columns={'a':f'{k}_lo','b':f'{k}_hi'}))
    
    out = pd.concat(out, axis=1)
    return out.reset_index()

def get_performance_bootstrapped(df, n=1000, anchor='threshold'):
    curves = []
    keys = ['threshold', 'sensitivity', 'specificity', 'below', 'above']
    asc = {'threshold':False, 'sensitivity':True, 'specificity':False, 'below':False, 'above':True}[anchor]
    ths = np.linspace(0,1,1001)
    
    for _ in range(n):
        tmp = df.sample(frac=1, replace=True).copy()
        tmp = get_performance(tmp)
        curve = {}
        for k in keys:
            if asc:
                curve[k] = np.interp(ths, tmp[anchor], tmp[k])
            else:
                curve[k] = np.interp(ths, tmp.iloc[::-1][anchor], tmp.iloc[::-1][k])
        
        curves.append(pd.DataFrame(curve))
    curves = pd.concat(curves).reset_index(drop=True)
    out = []
    for k in keys:
        if k != anchor:
            out.append(curves.groupby(anchor).agg(a=(k, lambda x:np.quantile(x, 0.025)), b=(k, lambda x:np.quantile(x, 0.975))).rename(columns={'a':f'{k}_lo','b':f'{k}_hi'}))
    
    out = pd.concat(out, axis=1)
    return out.reset_index()

def bootstrap_auc(df, n=1000, target_name='target', score_name='probability'):
    out = np.empty((n))
    for i in range(n):
        tmp = df.sample(frac=1, replace=True)
        out[i] = roc_auc_score(tmp[target_name], tmp[score_name])
    
    return out.mean(), np.quantile(out, [0.025, 0.975])

def bootstrap_roc(df, n=1000, target_name='target', score_name='probability'):
    out = np.empty((1001,n))
    fprhat = np.linspace(0,1,1001)
    for i in range(n):
        tmp = df.sample(frac=1, replace=True)
        fpr, tpr, thresholds = metrics.roc_curve(tmp[target_name], tmp[score_name])
        f = interpolate.interp1d(fpr, tpr, assume_sorted=True)
        tprhat = f(fprhat)
        out[:,i] = tprhat
    tprmin, tprmax = np.quantile(out, [0.025, 0.975], axis=1)
    return fprhat, tprmin, tprmax

def plot_roc(df, n=1000, target_name='target', score_name='probability', ax=None, index=0, label=None):
    fpr, tpr, thresholds = metrics.roc_curve(df[target_name], df[score_name])
    auc = metrics.auc(fpr, tpr)
    fprhat, tprmin, tprmax = bootstrap_roc(df, n=1000, target_name=target_name, score_name=score_name)
    lw = 2
    ax.fill_between(1-fprhat, tprmin, tprmax, color=sns.color_palette()[index], alpha=0.5)
    ax.plot(1-fpr, tpr, color=sns.color_palette()[index],
            lw=lw, label=label)

def simulation(df, se_max=0.5, sp_min=0.5, N=101):
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
            assisted = df.idylla.copy()
            assisted.loc[df.score<se_th] = 0
            assisted.loc[df.score>sp_th] = 1
            npv[i, j] = npv_score(df.target, assisted)
            ppv[i, j] = ppv_score(df.target, assisted)
            bac[i, j] = metrics.balanced_accuracy_score(df.target, assisted)
            sen[i, j] = se_score(df.target, assisted)
            spe[i, j] = sp_score(df.target, assisted)
            red[i, j] = (df.score<se_th).sum() + (df.score>sp_th).sum()
    
    red = red / len(df)
    X, Y = np.meshgrid(sp_arr,se_arr)
    return X, Y, npv, ppv, sen, spe, bac, red

def get_conf_matrix(df, se_th=0.5, sp_th=0.5):
    assisted = df.idylla.copy()
    assisted.loc[df.score<se_th] = 0
    assisted.loc[df.score>sp_th] = 1
    return pd.crosstab(df.target, assisted)

tn_idylla = 1059
fp_idylla = 7
tp_idylla = 568
fn_idylla = 51
se_idylla = 0.9176090
sp_idylla = 0.9934334
npv_idylla = 0.9540541
ppv_idylla = 0.9878261
bac_idylla = 0.9555212

# Load pre IRT data
#dfname = 'eagle_results_pre_IRT.csv'
dfname = 'eagle_results_06_2023_through_04_2024.csv'
df = pd.read_csv(dfname, usecols=['DMP_ASSAY_ID', 'ANY_EGFR_KINASE', 'ANY_EGFR_KINASE_IDYLLA', 'SampleType', 'gigapath_snumber_results']).rename(columns={
    'ANY_EGFR_KINASE':'target',
    'ANY_EGFR_KINASE_IDYLLA':'idylla',
    'SampleType':'sample',
    'gigapath_snumber_results':'score'
})
df = df[(df['sample']=='Primary')&(~df.idylla.isna())].reset_index(drop=True)
roc_auc_score(df.target, df.score)
tni, fpi, fni, tpi = pd.crosstab(df.target, df.idylla).values.ravel()

# Load IRT data
irt = pd.read_csv('eagle_results_2024_IRT.csv', usecols=[0,1,2,3,5,8,10,11,12], header=0, names=['case','target','idylla','sample','score','eagle_time','start_time','impact_time','idylla_time'])
irt = irt[(irt['sample']=='Primary')&(irt.target.isin([0,1]))&(~irt.score.isna())&(~irt.idylla.isna())]
roc_auc_score(irt.target, irt.score)

#####
# Bootstrap Idylla historical data
#idylla_history = pd.Series(['tn']*(tn_idylla+tni)+['fp']*(fp_idylla+fpi)+['tp']*(tp_idylla+tpi)+['fn']*(fn_idylla+fni), dtype="category")
idylla_history = pd.Series(['tn']*(tn_idylla)+['fp']*(fp_idylla)+['tp']*(tp_idylla)+['fn']*(fn_idylla), dtype="category")
def idylla_metrics(x):
    tmp = x.value_counts()
    npv = tmp['tn'] / (tmp['tn']+tmp['fn'])
    ppv = tmp['tp'] / (tmp['tp']+tmp['fp'])
    return {'ppv':ppv, 'npv':npv}

x = idylla_history
n = 1000
samples = []
for _ in range(n):
    samples.append(idylla_metrics(x.sample(frac=1, replace=True)))

samples = pd.DataFrame(samples)
ppv_idylla_50 = samples.ppv.quantile(0.5)
npv_idylla_50 = samples.npv.quantile(0.5)
ppv_idylla_10 = samples.ppv.quantile(0.1)
npv_idylla_10 = samples.npv.quantile(0.1)
ppv_idylla_05 = samples.ppv.quantile(0.05)
npv_idylla_05 = samples.npv.quantile(0.05)

ppv_idylla_95inf = samples.ppv.quantile(0.025)
npv_idylla_95inf = samples.npv.quantile(0.025)
ppv_idylla_95sup = samples.ppv.quantile(0.975)
npv_idylla_95sup = samples.npv.quantile(0.975)
ppv_idylla_avg = samples.ppv.mean()
npv_idylla_avg = samples.npv.mean()

#####

npv_idylla_curr = tni/(tni+fni)
ppv_idylla_curr = tpi/(tpi+fpi)

stophere
'''
# Idylla performance
tni, fpi, fni, tpi = pd.crosstab(df.target, df.idylla).values.ravel()
tmp = get_performance(df)
tmp['below'] = tmp.below/len(df)
tmp['above'] = tmp.above/len(df)

assisted = get_performance_assisted(df)
assisted['below'] = assisted.below/len(df)
assisted['above'] = assisted.above/len(df)

boot = get_performance_bootstrapped(df, anchor='threshold', n=1000)
boot['below_lo'] = boot.below_lo/len(df)
boot['below_hi'] = boot.below_hi/len(df)
boot['above_lo'] = boot.above_lo/len(df)
boot['above_hi'] = boot.above_hi/len(df)
'''

# Simulation: full window
X0, Y0, npv0, ppv0, sen0, spe0, bac0, red0 = simulation(df, se_max=0.5, sp_min=0.5, N=201)

f, ax = plt.subplots(1,1)
hmap = ax.imshow(red0, extent=(0.5, 1, 0.5, 0), interpolation='none', cmap=plt.cm.bone, vmin=0.1, vmax=1)
CSnpv = ax.contour(X0, Y0, npv0, levels=[0.94, 0.95, 0.96, 0.97], cmap=plt.cm.Reds, vmin=0.9, vmax=1)
CSppv = ax.contour(X0, Y0, ppv0, levels=[0.90, 0.92, 0.94, 0.97], cmap=plt.cm.Blues, vmin=0.8, vmax=1)
ax.set_xlabel('EAGLE High Threshold')
ax.set_ylabel('EAGLE Low Threshold')
ax.clabel(CSnpv, CSnpv.levels, inline=True, fontsize=10)
ax.clabel(CSppv, CSppv.levels, inline=True, fontsize=10)
cbar = f.colorbar(hmap, ax=ax)
cbar.set_label('Rapid Test Reduction')
proxy = [
    plt.Rectangle((0,0),1,1,fc = CSnpv.collections[-1].get_edgecolor()[0]),
    plt.Rectangle((0,0),1,1,fc = CSppv.collections[-1].get_edgecolor()[0])
]
ax.legend(proxy, ["NPV Isolines", "PPV Isolines"], loc='lower left')
f.tight_layout()
f.show()

f, ax = plt.subplots(1,1)
hmap = ax.imshow(red0, extent=(0.5, 1, 0.5, 0), interpolation='none', cmap=plt.cm.bone, vmin=0.1, vmax=1)
CSnpv_conf = ax.contour(X0, Y0, npv0, levels=[npv_idylla_95inf, npv_idylla_95sup], colors='red', linestyles='dashed')
CSppv_conf = ax.contour(X0, Y0, ppv0, levels=[ppv_idylla_95inf, ppv_idylla_95sup], colors='blue', linestyles='dashed')
CSnpv_med = ax.contour(X0, Y0, npv0, levels=[npv_idylla_avg], colors='red')
CSppv_med = ax.contour(X0, Y0, ppv0, levels=[ppv_idylla_avg], colors='blue')
ax.set_xlabel('EAGLE High Threshold')
ax.set_ylabel('EAGLE Low Threshold')
cbar = f.colorbar(hmap, ax=ax)
cbar.set_label('Rapid Test Reduction')
handles_npv_conf, labels_npv_conf = CSnpv_conf.legend_elements()
handles_ppv_conf, labels_ppv_conf = CSppv_conf.legend_elements()
handles_npv_med, labels_npv_med = CSnpv_med.legend_elements()
handles_ppv_med, labels_ppv_med = CSppv_med.legend_elements()
ax.legend(
    handles_npv_conf[0:1] + handles_npv_med + handles_ppv_conf[0:1] + handles_ppv_med,
    [f'NPV 95% CI\n{npv_idylla_95inf:.3f} - {npv_idylla_95sup:.3f}', f'NPV Avg\n{npv_idylla_avg:.3f}',
     f'PPV 95% CI\n{ppv_idylla_95inf:.3f} - {ppv_idylla_95sup:.3f}', f'PPV Avg\n{ppv_idylla_avg:.3f}'],
    ncol=2, title='Historical Idylla Data', loc='lower left'
)
f.tight_layout()
f.show()


# Simulation: small window
#X1, Y1, npv1, ppv1, sen1, spe1, bac1, red1 = simulation(df, se_max=0.1, sp_min=0.9, N=201)
X1, Y1, npv1, ppv1, sen1, spe1, bac1, red1 = simulation(df, se_max=0.05, sp_min=0.95, N=201)
# Choosing thresholds
i,j = np.where((npv1>=npv_idylla_avg)&(ppv1>=ppv_idylla_avg))
k = np.where(red1[i,j]==red1[i,j].max())
0.0035, 1
0.0035, 0.998
0.023, 0.997

f, ax = plt.subplots(1,1)
#hmap = ax.imshow(red1, extent=(0.9, 1, 0.1, 0), cmap=plt.cm.bone, vmin=0.1, vmax=1)
hmap = ax.imshow(red1, extent=(0.95, 1, 0.05, 0), cmap=plt.cm.bone, vmin=0.1, vmax=1)
CSnpv_conf = ax.contour(X1, Y1, npv1, levels=[npv_idylla_95inf, npv_idylla_95sup], colors='red', linestyles='dashed')
CSppv_conf = ax.contour(X1, Y1, ppv1, levels=[ppv_idylla_95inf, ppv_idylla_95sup], colors='blue', linestyles='dashed')
CSnpv_med = ax.contour(X1, Y1, npv1, levels=[npv_idylla_avg], colors='red')
CSppv_med = ax.contour(X1, Y1, ppv1, levels=[ppv_idylla_avg], colors='blue')
ax.scatter([1, 0.998 ,0.997], [0.0035, 0.0035, 0.023], c='yellow', zorder=101)
ax.set_xlabel('EAGLE High Threshold')
ax.set_ylabel('EAGLE Low Threshold')
cbar = f.colorbar(hmap, ax=ax)
cbar.set_label('Rapid Test Reduction')
#handles_npv_conf, labels_npv_conf = CSnpv_conf.legend_elements()
#handles_ppv_conf, labels_ppv_conf = CSppv_conf.legend_elements()
#handles_npv_med, labels_npv_med = CSnpv_med.legend_elements()
#handles_ppv_med, labels_ppv_med = CSppv_med.legend_elements()
#ax.legend(
#    handles_npv_conf[0:1] + handles_npv_med + handles_ppv_conf[0:1] + handles_ppv_med,
#    [f'NPV 95% CI\n{npv_idylla_95inf:.3f} - {npv_idylla_95sup:.3f}', f'NPV Avg\n{npv_idylla_avg:.3f}',
#     f'PPV 95% CI\n{ppv_idylla_95inf:.3f} - {ppv_idylla_95sup:.3f}', f'PPV Avg\n{ppv_idylla_avg:.3f}'],
#    ncol=2, title='Historical Idylla Data', loc='lower left'
#)
f.tight_layout()
f.show()

# Get performance along path: th0 0-0.1, th1 1-1
th0 = np.linspace(0,0.05,501)
th1 = np.ones(501)
path1 = get_performance_assisted(df, th0=th0, th1=th1)
path1boot = get_performance_assisted_bootstrapped(df, th0=th0, th1=th1, n=200)


f, axs = plt.subplots(2, 1)
axs[0].fill_between(path1boot.th0, path1boot.npv_lo, path1boot.npv_hi, color=sns.color_palette()[0], alpha=0.5)
axs[1].fill_between(path1boot.th0, path1boot.reduction_lo, path1boot.reduction_hi, color=sns.color_palette()[0], alpha=0.5)
axs[0].plot(path1.th0, path1.npv, color=sns.color_palette()[0])
axs[1].plot(path1.th0, path1.reduction, color=sns.color_palette()[0])
axs[0].axhline(npv_idylla_avg, c='red')
axs[0].axhline(npv_idylla_95inf, c='r', linestyle='dashed')
axs[0].axhline(npv_idylla_95sup, c='r', linestyle='dashed')
axs[0].annotate('Idylla NPV', (0.99, npv_idylla_avg), xycoords=('axes fraction','data'), rotation='horizontal', ha='right', va='bottom', c='red')
axs[0].axvline(0.0035, c=sns.color_palette()[2])
axs[1].axvline(0.0035, c=sns.color_palette()[2])
axs[0].set_ylabel('Assisted NPV')
axs[1].set_ylabel('Rapid Test Reduction')
axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
lab1 = axs[1].set_xlabel('NPV\nThreshold', horizontalalignment='right', x=-0.01)
axs[1].xaxis.set_label_coords(-0.02, 0.)
ax2 = axs[1].secondary_xaxis(-0.2, functions=(lambda x: x, lambda x: x))
lab2 = ax2.set_xlabel('PPV\nThreshold', horizontalalignment='right', x=-0.01)
ax2.xaxis.set_label_coords(-0.02, -0.2)
ax2.set_xticklabels(['1'] * len(axs[1].get_xticklabels()))
f.tight_layout()
f.show()

# Get performance along path: th0 0-0.023*3, th1 1-(1-0.997)*3
## pre-IRT
th0 = np.linspace(0, 0.023*3, 500)
th1 = np.linspace(1, 1-(1-0.997)*3, 500)
path2 = get_performance_assisted(df, th0=th0, th1=th1)
path2boot = get_performance_assisted_bootstrapped(df, th0=th0, th1=th1, n=1000)
## IRT
path3 = get_performance_assisted(irt, th0=th0, th1=th1)
path3boot = get_performance_assisted_bootstrapped(irt, th0=th0, th1=th1, n=1000)


### Overall Plot
def interp_ppv(x):
    line_ppv = np.linspace(1, 1-(1-0.997)*3, 100)
    line_npv = np.linspace(0, 0.023*3, 100)
    return np.interp(x, line_npv, line_ppv)

def interp_npv(x):
    line_ppv = np.linspace(1, 1-(1-0.997)*3, 100)
    line_npv = np.linspace(0, 0.023*3, 100)
    return np.interp(x, line_ppv, line_npv)

'''
f = plt.figure(layout="constrained", figsize=(18,6))
gs = GridSpec(3, 3, figure=f, width_ratios=[3, 3, 2])
ax1 = f.add_subplot(gs[:,0])
ax2 = f.add_subplot(gs[:,1])
ax3 = f.add_subplot(gs[0,2])
ax4 = f.add_subplot(gs[1,2])
ax5 = f.add_subplot(gs[2,2])

hmap = ax1.imshow(red0, extent=(0.5, 1, 0.5, 0), interpolation='none', cmap=plt.cm.bone, vmin=0.1, vmax=1)
CSnpv_conf = ax1.contour(X0, Y0, npv0, levels=[npv_idylla_95inf, npv_idylla_95sup], colors='red', linestyles='dashed')
CSppv_conf = ax1.contour(X0, Y0, ppv0, levels=[ppv_idylla_95inf, ppv_idylla_95sup], colors='blue', linestyles='dashed')
CSnpv_med = ax1.contour(X0, Y0, npv0, levels=[npv_idylla_avg], colors='red')
CSppv_med = ax1.contour(X0, Y0, ppv0, levels=[ppv_idylla_avg], colors='blue')
ax1.add_patch(Rectangle((0.95, 0.05), 0.05, -0.05, alpha=1, facecolor='none'))
ax1.set_xlabel('EAGLE PPV Threshold')
ax1.set_ylabel('EAGLE NPV Threshold')
handles_npv_conf, labels_npv_conf = CSnpv_conf.legend_elements()
handles_ppv_conf, labels_ppv_conf = CSppv_conf.legend_elements()
handles_npv_med, labels_npv_med = CSnpv_med.legend_elements()
handles_ppv_med, labels_ppv_med = CSppv_med.legend_elements()
ax1.legend(
    handles_npv_conf[0:1] + handles_npv_med + handles_ppv_conf[0:1] + handles_ppv_med,
    [f'NPV 95% CI\n{npv_idylla_95inf:.3f} - {npv_idylla_95sup:.3f}', f'NPV Avg\n{npv_idylla_avg:.3f}',
     f'PPV 95% CI\n{ppv_idylla_95inf:.3f} - {ppv_idylla_95sup:.3f}', f'PPV Avg\n{ppv_idylla_avg:.3f}'],
    ncol=2, title='Historical Idylla Data', loc='lower left'
)

line_ppv = np.linspace(1, 1-(1-0.997)*3, 100)
line_npv = np.linspace(0, 0.023*3, 100)
line_npv = line_npv[np.where(line_npv<=0.05)]
line_ppv = line_ppv[np.where(line_npv<=0.05)]
hmap = ax2.imshow(red1, extent=(0.95, 1, 0.05, 0), cmap=plt.cm.bone, vmin=0.1, vmax=1)
CSnpv_conf = ax2.contour(X1, Y1, npv1, levels=[npv_idylla_95inf, npv_idylla_95sup], colors='red', linestyles='dashed')
CSppv_conf = ax2.contour(X1, Y1, ppv1, levels=[ppv_idylla_95inf, ppv_idylla_95sup], colors='blue', linestyles='dashed')
CSnpv_med = ax2.contour(X1, Y1, npv1, levels=[npv_idylla_avg], colors='red')
CSppv_med = ax2.contour(X1, Y1, ppv1, levels=[ppv_idylla_avg], colors='blue')
ax2.plot(line_ppv, line_npv, c=sns.color_palette()[4], zorder=101)
ax2.set_xlabel('EAGLE PPV Threshold')
ax2.set_ylabel('EAGLE NPV Threshold')
cbar = f.colorbar(hmap, ax=ax2, shrink=0.6)
cbar.set_label('Rapid Test Reduction')

#ax3.fill_between(path2boot.th0, path2boot.npv_lo, path2boot.npv_hi, color=sns.color_palette()[1], alpha=0.5)
#ax4.fill_between(path2boot.th0, path2boot.ppv_lo, path2boot.ppv_hi, color=sns.color_palette()[1], alpha=0.5)
#ax5.fill_between(path2boot.th0, path2boot.reduction_lo, path2boot.reduction_hi, color=sns.color_palette()[1], alpha=0.5)
#ax3.fill_between(path3boot.th0, path3boot.npv_lo, path3boot.npv_hi, color=sns.color_palette()[2], alpha=0.5)
#ax4.fill_between(path3boot.th0, path3boot.ppv_lo, path3boot.ppv_hi, color=sns.color_palette()[2], alpha=0.5)
#ax5.fill_between(path3boot.th0, path3boot.reduction_lo, path3boot.reduction_hi, color=sns.color_palette()[2], alpha=0.5)
ax3.plot(path2.th0, path2.npv, color=sns.color_palette()[1])
ax4.plot(path2.th0, path2.ppv, color=sns.color_palette()[1])
ax5.plot(path2.th0, path2.reduction, color=sns.color_palette()[1], label='pre-IRT')
ax3.plot(path3.th0, path3.npv, color=sns.color_palette()[2])
ax4.plot(path3.th0, path3.ppv, color=sns.color_palette()[2])
ax5.plot(path3.th0, path3.reduction, color=sns.color_palette()[2], label='IRT')
ax3.axhline(npv_idylla_avg, c='red')
ax3.axhline(npv_idylla_95inf, c='r', linestyle='dashed')
ax3.axhline(npv_idylla_95sup, c='r', linestyle='dashed')
ax3.annotate('Idylla NPV', (0.99, npv_idylla_avg), xycoords=('axes fraction','data'), rotation='horizontal', ha='right', va='bottom', c='red')
ax4.axhline(ppv_idylla_avg, c='blue')
ax4.axhline(ppv_idylla_95inf, c='blue', linestyle='dashed')
ax4.axhline(ppv_idylla_95sup, c='blue', linestyle='dashed')
ax4.annotate('Idylla PPV', (0.99, ppv_idylla_avg), xycoords=('axes fraction','data'), rotation='horizontal', ha='right', va='top', c='blue')
ax3.axvline(0.023, c=sns.color_palette()[4])
ax4.axvline(0.023, c=sns.color_palette()[4])
ax5.axvline(0.023, c=sns.color_palette()[4])
ax3.axvline(0.0035, c=sns.color_palette()[4])
ax4.axvline(0.0035, c=sns.color_palette()[4])
ax5.axvline(0.0035, c=sns.color_palette()[4])

ax3.set_ylabel('Assisted NPV')
ax4.set_ylabel('Assisted PPV')
ax5.set_ylabel('Rapid Test Reduction')
ax5.legend(title='Cohort')
ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax4.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
lab1 = ax5.set_xlabel('NPV Thresh.', ha='right', x=-0.01)
ax5.xaxis.set_label_coords(-0.02, -0.04)
ax6 = ax5.secondary_xaxis(-0.25, functions=(lambda x: x, lambda x: x))
lab2 = ax6.set_xlabel('PPV Thresh.', ha='right', x=-0.01)
ax6.xaxis.set_label_coords(-0.02, 0)
f.canvas.draw()
ax5.get_xticklabels()
xticks = [f'{interp_ppv(x.get_position()[0]):.3f}' for x in ax5.get_xticklabels()]
ax6.set_xticklabels(xticks)

f.show()
'''
cmap = plt.cm.Greys
colnpv = 'orchid'#'magenta' 
colppv = 'aquamarine'#'cyan'
collin = 'gold'
colbor = 'crimson'
colirt = 'black'
f = plt.figure(layout="constrained", figsize=(18,5))
gs = GridSpec(3, 4, figure=f, width_ratios=[3, 3, 2, 2])
ax1 = f.add_subplot(gs[:,0])
ax2 = f.add_subplot(gs[:,1])
ax3 = f.add_subplot(gs[0,2])
ax4 = f.add_subplot(gs[1,2])
ax5 = f.add_subplot(gs[2,2])
ax6 = f.add_subplot(gs[0,3])
ax7 = f.add_subplot(gs[1,3])
ax8 = f.add_subplot(gs[2,3])
hmap = ax1.imshow(red0, extent=(0.5, 1, 0.5, 0), interpolation='none', cmap=cmap, vmin=0.1, vmax=1)
CSnpv_conf = ax1.contour(X0, Y0, npv0, levels=[npv_idylla_95inf, npv_idylla_95sup], colors=colnpv, linestyles='dashed')
CSppv_conf = ax1.contour(X0, Y0, ppv0, levels=[ppv_idylla_95inf, ppv_idylla_95sup], colors=colppv, linestyles='dashed')
CSnpv_med = ax1.contour(X0, Y0, npv0, levels=[npv_idylla_avg], colors=colnpv)
CSppv_med = ax1.contour(X0, Y0, ppv0, levels=[ppv_idylla_avg], colors=colppv)
ax1.add_patch(Rectangle((0.95, 0.05), 0.05, -0.05, facecolor=colbor, edgecolor='none', alpha=0.5, zorder=80, lw=2))
ax1.set_xlabel('EAGLE PPV Threshold')
ax1.set_ylabel('EAGLE NPV Threshold')
handles_npv_conf, labels_npv_conf = CSnpv_conf.legend_elements()
handles_ppv_conf, labels_ppv_conf = CSppv_conf.legend_elements()
handles_npv_med, labels_npv_med = CSnpv_med.legend_elements()
handles_ppv_med, labels_ppv_med = CSppv_med.legend_elements()
ax1.legend(
    handles_npv_conf[0:1] + handles_npv_med + handles_ppv_conf[0:1] + handles_ppv_med,
    [f'NPV 95% CI\n{npv_idylla_95inf:.3f} - {npv_idylla_95sup:.3f}', f'NPV Avg\n{npv_idylla_avg:.3f}',
     f'PPV 95% CI\n{ppv_idylla_95inf:.3f} - {ppv_idylla_95sup:.3f}', f'PPV Avg\n{ppv_idylla_avg:.3f}'],
    ncol=2, title='Historical Idylla Data', loc='lower left'
)
line_ppv = np.linspace(1, 1-(1-0.997)*3, 100)
line_npv = np.linspace(0, 0.023*3, 100)
line_npv = line_npv[np.where(line_npv<=0.05)]
line_ppv = line_ppv[np.where(line_npv<=0.05)]
hmap = ax2.imshow(red1, extent=(0.95, 1, 0.05, 0), cmap=cmap, vmin=0.1, vmax=1)
for spine in ax2.spines.values():
    spine.set_edgecolor(colbor)

CSnpv_conf = ax2.contour(X1, Y1, npv1, levels=[npv_idylla_95inf, npv_idylla_95sup], colors=colnpv, linestyles='dashed')
CSppv_conf = ax2.contour(X1, Y1, ppv1, levels=[ppv_idylla_95inf, ppv_idylla_95sup], colors=colppv, linestyles='dashed')
CSnpv_med = ax2.contour(X1, Y1, npv1, levels=[npv_idylla_avg], colors=colnpv)
CSppv_med = ax2.contour(X1, Y1, ppv1, levels=[ppv_idylla_avg], colors=colppv)
ax2.plot(line_ppv, line_npv, c=collin, zorder=101)
ax2.scatter(
    np.interp([0.0035, 0.023, 0.038], path2.th0, path2.th1),
    [0.0035, 0.023, 0.038],
    c=collin, zorder=101, label='Selected\nThresholds'
)
ax2.set_xlabel('EAGLE PPV Threshold')
ax2.legend(loc='lower left')
#cbar = f.colorbar(hmap, ax=ax2, shrink=0.6)
cbar = f.colorbar(hmap, ax=[ax1, ax2], shrink=0.6, location='bottom')
cbar.set_label('Rapid Test Reduction')
ax3.fill_between(path2boot.th0, path2boot.npv_lo, path2boot.npv_hi, color=colirt, alpha=0.5)
ax4.fill_between(path2boot.th0, path2boot.ppv_lo, path2boot.ppv_hi, color=colirt, alpha=0.5)
ax5.fill_between(path2boot.th0, path2boot.reduction_lo, path2boot.reduction_hi, color=colirt, alpha=0.5)
ax6.fill_between(path3boot.th0, path3boot.npv_lo, path3boot.npv_hi, color=colirt, alpha=0.5)
ax7.fill_between(path3boot.th0, path3boot.ppv_lo, path3boot.ppv_hi, color=colirt, alpha=0.5)
ax8.fill_between(path3boot.th0, path3boot.reduction_lo, path3boot.reduction_hi, color=colirt, alpha=0.5)
ax3.plot(path2.th0, path2.npv, color=colirt)
ax4.plot(path2.th0, path2.ppv, color=colirt)
ax5.plot(path2.th0, path2.reduction, color=colirt, label='Pre-Trial\nN=397')
ax6.plot(path3.th0, path3.npv, color=colirt)
ax7.plot(path3.th0, path3.ppv, color=colirt)
ax8.plot(path3.th0, path3.reduction, color=colirt, label='Silent Trial\nN=197')
ax3.axhline(npv_idylla_avg, c=colnpv)
ax3.axhline(npv_idylla_95inf, c=colnpv, linestyle='dashed')
ax3.axhline(npv_idylla_95sup, c=colnpv, linestyle='dashed')
#ax3.annotate('Idylla NPV', (0.99, npv_idylla_avg), xycoords=('axes fraction','data'), rotation='horizontal', ha='right', va='bottom', c='red')
ax4.axhline(ppv_idylla_avg, c=colppv)
ax4.axhline(ppv_idylla_95inf, c=colppv, linestyle='dashed')
ax4.axhline(ppv_idylla_95sup, c=colppv, linestyle='dashed')
#ax4.annotate('Idylla PPV', (0.99, ppv_idylla_avg), xycoords=('axes fraction','data'), rotation='horizontal', ha='right', va='bottom', c='blue')
ax3.set_ylim(0.9, 1.01)
ax6.set_ylim(0.9, 1.01)
ax4.set_ylim(0.94, 1.01)
ax7.set_ylim(0.94, 1.01)
ax5.set_ylim(-0.05, 0.65)
ax8.set_ylim(-0.05, 0.65)
ax6.axhline(npv_idylla_avg, c=colnpv)
ax6.axhline(npv_idylla_95inf, c=colnpv, linestyle='dashed')
ax6.axhline(npv_idylla_95sup, c=colnpv, linestyle='dashed')
ax6.annotate('Idylla NPV', (0.99, npv_idylla_95sup), xycoords=('axes fraction','data'), rotation='horizontal', ha='right', va='bottom', c=colnpv)
ax7.axhline(ppv_idylla_avg, c=colppv)
ax7.axhline(ppv_idylla_95inf, c=colppv, linestyle='dashed')
ax7.axhline(ppv_idylla_95sup, c=colppv, linestyle='dashed')
ax7.annotate('Idylla PPV', (0.99, ppv_idylla_95sup), xycoords=('axes fraction','data'), rotation='horizontal', ha='right', va='bottom', c=colppv)
for axs in [ax3,ax4,ax5,ax6,ax7,ax8]:
    axs.axvline(0.023, c=collin)
    axs.axvline(0.0035, c=collin)
    axs.axvline(0.038, c=collin)

ax3.set_ylabel('Assisted NPV')
ax4.set_ylabel('Assisted PPV')
ax5.set_ylabel('Rapid Test\nReduction')
#ax5.legend(title='Cohort')
#ax8.legend(title='Cohort')
ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax4.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax6.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax7.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
lab1 = ax5.set_xlabel('NPV Thresh.', ha='right', x=-0.01)
ax5.xaxis.set_label_coords(-0.02, -0.04)
ax5b = ax5.secondary_xaxis(-0.25, functions=(lambda x: x, lambda x: x))
lab2 = ax5b.set_xlabel('PPV Thresh.', ha='right', x=-0.01)
ax5b.xaxis.set_label_coords(-0.02, 0)
f.canvas.draw()
xticks = [f'{interp_ppv(x.get_position()[0]):.3f}' for x in ax5.get_xticklabels()]
ax5b.set_xticklabels(xticks)
ax8b = ax8.secondary_xaxis(-0.25, functions=(lambda x: x, lambda x: x))
f.canvas.draw()
xticks = [f'{interp_ppv(x.get_position()[0]):.3f}' for x in ax8.get_xticklabels()]
ax8b.set_xticklabels(xticks)

ax1.set_title('a)', loc='left')
ax2.set_title('b)', loc='left')
ax3.set_title('c)', loc='left')
ax6.set_title('d)', loc='left')

f.savefig('paper_trial_results2.png', dpi=300)
f.show()

########### SPLIT 1
f = plt.figure(layout="constrained", figsize=(10,6))
gs = GridSpec(1, 2, figure=f, width_ratios=[1, 1])
ax1 = f.add_subplot(gs[:,0])
ax2 = f.add_subplot(gs[:,1])
hmap = ax1.imshow(red0, extent=(0.5, 1, 0.5, 0), interpolation='none', cmap=cmap, vmin=0.1, vmax=1)
CSnpv_conf = ax1.contour(X0, Y0, npv0, levels=[npv_idylla_95inf, npv_idylla_95sup], colors=colnpv, linestyles='dashed')
CSppv_conf = ax1.contour(X0, Y0, ppv0, levels=[ppv_idylla_95inf, ppv_idylla_95sup], colors=colppv, linestyles='dashed')
CSnpv_med = ax1.contour(X0, Y0, npv0, levels=[npv_idylla_avg], colors=colnpv)
CSppv_med = ax1.contour(X0, Y0, ppv0, levels=[ppv_idylla_avg], colors=colppv)
ax1.add_patch(Rectangle((0.95, 0.05), 0.05, -0.05, facecolor=colbor, edgecolor='none', alpha=0.5, zorder=80))#edgecolor='none'
#ax1.add_patch(Rectangle((0.95, 0.05), 0.05, -0.05, facecolor='none', edgecolor=colbor, zorder=85, lw=3))
ax1.set_xlabel('EAGLE PPV Threshold')
ax1.set_ylabel('EAGLE NPV Threshold')
handles_npv_conf, labels_npv_conf = CSnpv_conf.legend_elements()
handles_ppv_conf, labels_ppv_conf = CSppv_conf.legend_elements()
handles_npv_med, labels_npv_med = CSnpv_med.legend_elements()
handles_ppv_med, labels_ppv_med = CSppv_med.legend_elements()
ax1.legend(
    handles_npv_conf[0:1] + handles_npv_med + handles_ppv_conf[0:1] + handles_ppv_med,
    [f'NPV 95% CI\n{npv_idylla_95inf:.3f} - {npv_idylla_95sup:.3f}', f'NPV Avg\n{npv_idylla_avg:.3f}',
     f'PPV 95% CI\n{ppv_idylla_95inf:.3f} - {ppv_idylla_95sup:.3f}', f'PPV Avg\n{ppv_idylla_avg:.3f}'],
    ncol=2, title='Historical Idylla Data', loc='lower left'
)
line_ppv = np.linspace(1, 1-(1-0.997)*3, 100)
line_npv = np.linspace(0, 0.023*3, 100)
line_npv = line_npv[np.where(line_npv<=0.05)]
line_ppv = line_ppv[np.where(line_npv<=0.05)]
hmap = ax2.imshow(red1, extent=(0.95, 1, 0.05, 0), cmap=cmap, vmin=0.1, vmax=1)
for spine in ax2.spines.values():
    spine.set_edgecolor(colbor)
    spine.set_linewidth(3)

CSnpv_conf = ax2.contour(X1, Y1, npv1, levels=[npv_idylla_95inf, npv_idylla_95sup], colors=colnpv, linestyles='dashed')
CSppv_conf = ax2.contour(X1, Y1, ppv1, levels=[ppv_idylla_95inf, ppv_idylla_95sup], colors=colppv, linestyles='dashed')
CSnpv_med = ax2.contour(X1, Y1, npv1, levels=[npv_idylla_avg], colors=colnpv)
CSppv_med = ax2.contour(X1, Y1, ppv1, levels=[ppv_idylla_avg], colors=colppv)
ax2.plot(line_ppv, line_npv, c=collin, zorder=101)
ax2.scatter(
    np.interp([0.0035, 0.023, 0.038], path2.th0, path2.th1),
    [0.0035, 0.023, 0.038],
    c=collin, zorder=101, label='Selected\nThresholds'
)
ax2.set_xlabel('EAGLE PPV Threshold')
ax2.legend(loc='lower left')
cbar = f.colorbar(hmap, ax=[ax1, ax2], shrink=0.6, location='bottom')
cbar.set_label('Rapid Test Reduction')
ax1.set_title('a)', loc='left')
ax2.set_title('b)', loc='left')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('EAGLE NPV Threshold')
#f.show()
f.savefig('paper_trial_results2_ab.png', dpi=300)

########### SPLIT 2
f = plt.figure(layout="constrained", figsize=(8,6))
gs = GridSpec(3, 2, figure=f)
ax3 = f.add_subplot(gs[0,0])
ax4 = f.add_subplot(gs[1,0])
ax5 = f.add_subplot(gs[2,0])
ax6 = f.add_subplot(gs[0,1])
ax7 = f.add_subplot(gs[1,1])
ax8 = f.add_subplot(gs[2,1])
#ax_legend = plt.subplot(gs[0,:])
ax3.fill_between(path2boot.th0, path2boot.npv_lo, path2boot.npv_hi, color=colirt, alpha=0.5)
ax4.fill_between(path2boot.th0, path2boot.ppv_lo, path2boot.ppv_hi, color=colirt, alpha=0.5)
ax5.fill_between(path2boot.th0, path2boot.reduction_lo, path2boot.reduction_hi, color=colirt, alpha=0.5)
ax6.fill_between(path3boot.th0, path3boot.npv_lo, path3boot.npv_hi, color=colirt, alpha=0.5)
ax7.fill_between(path3boot.th0, path3boot.ppv_lo, path3boot.ppv_hi, color=colirt, alpha=0.5)
ax8.fill_between(path3boot.th0, path3boot.reduction_lo, path3boot.reduction_hi, color=colirt, alpha=0.5)
ax3.plot(path2.th0, path2.npv, color=colirt)
ax4.plot(path2.th0, path2.ppv, color=colirt)
ax5.plot(path2.th0, path2.reduction, color=colirt, label='Pre-Trial\nN=397')
ax6.plot(path3.th0, path3.npv, color=colirt)
ax7.plot(path3.th0, path3.ppv, color=colirt)
ax8.plot(path3.th0, path3.reduction, color=colirt, label='Silent Trial\nN=197')
ax3.axhline(npv_idylla_avg, c=colnpv, label='Idylla NPV')
ax3.axhline(npv_idylla_95inf, c=colnpv, linestyle='dashed')
ax3.axhline(npv_idylla_95sup, c=colnpv, linestyle='dashed')
#ax3.annotate('Idylla NPV', (0.99, npv_idylla_avg), xycoords=('axes fraction','data'), rotation='horizontal', ha='right', va='bottom', c='red')
ax4.axhline(ppv_idylla_avg, c=colppv, label='Idylla PPV')
ax4.axhline(ppv_idylla_95inf, c=colppv, linestyle='dashed')
ax4.axhline(ppv_idylla_95sup, c=colppv, linestyle='dashed')
#ax4.annotate('Idylla PPV', (0.99, ppv_idylla_avg), xycoords=('axes fraction','data'), rotation='horizontal', ha='right', va='bottom', c='blue')
ax3.set_ylim(0.9, 1.01)
ax6.set_ylim(0.9, 1.01)
ax4.set_ylim(0.94, 1.01)
ax7.set_ylim(0.94, 1.01)
ax5.set_ylim(-0.05, 0.65)
ax8.set_ylim(-0.05, 0.65)
ax6.axhline(npv_idylla_avg, c=colnpv, label='Idylla NPV')
ax6.axhline(npv_idylla_95inf, c=colnpv, linestyle='dashed')
ax6.axhline(npv_idylla_95sup, c=colnpv, linestyle='dashed')
#ax6.annotate('Idylla NPV', (0.95, npv_idylla_avg), xycoords=('axes fraction','data'), rotation='horizontal', ha='right', va='bottom', c=colnpv)
ax7.axhline(ppv_idylla_avg, c=colppv, label='Idylla PPV')
ax7.axhline(ppv_idylla_95inf, c=colppv, linestyle='dashed')
ax7.axhline(ppv_idylla_95sup, c=colppv, linestyle='dashed')
#ax7.annotate('Idylla PPV', (0.95, ppv_idylla_avg), xycoords=('axes fraction','data'), rotation='horizontal', ha='right', va='bottom', c=colppv)
for axs in [ax3,ax4,ax5,ax6,ax7,ax8]:
    axs.axvline(0.023, c=collin)
    axs.axvline(0.0035, c=collin)
    axs.axvline(0.038, c=collin)

ax3.set_ylabel('Assisted NPV')
ax4.set_ylabel('Assisted PPV')
ax5.set_ylabel('Rapid Test\nReduction')
ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax4.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax6.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax7.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
lab1 = ax5.set_xlabel('NPV Thresh.', ha='right', x=-0.01)
ax5.xaxis.set_label_coords(-0.02, -0.04)
ax5b = ax5.secondary_xaxis(-0.25, functions=(lambda x: x, lambda x: x))
lab2 = ax5b.set_xlabel('PPV Thresh.', ha='right', x=-0.01)
ax5b.xaxis.set_label_coords(-0.02, 0)
f.canvas.draw()
xticks = [f'{interp_ppv(x.get_position()[0]):.3f}' for x in ax5.get_xticklabels()]
ax5b.set_xticklabels(xticks)
ax8b = ax8.secondary_xaxis(-0.25, functions=(lambda x: x, lambda x: x))
f.canvas.draw()
xticks = [f'{interp_ppv(x.get_position()[0]):.3f}' for x in ax8.get_xticklabels()]
ax8b.set_xticklabels(xticks)
ax3.set_title('c)', loc='left')
ax6.set_title('d)', loc='left')
ax3.legend(loc='lower left')
ax4.legend(loc='lower left')
ax6.legend(loc='lower left')
ax7.legend(loc='lower left')
ax5.legend()
ax8.legend()
#handles, labels = ax6.get_legend_handles_labels()
#handles2, labels2 = ax7.get_legend_handles_labels()
#ax6.legend(handles + handles2, labels + labels2, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1))
f.savefig('paper_trial_results2_cd.png', dpi=300)


### Thresholds
# NPV
# 0.0035, 0.023, 0.038
# PPV
# np.interp([0.0035, 0.023, 0.038], path2.th0, path2.th1)
# 0.99954348, 0.997     , 0.99504348

# Tables

ths = pd.Series([0.0035, 0.023, 0.038])
tab1 = pd.DataFrame({
    'Cohort': 'Pre-Trial',
    'NPV Threshold': ths,
    'PPV Threshold': np.interp([0.0035, 0.023, 0.038], path2.th0, path2.th1),
    'npv_avg': np.interp([0.0035, 0.023, 0.038], path2.th0, path2.npv),
    'npv_low': np.interp([0.0035, 0.023, 0.038], path2boot.th0, path2boot.npv_lo),
    'npv_hig': np.interp([0.0035, 0.023, 0.038], path2boot.th0, path2boot.npv_hi),
    'ppv_avg': np.interp([0.0035, 0.023, 0.038], path2.th0, path2.ppv),
    'ppv_low': np.interp([0.0035, 0.023, 0.038], path2boot.th0, path2boot.ppv_lo),
    'ppv_hig': np.interp([0.0035, 0.023, 0.038], path2boot.th0, path2boot.ppv_hi),
    'red_avg': np.interp([0.0035, 0.023, 0.038], path2.th0, path2.reduction),
    'red_low': np.interp([0.0035, 0.023, 0.038], path2boot.th0, path2boot.reduction_lo),
    'red_hig': np.interp([0.0035, 0.023, 0.038], path2boot.th0, path2boot.reduction_hi),
})
tab2 = pd.DataFrame({
    'Cohort': 'Silent Trial',
    'NPV Threshold': ths,
    'PPV Threshold': np.interp([0.0035, 0.023, 0.038], path2.th0, path2.th1),
    'npv_avg': np.interp([0.0035, 0.023, 0.038], path3.th0, path3.npv),
    'npv_low': np.interp([0.0035, 0.023, 0.038], path3boot.th0, path3boot.npv_lo),
    'npv_hig': np.interp([0.0035, 0.023, 0.038], path3boot.th0, path3boot.npv_hi),
    'ppv_avg': np.interp([0.0035, 0.023, 0.038], path3.th0, path3.ppv),
    'ppv_low': np.interp([0.0035, 0.023, 0.038], path3boot.th0, path3boot.ppv_lo),
    'ppv_hig': np.interp([0.0035, 0.023, 0.038], path3boot.th0, path3boot.ppv_hi),
    'red_avg': np.interp([0.0035, 0.023, 0.038], path3.th0, path3.reduction),
    'red_low': np.interp([0.0035, 0.023, 0.038], path3boot.th0, path3boot.reduction_lo),
    'red_hig': np.interp([0.0035, 0.023, 0.038], path3boot.th0, path3boot.reduction_hi),
})
tab = pd.concat([tab1, tab2])
tab['NPV Threshold'] = tab['NPV Threshold'].map('{:.4f}'.format)
tab['PPV Threshold'] = tab['PPV Threshold'].map('{:.4f}'.format)
tab['npv_avg'] = tab['npv_avg'].map('{:.3f}'.format)
tab['ppv_avg'] = tab['ppv_avg'].map('{:.3f}'.format)
tab['red_avg'] = tab['red_avg'].map('{:.3f}'.format)
tab['npv_ci'] = tab.npv_low.map('{:.3f}'.format)+'-'+tab.npv_hig.map('{:.3f}'.format)
tab['ppv_ci'] = tab.ppv_low.map('{:.3f}'.format)+'-'+tab.ppv_hig.map('{:.3f}'.format)
tab['red_ci'] = tab.red_low.map('{:.3f}'.format)+'-'+tab.red_hig.map('{:.3f}'.format)
print(tab[['Cohort','NPV Threshold','PPV Threshold','npv_avg','npv_ci','ppv_avg','npv_ci','red_avg','red_ci']].to_latex(index=False))
'''
\begin{tabular}{lllllllll}
\toprule
      Cohort & NPV Threshold & PPV Threshold & npv\_avg &      npv\_ci & ppv\_avg &      npv\_ci & red\_avg &      red\_ci \\
\midrule
   Pre-Trial &        0.0035 &        0.9995 &   0.954 & 0.929-0.976 &   0.991 & 0.929-0.976 &   0.250 & 0.210-0.293 \\
   Pre-Trial &        0.0230 &        0.9970 &   0.952 & 0.925-0.974 &   0.981 & 0.925-0.974 &   0.389 & 0.341-0.437 \\
   Pre-Trial &        0.0380 &        0.9950 &   0.952 & 0.924-0.975 &   0.981 & 0.924-0.975 &   0.433 & 0.383-0.481 \\
Silent Trial &        0.0035 &        0.9995 &   0.971 & 0.941-0.993 &   1.000 & 0.941-0.993 &   0.178 & 0.127-0.229 \\
Silent Trial &        0.0230 &        0.9970 &   0.970 & 0.941-0.993 &   0.984 & 0.941-0.993 &   0.371 & 0.310-0.442 \\
Silent Trial &        0.0380 &        0.9950 &   0.963 & 0.930-0.992 &   0.984 & 0.930-0.992 &   0.431 & 0.360-0.503 \\
\bottomrule
\end{tabular}
'''





















# Get performance along path: th0 0-0, th1 0.9-1
th0 = np.zeros(1001)
th1 = np.linspace(0.9,1,1001)[::-1]
path2 = get_performance_assisted(df, th0=th0, th1=th1)
path2boot = get_performance_assisted_bootstrapped(df, th0=th0, th1=th1, n=100)

# Get performance along path: th0 0-0.1, th1 0.9-1
th0 = np.linspace(0,0.1,1001)
th1 = np.linspace(0.9,1,1001)[::-1]
path3 = get_performance_assisted(df, th0=th0, th1=th1)
path3boot = get_performance_assisted_bootstrapped(df, th0=th0, th1=th1, n=100)

f, axs = plt.subplots(2, 3)

axs[0,0].fill_between(path1boot.th0, path1boot.npv_lo, path1boot.npv_hi, color=sns.color_palette()[0], alpha=0.5)
axs[1,0].fill_between(path1boot.th0, path1boot.reduction_lo, path1boot.reduction_hi, color=sns.color_palette()[1], alpha=0.5)
axs[0,0].plot(path1.th0, path1.npv, color=sns.color_palette()[0])
axs[1,0].plot(path1.th0, path1.reduction, color=sns.color_palette()[1])
axs[0,0].set_ylabel('Assisted NPV')
axs[1,0].set_ylabel('Rapid Test Reduction')
axs[0,0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# First axis
lab1 = axs[1,0].set_xlabel('Sensitivity\nThreshold', horizontalalignment='right', x=-0.01)
axs[1,0].xaxis.set_label_coords(-0.02, 0.)
# Second axis
ax2 = axs[1,0].secondary_xaxis(-0.2, functions=(lambda x: x, lambda x: x))
lab2 = ax2.set_xlabel('Specificity\nThreshold', horizontalalignment='right', x=-0.01)
ax2.xaxis.set_label_coords(-0.02, -0.2)
ax2.set_xticklabels(['1'] * len([item.get_text() for item in ax2.get_xticklabels()]))

axs[0,1].fill_between(path2boot.th1, path2boot.npv_lo, path2boot.npv_hi, color=sns.color_palette()[0], alpha=0.5)
axs[1,1].fill_between(path2boot.th1, path2boot.reduction_lo, path2boot.reduction_hi, color=sns.color_palette()[1], alpha=0.5)
axs[0,1].plot(path2.th1, path2.npv, color=sns.color_palette()[0])
axs[1,1].plot(path2.th1, path2.reduction, color=sns.color_palette()[1])
axs[0,1].set_ylabel('Assisted NPV')
axs[1,1].set_ylabel('Rapid Test Reduction')
axs[0,1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# First axis
lab1 = axs[1,1].set_xlabel('Sensitivity\nThreshold', horizontalalignment='right', x=-0.01)
axs[1,1].xaxis.set_label_coords(-0.02, 0.)
# Second axis
ax2 = axs[1,1].secondary_xaxis(-0.2, functions=(lambda x: x, lambda x: x))
lab2 = ax2.set_xlabel('Specificity\nThreshold', horizontalalignment='right', x=-0.01)
ax2.xaxis.set_label_coords(-0.02, -0.2)
axs[1,1].set_xticklabels(['0'] * len([item.get_text() for item in axs[1,1].get_xticklabels()]))

f.tight_layout()
f.show()





f, ax = plt.subplots(1,1)
ax.plot(tmp.threshold, tmp.sensitivity, label='Sensitivity')
ax.plot(tmp.threshold, tmp.specificity, label='Specificity')
ax.plot(tmp.threshold, tmp.npv, label='NPV')
ax.plot(tmp.threshold, tmp.ppv, label='PPV')
ax.legend()
ax.set_xlabel('EAGLE Score')
f.show()


se_idylla = 0.9176090
f, ax = plt.subplots(1,1)
ax.plot(tmp[tmp.sensitivity>0.8].sensitivity, tmp[tmp.sensitivity>0.8].below)
ax.axvline(x=se_idylla, color='red')
ax.annotate('Rapid Test Sensitivity', (se_idylla,0.01), xycoords=('data','axes fraction'), rotation='vertical', ha='right', va='bottom', c='red')
ax.set_xlabel('Sensitivity')
ax.set_ylabel('Reduction in Rapid Tests')
f.tight_layout()
f.show()

f, ax = plt.subplots(1,1)
ax.plot(tmp.threshold, tmp.below)
ax.fill_between(boot.threshold, boot.below_lo, boot.below_hi, alpha=0.5)
ax.axvline(x=se_idylla, color='red')
ax.annotate('Rapid Test Sensitivity', (se_idylla,0.01), xycoords=('data','axes fraction'), rotation='vertical', ha='right', va='bottom', c='red')
ax.set_xlabel('Sensitivity')
ax.set_ylabel('Reduction in Rapid Tests')
f.tight_layout()
f.show()

sp_idylla = 0.9934334
f, ax = plt.subplots(1,1)
ax.plot(tmp[tmp.specificity>0.8].specificity, tmp[tmp.specificity>0.8].above)
ax.axvline(x=sp_idylla, color='red')
ax.annotate('Rapid Test Specificity', (sp_idylla,0.99), xycoords=('data','axes fraction'), rotation='vertical', ha='right', va='top', c='red')
ax.set_xlabel('Specificity')
ax.set_ylabel('Reduction in Rapid Tests')
f.tight_layout()
f.show()

f, axs = plt.subplots(1,2)
ax = axs[0]
ax.plot(tmp[tmp.sensitivity>0.8].sensitivity, tmp[tmp.sensitivity>0.8].below)
ax.axvline(x=se_idylla, color='red')
ax.annotate('Rapid Test Sensitivity', (se_idylla,0.01), xycoords=('data','axes fraction'), rotation='vertical', ha='right', va='bottom', c='red')
ax.set_xlabel('Sensitivity')
ax.set_ylabel('Reduction in Rapid Tests')
ax = axs[1]
ax.plot(tmp[tmp.specificity>0.8].specificity, tmp[tmp.specificity>0.8].above)
ax.axvline(x=sp_idylla, color='red')
ax.annotate('Rapid Test Specificity', (sp_idylla,0.99), xycoords=('data','axes fraction'), rotation='vertical', ha='right', va='top', c='red')
ax.set_xlabel('Specificity')
ax.set_ylabel('Reduction in Rapid Tests')
f.tight_layout()
f.show()

# Tables
se_check = pd.Series([0.995, 0.99, 0.95, 0.9, 0.85, se_idylla]).sort_values(ascending=False)
tab1 = pd.DataFrame({
    'threshold': np.interp(se_check, tmp.sensitivity, tmp.threshold),
    'sensitivity': se_check,
    'specificity': np.interp(se_check, tmp.sensitivity, tmp.specificity),
    'ppv': np.interp(se_check, tmp.sensitivity, tmp.ppv),
    'npv': np.interp(se_check, tmp.sensitivity, tmp.npv),
    'below': np.interp(se_check, tmp.sensitivity, tmp.below),
    'above': np.interp(se_check, tmp.sensitivity, tmp.above),
})
print(tab1.to_latex(index=False, float_format="{:.3f}".format))
'''
\begin{tabular}{rrrrrrr}
\toprule
 threshold &  sensitivity &  specificity &   ppv &   npv &  below &  above \\
\midrule
     0.001 &        0.995 &        0.241 & 0.359 & 0.991 &  0.171 &  0.829 \\
     0.003 &        0.990 &        0.338 & 0.389 & 0.988 &  0.240 &  0.760 \\
     0.057 &        0.950 &        0.592 & 0.498 & 0.965 &  0.430 &  0.570 \\
     0.116 &        0.918 &        0.673 & 0.545 & 0.950 &  0.497 &  0.503 \\
     0.144 &        0.900 &        0.688 & 0.552 & 0.942 &  0.512 &  0.488 \\
     0.282 &        0.850 &        0.770 & 0.612 & 0.923 &  0.584 &  0.416 \\
\bottomrule
\end{tabular}
'''

sp_check = pd.Series([0.995, 0.99, 0.95, 0.9, 0.85, sp_idylla]).sort_values(ascending=True)
tab2 = pd.DataFrame({
    'threshold': np.interp(sp_check, tmp.specificity.iloc[::-1], tmp.threshold.iloc[::-1]),
    'sensitivity': np.interp(sp_check, tmp.specificity.iloc[::-1], tmp.sensitivity.iloc[::-1]),
    'specificity': sp_check,
    'ppv': np.interp(sp_check, tmp.specificity.iloc[::-1], tmp.ppv.iloc[::-1]),
    'npv': np.interp(sp_check, tmp.specificity.iloc[::-1], tmp.npv.iloc[::-1]),
    'below': np.interp(sp_check, tmp.specificity.iloc[::-1], tmp.below.iloc[::-1]),
    'above': np.interp(sp_check, tmp.specificity.iloc[::-1], tmp.above.iloc[::-1]),
})
print(tab2.to_latex(index=False, float_format="{:.3f}".format))
'''
\begin{tabular}{rrrrrrr}
\toprule
 threshold &  sensitivity &  specificity &   ppv &   npv &  below &  above \\
\midrule
     0.478 &        0.804 &        0.850 & 0.696 & 0.910 &  0.654 &  0.346 \\
     0.654 &        0.652 &        0.900 & 0.736 & 0.858 &  0.735 &  0.265 \\
     0.861 &        0.461 &        0.950 & 0.797 & 0.805 &  0.827 &  0.173 \\
     0.985 &        0.225 &        0.990 & 0.906 & 0.750 &  0.926 &  0.074 \\
     0.986 &        0.225 &        0.993 & 0.937 & 0.750 &  0.928 &  0.072 \\
     0.986 &        0.225 &        0.995 & 0.951 & 0.751 &  0.929 &  0.071 \\
\bottomrule
\end{tabular}
'''

pippo = tmp.copy()
pippo = pippo.dropna().sort_values('npv')
npv_check = [0.99, 0.98, 0.97, 0.95]
tab3 = pd.DataFrame({
    'threshold': np.interp(npv_check, pippo.npv, pippo.threshold),
    'sensitivity': np.interp(npv_check, pippo.npv, pippo.sensitivity),
    'specificity': np.interp(npv_check, pippo.npv, pippo.specificity),
    'ppv': np.interp(npv_check, pippo.npv, pippo.ppv),
    'npv': npv_check,
    'below': np.interp(npv_check, pippo.npv, pippo.below),
    'above': np.interp(npv_check, pippo.npv, pippo.above),
})
print(tab3.to_latex(index=False, float_format="{:.3f}".format))
'''
\begin{tabular}{rrrrrrr}
\toprule
 threshold &  sensitivity &  specificity &   ppv &   npv &  below &  above \\
\midrule
     0.000 &        0.995 &        0.207 & 0.349 & 0.990 &  0.147 &  0.853 \\
     0.017 &        0.977 &        0.484 & 0.448 & 0.980 &  0.346 &  0.654 \\
     0.043 &        0.958 &        0.577 & 0.491 & 0.970 &  0.417 &  0.583 \\
     0.118 &        0.917 &        0.676 & 0.547 & 0.950 &  0.499 &  0.501 \\
\bottomrule
\end{tabular}
'''

# Thresholds
''' Previous
# match idylla se
tab1.iloc[3].threshold
0.11607701866666638
# match idylla sp
tab2.iloc[4].threshold
0.9856373787333331
# eagle 98% NPV
tab3.iloc[1].threshold
0.01744255102040681
'''

'''New?
# match idylla se
tab1.iloc[3].threshold
0.126254051
# match idylla sp
tab2.iloc[4].threshold
0.9981019389647277
# eagle 98% NPV
tab3.iloc[1].threshold
0.004422000195999985
'''

# Plot
fig = plt.figure(figsize=(12,4), dpi=300)
gs = GridSpec(1, 4, width_ratios=[3, 1, 2, 2], height_ratios=[1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])

plot_roc(df, target_name='target', score_name='score', ax=ax1, index=0, label=f'Pre-Trial Cohort\nAUC={roc_auc_score(df.target, df.score):.3f}\nN={len(df)}')
ax1.plot([1, 0], [0, 1], color='gray', lw=2, linestyle='--')
ax1.scatter(tab3.iloc[1].specificity, tab3.iloc[1].sensitivity, color=sns.color_palette()[3], label='EAGLE 98% NPV',zorder=120)
ax1.scatter(tab1.iloc[3].specificity, tab1.iloc[3].sensitivity, color=sns.color_palette()[1], label='Idylla Sensitivity',zorder=110)
ax1.scatter(tab2.iloc[4].specificity, tab2.iloc[4].sensitivity, color=sns.color_palette()[2], label='Idylla Specificity',zorder=100)
ax1.set_xlabel('Specificity')
ax1.set_ylabel('Sensitivity')
ax1.legend(loc="lower right")#, title=legend)
ax1.invert_xaxis()
ax1.set_aspect('equal', adjustable='box')
ax1.set_title('a)', loc='left')

reduction = (tab1.iloc[3].below + tab2.iloc[4].above) * 100
sns.kdeplot(data=df, y='score', color=sns.color_palette()[0], ax=ax2, cut=0, bw_adjust=0.2)
ax2.axhline(y=tab3.iloc[1].threshold, color=sns.color_palette()[3])
ax2.axhline(y=tab1.iloc[3].threshold, color=sns.color_palette()[1])
ax2.axhline(y=tab2.iloc[4].threshold, color=sns.color_palette()[2])
ax2.set_ylabel('EAGLE Score')
ax2.invert_yaxis()
ax2.set_title('b)', loc='left')
#ax2.annotate(f"Idylla\nAvoided\n{reduction:.1f}%", (0.5, 0.5), xycoords='axes fraction', ha='center', va='center')

ax3.plot(tmp[tmp.sensitivity>0.85].sensitivity, tmp[tmp.sensitivity>0.85].below)
ax3.axvline(x=se_idylla, color=sns.color_palette()[1])
ax3.annotate('Idylla Sensitivity', (se_idylla,0.01), xycoords=('data','axes fraction'), rotation='vertical', ha='right', va='bottom', c=sns.color_palette()[1])
ax3.axvline(x=tab3.iloc[1].sensitivity, color=sns.color_palette()[3])
ax3.annotate('EAGLE 98% NPV', (tab3.iloc[1].sensitivity,0.01), xycoords=('data','axes fraction'), rotation='vertical', ha='right', va='bottom', c=sns.color_palette()[3])
ax3.set_xlabel('Sensitivity')
ax3.set_ylabel('Reduction in Rapid Tests')
ax3.invert_xaxis()
ax3.set_title('c)', loc='left')

ax4.plot(tmp[tmp.specificity>0.85].specificity, tmp[tmp.specificity>0.85].above)
ax4.axvline(x=sp_idylla, color=sns.color_palette()[2])
ax4.annotate('Idylla Specificity', (sp_idylla,0.99), xycoords=('data','axes fraction'), rotation='vertical', ha='right', va='top', c=sns.color_palette()[2])
ax4.set_xlabel('Specificity')
ax4.set_ylabel('Reduction in Rapid Tests')
ax4.set_title('d)', loc='left')

fig.show()
fig.tight_layout()
fig.savefig('plot_threshold_analysis.png')
