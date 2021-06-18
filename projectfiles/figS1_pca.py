import json
import os.path as op
import pickle
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, cm
from matplotlib.gridspec import GridSpec
from scipy import linalg

from mtrf_python import STRF_utils

agg_subject = 'Hamilton_Agg_LH_no63_131_143' # 'Hamilton_Agg_LH_9subjects'
strf_config_flag = 'regression_STRF'
config_flag = 'regression_SO1_PR_phnfeats_750msdelays'
irrr_flag = 'iRRR_210111' # 10**np.array([-1.5, -1.1, -0.7, -0.3, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
col_ctr_RR = True

# Load paths
with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
    config = json.load(f)
subjects_dir = config['subjects_dir']
loadpath = op.join(subjects_dir, agg_subject, config_flag)

blnsave = True

cv_folder = 'cv-10fold'

# [Data prep in run_cv.py]

#%%
figurepath = op.join(loadpath, 'Figures', 'Manuscript Figures')
if not op.exists(figurepath) and blnsave:
    makedirs(figurepath)

#%% Load config

# Regression data: fields and validation data
datapath = op.join(loadpath, f'{agg_subject}_RegressionSetup.pkl')
with open(datapath,'rb') as f:
    data = pickle.load(f)

phnstimnames = data['phnstimnames']
sentence_sel = data['sentence_sel']
electrodes = data['electrodes']
electrode_info = data['electrode_info']
delays = data['delays']
time_lags = data['time_lags']
fs = data['fs']

subjects = data['df'].subject.unique()
Nsubjects = len(subjects)

nchans = len(electrodes) # The number of channels (e.g. electrodes, neurons)
ndim = len(phnstimnames)
ndelays = [len(d) for d in delays]
nsentences = len(sentence_sel)

df = data['df']
del data

#%% Load data
loadpath = op.join(subjects_dir, agg_subject, config_flag)
filepath = op.join(loadpath,f'{agg_subject}_RegressionSetup.pkl')
with open(filepath,'rb') as f:
    data = pickle.load(f)

dstim = data['dstim']
resp = data['resp']
time_in_trial = data['time_in_trial']
sentence_ind = data['sentence_ind']
resp = resp-resp.mean(0,keepdims=True)


#%% sentence details
sentdet_df, featurenames, phnnames = STRF_utils.load_sentence_details(subjects_dir)
sentdet_df = sentdet_df.set_index(sentdet_df.name)

#%% Get 2d surface with coordinates

hemi = 'lh'
surfpath = op.join(loadpath,f'{agg_subject}_{hemi}_pial_lateral_2dSurf.pkl')

if not op.isfile(surfpath):
    from mtrf_python.scripts import prepare_2d_surface_script
    prepare_2d_surface_script.prepare_2d_surface(agg_subject,config_flag,hemi=hemi)

with open(surfpath,'rb') as f:
    surfdict = pickle.load(f)
img=surfdict['img']
coords_2d=surfdict['coords_2d']
blnplot = np.logical_not(np.isnan(coords_2d.sum(axis=1)))
coords_2d = coords_2d[blnplot,:]


#%% Plotting functions
def plot_aligned(toplot,align_window,align_inds,axs,plotorder=None,titles=None,
                durations=None):
    nevents = len(align_inds)
    [T,nplot] = toplot.shape
    ta = align_window/fs

    if plotorder is None:
        plotorder = range(nevents)
    if titles is None:
        titles = ['']*nplot

    toplot_aligned = np.zeros((nevents,len(align_window),nplot))
    for i in range (nevents):
        if align_window.max()+align_inds[i]<T:
            toplot_aligned[i,:,:] = toplot[align_window+align_inds[i],:nplot]
            if durations is not None:
                toplot_aligned[i,ta>durations[i]+0.35,:] = 0

    vmax = np.percentile(np.abs(toplot),99)
    for i,(title,ax) in enumerate(zip(titles,axs)):
        h = ax.imshow(toplot_aligned[plotorder,:,i],aspect='auto',
                      vmin=-vmax,vmax=vmax,cmap='RdBu_r',
                      extent=[ta[0]-1/fs,ta[-1]+1/fs,nevents+0.5,-0.5])
        ax.set_title(title)

    return h

def plot_loadings(toplot,axs,titles=None):
    [N,nplot] = toplot.shape

    if titles is None:
        titles = ['']*nplot

    vmax = np.percentile(np.abs(toplot),99)
    norm = colors.Normalize(vmin=-vmax, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdBu_r)

    for j,(title,ax) in enumerate(zip(titles,axs)):
        toplotj = toplot[:,j]
        order = np.argsort(toplotj**2)
        ax.imshow(img)
        sk = dict(s=4,marker='o',
                  c=np.stack([mapper.to_rgba(v) for v in toplotj[order]]))
        ax.scatter(coords_2d[order, 0], coords_2d[order, 1],**sk)
        ax.axis('off')
        ax.set_title(title)

    return mapper


#%% PCA on whole data
Y = resp
[U,S,Vh] = linalg.svd(Y,full_matrices=False)
total_var = (S ** 2).sum()
explained_variance_ratio = (S ** 2) / (S ** 2).sum()

#%% Figure
plt.close('all')
fig = plt.figure(figsize=[8,8])
gs = GridSpec(3,2,wspace=0.4,hspace=0.3,height_ratios=[0.8,0.8,1])

#% Plot explained variance and cumulative explained variance
ax = fig.add_subplot(gs[0, 0])
ax.plot(np.arange(nchans)+1,explained_variance_ratio*100,'k.')
ax.set_ylabel('% Explained Variance')
ax.set_xlabel('Component #')
ax.set_xlim([0,10])
ax = fig.add_subplot(gs[0, 1])
ax.plot(np.arange(nchans)+1,np.cumsum(explained_variance_ratio)*100,'k.')
ax.set_ylabel('Cumulative % Explained Variance')
ax.set_xlabel('# of components')
ax.axhline(80,linestyle='--',color='k')
ndims_capture80 = np.argwhere(np.cumsum(explained_variance_ratio) > 0.8)[0][0] + 1
ax.axvline(ndims_capture80, linestyle='--', color='k')

print(f'# dims to capture 80%: {ndims_capture80}')
print(f'% explained in first 2 dimensions: {np.sum(explained_variance_ratio[:2]*100):0.1f}%')

#% Plot timeseries aligned to sentence onset
nplot = 2
align_i = 0
align_window = np.arange(-0.5*fs,3*fs,dtype=int)

align_name = phnstimnames[align_i]
align_to = dstim[align_i][:,0]>0
align_inds = np.argwhere(align_to)
durations = np.array([sentdet_df.loc[sentence_sel[i]].duration for i in range(len(align_inds))])-1
plotorder = np.argsort(durations)

axs = [fig.add_subplot(gs[1, 0]),fig.add_subplot(gs[1, 1])]
h = plot_aligned(U[:,:nplot],align_window,align_inds,axs,plotorder,durations=durations)
for ax in axs:
    ax.plot(np.zeros(nsentences),np.arange(nsentences),'k--')
    ax.plot(durations[plotorder],np.arange(nsentences),'k--')

cfig,cax = plt.subplots(figsize=[1.5,3],clear=True)
cfig.colorbar(h,cax=cax,aspect=15)
cfig.subplots_adjust(right=0.3)

#% Plot electrode loadings
axs = [fig.add_subplot(gs[2, 0]),fig.add_subplot(gs[2, 1])]
mapper = plot_loadings(Vh[:nplot,:].T,axs)
for ax in axs:
    ax.set_xlim([30,760])
    ax.set_ylim([680,120])

# make separate colorbar
cfig2,cax = plt.subplots(figsize=[1.5,3],clear=True)
cfig2.colorbar(mapper,cax=cax,aspect=15)
cfig2.subplots_adjust(right=0.3)

if blnsave:
    fig.savefig(op.join(figurepath,f'{agg_subject}_pca.svg'))
    cfig.savefig(op.join(figurepath,f'{agg_subject}_pca_colorbar1.svg'))
    cfig2.savefig(op.join(figurepath,f'{agg_subject}_pca_colorbar2.svg'))

#%%
plt.figure()
plt.scatter(Vh[0,:],Vh[1,:])


#%%
# Plot aligned to peak rate
align_i = 1
align_window = np.arange(-0.1*fs,0.5*fs,dtype=int)

tTimes = time_in_trial[:,0]

align_name = phnstimnames[align_i]
# align_to = dstim[align_i][:,0]
align_to = np.logical_and(dstim[align_i][:,0]>0,tTimes>0.5)
align_inds = np.argwhere(align_to)
prvals = dstim[align_i][align_inds[:,0],0]
plotorder = np.argsort(prvals)

fig,axs = plt.subplots(1,2,figsize=[8, 4])
plot_aligned(U[:,:nplot],align_window,align_inds,axs,plotorder)



