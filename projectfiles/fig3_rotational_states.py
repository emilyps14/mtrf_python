import json
import os.path as op
import pickle
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, cm
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat

from mtrf_python import STRF_utils
from mtrf_python.scripts.prepare_regression_data_script import get_delay_matrices, get_data_from_df

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

# [Data prep in run_cv_750msdelays.py]

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
sentence_ind = data['sentence_ind']
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

df = data['df']
del data

# iRRR config
iRRR_config_path = op.join('.','config',f'{irrr_flag}.json')
with open(iRRR_config_path, "r") as f:
    irrr_config = json.load(f)

lambdas = np.array(irrr_config['lambdas']) # 10**np.array([-1.5, -1.1, -0.7, -0.3, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
nlambdas = len(lambdas)

#%% Load results

cv_summary_path = op.join(loadpath, f'{cv_folder}_cv_summary.pkl')
with open(cv_summary_path, 'rb') as f:
    data = pickle.load(f)
cv_summaries=data['cv_summaries']

cv_fold1 = cv_summaries[0]

# Use the first CV fold to pull out specific model fits
cv1_best_irrr = cv_fold1['best_irrr']
cv1_strfs_irrr = cv1_best_irrr['strfs']
cv1_best_ndims_by_feature = cv1_best_irrr['details']['rec_nonzeros'][-1, :].astype(int)

#%% Load sentence details

preproc_dir = config['local_timit_dir']
sentdet_df, featurenames, phnnames = STRF_utils.load_sentence_details(op.join(preproc_dir))

freqs = loadmat(op.join(preproc_dir,'mel_centerF.mat'), squeeze_me=True,
                  variable_names=['binfrqs'])['binfrqs']
fa_corners = freqs[:81]
fa = (fa_corners[1:]+fa_corners[:-1])/2

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


#%% Compute predicted trajectories

name_sent = 'fbcg1_si1612'
soi = 0
pri = 1
nplot = 2
befaft = [0,0]
df_sent = df.loc[df.name == name_sent]

# Audio timeseries
sentence = sentdet_df.loc[np.equal(sentdet_df.name, name_sent)].iterrows().__next__()[1]
source_befaft = sentence.befaft
soundf = sentence.soundf
sound_padleft = ((source_befaft[0] - befaft[0]) * soundf).astype(int)
sound_padright = ((source_befaft[1] - befaft[1]) * soundf).astype(int)
sound = STRF_utils.remove_pad(sentence.sound[None,:], sound_padleft, sound_padright)[0, :]
sound_ta = np.arange(len(sound))/soundf

# Specgram
padleft = ((source_befaft[0] - befaft[0]) * fs).astype(int)
padright = ((source_befaft[1] - befaft[1]) * fs).astype(int)
melspec = STRF_utils.remove_pad(sentence.aud,padleft,padright).transpose([1,0]) # ntimes x nfreqs

# Features
df_sent = df.loc[df.name == name_sent]
txt = df_sent.iloc[0].txt
phnstim, ta, _, resp, _ = \
    get_data_from_df(df_sent,phnstimnames,subjects,
                     padleft,padright,
                     electrodes_sel=electrodes)
ta = ta[:,0]
onset_times = ta[phnstim[:, soi] > 0]
peakRate_times = ta[phnstim[:, pri] > 0]
dstims, _ = get_delay_matrices(phnstim, phnstimnames, time_lags, fs)

#%% Static plot with jPCA projection
plt.close('all')

norm = plt.Normalize(ta.min(),ta.max())
cmap = 'jet'

# Set up figure
fig = plt.figure(figsize=[4,5])
gs = GridSpec(3,1,height_ratios=[0.5,1,1],hspace=0.2)

audax = fig.add_subplot(gs[0])
audax.plot(sound_ta, sound, 'k')
for pt in onset_times:
    audax.axvline(pt,linestyle='-',color='k')
for pt in peakRate_times:
    audax.axvline(pt,linestyle='--',color='k')
audax.set_title(txt)
audax.spines['top'].set_visible(False)
audax.spines['right'].set_visible(False)
# audax.set_xticklabels([])
audax.set_yticks([])
STRF_utils.colorline(ta,np.ones(ta.shape)*audax.get_ylim()[0],c=ta,cmap=cmap,norm=norm,linewidth=10)
audax.set_xlim(ta[[0,-1]])

# specax = fig.add_subplot(gs[1, :],sharex=audax)
# specim = STRF_utils.plot_aud(melspec, ta, fa, ax=specax,
#                              onset_times=onset_times,
#                              peakRate_times=peakRate_times,
#                              pcolor_kwargs={'cmap':cm.Greys,
#                                             'shading':'gouraud'})
# specax.spines['top'].set_visible(False)
# specax.spines['right'].set_visible(False)

# Projection onto iRRR feature subspaces
for i,(name,stim,mtrf_approx,ncomponents) in enumerate(zip(phnstimnames[:nplot], dstims, cv1_strfs_irrr, cv1_best_ndims_by_feature)):
    U,S,Vh = STRF_utils.get_svd_of_mtrf(mtrf_approx,ncomponents,True)

    pred_jpca = stim @ (U * S)
    pred_jpca -= pred_jpca.min(axis=0)
    # proj = resp @ Vh.T

    projax = fig.add_subplot(gs[i+1])
    toplot = pred_jpca - np.cumsum(pred_jpca.max(axis=0) - pred_jpca.min(axis=0) + 1)
    for j in range(ncomponents):
        projax.axhline(toplot[0,j],color='k',linewidth=0.5)
    projax.plot(ta,toplot)

    projax.spines['top'].set_visible(False)
    projax.spines['right'].set_visible(False)

    ticks = toplot[0,0] + np.array([0,5,10])
    ticklabels = np.array([0,5,10])
    projax.spines["left"].set_position(("axes", -0.02))
    projax.spines["left"].set_bounds(ticks[0],ticks[-1])
    projax.set_yticks(ticks)
    projax.set_yticklabels(ticklabels)

    if i<nplot-1:
        projax.set_xticks([])
        projax.spines['bottom'].set_visible(False)
    else:
        projax.spines['bottom'].set_position(("axes", -0.02))
    projax.set_xlim(ta[[0,-1]])


if blnsave:
    fig.savefig(op.join(figurepath,f'{agg_subject}_predicted_states.svg'))

#%% 3d space and projection onto jPCA subspaces
jpcas = STRF_utils.compute_jPCA_on_MTRFs(phnstimnames, delays, cv1_best_ndims_by_feature, cv1_strfs_irrr)

fig = plt.figure(num=2,figsize=[5.5,5])
fig.clear()
gs = GridSpec(nplot,2,wspace=0.5)

for i,(name,stim,mtrf_approx,ncomponents) in enumerate(zip(phnstimnames[:nplot], dstims, cv1_strfs_irrr, cv1_best_ndims_by_feature)):

    ### 3d
    U,S,Vh = STRF_utils.get_svd_of_mtrf(mtrf_approx,ncomponents,
                                    reduced=True)
    pred_3d = stim @ (U * S)

    ax = fig.add_subplot(gs[i,0],projection='3d')
    STRF_utils.colorline(pred_3d[:, 0], pred_3d[:, 1], pred_3d[:, 2], c=ta, cmap=cmap, norm=norm, ax=ax)
    # ax.scatter(pred_state_3d[pr_events[pr_events_toplot],0],pred_state_3d[pr_events[pr_events_toplot],1],pred_state_3d[pr_events[pr_events_toplot],2],color='k')
    ax.set_xlim(pred_3d[:, 0].min() - pred_3d[:, 0].std() / 2, pred_3d[:, 0].max() + pred_3d[:, 0].std() / 2)
    ax.set_ylim(pred_3d[:, 1].min() - pred_3d[:, 1].std() / 2, pred_3d[:, 1].max() + pred_3d[:, 1].std() / 2)
    ax.set_zlim(pred_3d[:, 2].min() - pred_3d[:, 2].std() / 2, pred_3d[:, 2].max() + pred_3d[:, 2].std() / 2)

    ax.view_init(elev=44.6, azim=-76.6)
    ax.set_xlabel('#1')
    ax.set_ylabel('#2')
    ax.set_zlabel('#3')

    ax.set_xticks(np.arange(0, pred_3d[:, 0].max(), 5))

    ### jpca
    pred_jpca = stim @ mtrf_approx @ jpcas[name]['jpca'].jpcs

    ax3 = fig.add_subplot(gs[i,1])
    # ax3.axvline(0,color='k',linewidth=0.5)
    # ax3.axhline(0,color='k',linewidth=0.5)
    STRF_utils.colorline(pred_jpca[:, 0], pred_jpca[:, 1], c=ta, cmap=cmap, norm=norm)
    lb = pred_jpca[:, :2].min() - pred_jpca[:, :2].std() / 2
    ub = pred_jpca[:, :2].max() + pred_jpca[:, :2].std() / 2
    ax3.set_xlim(lb,ub)
    ax3.set_ylim(lb,ub)
    ax3.set_aspect('equal','box')
    ax3.set_xlabel(f'jPC1')#: {jpcas[name]["pct_var"][0]:0.1f}%')
    ax3.set_ylabel(f'jPC2')#: {jpcas[name]["pct_var"][1]:0.1f}%')

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ticks = np.array([-2,0,2])
    ax3.set_xticks(ticks)
    ax3.set_yticks(ticks)
    ax3.spines['bottom'].set_bounds(ticks[[0,-1]])
    ax3.spines['left'].set_bounds(ticks[[0,-1]])
    # ax3.set_title(name)

    print(f'{name} top 3: {100*np.sum(S[:3]**2)/np.sum(S**2):0.1f}%')
    print(f'{name} jpca: {np.sum(jpcas[name]["pct_var"][:2]):0.1f}%')

# fig.suptitle(title)

if blnsave:
    fig.savefig(op.join(figurepath,f'{agg_subject}_predicted_states_jpca.svg'))

#%% Plot electrode weights for first two jpcs
plt.close('all')
vmax=0.15
for i,(name,mtrf_approx) in enumerate(zip(phnstimnames[:nplot], cv1_strfs_irrr)):
    assert(name in jpcas)

    jpcs = jpcas[name]['jpca'].jpcs

    # vmax = np.percentile(np.abs(jpcs),98)
    norm_el = colors.Normalize(vmin=-vmax, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm_el, cmap=cm.RdBu_r)

    for j in range(jpcs.shape[1]):
        fig2, ax = plt.subplots(figsize=[4,3.5],clear=True)
        toplot = jpcs[:,j]
        order = np.argsort(toplot**2)
        ax.imshow(img)
        sk = dict(s=10,marker='o',
                  c=np.stack([mapper.to_rgba(v) for v in toplot[order]]))
        ax.scatter(coords_2d[order, 0], coords_2d[order, 1],**sk)
        ax.axis('off')
        ax.set_title(f'{name} jPC{j+1}')
        ax.set_xlim([30,760])
        ax.set_ylim([680,120])

        if blnsave:
            fig2.savefig(op.join(figurepath,f'{agg_subject}_{name}_jpc{j}_surf.svg'))
