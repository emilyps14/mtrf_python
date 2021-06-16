import json
import os.path as op
import pickle
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, cm
from matplotlib.gridspec import GridSpec

from mtrf_python import STRF_utils
from mtrf_python.scripts.prepare_regression_data_script import get_delay_matrices, get_data_from_df

agg_subject = 'Hamilton_Agg_LH_no63_131_143' # 'Hamilton_Agg_LH_9subjects'
config_flag = 'regression_SO1_PR_phnfeats_750msdelays'
irrr_flag = 'iRRR_210111' # 10**np.array([-1.5, -1.1, -0.7, -0.3, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9])

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


preproc_dir = config['local_timit_dir']
sentdet_df, featurenames, phnnames = STRF_utils.load_sentence_details(op.join(preproc_dir))

#%% Load results

cv_summary_path = op.join(loadpath, f'{cv_folder}_cv_summary.pkl')
with open(cv_summary_path, 'rb') as f:
    data = pickle.load(f)
cv_summaries=data['cv_summaries']

nfolds_outer = len(cv_summaries)
nfolds_inner = cv_summaries[0]['nfolds_inner']


soi = [i for i,x in enumerate(phnstimnames) if x=='sentence_onset'][0]
pri = [i for i,x in enumerate(phnstimnames) if x=='peakRate'][0]
loo_i = pri

#%% make map of sentence_ind to sentence name

sentence_ind_to_name = {x['sentence_ind'][0,0,0].split('_')[0]:x['name'] for i,x in df.iterrows()}
sentence_name_to_ind = {v:k for k,v in sentence_ind_to_name.items()}

sentence_name_to_summary = {}
for i,cv_summary in enumerate(cv_summaries):
    cv_val_sentences = [sentence_ind_to_name[x.split('_')[0]] for x in np.unique(sentence_ind[cv_summary['test_inds']])]
    for name_sent in cv_val_sentences:
        sentence_name_to_summary[name_sent] = i

#%%

cmap = 'jet'
mplcmap = cm.get_cmap(cmap)

# name_sent = 'fdms0_si1218' # poets, moreover
# pr_events_toplot = [0,3,5]
name_sent = 'fcaj0_si1804' # clockwork
pr_events_toplot = [0,1,2,3]
# name_sent = 'mfwk0_si1249' # twenty thirty fifty
# pr_events_toplot = [0,1,2,3]
# name_sent = 'fbcg1_si1612' # they've never met, you know
# pr_events_toplot = None

blnPrediction = True # whether to plot prediction or projected responses

befaft = [0,0] #reg_config.get('befaft', [0, 0])
feati = pri

sentence = sentdet_df.loc[np.equal(sentdet_df.name, name_sent)].iterrows().__next__()[1]
source_befaft = sentence.befaft
soundf = sentence.soundf
sound_padleft = ((source_befaft[0] - befaft[0]) * soundf).astype(int)
sound_padright = ((source_befaft[1] - befaft[1]) * soundf).astype(int)
sound = STRF_utils.remove_pad(sentence.sound[None,:], sound_padleft, sound_padright)[0, :]
sound_ta = np.arange(len(sound))/soundf

padleft = ((source_befaft[0] - befaft[0]) * fs).astype(int)
padright = ((source_befaft[1] - befaft[1]) * fs).astype(int)

plt.close('all')

def plot_jpcs(ax,ta,latent_state,norm,**kwargs):
    # STRF_utils.colorline(latent_state[:,0],latent_state[:,1],c=ta,cmap=cmap,norm=norm,ax=ax)
    lb = latent_state[:,:2].min()*1.1
    ub = latent_state[:,:2].max()*1.1
    ax.set_xlim(lb,ub)
    ax.set_ylim(lb,ub)
    # ax.axhline(0,color='k')
    # ax.axvline(0,color='k')
    ax.plot(latent_state[:,0],latent_state[:,1],**kwargs)



# Get model
cv_summary = cv_summaries[sentence_name_to_summary[name_sent]]
best_irrr = cv_summary['best_irrr']
best_ndims_by_feature = best_irrr['details']['rec_nonzeros'][-1, :].astype(int)

# Get sentence data
df_sent = df.loc[df.name == name_sent]
txt = df_sent.iloc[0].txt
phnstim, ta, _, resp, _ = \
    get_data_from_df(df_sent,phnstimnames,subjects,
                     padleft,padright,
                     electrodes_sel=electrodes)
dstim, _ = get_delay_matrices(phnstim,phnstimnames,time_lags,fs)
pr_events = np.where(phnstim[:,feati]>0)[0]
if pr_events_toplot is None:
    pr_events_toplot = range(len(pr_events))
feat_events = np.where(phnstim[:,2:].sum(axis=1)>0)[0]

ta = ta[:,0]
norm = plt.Normalize(ta.min(),ta.max())

jpcas = STRF_utils.compute_jPCA_on_MTRFs(phnstimnames, delays, best_ndims_by_feature, best_irrr['strfs'])

# Plot sentence audio and peak rate event magnitude
fig = plt.figure(figsize=[8.5,4])
gs = GridSpec(2,len(pr_events_toplot),height_ratios=[0.8,1],hspace=0.3)#[0.8,0.2,1])
audax = fig.add_subplot(gs[0, :])
audax.plot(sound_ta, sound,'k')
STRF_utils.colorline(ta, np.ones(ta.shape) * audax.get_ylim()[0], c=ta,
                     cmap=cmap, norm=norm, linewidth=10, ax=audax)
audax.set_yticks([])
for pt in ta[feat_events]:
    audax.axvline(pt,linestyle=':',color=mplcmap(norm(pt)), linewidth=1)
for pt in ta[pr_events[pr_events_toplot]]:
    audax.axvline(pt,linestyle='-',color='k')

audax.set_title(txt)
audax.set_xlim(ta[[0,-1]])

# prax = fig.add_subplot(gs[1, :], sharex=audax)
# prax.plot(ta, phnstim[:,feati],color='grey')
# prax.axis('off')


mtrf_approx = best_irrr['strfs'][feati]
jpca_sel = jpcas[phnstimnames[feati]]['jpca']
pred_state = dstim[feati] @ mtrf_approx @ jpca_sel.jpcs
resp_state = np.tensordot(resp,jpca_sel.jpcs,axes=[1,0])
if blnPrediction:
    toplot = pred_state
else:
    toplot = resp_state

for i,(event_toplot,next_event) in enumerate(zip(pr_events[pr_events_toplot],np.append(pr_events[pr_events_toplot[1:]],len(ta)))):
    latentax = fig.add_subplot(gs[1,i])
    inds = np.arange(event_toplot-1,next_event) # start an index early so that we capture the jump
    inds = inds[inds<len(ta)]
    feat_inds = np.intersect1d(inds,feat_events)
    plot_jpcs(latentax,ta[inds],toplot[inds,:],norm,color='grey',linestyle='-')
    latentax.scatter(toplot[feat_inds,0],toplot[feat_inds,1],c=ta[feat_inds],
                     cmap=cmap,norm=norm)
    lb = toplot[:, :2].min(axis=0) - toplot[:,:2].std(axis=0)/2
    ub = toplot[:, :2].max(axis=0) + toplot[:,:2].std(axis=0)/2
    latentax.set_ylim(lb[1],ub[1])
    latentax.set_xlim(lb[0],ub[0])

    ticks = np.array([-1,0,1])
    latentax.set_xticks(ticks)
    latentax.spines['top'].set_visible(False)
    latentax.spines['right'].set_visible(False)
    latentax.spines['bottom'].set_bounds(ticks[[0,-1]])

    if i==0:
        latentax.set_yticks(ticks)
        latentax.spines['left'].set_bounds(ticks[[0,-1]])
        pass
    else:
        latentax.spines['left'].set_visible(False)
        latentax.set_yticks([])

# fig.text(0.5,0.05,f'{phnstimnames[feati]} Latent State (jPCA projection)',horizontalalignment='center')

if blnsave:
    fig.savefig(op.join(figurepath,f'{name_sent}_scaffolding.png'))
    fig.savefig(op.join(figurepath,f'{name_sent}_scaffolding.svg'))

#%%


fig,ax = plt.subplots()
for i in range(ndim):
    inds = phnstim[:,i]>0
    ax.scatter(ta[inds],np.ones(inds.sum())*i,c=ta[inds],
                     cmap=cmap,norm=norm)
for pt in ta[feat_events]:
    ax.axvline(pt,linestyle=':',color=mplcmap(norm(pt)), linewidth=1)
for pt in ta[pr_events[pr_events_toplot]]:
    ax.axvline(pt,linestyle='-',color='k')

# ax.scatter(phnstim[:,2:].T,aspect='auto')
ax.set_yticks(range(ndim))
ax.set_yticklabels(phnstimnames)

#%%
fig,ax = plt.subplots()
STRF_utils.colorline(toplot[:,0],toplot[:,1],c=ta,cmap=cmap,norm=norm,ax=ax)

lb = toplot[:,:2].min()*1.1
ub = toplot[:,:2].max()*1.1
ax.scatter(toplot[pr_events[pr_events_toplot],0],toplot[pr_events[pr_events_toplot],1],c='k')

for i,(event_toplot,next_event) in enumerate(zip(pr_events[pr_events_toplot],np.append(pr_events[pr_events_toplot[1:]],len(ta)))):
    inds = np.arange(event_toplot,next_event)
    inds = inds[inds<len(ta)]
    feat_inds = np.intersect1d(inds,feat_events)
    plot_jpcs(ax,ta[inds],toplot[inds,:],norm,color='grey',linestyle='-')

ax.set_xlim(lb,ub)
ax.set_ylim(lb,ub)

#%%
ncomponents = best_ndims_by_feature[feati]
U,S,Vh = STRF_utils.get_svd_of_mtrf(mtrf_approx,ncomponents,
                                reduced=True)
pred_state_3d = dstim[feati] @ (U*S)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
STRF_utils.colorline(pred_state_3d[:,0],pred_state_3d[:,1],pred_state_3d[:,2],c=ta,cmap=cmap,norm=norm,ax=ax)
# ax.scatter(pred_state_3d[pr_events[pr_events_toplot],0],pred_state_3d[pr_events[pr_events_toplot],1],pred_state_3d[pr_events[pr_events_toplot],2],color='k')
ax.set_xlim(pred_state_3d[:,0].min() - pred_state_3d[:,0].std()/2,pred_state_3d[:,0].max() + pred_state_3d[:,0].std()/2)
ax.set_ylim(pred_state_3d[:,1].min() - pred_state_3d[:,1].std()/2,pred_state_3d[:,1].max() + pred_state_3d[:,1].std()/2)
ax.set_zlim(pred_state_3d[:,2].min() - pred_state_3d[:,2].std()/2,pred_state_3d[:,2].max() + pred_state_3d[:,2].std()/2)

ax.view_init(elev=44.6, azim=-76.6)
ax.set_xlabel('#1')
ax.set_ylabel('#2')
ax.set_zlabel('#3')

ax.set_xticks(np.arange(0,pred_state_3d[:,0].max(),5))

if blnsave:
    fig.savefig(op.join(figurepath,f'{name_sent}_scaffolding_3d.svg'))
