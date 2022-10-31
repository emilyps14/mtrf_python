import json
import os.path as op
import pickle
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, cm
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
from scipy.stats import ttest_rel, t as tdist


from mtrf_python import STRF_utils, peakRate
from mtrf_python.scripts.prepare_regression_data_script import get_data_from_df

agg_subject = 'EC36' # 'Hamilton_Agg_LH_9subjects'
strf_config_flag = 'regression_STRF'
config_flag = 'regression_SO1_PR_phnfeats_750msdelays'
loo_config_flags = ['regression_justPhonetic_750msdelays',
                        'regression_SO1_PR_750msdelays']
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

df = data['df']
del data

#%% Load STRF model
datapath = op.join(subjects_dir, agg_subject,
                   strf_config_flag,
                   "strf_r2_threshold_Ridge.pkl")
with open(datapath,'rb') as f:
    strf_data = pickle.load(f)

#%% Load results

cv_summary_path = op.join(loadpath, f'{cv_folder}_cv_summary.pkl')
with open(cv_summary_path, 'rb') as f:
    data = pickle.load(f)
cv_summaries = data['cv_summaries']

nfolds_outer = len(cv_summaries)
nfolds_inner = cv_summaries[0]['nfolds_inner']

validate_field = 'testing_overall_R2'

def mean_and_cis(cv):
    m = cv.mean(-1,keepdims=True)
    s = cv.std(-1,keepdims=True)
    n = cv.shape[-1]
    cis = m + tdist.ppf(0.975,n-1)/np.sqrt(n)*s*np.array([[-1,1]])
    return np.squeeze(m),np.squeeze(cis)


#%% Load sentence details

sentdet_df, featurenames, phnnames = STRF_utils.load_sentence_details(subjects_dir)

freqs = loadmat(op.join(subjects_dir,'mel_centerF.mat'), squeeze_me=True,
                  variable_names=['binfrqs'])['binfrqs']
fa_corners = freqs[:81]
fa = (fa_corners[1:]+fa_corners[:-1])/2

#%% Get 2d surface with coordinates

hemi = 'lh'
surfpath = op.join(loadpath,f'{agg_subject}_{hemi}_pial_lateral_2dSurf.pkl')

if not op.isfile(surfpath):
    with open(op.join(subjects_dir, agg_subject,f'{agg_subject}_{hemi}_pial_lateral_2dSurf_grid.pkl'),'rb') as f:
        fullsurf = pickle.load(f)
    surfdict = {}
    surfdict['img'] = fullsurf['img']
    surfdict['trans_dict'] = fullsurf['trans_dict']
    surfdict['coords_2d'] = fullsurf['coords_2d'][[int(ch.split('_')[1]) for ch in electrodes],:]

    fig, ax = plt.subplots()
    ax.imshow(surfdict['img'])
    ax.scatter(surfdict['coords_2d'][:, 0], surfdict['coords_2d'][:, 1])

    with open(surfpath,'wb') as f:
        pickle.dump(surfdict,f)

with open(surfpath,'rb') as f:
    surfdict = pickle.load(f)
img=surfdict['img']
coords_2d=surfdict['coords_2d']
blnplot = np.logical_not(np.isnan(coords_2d.sum(axis=1)))
coords_2d = coords_2d[blnplot,:]

#%% Electrode selection
plt.close('all')
fig, ax = plt.subplots()

speech_resp = [e in electrodes for e in strf_data['electrodes_all']]
toplot = strf_data['model']['testing_electrode_R2s'][speech_resp]
order = np.argsort(toplot)

vmax = toplot.max() #0.5 #np.percentile(toplot,98)
norm = colors.Normalize(vmin=0, vmax=vmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)

sk = dict(s=4,marker='o',
          c=np.stack([mapper.to_rgba(v) for v in toplot[order]]))

ax.imshow(img)
ax.scatter(coords_2d[order, 0], coords_2d[order, 1],**sk)
ax.axis('off')

cfig,cax = plt.subplots(figsize=[1.5,3],clear=True)
cfig.colorbar(mapper,cax=cax,aspect=15)
cfig.subplots_adjust(right=0.3)

if blnsave:
    fig.savefig(op.join(figurepath,f'{agg_subject}_electrode_selection.svg'))
    cfig.savefig(op.join(figurepath,f'{agg_subject}_electrode_selection_colorbar.svg'))

#%% Timeseries and features
name_sent = 'fbcg1_si1612'
befaft = [0,0] #reg_config.get('befaft', [0, 0])
soi = 0
pri = 1
df_sent = df.loc[df.name == name_sent]

# Audio timeseries
sentence = sentdet_df.loc[np.equal(sentdet_df.name, name_sent)].iterrows().__next__()[1]
source_befaft = sentence.befaft
soundf = sentence.soundf
sound_padleft = ((source_befaft[0] - befaft[0]) * soundf).astype(int)
sound_padright = ((source_befaft[1] - befaft[1]) * soundf).astype(int)
sound = STRF_utils.remove_pad(sentence.sound[None,:], sound_padleft, sound_padright)[0, :]
sound_ta = np.arange(len(sound))/soundf

# Acoustic amplitude and derivative
env = peakRate.Syl_1984_Schotola(sound, soundf, fs)
onsOffenv = np.zeros([2, len(env)])
onsOffenv[0,0] = 1
onsOffenv[1,-1] = 1
allTS, varNames = peakRate.find_landmarks(env, onsOffenv, cleanup_flag=False)
dtLoudness = allTS[1,:]
dtLoudness[dtLoudness<0] = 0

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
so_events = np.where(phnstim[:,soi]>0)[0]
pr_events = np.where(phnstim[:,pri]>0)[0]
ta = ta[:,0]

# Plot
# Note: the spectrogram axis takes up a ton of memory if I save it as a svg
# Try saving it separately as a png
plt.close('all')
fig = plt.figure(figsize=[6,6])
fig2 = plt.figure(figsize=[6,6])
gs = GridSpec(4,1,height_ratios=[0.7,1,0.3,4],left=0.22,hspace=0.1)

audax = fig.add_subplot(gs[0, 0])
audax.plot(sound_ta, sound, 'k')
for pt in ta[so_events]:
    audax.axvline(pt,linestyle='-',color='k')
for pt in ta[pr_events]:
    audax.axvline(pt,linestyle='--',color='k')
audax.set_title(txt)
audax.spines['top'].set_visible(False)
audax.spines['right'].set_visible(False)
audax.set_xticklabels([])
audax.set_yticks([])
# audax.spines['bottom'].set_visible(False)

prax = fig.add_subplot(gs[2, 0],sharex=audax)
prax.plot(ta,env,'k')
prax2 = prax.twinx()  # instantiate a second axes that shares the same x-axis
prax2.plot(ta,dtLoudness,'r')
prax2.tick_params(axis='y', labelcolor='r')
prax.spines['top'].set_visible(False)
prax.spines['right'].set_visible(False)
# prax.spines['bottom'].set_visible(False)
prax2.spines['top'].set_visible(False)
# prax2.spines['bottom'].set_visible(False)
for pt in ta[so_events]:
    prax.axvline(pt,linestyle='-',color='k')
for pt in ta[pr_events]:
    prax.axvline(pt,linestyle='--',color='k')


specax = fig2.add_subplot(gs[1, 0],sharex=audax)
specim = STRF_utils.plot_aud(melspec, ta, fa, ax=specax,
                             onset_times=ta[so_events],
                             peakRate_times=ta[pr_events],
                             pcolor_kwargs={'cmap':cm.Greys,
                                            'shading':'gouraud'})
specax.spines['top'].set_visible(False)
specax.spines['right'].set_visible(False)

featax = fig.add_subplot(gs[3, 0],sharex=audax)
for i,name in enumerate(phnstimnames):
    eventinds = phnstim[:,i]>0
    if i==pri:
        vmax = phnstim[:,i].max()*1.1
        norm = colors.Normalize(vmin=0, vmax=vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
        featax.scatter(ta[eventinds],np.ones(sum(eventinds))*i,s=200,marker='|',
                       c=phnstim[eventinds,i],cmap=cm.Reds,norm=norm)
        for j in np.where(eventinds)[0]:
            featax.text(ta[j]+0.01, i, f'{phnstim[j, i]:0.2f}',
                        color=mapper.to_rgba(phnstim[j, i]),
                        verticalalignment='center',
                        size=8)
    else:
        featax.scatter(ta[eventinds],np.ones(sum(eventinds))*i,s=200,marker='|',color='k')
featax.spines['top'].set_visible(False)
featax.spines['right'].set_visible(False)
featax.set_ylim(ndim-0.5,-0.5) # flip axis direction
featax.set_yticks(range(ndim))
featax.set_yticklabels(phnstimnames,rotation=45)
featax.set_xticks([0,0.5,1,1.5])

if blnsave:
    fig.savefig(op.join(figurepath,f'{agg_subject}_stimulus_example_{name_sent}_nospec.svg'))
    fig2.savefig(op.join(figurepath,f'{agg_subject}_stimulus_example_{name_sent}_justspec.png'))

#%% Compare penalty terms across models

plt.close('all')
fig1 = plt.figure(figsize=[4.5,6])
plt.clf()
gs = GridSpec(2,3,wspace=0.4,hspace=0.5,height_ratios=[1,2],
              bottom=0.17,left=0.15)


# Explained Variance
R2_irrr_best_cv = np.array([b['best_irrr'][validate_field] for b in cv_summaries])
R2_irrr_best,R2_irrr_best_cis = mean_and_cis(R2_irrr_best_cv)

R2_ols_cv = np.array([b['ols_model'][validate_field] for b in cv_summaries])
R2_ols,R2_ols_cis = mean_and_cis(R2_ols_cv)

R2_ridge_best_cv = np.array([b['best_ridge_model'][validate_field] for b in cv_summaries])
R2_ridge_best,R2_ridge_best_cis = mean_and_cis(R2_ridge_best_cv)


ax0 = fig1.add_subplot(gs[0,0])
ax0.bar(range(3),[R2_ols,R2_ridge_best,R2_irrr_best],color='grey')
plt.vlines(range(3),
           ymin=[R2_ols_cis[0],R2_ridge_best_cis[0],R2_irrr_best_cis[0]],
           ymax=[R2_ols_cis[1],R2_ridge_best_cis[1],R2_irrr_best_cis[1]],
           colors='k',linewidth=1)
ax0.axhline(0,color='k',linewidth=1)
ax0.set_xticks(range(3))
ax0.set_xticklabels(['OLS','Ridge','iRRR'],rotation=45,ha='right')
ax0.set_ylabel('Testing R^2')
ax0.set_title('Explained Variance')
ax0.set_ylim([0,0.28])

comps = [R2_ols_cv,R2_ridge_best_cv,R2_irrr_best_cv]
ctr = 0
for i in range(3):
    for j in range(i+1,3):
        stats,pvals = ttest_rel(comps[i],comps[j])
        # stats,pvals = ranksums(comps[i],comps[j])
        print(f'{i},{j}: t={stats}, p={pvals}')
        if pvals<0.05:
            ctr +=1
            STRF_utils.barplot_annotate_brackets(i,j,pvals,range(3),
                                      [R2_ols,R2_ridge_best,R2_irrr_best],
                                      dh=.05*ctr, barh=.02, fs=8,
                                      maxasterix=3)

# parameters
ndims_byFeature_cv = np.stack([b['best_irrr']['details']['rec_nonzeros'][-1, :]
                               for b in cv_summaries], axis=-1)
nparams_iRRR_cv = np.array([np.sum([ncomps * (ndel + nchans + 1)
                                    for ncomps, ndel in
                                    zip(ndims_byFeature_iRRR, ndelays)])
                            for ndims_byFeature_iRRR in ndims_byFeature_cv])
nparams_iRRR, nparams_iRRR_ci = mean_and_cis(nparams_iRRR_cv)

nparams_full = sum(ndelays)*nchans

ax0 = fig1.add_subplot(gs[0,2])
ax0.bar(range(3),[nparams_full,
                  nparams_full,
                  nparams_iRRR],color='grey')
ax0.vlines(2,
           ymin=nparams_iRRR_ci[0],
           ymax=nparams_iRRR_ci[1],
           colors='k',linewidth=1)
ax0.set_xticks(range(3))
ax0.set_xticklabels(['OLS','Ridge','iRRR'],rotation=45,ha='right')
ax0.set_title('# Parameters')

print(f'iRRR uses {100*(nparams_full-nparams_iRRR)/nparams_full:1.2f}% fewer parameters than OLS/Ridge')

# Group nuclear norm (iRRR penalty)
# Note that the weights depend on the data, and they are computed in the irrr fit for that cv fold
from scipy.linalg import svdvals
def compute_irrr_penalty(strfs, penalty_weights=None):
    if penalty_weights is None:
        penalty_weights = [1]*len(strfs)
    return np.sum([np.sum(svdvals(strf))*weight
                   for strf,weight in zip(strfs, penalty_weights)])

blnWeightedNN = True
if blnWeightedNN:
    group_nn_cv = np.array([[compute_irrr_penalty(b[model]['strfs'],b['best_irrr']['params']['weight'])
                                    for b in cv_summaries]
                                   for model in ['ols_model', 'best_ridge_model', 'best_irrr']])
else:
    group_nn_cv = np.array([[compute_irrr_penalty(b[model]['strfs'])
                                    for b in cv_summaries]
                                   for model in ['ols_model', 'best_ridge_model', 'best_irrr']])

group_nn,group_nn_cis = mean_and_cis(group_nn_cv)

ax2 = fig1.add_subplot(gs[0,1])
ax2.bar(range(3),group_nn,color='grey')
ax2.vlines(range(3),
           ymin=group_nn_cis[:,0],
           ymax=group_nn_cis[:,1],
           colors='k',linewidth=1)
ax2.set_xticks(range(3))
ax2.set_xticklabels(['OLS','Ridge','iRRR'],rotation=45,ha='right')
ax2.set_title('Group Nuclear Norm')
# ax2.set_ylim([0,15])
#
# ctr = 0
# for i in range(3):
#     for j in range(i+1,3):
#         stats,pvals = ttest_rel(group_nn_cv[i,:],group_nn_cv[j,:])
#         print(f'{i},{j}: t={stats}, p={pvals}')
#         if pvals<0.05:
#             ctr +=1
#             STRF_utils.barplot_annotate_brackets(i,j,pvals,range(3),
#                                       group_nn,
#                                       dh=.05*ctr, barh=.02, fs=8,
#                                       maxasterix=3)


# Unique explained variance
nl = len(loo_config_flags)
nsubmodels = ndim+nl
R2_irrr_best_cv = np.array([b['best_irrr'][validate_field] for b in cv_summaries])
R2_iRRR_loo_cv = np.zeros((nsubmodels,nfolds_outer))
for i in range(ndim):
    R2_iRRR_loo_cv[i, :] = [b['loo_irrrs'][i][validate_field] for b in cv_summaries]
for i,l in enumerate(loo_config_flags):
    R2_iRRR_loo_cv[i+ndim, :] = [b['group_loo_irrrs'][l][validate_field] for b in cv_summaries]
R2_iRRR_loo_pct_cv = (R2_irrr_best_cv-R2_iRRR_loo_cv)/R2_irrr_best_cv
R2_iRRR_loo_pct,R2_iRRR_loo_pct_cis = mean_and_cis(R2_iRRR_loo_pct_cv)

names_submodels = list(phnstimnames)
if nl==2:
    names_submodels += ['All Timing','All Phonetic']


ax3 = fig1.add_subplot(gs[1,:])
ax3.bar(range(nsubmodels),R2_iRRR_loo_pct*100, color=['C0'] * ndim + ['gray'] * nl)
plt.vlines(range(nsubmodels),
           ymin=R2_iRRR_loo_pct_cis[:,1]*100,
           ymax=R2_iRRR_loo_pct_cis[:,0]*100,
           colors='k',linewidth=1)
ax3.axhline(0,color='k',linewidth=1)
ax3.set_xticks(range(nsubmodels))
ax3.set_xticklabels(names_submodels,rotation=45,ha='right')
ax3.set_ylabel('% of full model')
ax3.set_title('Unique Explained Variance')
ax3.set_ylim([-1,26])

# one-sample t test for individual features vs 0
# for i in range(ndim):
#     stat,pval = ttest_1samp(R2_iRRR_loo_pct_cv[i,:],0)
#     print(f'{i}: t={stat}, p={pval}')
#     if pval<0.05:
#         STRF_utils.barplot_annotate_brackets(i,i,pval,range(nsubmodels),
#                                   R2_iRRR_loo_pct_cis[:,1]*100,
#                                   dh=.02, barh=0, fs=8,
#                                   maxasterix=3)

# compare the groups
i=nsubmodels-1
j=nsubmodels-2
stats,pvals = ttest_rel(R2_iRRR_loo_pct_cv[i],R2_iRRR_loo_pct_cv[j])
# stats,pvals = ranksums(R2_iRRR_loo_pct_cv[i],R2_iRRR_loo_pct_cv[j])
print(f'{names_submodels[i]},{names_submodels[j]}: t={stats}, p={pvals}')
if pvals<0.05:
    STRF_utils.barplot_annotate_brackets(i,j,pvals,range(nsubmodels),
                              R2_iRRR_loo_pct_cis[:,1]*100,
                              dh=.02, barh=0.02, fs=8,
                              maxasterix=3)

# Compare SO and PR to all other features (don't plot)
pvals_feats = []
for i in range(2):
    for j in range(2,ndim):
        stats,pvals = ttest_rel(R2_iRRR_loo_pct_cv[i],R2_iRRR_loo_pct_cv[j])
        # stats,pvals = ranksums(R2_iRRR_loo_pct_cv[i],R2_iRRR_loo_pct_cv[j])
        # print(f'{names_submodels[i]},{names_submodels[j]}: t={stats}, p={pvals}, p (corrected)={pvals*2*ndim}')
        pvals_feats.append(pvals)

print(f'Max corrected pvalue between timing and phonetic: {np.max(pvals_feats)*2*ndim}')

if blnsave:
    fig1.savefig(op.join(figurepath,f'{agg_subject}_performance_summary2_wsig.svg'))
