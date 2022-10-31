import json
import os.path as op
import pickle
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, cm
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
from scipy.stats import t as tdist

from mtrf_python import STRF_utils
from mtrf_python.scripts.prepare_regression_data_script import get_delay_matrices, get_data_from_df

agg_subject = 'EC36' # 'Hamilton_Agg_LH_9subjects'
strf_config_flag = 'regression_STRF'
config_flag = 'regression_SO1_PR_phnfeats_750msdelays'
col_ctr_RR = True

# Load paths
with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
    config = json.load(f)
subjects_dir = config['subjects_dir']
loadpath = op.join(subjects_dir, agg_subject, config_flag)

blnsave = True

cv_folder = 'cv-10fold-1'

# [Data prep in run_cv_justEC36.py]

#%%
figurepath = op.join(loadpath, 'Figures')
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

#%% Load results

cv_summary_path = op.join(loadpath, f'{cv_folder}_cv_summary.pkl')
with open(cv_summary_path, 'rb') as f:
    data = pickle.load(f)
cv_summaries = data['cv_summaries']

nfolds_outer = len(cv_summaries)
nfolds_inner = cv_summaries[0]['nfolds_inner']

validate_field = 'testing_overall_R2'

# Use the first CV fold to pull out specific model fits
cv1_best_irrr = cv_summaries[0]['best_irrr']
cv1_strfs_irrr = cv1_best_irrr['strfs']
cv1_best_ndims_by_feature = cv1_best_irrr['details']['rec_nonzeros'][-1, :].astype(int)

def mean_and_cis(cv):
    m = cv.mean(-1,keepdims=True)
    s = cv.std(-1,keepdims=True)
    n = cv.shape[-1]
    cis = m + tdist.ppf(0.975,n-1)/np.sqrt(n)*s*np.array([[-1,1]])
    return np.squeeze(m),np.squeeze(cis)

#%% Load sentence details

sentdet_df, featurenames, phnnames = STRF_utils.load_sentence_details(op.join(subjects_dir))

freqs = loadmat(op.join(subjects_dir,'mel_centerF.mat'), squeeze_me=True,
                  variable_names=['binfrqs'])['binfrqs']
fa_corners = freqs[:81]
fa = (fa_corners[1:]+fa_corners[:-1])/2

#%% Compute predicted trajectories

# name_sent = 'fbcg1_si1612'
name_sent = 'fcaj0_si1804' # clockwork
soi = 0
pri = 1
nplot = 2
df_sent = df.loc[df.name == name_sent]
befaft = [0,0]


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

#%% Compare penalty terms across models

plt.close('all')
fig1 = plt.figure(figsize=[5,2])
plt.clf()
gs = GridSpec(1,2,wspace=0.4,hspace=0.5,bottom=0.3)

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

# parameters
ndims_byFeature_cv = np.stack([b['best_irrr']['details']['rec_nonzeros'][-1, :]
                               for b in cv_summaries], axis=-1)
nparams_iRRR_cv = np.array([np.sum([ncomps * (ndel + nchans + 1)
                                    for ncomps, ndel in
                                    zip(ndims_byFeature_iRRR, ndelays)])
                            for ndims_byFeature_iRRR in ndims_byFeature_cv])
nparams_iRRR, nparams_iRRR_ci = mean_and_cis(nparams_iRRR_cv)

nparams_full = sum(ndelays)*nchans

ax0 = fig1.add_subplot(gs[0,1])
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

print(f'iRRR uses {(nparams_full-nparams_iRRR)/nparams_full}% fewer parameters than OLS/Ridge')

if blnsave:
    fig1.savefig(op.join(figurepath,f'{agg_subject}_summary_1.svg'))

#%% Latent states
norm = plt.Normalize(ta.min(),ta.max())
cmap = 'jet'

# Aud
fig = plt.figure(num=2,figsize=[5,1.5])
gs = GridSpec(1,1,hspace=0.5,bottom=0.3,top=0.8)

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


if blnsave:
    fig.savefig(op.join(figurepath,f'{agg_subject}_summary_2.svg'))

#%% Top 3

fig = plt.figure(num=3,figsize=[5,2.5])
fig.clear()
gs = GridSpec(1,nplot,wspace=0.5)

for i,(name,stim,mtrf_approx,ncomponents) in enumerate(zip(phnstimnames[:nplot], dstims, cv1_strfs_irrr, cv1_best_ndims_by_feature)):
    U,S,Vh = STRF_utils.get_svd_of_mtrf(mtrf_approx,ncomponents,True)

    pred = stim @ (U*S)[:,:3]

    if ncomponents>2:
        ax = fig.add_subplot(gs[i],projection='3d')
        STRF_utils.colorline(pred[:,0],pred[:,1],pred[:,2],c=ta,cmap=cmap,norm=norm,ax=ax)
        ax.set_zlim(pred[:,2].min() - pred[:,2].std()/2,pred[:,2].max() + pred[:,2].std()/2)
        ax.set_xlim(pred[:,0].min() - pred[:,0].std()/2,pred[:,0].max() + pred[:,0].std()/2)
        ax.set_ylim(pred[:,1].min() - pred[:,1].std()/2,pred[:,1].max() + pred[:,1].std()/2)

        ax.view_init(elev=47, azim=-65)
        ax.set_xlabel('#1')
        ax.set_ylabel('#2')
        ax.set_zlabel('#3')
        ax.set_xticks(np.arange(0,pred[:,0].max(),5))
    elif ncomponents>1:
        ax = fig.add_subplot(gs[i])
        STRF_utils.colorline(pred[:,0],pred[:,1],c=ta,cmap=cmap,norm=norm,ax=ax)

        ax.set_xlim(pred[:,0].min() - pred[:,0].std()/2,pred[:,0].max() + pred[:,0].std()/2)
        ax.set_ylim(pred[:,1].min() - pred[:,1].std()/2,pred[:,1].max() + pred[:,1].std()/2)
        ax.set_xlabel('#1')
        ax.set_ylabel('#2')
        ax.set_xticks(np.arange(0,pred[:,0].max(),5))



# fig.suptitle(title)

if blnsave:
    fig.savefig(op.join(figurepath,f'{agg_subject}_summary_3.svg'))
