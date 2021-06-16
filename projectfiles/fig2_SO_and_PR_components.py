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
cv1_best_ndims_by_feature = cv1_best_irrr['details']['rec_nonzeros'][-1, :]

#%% Load sentence details

preproc_dir = config['local_timit_dir']
sentdet_df, featurenames, phnnames = STRF_utils.load_sentence_details(op.join(preproc_dir))

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


#%% Components in time

plt.close('all')
nfeat_toplot = 2
fig,axs = plt.subplots(1,nfeat_toplot,figsize=[8.5,2.75],clear=True,
                       gridspec_kw={'width_ratios':ndelays[:nfeat_toplot]})
for i, (name,feat_delays,ndelay,mtrf_approx,ax) in enumerate(zip(phnstimnames[:nfeat_toplot], delays, ndelays, cv1_strfs_irrr,axs)):
    ncomponents = int(cv1_best_ndims_by_feature[i])

    # plot top # singular vectors
    if ncomponents>0:
        U,S,Vh = STRF_utils.get_svd_of_mtrf(mtrf_approx,ncomponents)
        ax.plot(feat_delays*10,U[:,:ncomponents]*S[:ncomponents])
        ax.legend([f'#{j+1}' for j in range(ncomponents)])
        # ax.set_aspect(ndelay / 50, 'box')
        ax.set_xlabel('Delay (ms)')

if blnsave:
    fig.savefig(op.join(figurepath,f'{agg_subject}_MTRF_time_components.png'))
    fig.savefig(op.join(figurepath,f'{agg_subject}_MTRF_time_components.svg'))

#%% Components on surface

plt.close('all')
nfeat_toplot = 2
vmax = 0.15
for i, (name,feat_delays,mtrf_approx) in enumerate(zip(phnstimnames[:nfeat_toplot], delays, cv1_strfs_irrr)):
    ncomponents = int(cv1_best_ndims_by_feature[i])
    ndelay = len(feat_delays)

    # plot top # singular vectors
    if ncomponents>0:
        U,S,Vh = STRF_utils.get_svd_of_mtrf(mtrf_approx,ncomponents)
        explained_variance = (S ** 2) / (ndelay - 1)
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var

        nplot = min(ncomponents,2)
        # vmax = np.percentile(np.abs(Vh[:nplot,:]),98)
        norm = colors.Normalize(vmin=-vmax, vmax=vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdBu_r)

        # color bar
        # fig,ax = plt.subplots(figsize=[1.5,3],clear=True)
        # plt.colorbar(mapper,cax=ax,aspect=15)
        # plt.subplots_adjust(right=0.3)
        fig,ax = plt.subplots(figsize=[3.5,0.6],clear=True,
                              gridspec_kw={'bottom':0.4})
        plt.colorbar(mapper,cax=ax,aspect=15,orientation='horizontal')

        fig2,axs = plt.subplots(2,2,figsize=[8.5,6.6],sharey='row',clear=True,
                                gridspec_kw={'height_ratios':[3,2.5],
                                             'wspace':0.05})
        for j in range(nplot):
            toplot = Vh[j,blnplot]
            order = np.argsort(toplot**2)
            imgax = axs[0,j]
            erpax = axs[1,j]
            imgax.imshow(img)
            sk = dict(s=10,marker='o',
                      c=np.stack([mapper.to_rgba(v) for v in toplot[order]]))
            imgax.scatter(coords_2d[order, 0], coords_2d[order, 1],**sk)
            imgax.axis('off')
            imgax.set_xlim([30,760])
            imgax.set_ylim([680,120])
            # imgax.set_title(f'{explained_variance_ratio[j]*100:0.1f}%')

            for k in range(nchans):
                erpax.plot(feat_delays*10,mtrf_approx[:,order[k]],color=mapper.to_rgba(toplot[order[k]]),linewidth=1)

        # fig2.suptitle(name)

        if blnsave:
            sfx = 'MTRF_surf_and_ERP'
            fig2.savefig(op.join(figurepath,f'{agg_subject}_{name}_{sfx}.png'))
            fig2.savefig(op.join(figurepath,f'{agg_subject}_{name}_{sfx}.svg'))
            fig.savefig(op.join(figurepath,f'{agg_subject}_{name}_colorbar.svg'))

#%% <Below is experimenting with electrode traces>




sentence_ind_to_name = {x['sentence_ind'][0,0,0].split('_')[0]:x['name'] for i,x in df.iterrows()}

sentence_name_to_summary = {}
for i,cv_summary in enumerate(cv_summaries):
    cv_test_sentences = [sentence_ind_to_name[x.split('_')[0]] for x in np.unique(sentence_ind[cv_summary['test_inds']])]
    for name_sent in cv_test_sentences:
        sentence_name_to_summary[name_sent] = i

#%%

import pandas as pd
mtrf_approx = cv1_strfs_irrr[1]
ncomponents = int(cv1_best_ndims_by_feature[i])
U,S,Vh = STRF_utils.get_svd_of_mtrf(mtrf_approx,ncomponents)

df_pr = pd.DataFrame(index=electrodes,data={'R2':cv1_best_irrr['testing_electrode_R2s'],'comp2':Vh[1,:]})

print(df_pr.loc[np.logical_and(df_pr.R2>0.3,df_pr.comp2<-0.05)])

plt.figure()
plt.scatter(cv1_best_irrr['testing_electrode_R2s'],Vh[1,:])

#%% Example electrodes

# Good SO electrodes: *'EC92_6', 'EC92_22', 'EC82_66', 'EC82_83'
# Good PR electrodes (positive): 'EC55_198', *'EC36_71'
# Good PR electrodes (negative): 'EC2_69', 'EC35_131', 'EC36_22', 'EC113_53', 'EC113_132', 'EC113_149'

elnames = ['EC92_6', 'EC36_71',  'EC36_22']
name_sent = 'fcaj0_si1804'#'fbcg1_si1612' #'fdms0_si1218' #fbcg1_si1612'


# TODO use the same CV fold for all panels?
cv_summary = cv_summaries[sentence_name_to_summary[name_sent]]


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

# Features and response
padleft = ((source_befaft[0] - befaft[0]) * fs).astype(int)
padright = ((source_befaft[1] - befaft[1]) * fs).astype(int)
df_sent = df.loc[df.name == name_sent]
txt = df_sent.iloc[0].txt
phnstim, ta, _, resp, _ = \
    get_data_from_df(df_sent,phnstimnames,subjects,
                     padleft,padright,
                     electrodes_sel=electrodes)
so_events = np.where(phnstim[:,soi]>0)[0]
pr_events = np.where(phnstim[:,pri]>0)[0]
ta = ta[:,0]

dstims, _ = get_delay_matrices(phnstim, phnstimnames, time_lags, fs)
pred = STRF_utils.compute_pred(np.concatenate(dstims,axis=1),cv_summary['best_irrr']['wts'],
                               cv_summary['best_irrr']['mu'].T)

# Figure
plt.close('all')
fig = plt.figure(figsize=[7,6])
gs = GridSpec(len(elnames)+1,1,hspace=0,left=0.2)

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

for i,elname in enumerate(elnames):
    ax = fig.add_subplot(gs[i+1,0])
    eli = np.argwhere(np.array(electrodes)==elname)[0]
    # ax.plot(ta,resp[:,eli],'k')

    subj = elname.split('_')[0]
    df_sent_subj = df_sent[df_sent.subject==subj]
    _, _, _, resp_subj, _ = \
        get_data_from_df(df_sent_subj,phnstimnames,[subj],
                         padleft,padright,
                         electrodes_sel=[elname],
                         repeat_sel=range(df_sent_subj.nrepeats.min()))
    ax.plot(ta,resp_subj[:,0,:],color='grey')
    ax.plot(ta,resp_subj[:,0,:].mean(1),color='k',linewidth=2)

    ax.plot(ta,pred[:,eli],'r')

    ax.set_ylabel(elname,rotation=0,labelpad=30)
