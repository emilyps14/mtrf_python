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

sentdet_df, featurenames, phnnames = STRF_utils.load_sentence_details(subjects_dir)

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
