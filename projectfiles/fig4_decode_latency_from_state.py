import json
import os.path as op
import pickle
from os import makedirs

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.neural_network import MLPRegressor
from scipy.stats import t as tdist

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

#%% Load results

cv_summary_path = op.join(loadpath, f'{cv_folder}_cv_summary.pkl')
with open(cv_summary_path, 'rb') as f:
    data = pickle.load(f)
cv_summaries=data['cv_summaries']

nfolds_outer = len(cv_summaries)
nfolds_inner = cv_summaries[0]['nfolds_inner']

# Use the first CV fold to pull out specific model fits
cv1_best_irrr = cv_summaries[0]['best_irrr']
cv1_strfs_irrr = cv1_best_irrr['strfs']
cv1_best_ndims_by_feature = cv1_best_irrr['details']['rec_nonzeros'][-1, :].astype(int)

cv1_train_inds = cv_summaries[0]['train_inds']
cv1_test_inds = cv_summaries[0]['test_inds']

#%% Functions

# compute time since sentence onset and peak rate events for training and validation data
def trial_latency(stimdata,max_latency=0.5):
    latencies = np.arange(0,max_latency,1/fs)
    Nl = len(latencies)
    out = np.zeros_like(stimdata)*np.nan
    for i,j in np.argwhere(stimdata):
        endpt = j+Nl
        if endpt>stimdata.shape[1]:
            endpt = stimdata.shape[1]
        out[i,j:endpt] = latencies[:endpt-j]
    return out

# get projection matrices for latent states
def get_latent_projection(mtrf_approx,ncomponents):
    U,S,Vh = STRF_utils.get_svd_of_mtrf(mtrf_approx,ncomponents)
    return Vh[:ncomponents,:].T

def decode_helper(X_train, Y_train,X_test,Y_test,swtScore):
    regr = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(20,),
                        alpha=1e-5,activation='logistic').fit(X_train, Y_train)
    y_hat = regr.predict(X_test)
    if swtScore=='score':
        r2 = regr.score(X_test, Y_test)
    else:
        r2 = (np.corrcoef(Y_test.ravel(), y_hat.ravel())[0,1])**2
    return regr,y_hat,r2


def decode_latency(phnstimnames_to_train,
                   phnstim,resp,timesel,
                   phnstim_val,resp_val,timesel_val,
                   list_of_projs_to_train,proj_names,blnCenter,swtScore,
                   blnplot=True, ):
    # list_of_projs_to_train: list containing lists with the projection matrices for each feature (nprojs,) list of (ndim,) lists of [Nxk] projection matrices
    # e.g. [[np.eye(nchans)]*ndim, latent_projs_to_train, jpcs_to_train] will train each feature on [all electrodes, latent projection, jpca projection]
    # proj_names: (nprojs,) list of the names for the projection types
    # e.g. ['All Electrodes','Latent Projection','jPCA Projection']
    r2s = np.zeros((len(feats_to_train),len(list_of_projs_to_train)))
    for i,(name,stimcol,tselcol,stimcol_val,tselcol_val) \
            in enumerate(zip(phnstimnames_to_train,
                               phnstim.T,timesel.T,
                               phnstim_val.T,timesel_val.T)):

        X_train = resp[tselcol,:]
        Y_train = stimcol[tselcol]
        X_test = resp_val[tselcol_val,:]
        Y_test = stimcol_val[tselcol_val]

        if blnCenter:
            X_train -= np.mean(X_train,0,keepdims=True)
            Y_train -=  np.mean(Y_train,0,keepdims=True)
            X_test -= np.mean(X_test,0,keepdims=True)
            Y_test -=  np.mean(Y_test,0,keepdims=True)

        yhats_feat = []
        r2s_feat = []
        for j,projs_to_train in enumerate(list_of_projs_to_train):
            proj = projs_to_train[i]
            regr,y_hat,r2 = decode_helper(X_train@proj,Y_train,
                                          X_test@proj,Y_test,
                                          swtScore)
            yhats_feat.append(y_hat)
            r2s_feat.append(r2)
            r2s[i,j] = r2

        if blnplot:
            fig,ax = plt.subplots(clear=True)
            ax.plot(Y_test,label='True Latency')
            for feat_name,y_hat,r2 in zip(proj_names,yhats_feat,r2s_feat):
                print(f'{name}, {feat_name}: {r2}')
                ax.plot(y_hat,label=f'Prediction from {feat_name} (r={r2})')
            ax.legend()
            ax.set_ylabel('Time')
            ax.set_xlabel('Latency (s)')
            fig.suptitle(name)

    return r2s

#%% Decode time since sentence onset / peak rate using projected latent states
# (Run it for CV fold 1 to make sure everything is working)

feats_to_train = list(range(ndim))
blnCenter = False
swtScore = 'score' #

phnstimnames_to_train = [f'{name}_latency' for name in phnstimnames[feats_to_train]]
jpcas = STRF_utils.compute_jPCA_on_MTRFs(phnstimnames, delays, cv1_best_ndims_by_feature, cv1_strfs_irrr)
jpcs_to_train = [jpcas[phnstimnames[i]]['jpca'].jpcs for i in feats_to_train]
latent_projs_to_train = [get_latent_projection(cv1_strfs_irrr[i], cv1_best_ndims_by_feature[i]) for i in feats_to_train]

for name,colname,lag in zip(phnstimnames[feats_to_train],phnstimnames_to_train,time_lags):
    df[colname] = df.apply(lambda x: trial_latency(x[name],lag),axis=1)

phnstim_all, ta_all, _, resp_all, _ = get_data_from_df(df,phnstimnames_to_train,
                                                       subjects,
                                                       0,0,
                                                       electrodes_sel=electrodes)

# get times that aren't within the delay time of a relevant event
timesel_all = np.logical_not(np.isnan(phnstim_all))

## train on training data and test on validation set (perceptron)
ta,phnstim,resp,timesel = ta_all[cv1_train_inds,:],phnstim_all[cv1_train_inds,:],resp_all[cv1_train_inds,:],timesel_all[cv1_train_inds,:]
ta_val,phnstim_val,resp_val,timesel_val = ta_all[cv1_test_inds,:],phnstim_all[cv1_test_inds,:],resp_all[cv1_test_inds,:],timesel_all[cv1_test_inds,:]

plt.close('all')
proj_list = [[np.eye(nchans)]*ndim, latent_projs_to_train, jpcs_to_train]
proj_names = ['All Electrodes','Latent Projection','jPCA Projection']
r2s = decode_latency(phnstimnames_to_train,
                   phnstim,resp,timesel,
                   phnstim_val,resp_val,timesel_val,
                   proj_list,
                   proj_names,
                    blnCenter,swtScore)

#%%

plt.close('all')

width = 0.35  # the width of the bars

fig = plt.figure(figsize=[5,3])
plt.clf()
gs = GridSpec(1,1,wspace=0.4,bottom=0.4)
ax = fig.add_subplot(gs[0,0])
x = np.arange(len(feats_to_train))
rects1 = ax.bar(x - width/2, r2s[:,0], width, label='Prediction from all electrodes')
rects2 = ax.bar(x + width/2, r2s[:,1], width, label='Prediction from latent projection')
ax.axhline(0,color='k',linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([name for name in phnstimnames[feats_to_train]],rotation=45,ha='right')
ax.set_ylabel('Testing R^2')
ax.set_title('Decoding time relative to event')
ax.legend()

#%% Compute for all cv folds (for mean and cis)
proj_names_b =['All Electrodes','Latent Projection']

r2s_cv = np.zeros((nfolds_outer, len(feats_to_train), 2))
for i,b in enumerate(cv_summaries):
    print(i)
    train_inds = b['train_inds']
    test_inds = b['test_inds']
    best_ndims_by_feature_b = b['best_irrr']['details']['rec_nonzeros'][-1,:].astype(int)
    strfs_irrr_b = b['best_irrr']['strfs']

    # jpcas_b = STRF_utils.compute_jPCA_on_MTRFs(phnstimnames,delays,best_ndims_by_feature_b,strfs_irrr_b)
    # jpcas_to_train_b = [jpcas_b[phnstimnames[i]]['jpca'] for i in feats_to_train]
    latent_projs_to_train_b = [get_latent_projection(strfs_irrr_b[i], best_ndims_by_feature_b[i]) for i in feats_to_train]
    proj_list_b = [[np.eye(nchans)]*ndim, latent_projs_to_train_b]

    r2s_cv[i, :, :] = \
        decode_latency(phnstimnames_to_train,
                       phnstim_all[train_inds,:],resp_all[train_inds,:],timesel_all[train_inds,:],
                       phnstim_all[test_inds, :], resp_all[test_inds, :], timesel_all[test_inds, :],
                       proj_list_b,
                       proj_names_b, blnCenter, swtScore,
                       blnplot=False)

def mean_and_cis(cv):
    m = cv.mean(-1,keepdims=True)
    s = cv.std(-1,keepdims=True)
    n = cv.shape[-1]
    cis = m + tdist.ppf(0.975,n-1)/np.sqrt(n)*s*np.array([[-1,1]])
    return np.squeeze(m),np.squeeze(cis)

r2s_all_cv = r2s_cv[:,:,0].T
r2s_latent_cv = r2s_cv[:,:,1].T

r2s_all,r2s_all_cis = mean_and_cis(r2s_all_cv)
r2s_latent,r2s_latent_cis = mean_and_cis(r2s_latent_cv)

#%%
from scipy.stats import ttest_rel

stats,pvals = ttest_rel(r2s_all_cv,r2s_latent_cv,axis=0)

for i, (name,stat,pval) in enumerate(zip(phnstimnames,stats,pvals)):
    print(f'{name}: t stat: {stat}, pval (bonf): {pval*ndim}')

#%%

plt.close('all')


width = 0.35  # the width of the bars

fig = plt.figure(figsize=[5,3])
plt.clf()
gs = GridSpec(1,1,wspace=0.4,bottom=0.4)
ax = fig.add_subplot(gs[0,0])
x = np.arange(len(feats_to_train))
rects1 = ax.bar(x - width/2, r2s_all, width, label='Prediction from all electrodes')
rects2 = ax.bar(x + width/2, r2s_latent, width, label='Prediction from latent projection')
plt.vlines(x - width/2,
           ymin=r2s_all_cis[:,0],
           ymax=r2s_all_cis[:,1],
           colors='k',linewidth=1)
plt.vlines(x + width/2,
           ymin=r2s_latent_cis[:,0],
           ymax=r2s_latent_cis[:,1],
           colors='k',linewidth=1)
ax.axhline(0,color='k',linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([name for name in phnstimnames[feats_to_train]],rotation=45,ha='right')
ax.set_ylabel('Testing R^2')
ax.set_title('Decoding time relative to event')
ax.legend()

if blnsave:
    fig.savefig(op.join(figurepath,f'decode_perf_w_cis_{swtScore}.svg'))

#%%

plt.close('all')


width = 0.35  # the width of the bars

fig = plt.figure(figsize=[5,3])
plt.clf()
gs = GridSpec(1,1,wspace=0.4,bottom=0.4)
ax = fig.add_subplot(gs[0,0])
x = np.arange(len(feats_to_train))
rects1 = ax.bar(x - width/2, r2s_all, width, label='Prediction from all electrodes')
rects2 = ax.bar(x + width/2, r2s_latent, width, label='Prediction from latent projection')
ax.plot(x - width/2,r2s_all_cv,'k.',markersize=3)
ax.plot(x + width/2,r2s_latent_cv,'k.',markersize=3)
ax.axhline(0,color='k',linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([name for name in phnstimnames[feats_to_train]],rotation=45,ha='right')
ax.set_ylabel('Testing R^2')
ax.set_title('Decoding time relative to event')
ax.legend()

if blnsave:
    fig.savefig(op.join(figurepath,f'decode_perf_w_cvpts_{swtScore}.svg'))

#%% Save

tosave = dict(feats_to_train=feats_to_train,
              blnCenter=blnCenter,
              swtScore=swtScore,
              phnstimnames_to_train=phnstimnames_to_train,
              jpcs_to_train=jpcs_to_train,
              latent_projs_to_train=latent_projs_to_train,
              r2s=r2s,
              r2s_cv=r2s_cv,
              ta=ta, phnstim=phnstim, resp=resp, timesel=timesel,
              ta_val=ta_val, phnstim_val=phnstim_val, resp_val=resp_val,
              timesel_val=timesel_val)

if blnsave:
    filepath = op.join(figurepath,f'decode_latency_{swtScore}.pkl')
    with open(filepath,'wb') as f:
        pickle.dump(tosave,f)

#%%

def plot_cond_dens(X, Y, xa, ya, ax): # X|Y
    dens_c = sm.nonparametric.KDEMultivariateConditional(endog=X,
        exog=Y, dep_type='c', indep_type='c', bw='normal_reference')
    xs,ys = np.meshgrid(xa, ya)
    cond_dens = dens_c.pdf(xs.reshape([np.prod(xs.shape),1]),ys.reshape([np.prod(xs.shape),1])).reshape(xs.shape)

    pc = ax.pcolormesh(xa, ya, cond_dens.T)
    ax.set_xlim(xa[0], xa[-1])
    ax.set_ylim(ya[0], ya[-1])

    return pc

#%%
plt.close('all')
feati=0
proji=1

tselcol = timesel[:,feati]
stimcol = phnstim[:,feati]
tselcol_val = timesel_val[:,feati]
stimcol_val = phnstim_val[:,feati]

X_train = resp[tselcol,:]
Y_train = stimcol[tselcol]
X_test = resp_val[tselcol_val,:]
Y_test = stimcol_val[tselcol_val]

featname = phnstimnames_to_train[feati]

latent_proj = latent_projs_to_train[proji]
jpca = jpcs_to_train[proji]
projname = phnstimnames_to_train[proji]

if blnCenter:
    X_train -= np.mean(X_train,0,keepdims=True)
    Y_train -=  np.mean(Y_train,0,keepdims=True)
    X_test -= np.mean(X_test,0,keepdims=True)
    Y_test -=  np.mean(Y_test,0,keepdims=True)

regr_all,y_hat_all,r2_all = decode_helper(X_train,Y_train,
                                          X_test,Y_test,
                                          swtScore)
regr_latent,y_hat_latent,r2_latent = decode_helper(X_train @ latent_proj,Y_train,
                                                   X_test @ latent_proj,Y_test,
                                                   swtScore)
regr_jpca,y_hat_jpca,r2_jpca = decode_helper(X_train @ jpca,Y_train,
                                             X_test @ jpca,Y_test,
                                             swtScore)

y_hat_train_all = regr_all.predict(X_train)
y_hat_train_latent = regr_latent.predict(X_train@latent_proj)
y_hat_train_jpca = regr_jpca.predict(X_train@jpca)


bins = np.unique(Y_test)
fig,axs = plt.subplots(3,2,clear=True,sharex=True,sharey='row')
fig2,axs2 = plt.subplots(2,2,clear=True,sharex=True,sharey=True)
h = [[]]*2
for i,(y_hat,y_hat_train,title) in enumerate(zip([y_hat_all,y_hat_latent],#,y_hat_jpca],
                                           [y_hat_train_all,y_hat_train_latent], #,y_hat_train_jpca],
                                           ['All Electrodes','Latent Projection'])):
    axs[0,i].hist2d(Y_train,y_hat_train,bins,density=True)#,vmin=0,vmax=60)
    axs[0,i].axline([0,0],slope=1,color='k')
    axs[0,i].set_title(title)
    h[i]=axs[1,i].hist2d(Y_test,y_hat,bins,density=True)#,vmin=0,vmax=60)
    axs[1,i].axline([0,0],slope=1,color='k')

    # testing MSE as a function of Y_test
    axs[2,i].scatter(Y_test,(Y_test-y_hat)**2,3)
    mean_mse = np.array([np.mean((b-y_hat[Y_test==b])**2) for b in bins])
    axs[2,i].plot(bins,mean_mse,color='k')
    axs[2,i].set_xlabel('True Latency')

    if i==0:
        axs[0,i].set_ylabel('Training Yhat')
        axs[1,i].set_ylabel('Testing Yhat')
        axs[2,i].set_ylabel('MSE')

    _ = plot_cond_dens(y_hat_train,Y_train,h[i][1],h[i][2],axs2[0,i])
    _ = plot_cond_dens(y_hat,Y_test,h[i][1],h[i][2],axs2[1,i]) # y_hat|Y_test
    axs2[0,i].set_title(title)
    if i==0:
        axs2[0,i].set_ylabel('Training Yhat')
        axs2[1,i].set_ylabel('Testing Yhat')

fig.suptitle(f'decode {featname} using projection onto {projname}')
fig2.suptitle(f'decode {featname} using projection onto {projname}')

print(f'decode {featname} using all electrodes: {r2_all}')
print(f'decode {featname} using projection onto {projname}, latent projection: {r2_latent}')
print(f'decode {featname} using projection onto {projname}, jpca projection: {r2_jpca}')
