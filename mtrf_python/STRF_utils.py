import numpy as np
import matplotlib.animation as manimation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os.path as op
import pandas as pd
import re
from functools import reduce
from scipy.io import loadmat
from scipy import linalg

from es_utils.ecog_utils import gridOrient,electrode_grid_subplots,imshow_on_electrode_grid

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_electrode_info(imaging_dir,subject,max_el=256):
    e = loadmat(op.join(imaging_dir,subject,'elecs','TDT_elecs_all.mat'),squeeze_me=True)
    e_w = loadmat(op.join(imaging_dir,subject,'elecs','TDT_elecs_all_warped.mat'),squeeze_me=True)
    if max_el==256:
        assert(np.all(np.equal(e['eleclabels'][:max_el,2],'grid')))
        assert(np.all(np.equal(e_w['anatomy'][:max_el,2],'grid')))
        assert(np.all(np.equal(e['eleclabels'][:max_el,0],e_w['anatomy'][:256,0])))
    electrode_info = {}
    electrode_info['ch_name'] = e['eleclabels'][:max_el, 0]
    electrode_info['loc_x'] = e['elecmatrix'][:max_el, 0]
    electrode_info['loc_y'] = e['elecmatrix'][:max_el, 1]
    electrode_info['loc_z'] = e['elecmatrix'][:max_el, 2]
    electrode_info['rois'] = e['anatomy'][:max_el, 3]
    electrode_info['loc_warped_x'] = e_w['elecmatrix'][:max_el, 0]
    electrode_info['loc_warped_y'] = e_w['elecmatrix'][:max_el, 1]
    electrode_info['loc_warped_z'] = e_w['elecmatrix'][:max_el, 2]

    return electrode_info


def load_sentence_details(sentdet_dir, sentdet_name='out_sentence_details_timit_all_loudness.mat'):
    ### Features
    sentdet_file = op.join(sentdet_dir,sentdet_name)
    sentdet = loadmat(sentdet_file, squeeze_me=True,
                      variable_names=['sentdet'])['sentdet']
    sentdet_df = pd.DataFrame(sentdet)
    del sentdet
    featurenames = loadmat(sentdet_file, squeeze_me=False,
                      variable_names=['features'])['features'][0,0]['names'][:,0]
    featurenames = [x[0] for x in featurenames]
    phnnames = loadmat(sentdet_file,squeeze_me=True,variable_names=['phnnames'])['phnnames']

    # peakRate is maxDtL (the sixth row of loudnessall)
    peakRate = 'maxDtL'
    sentdet_df['peakRate'] = sentdet_df.apply(lambda x: x['loudnessall'][x['loudnessallNames']==peakRate,:],axis=1)

    # Correct dimensions on sentence_onset (should be 1xtime, but it was squeezed)
    sentdet_df['sentence_onset'] = sentdet_df['sentence_onset'].apply(np.atleast_2d)

    ### Sentence text
    sentence_text_file = op.join(sentdet_dir,'sentence_text.mat')
    sentence_text = loadmat(sentence_text_file,squeeze_me=True)
    senttext_df = pd.DataFrame(sentence_text['sentences'])

    # strip the first two numbers from the sentence text
    senttext_df['txt'] = senttext_df['txt'].apply(lambda x: ' '.join(re.split(' ',x)[2:]))

    sentdet_df = sentdet_df.merge(senttext_df,on='name',how='left',validate='one_to_one')

    return sentdet_df, featurenames, phnnames

#%% Matrix Wrangling and Regression

def get_training_mat(series,padleft,padright,blnMean=False,sel=[]):
    # returns concatenated sentences (time in axis 0)

    # data are either N x T or N x T x # repeats
    # _reduce should put the time axis into axis 0
    if blnMean:
        def _reduce(x):
            if np.ndim(x)>2:
                if len(sel)==0:
                    return x.mean(2).T
                else:
                    return x[:,:,sel].mean(2).T
            else:
                return x.T
    else:
        def _reduce(x):
            if len(sel)==0:
                return np.atleast_3d(x)[:,:,0].T
            else:
                return np.atleast_3d(x)[:,:,sel].transpose([1,0,2])

    mat = np.concatenate([_reduce(remove_pad(tr,padleft,padright)) for tr in series],axis=0)
    return mat

def remove_pad(mat,padleft,padright):
    # time in axis 1
    if padright<=1:
        return mat[:,padleft:,]
    else:
        return mat[:,padleft:-padright+1,]

def build_delay_matrices(stim,fs,delay_time=0.3,blnZscore=True):
    # First, choose the number of delays to use (remember this is in bins)
    delays = np.arange(np.floor(delay_time*fs)+1).astype(int)

    # Create the delay matrix
    zs = lambda x: (x-x.mean(0))/x.std(0)

    # z-score the stimulus (if not a binary matrix, otherwise comment out)
    if blnZscore:
        zstim = zs(stim)
    else:
        zstim = stim # don't z-score

    nt,ndim = zstim.shape # you could also replace all instances of "stim" here with "phnstim"
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d] = zstim[-d:] # The last d elements until the end
        elif d>0:
            dstim[d:] = zstim[:-d] # All but the last d elements
        else:
            dstim = zstim.copy()
        dstims.append(dstim)
    dstims = np.hstack(dstims)

    return delays,dstims

def split_training_inds(nt,split=(0.6,0.2,0.2)):
    assert(sum(split)==1)
    inds = np.concatenate([[0],[np.int(nt*x) for x in np.cumsum(split)]])
    return [np.arange(s,e) for s,e in zip(inds,inds[1:])]

def split_training_sentences(sentence_offset_inds, split=(0.6, 0.2, 0.2)):
    # sentence_offset_inds: (Ntrials array) the offset indices for every trial
    #  e.g. [2,6,8,12,15]
    #
    # Output: indices from 0 to sentence_offset_inds[-1], split into groups
    #         with # trials indicated by split
    #   e.g. [[0,1,2,3,4,5,6,7,8],[9,10,11,12],[13,14,15]]
    assert(sum(split)==1)
    Ntrials = len(sentence_offset_inds)
    trial_splits = split_training_inds(Ntrials,split=split)

    ind1 = 0
    splits = []
    for s in trial_splits:
        s_inds = []
        for trial_ind in s:
            ind2 = sentence_offset_inds[trial_ind]
            s_inds += list(range(ind1,ind2+1))
            ind1 = ind2+1
        splits.append(s_inds)

    return splits

def get_sentence_offset_inds(time_in_trial):
    flat_time = np.squeeze(time_in_trial)
    assert(len(flat_time.shape)==1)
    return list(np.nonzero(np.diff(flat_time)<0)[0])+[len(time_in_trial)-1]

def resample_sentences(groups, sample=None):
    # groups: an array labeling the group for each data point
    #   groups will be kept together in the output
    #   e.g. the sentence index, if there are no sentence repeats
    #   e.g. [6,6,6,2,2,2,2,3,3,4,4,4,4]
    # sample: the new sample, e.g. [3,4,6,3]
    #   if None, will be sampled randomly with replacement
    #
    # Output:
    #   order: the indices to execute the sample
    #         e.g. [7,8,9,10,11,12,0,1,2,7,8]
    #   new_groups: the new group labels (*non-unique*)
    #         e.g. [3,3,4,4,4,4,6,6,6,3,3]
    if sample is None:
        # Sample the same number of groups, with replacement
        sample = np.unique(groups)
        Ngroups = len(sample)
        sample = sample[np.random.randint(Ngroups,size=Ngroups)]

    groups_arr = np.array(groups)

    order = []
    new_groups = []
    for grp in sample:
        grp_inds = np.nonzero(groups_arr==grp)[0]
        order += [int(j) for j in grp_inds]
        new_groups += list(groups_arr[grp_inds])

    return order, new_groups


def ridge_regression(tStim,tResp,rStim=np.array([]),rResp=np.array([]),
                     alphas=[0.],dtype=np.single,corrmin=0.1):
    '''
    Run ridge regression on training data, compute R^2 on testing data

    :param tStim: np.ndarray (ntimes, nfeatures)
    :param tResp: np.ndarray (ntimes, nchannels)
    :param rStim: np.ndarray (ntimes_test, nfeatures) or empty
    :param rResp: np.ndarray (ntimes_test, nchannels) or empty
    :param alphas: list of float (nalphas)
    :param dtype:
    :param corrmin:

    :return:
    covmat:
    wts: np.ndarray (nfeatures, response dims, nalphas)
    electrode_R2s: (nalphas, nchannels)
    '''
    # Calculate covariance matrix for training data
    covmat = np.array(np.dot(tStim.astype(dtype).T, tStim.astype(dtype)))

    # Do eigenvalue decomposition on the covariance matrix
    [S,U] = np.linalg.eigh(covmat)
    Usr = np.dot(U.T, np.dot(tStim.T, tResp))
    wts = []
    electrode_R2s = []
    for a in alphas:
        print("Running alpha {:.3f}".format(a))
        D = np.diag(1/(S+a)).astype(dtype)

        wt = np.array(reduce(np.dot, [U, D, Usr]).astype(dtype))

        ## Predict test responses
        if len(rStim)>0:
            pred = compute_pred(rStim, wt)
            # calculate correlation between actual response in ridge set and predicted response
            R2 = compute_electrode_rsquared(rResp, pred)
            electrode_R2s.append(R2)
            print("Training: alpha={:.3f}, \
            mean R^2={:.3f}, \
            max R^2={:.3f}".format(a, np.mean(R2), np.max(R2)))

        wts.append(wt)

    # wts matrix is the matrix of STRFs for each alpha value
    wts = np.dstack(wts)
    # electrode_R2s is the R^2 values for each electrode and alpha on the ridge set
    if len(rStim)>0:
        electrode_R2s = np.stack(electrode_R2s,axis=0)

    return covmat, wts, electrode_R2s

def compute_pred(stim,wt_array,mu=0):
    # stim: ntimes x nfeatures
    # wt_array: nfeatures x (...)
    pred = np.tensordot(stim,wt_array,axes=(1,0))
    pred += mu
    return pred # ntimes x (...)

def compute_electrode_rsquared(resp, pred):
    # resp: ntimes x nchannels
    # pred: ntimes x nchannels
    SS_res = ((resp-pred)**2).sum(0)
    SS_tot = ((resp-resp.mean(axis=0,keepdims=True))**2).sum(0)
    R2 = 1-SS_res/SS_tot
    R2[np.isnan(R2)] = 0
    return R2 # nchannels

def compute_overall_rsquared(resp, pred):
    # resp: ntimes x nchannels
    # pred: ntimes x nchannels
    SS_res = ((resp-pred)**2).sum()
    SS_tot = ((resp-resp.mean())**2).sum()
    return 1-SS_res/SS_tot

def get_svd_of_mtrf(mtrf_approx,ncomponents=None,
                    reduced=False):
    U,S,Vh = linalg.svd(mtrf_approx)

    if ncomponents is None:
        ncomponents = U.shape[1]

    # flip components that have a negative peak
    for i in range(ncomponents):
        if U[np.argmax(np.abs(U[:,i])),i]<0:
            U[:,i] = -U[:,i]
            Vh[i,:] = -Vh[i,:]

    if reduced:
        return U[:,:ncomponents],S[:ncomponents],Vh[:ncomponents,:]
    else:
        return U,S,Vh

def compute_jPCA_on_MTRFs(phnstimnames, delays, ndims_byFeature, mtrfs,
                          var_thresh=0.98,jpca_ncomp=None):
    import jPCA

    jpcas = {}
    for i, (name,feat_delays,ncomponents,mtrf_approx) in \
            enumerate(zip(phnstimnames, delays, ndims_byFeature, mtrfs)):

        ncomponents = int(ncomponents)
        U,S,Vh = get_svd_of_mtrf(mtrf_approx,ncomponents)

        explained_ss = (S ** 2)
        total_ss = explained_ss.sum()

        if jpca_ncomp is None:
            if var_thresh is None:
                jpca_ncomp = ncomponents
            else:
                jpca_ncomp = np.nonzero(np.cumsum(explained_ss / total_ss) > var_thresh)[0].min() + 1
        mtrf_reduced = (U[:,:jpca_ncomp]*S[:jpca_ncomp])@Vh[:jpca_ncomp,:]

        if jpca_ncomp>1:
            # Fit a jPCA object to data
            # pca=False because the MTRF is already low-rank
            # subtract_cc_mean=False because there are no conditions
            jpca = jPCA.JPCA(num_jpcs=2)
            (projected,_,_,_) = jpca.fit([mtrf_reduced],times=list(feat_delays),
                                           tstart=feat_delays[0],
                                           tend=feat_delays[-1],
                                           pca=False,subtract_cc_mean=False,
                                           soft_normalize=-1,
                                           align_axes_to_data=False)

            jpca_ss = (projected[0] ** 2).sum(axis=0)
            pct_var = jpca_ss / total_ss * 100

            # figure out the transform between latent space and the first two jPCs
            transform = S[:ncomponents,None]*Vh[:ncomponents,:]@jpca.jpcs

            # # vector perpendicular to projection plane in 3d DOESN'T WORK
            # view_vect = np.cross(transform[:3,0],transform[:3,1])
            # az,el,r = cart2sph(view_vect[0],view_vect[1],view_vect[2])

            jpcas[name] = {'jpca':jpca,
                           'jpca_ncomp':jpca_ncomp,
                           'mtrf_reduced':mtrf_reduced,
                           'projected':projected,
                           'transform':transform,
                           'view':None,
                           'pct_var':pct_var}

    return jpcas

# Note that the weights depend on the data, and they are computed in the irrr fit for that cv fold
from scipy.linalg import svdvals
def compute_irrr_penalty(strfs, penalty_weights=None,blnByFeature=False):
    if penalty_weights is None:
        penalty_weights = [1]*len(strfs)
    feature_penalties = [np.sum(svdvals(strf))*weight
                   for strf,weight in zip(strfs, penalty_weights)]
    if blnByFeature:
        return feature_penalties
    else:
        return np.sum(feature_penalties)

#%% Plotting

def plot_stim_resp(stim, phnstim, resp, phnstimnames=None, ntimes=1000, fignum=1):
    fig = plt.figure(fignum,figsize=(12,8)) # make a figure of size 12 x 8
    fig.clf()
    ax = plt.subplot(3,1,1)
    plot_aud(stim[:ntimes, :],ax=ax)
    plt.ylabel('Frequency bin')
    plt.title('Spectrogram')

    plt.subplot(3,1,2,sharex=ax)
    plt.imshow(phnstim[:ntimes, :].T, cmap = plt.cm.Greys, aspect='auto')
    plt.title('Phoneme matrix')
    if phnstimnames is not None:
        plt.yticks(range(len(phnstimnames)),phnstimnames)
        plt.ylabel('Phoneme')
    else:
        plt.ylabel('Phoneme #')


    plt.subplot(3,1,3,sharex=ax)
    if resp.shape[1]>4: # For a large number of channels, show an image
        plt.imshow(resp[:ntimes, :].T, vmin=-4, vmax=4, cmap = plt.cm.RdBu_r, aspect='auto')
        plt.ylabel('Electrode')
    else: # For a small number of channels, plot the time series
        plt.plot(resp[:ntimes, :])
        plt.ylabel('Z')
    plt.xlabel('Time bin')
    plt.title('High gamma response')

    fig.subplots_adjust(hspace=.5) # Put some space between the plots for ease of viewing

    return fig

def plot_regression_schematic(stim,resp,chan_to_plot,ntimes=500,fignum=2):
    fig = plt.figure(fignum,figsize=(17,3)) # make a figure of size 12 x 8
    fig.clf()
    ax1 = fig.add_subplot(111)
    ax1.imshow(stim[:ntimes,:].T, cmap = plt.cm.Reds,
               origin='lower')
    plt.xlabel('Time bin')
    ax1.set_ylabel('Freq. band')
    ax1.set_xlim(0,ntimes)
    ax1.set_ylim(0,stim.shape[1])

    # Plot the response overlayed (with a separate y axis scaled appropriately)
    ax2 = ax1.twinx()
    ax2.plot(resp[:ntimes,chan_to_plot], 'b')
    ax2.set_ylabel('Z scored high gamma', color='b')
    for tl in ax2.get_yticklabels():
        tl.set_color('b')
    ax2.set_xlim(0,ntimes)

    return fig

def plot_delayed_stim(dstims, nfeatures=None, fignum=4):
    fig = plt.figure(fignum,figsize=(12,8))
    fig.clf()
    plt.imshow(dstims[:1000, :nfeatures].T, cmap = plt.cm.Greys, aspect='auto')
    plt.gca().xaxis.grid(b=True, which='major', color='r', linestyle='-', linewidth=2)
    plt.xlabel('Time bin')
    plt.ylabel('Delayed features')

    return fig

def plot_Rcorrs(Rcorrs,alphas,fignum=6):
    # Plot correlations vs. alpha regularization value
    fig=plt.figure(fignum,figsize=(7,5))
    fig.clf()
    plt.subplot(1,2,1)
    plt.plot(alphas,Rcorrs,'k')
    plt.gca().set_xscale('log')

    # Plot the best average alpha
    plt.axvline(alphas[Rcorrs.mean(1).argmax()])
    plt.plot(alphas,np.array(Rcorrs).mean(1),'r',linewidth=5)
    plt.xlabel('Regularization parameter, alpha')
    plt.ylabel('Correlation for ridge set')

    plt.subplot(1,2,2)
    plt.plot(alphas,np.array(Rcorrs).mean(1),'r',linewidth=5)
    plt.xlabel('Regularization parameter, alpha')
    plt.gca().set_xscale('log')
    return fig

def plot_pred_vs_actual(pred, resp, chans, ntimes=500, ta=None, fignum=7, figsize=(15,6)):
    if ta is None:
        ta = np.arange(pred.shape[0])
    nchans = len(chans)
    fig = plt.figure(fignum,figsize=figsize)
    fig.clf()
    for i,ch in enumerate(chans):
        plt.subplot(nchans,1,i+1)
        plt.plot(ta[:ntimes],pred[:ntimes, ch, ], 'k')
        plt.plot(ta[:ntimes],resp[:ntimes, ch], 'r')
        plt.title(f'Channel {ch+1}')
    fig.subplots_adjust(hspace=.5)

    return fig

def plot_strf(wts,delays,blnLabels=True,featurelabels=None):
    ax = plt.gca()
    strf = wts.reshape(len(delays),-1)
    smax = np.max(np.abs(strf))
    if not np.all(np.equal(strf,0)):
        ax.imshow(strf.T,vmin=-smax, vmax=smax, cmap = plt.cm.RdBu_r, aspect='auto', origin='upper')
    if blnLabels:
        ax.set_xlabel('Delay')
        ax.set_ylabel('Feature')
    if featurelabels is not None:
        ax.set_yticks(range(len(featurelabels)))
        ax.set_yticklabels(featurelabels)
        ax.set_ylim([strf.shape[-1]-0.5,-0.5])

        ax.set_xticks([0,len(delays)])
        ax.set_xticklabels(delays[[0,-1]])
    else:
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])

def plot_aud(aud,ta=None,fa=None,onset_times=[],peakRate_times=[],ax=None,
             pcolor_kwargs={'cmap':plt.cm.Greys}):
    if ax is None:
        ax = plt.gca()
    [T,F] = aud.shape
    if ta is None:
        ta = np.linspace(-0.5,T+0.5,T+1,endpoint=True)
    if fa is None:
        fa = np.linspace(-0.5,F+0.5,F+1,endpoint=True)
    stimim = ax.pcolormesh(ta, fa, aud.T, **pcolor_kwargs)

    for t in onset_times:
        ax.axvline(t,color='k')
    for t in peakRate_times:
        ax.axvline(t,color='k',linestyle='--')

    return stimim

def plot_strfs(wts,delays,corr,gridOrient=None,blnSTG=None,fignum=8):
    # wts: # features x # channels
    # corr: # channels
    def plot_fun(c):
        plot_strf(wts[:,c],delays,blnLabels=c==0)
        ax = plt.gca()
        ax.text(0.5, 0.5,'r=%2.2f'%(corr[c]),
                horizontalalignment='center',verticalalignment='center',
                transform=ax.transAxes)
        if blnSTG is not None:
            if not blnSTG[c]:
                ax.axis('off')

    nchans = wts.shape[1]
    if gridOrient is None:
        if nchans>4:
            fsize=(15,15)
            nrow = np.ceil(np.sqrt(nchans))
            ncol = np.ceil(nchans/nrow)
        else:
            fsize=(6,3)
            nrow = 1
            ncol = nchans

        fig = plt.figure(fignum,figsize=fsize)
        fig.clf()
        # smax = np.percentile(np.abs(wt_array),99)
        # Use separate regularization parameters for each STRF
        for c in np.arange(nchans):
            ax = fig.add_subplot(nrow,ncol,c+1)
            plot_fun(c)
    else:
        fig = electrode_grid_subplots(range(nchans),gridOrient,plot_fun,fignum=fignum,
                                blnTitleNumber=False,blnAxesOff=False)
    return fig

def plot_strfs_by_alpha(wt_array,delays,alphas,corr,best_alpha_overall=None,
                        best_alphas_indiv=None,fignum=10):
    # wt_array: # features x # channels x # alphas
    # corr: # channels x # alphas
    nchans = wt_array.shape[1]

    if best_alphas_indiv is None:
        best_alphas_indiv = [None]*nchans

    fig = plt.figure(fignum,figsize=(20,10))
    fig.clf()
    axes = [fig.add_subplot(nchans,len(alphas),ii+1) for ii in range(len(alphas)*nchans)]
    p = 0
    for c in np.arange(nchans): # loop through the best channels
        for a in np.arange(len(alphas)): # loop through the alpha regularization parameter
            strf = wt_array[:,c,a].reshape(len(delays),-1)
            smax = np.abs(strf).max()
            axes[p].imshow(strf.T,vmin=-smax, vmax=smax, cmap = plt.cm.RdBu_r,
                           aspect='auto')
            axes[p].xaxis.set_ticks([])
            axes[p].yaxis.set_ticks([])
            if a==best_alphas_indiv[c]:
                axes[p].set_title('a=%1.1g'%(alphas[a]),fontsize=10,
                                  fontstyle='italic',color='r')
            elif a==best_alpha_overall:
                axes[p].set_title('a=%1.1g'%(alphas[a]),fontsize=10,
                                  fontstyle='italic',color='b')
            else:
                axes[p].set_title('a=%1.1g'%(alphas[a]),fontsize=10)
            axes[p].set_xlabel('r=%2.3f'%(corr[c,a]))
            p+=1

    fig.subplots_adjust(hspace=.5)

    return fig

def plot_imshow_on_electrode_grid(toplot,gridOrient,blnSTG=None,**kwargs):
    nrows,ncols,N = toplot.shape

    fig,cb = imshow_on_electrode_grid(toplot,gridOrient,**kwargs)
    if blnSTG is not None:
        for ax in fig.get_axes()[:-1]:
            for k in range(gridOrient.shape[0]):
                for l in range(gridOrient.shape[1]):
                    if blnSTG[gridOrient[k,l]]:
                        ax.text(l,k,int(gridOrient[k,l])+1,
                                horizontalalignment='center',
                                verticalalignment='center',fontsize=6)
            add_mask_border(ax,blnSTG[gridOrient].T)
    return fig,cb

def plot_residuals_movie(grid_signals, stim, ta, gridOrient, title=None,
                         grid_names=None,onset_times=[], peakRate_times=[], blnSTG=None,
                         cmap='RdBu_r', cl=None,
                         fignum=1):
    # grid_signals: list of trials x time x channels
    # stim: time x frequency
    ngrids = len(grid_signals)
    Ntr_plot,T,N = grid_signals[0].shape

    fig = plt.figure(fignum)
    fig.clf()
    fig.set_size_inches([14.4,6])
    gs = gridspec.GridSpec(2+ngrids,Ntr_plot, height_ratios=[1.5]+[1]*ngrids+[0.1], hspace=0.5)
    if cl is None:
        cl = np.array([-1,1])*np.percentile(np.abs(np.r_[grid_signals]), [99])

    stimax = fig.add_subplot(gs[0,:])
    stimim = plot_aud(stim,ta,onset_times=onset_times,peakRate_times=peakRate_times,ax=stimax)
    tline = stimax.axvline(ta[0],color='r')
    stimax.set_yticks([])

    axs = np.zeros((ngrids,Ntr_plot),dtype=np.object)
    ims = np.zeros((ngrids,Ntr_plot),dtype=np.object)
    for i,grid_signal in enumerate(grid_signals):
        for j in range(Ntr_plot):
            ax = fig.add_subplot(gs[i+1,j])
            im = ax.imshow(grid_signal[j, [0], gridOrient], interpolation='nearest', vmin=cl[0], vmax=cl[1], cmap=cmap, animated=True)
            ax.set_yticks([])
            ax.set_xticks([])
            if blnSTG is not None:
                add_mask_border(ax,blnSTG[gridOrient].T)
            if j==0 and grid_names is not None:
                ylab1 = ax.set_ylabel(grid_names[i])
            axs[i,j] = ax
            ims[i,j] = im

            if i==ngrids-1:
                subti = ax.set_xlabel(f'Trial #{j}')

    cax = fig.add_subplot(gs[ngrids+1,:])
    cb = fig.colorbar(im,cax=cax,orientation='horizontal')
    cax.set_xlabel('High Gamma')
    if title is not None:
        ti = fig.suptitle(title)
    def animate(k):
        for i,grid_signal in enumerate(grid_signals):
            for j in range(Ntr_plot):
                ims[i,j].set_data(grid_signal[j, [k], gridOrient])
            tline.set_xdata(ta[k])
        return [ims,ti]

    ani = manimation.FuncAnimation(fig,animate,frames=T,interval=50,repeat=True,blit=False)

    plt.show()

    writer = manimation.writers['ffmpeg'](fps=10, metadata=dict(artist='Me'), bitrate=1800)

    return ani, writer


def add_mask_border(ax,mask,xaxis=None,yaxis=None,color='k'):
    from matplotlib.lines import Line2D
    n,m = mask.shape
    if xaxis is None:
        xpoints = np.linspace(*ax.get_xaxis().get_data_interval(),n+1)
    else:
        dx = xaxis[1]-xaxis[0]
        xpoints = np.concatenate([xaxis,[xaxis[-1]+dx]])-dx/2
    if yaxis is None:
        ypoints = np.linspace(*ax.get_yaxis().get_data_interval(),m+1)
    else:
        dy = yaxis[1]-yaxis[0]
        ypoints = np.concatenate([yaxis,[yaxis[-1]+dy]])-dy/2



    for i,(x1,x2) in enumerate(zip(xpoints[:-1],xpoints[1:])):
        for j,(y1,y2) in enumerate(zip(ypoints[:-1],ypoints[1:])):
            if i<n-1:
                if mask[i,j]!=mask[i+1,j]:
                    ax.add_line(Line2D([x2, x2],[y1,y2],color=color))
            if j<m-1:
                if mask[i,j]!=mask[i,j+1]:
                    ax.add_line(Line2D([x1,x2],[y2,y2],color=color))

#%% https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

# Topics: line, color, LineCollection, cmap, colorline, codex
'''
Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
The color is taken from optional data in z, and creates a LineCollection.

z can be:
- empty, in which case a default coloring will be used based on the position along the input arrays
- a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
- an array of the length of at least the same length as x, to color according to this data
- an array of a smaller length, in which case the colors are repeated along the curve

The function colorline returns the LineCollection created, which can be modified afterwards.

See also: plt.streamplot
'''

from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


# Data manipulation:

def make_segments(x, y, z=None):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    if z is None:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
    else:
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, c=None, cmap=plt.get_cmap('copper'),
              norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0, ax=None):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if c is None:
        c = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(c, "__iter__"):  # to check for numerical input -- this is a hack
        c = np.array([c])

    c = np.asarray(c)

    segments = make_segments(x, y, z)
    if z is None:
        lc = LineCollection(segments, array=c, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    else:
        lc = Line3DCollection(segments, array=c, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc

def colorline3d(x, y, z, c=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x, y, z
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if c is None:
        c = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(c, "__iter__"):  # to check for numerical input -- this is a hack
        c = np.array([c])

    c = np.asarray(c)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=c, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

#%% Modified from Stack Overflow:
# https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)

    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    if num1!=num2:
        plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)
