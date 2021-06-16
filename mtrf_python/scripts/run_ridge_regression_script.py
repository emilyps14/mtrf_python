import json
import numpy as np
import os.path as op
import pickle

from mtrf_python import STRF_utils
from mtrf_python.scripts.script_utils import load_training_data, load_testing_data, _add_testing_metrics


def run_ridge_regression(subject, config_flag, alphas=[0], column_center=True,
                         nfolds=1, sample_inds=None, outpath=None):
    # n_folds: the number of folds to use for the inner CV
    #   if nfolds=1, the regression will be run without CV (on all data, subject to sample_inds)
    # sample_inds: the indices to subsample the full data (for the outer CV)

    for j,(loadpath, train_inds, validate_inds, tStim, tResp, vStim, vResp) in \
        enumerate(load_training_data(subject, config_flag,
                                     column_center=column_center,
                                     sample_inds=sample_inds,
                                     nfolds=nfolds)):

        if outpath is None:
            outpath = loadpath

        #% Run regression for full model
        ndelays = [t.shape[1] for t in tStim]
        feat_inds = np.cumsum([0]+ndelays)

        covmat, wt_array, electrode_R2s = STRF_utils.ridge_regression(
            np.concatenate(tStim,axis=1),tResp,
            np.concatenate(vStim,axis=1) if len(vStim)>0 else vStim,vResp,
            alphas)

        for i,alpha in enumerate(alphas):
            wts = wt_array[:,:,i]

            strfs = []
            for ndelay,starti,endi in zip(ndelays,feat_inds[:-1],feat_inds[1:]):
                wts_feat = wts[starti:endi,:]
                strfs.append(wts_feat)

            if nfolds>1:
                rc = electrode_R2s[i,:]
                pred = STRF_utils.compute_pred(np.concatenate(vStim,axis=1), wts, 0)
                overall_R2 = STRF_utils.compute_overall_rsquared(vResp,pred)
            else:
                # if nfolds=1, there is no testing data so the electrode_R2s is []
                rc = []
                overall_R2 = []

            out = dict(
                alpha=alpha,
                column_center=column_center,
                sample_inds=sample_inds,
                train_inds=train_inds,
                validate_inds=validate_inds,
                covmat=covmat,
                wts=wts,
                electrode_R2s=rc,
                overall_R2=overall_R2,
                strfs=strfs,
                feat_inds=feat_inds
                )

            filename = get_ridge_filename(subject, alpha, column_center=column_center)
            if nfolds>1:
                filepath = op.join(outpath,f'{filename}_cv{j+1}of{nfolds}.pkl')
            else:
                filepath = op.join(outpath,f'{filename}.pkl')
            print(f'Saving {filepath}...')
            with open(filepath,'wb') as f:
                pickle.dump(out,f)


def get_ridge_filename(subject, alpha, column_center=True):
    if column_center:
        return f'{subject}_RidgeResults_ctr_alpha{alpha:0.1e}'.replace('+','').replace('.','_')
    else:
        return f'{subject}_RidgeResults_alpha{alpha:0.1e}'.replace('+','').replace('.','_')

def get_best_ridge_model(subject, config_flag, alphas, nfolds,
                         column_center=True, testing_inds=None,
                         resultspath=None, return_first=False):
    # assumes that run_ridge_regression has been run both with cv (nfolds=nfolds) and without (nfolds=1)
    assert(nfolds>1)

    # Load testing data
    loadpath, rStim, rResp = load_testing_data(subject, config_flag,
                                               testing_inds,
                                               column_center=column_center)

    if resultspath is None:
        resultspath = loadpath

    # Load models, collect R2 (from cv models) and weights (from models with no cv)
    nalphas = len(alphas)
    electrode_R2s_w_cv = []
    ridge_models_no_cv = []
    wts_ridge_no_cv = []
    for i,alpha in enumerate(alphas):
        # Load cv models and pull out electrode_R2s
        r2_cv = []
        for j in range(nfolds):
            ridge_path = op.join(resultspath, f'{get_ridge_filename(subject, alpha, column_center=column_center)}_cv{j+1}of{nfolds}.pkl')
            with open(ridge_path,'rb') as f:
                ridge = pickle.load(f)
            assert(len(np.intersect1d(ridge['sample_inds'],testing_inds))==0) # paranoid about mixing training and testing data
            r2_cv.append(ridge['electrode_R2s'])
        electrode_R2s_w_cv.append(np.stack(r2_cv,axis=0)) # (nfolds,nchans)

        # Load no cv model and pull out weights
        ridge_path = op.join(resultspath, f'{get_ridge_filename(subject, alpha, column_center=column_center)}.pkl')
        with open(ridge_path,'rb') as f:
            ridge = pickle.load(f)
        assert(len(np.intersect1d(ridge['sample_inds'],testing_inds))==0) # paranoid about mixing training and testing data
        ridge_models_no_cv.append(ridge)
        wts_ridge_no_cv.append(ridge['wts'])
    electrode_R2s_w_cv = np.stack(electrode_R2s_w_cv,axis=0) # (nalphas,nfolds,nchans)
    wts_ridge_no_cv = np.stack(wts_ridge_no_cv,axis=0) # (nalphas,np.sum(ndelays),nchans)

    # for each electrode, compute the R^2 as the mean over cv folds for each alpha
    cv_r2 = (electrode_R2s_w_cv).mean(axis=1) # (nalphas,nchans)

    # choose the best alpha
    best_models_ridge = cv_r2.argmax(axis=0)

    # collect the weights for the best model using the model trained without CV (using the best alpha for each electrode)
    best_wts_ridge = np.zeros(wts_ridge_no_cv.shape[1:]) # (np.sum(ndelays),nchans)
    for i,modi in enumerate(best_models_ridge):
        best_wts_ridge[:,i] = wts_ridge_no_cv[modi,:,i]

    # create output dict
    ndelays = [t.shape[1] for t in rStim]
    feat_inds = ridge_models_no_cv[0]['feat_inds']
    strfs = []
    for ndelay,starti,endi in zip(ndelays,feat_inds[:-1],feat_inds[1:]):
        wts_feat = best_wts_ridge[starti:endi,:]
        strfs.append(wts_feat)

    best = dict(
        alpha='mixed',
        column_center=column_center,
        wts=best_wts_ridge,
        best_models = best_models_ridge,
        training_cv_electrode_R2s=electrode_R2s_w_cv,
        strfs=strfs,
        feat_inds=feat_inds
        )

    # compute the testing R^2 using the best weights on the testing dataset
    best['testing_inds'] = testing_inds
    _add_testing_metrics(best,rStim,rResp)

    if return_first:
        first = ridge_models_no_cv[0]
        first['testing_inds'] = testing_inds
        _add_testing_metrics(first,rStim,rResp)
        return best, first
    else:
        return best

def run_ridge_for_electrode_selection(subject, strf_config_flag, alphas=[0],
                                      column_center=True, nfolds=1, r2_threshold=0.05, datapath=None,
                                      savename='strf_r2_threshold.pkl',
                                      skip_fitting=False):
    if nfolds==1:
        assert(len(alphas)==1)

    if datapath is None:
        # Load paths
        with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
            config = json.load(f)
        subjects_dir = config['subjects_dir']
        datapath = op.join(subjects_dir, subject, strf_config_flag)

    # split into testing and training data 80/20
    with open(op.join(datapath, f'{subject}_RegressionSetup.pkl'),'rb') as f:
        data = pickle.load(f)
    electrodes_all = data['electrodes']
    sentence_offset_inds = STRF_utils.get_sentence_offset_inds(data['time_in_trial'])
    train_sample_inds,test_sample_inds = STRF_utils.split_training_sentences(sentence_offset_inds,split=(0.8,0.2))

    # Fit ridge on training data
    if not skip_fitting:
        run_ridge_regression(subject, strf_config_flag, alphas, column_center=column_center,
                             nfolds=nfolds,sample_inds=train_sample_inds)

    if nfolds==1:
        # Load the model that was run
        ridge_path = op.join(datapath, f'{get_ridge_filename(subject, alphas[0], column_center=column_center)}.pkl')
        with open(ridge_path,'rb') as f:
            model = pickle.load(f)
        assert(len(np.intersect1d(model['sample_inds'],test_sample_inds))==0)

        # Add R^2 for testing data
        _, rStim, rResp = load_testing_data(subject, strf_config_flag,
                                         test_sample_inds,
                                         column_center=column_center)
        model['testing_inds'] = test_sample_inds
        _add_testing_metrics(model,rStim,rResp)
    else:
        # Run the fits on the entire training data (to get weights for best alpha)
        if not skip_fitting:
            run_ridge_regression(subject, strf_config_flag, alphas, column_center=column_center,
                                 nfolds=1,sample_inds=train_sample_inds)

        # pick the best model and compute testing R^2
        model = get_best_ridge_model(subject, strf_config_flag, alphas, nfolds,
                                     column_center=column_center,
                                     testing_inds=test_sample_inds)

    speech_responsive_all = (model['testing_electrode_R2s']) > r2_threshold
    print(f'Speech Responsive: {electrodes_all[speech_responsive_all]}')
    print(f'Count: {sum(speech_responsive_all)} of {len(electrodes_all)}')

    # Save results
    out = dict(
        subject=subject,
        strf_config_flag=strf_config_flag,
        alphas=alphas,
        column_center=column_center,
        nfolds=nfolds,
        r2_threshold=r2_threshold,
        train_sample_inds=train_sample_inds,
        test_sample_inds=test_sample_inds,
        electrodes_all=electrodes_all,
        model=model,
        speech_responsive_all=speech_responsive_all
        )
    filepath = op.join(datapath,f'{savename}')
    print(f'Saving {filepath}...')
    with open(filepath,'wb') as f:
        pickle.dump(out,f)



# %%

def main():
    import sys
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument(
      "subject",  # required
      type=str,
    )
    parser.add_argument(
      "config_flag",  # required
      type=str,
    )
    parser.add_argument(
      "--column_center",  # optional
      dest='column_center',
      action='store_true',
      default=False,  # default if nothing is provided
    )
    parser.add_argument(
      "--alphas",  # optional
      nargs="*",  # 0 or more values expected => creates a list
      type=float,
      default=[0],  # default if nothing is provided
    )

    args = parser.parse_args()
    run_ridge_regression(args.subject, args.config_flag, args.alphas,
                         column_center=args.column_center)


if __name__ == '__main__':
    main()
