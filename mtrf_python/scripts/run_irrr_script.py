import json
import numpy as np
import os.path as op
import pickle
from iRRR.iRRR_normal import irrr_normal
from scipy import linalg
from shutil import copyfile

from mtrf_python import STRF_utils
from mtrf_python.scripts.script_utils import load_training_data, load_testing_data, _add_testing_metrics


def run_iRRR(subject, config_flag, lam=0, warm_start_path=None, weightflag='theo',
             params=None, leave_one_out=False,
             nfolds=1, sample_inds=None, outpath=None):
    # n_folds: the number of folds to use for the inner CV
    #   if nfolds=1, the regression will be run without CV (on all data, subject to sample_inds)
    # sample_inds: the indices to subsample the full data (for the outer CV)

    for j,(loadpath, train_inds, validate_inds, tStim, tResp, vStim, vResp) in \
        enumerate(load_training_data(subject, config_flag,
                                     column_center=False,
                                     sample_inds=sample_inds,
                                     nfolds=nfolds)):
        # Note: column centering happens in irrr_normal
        print(f'CV fold {j+1}')

        #### Set up parameters
        if j==0:
            if outpath is None:
                outpath = loadpath

            nchans = tResp.shape[1]
            ndim = len(tStim)
            p = [t.shape[1] for t in tStim]
            cumsum_p = np.concatenate([[0],np.cumsum(p)])

            if params is None:
                params = {'varyrho':True,
                          'Niter':100,
                          'Tol':0.01,
                          'fig':False}

            # Pick weights
            if weightflag == 'ones':
                weight = np.ones((ndim,1))
                params['weight'] = weight
            elif weightflag == 'theo':
                weight = [np.max(linalg.svdvals(x))*(np.sqrt(nchans)+np.sqrt(np.linalg.matrix_rank(x)))/x.shape[0]
                          for x in tStim]
                params['weight'] = weight

        # Load warm start result
        if warm_start_path is not None:
            if nfolds>1:
                warm_start_cv = warm_start_path.replace('.pkl',f'_cv{j+1}of{nfolds}.pkl')
                with open(warm_start_cv,'rb') as f:
                    warm_start = pickle.load(f)
                print(f' Using warm start: {warm_start_cv}')
            else:
                with open(warm_start_path,'rb') as f:
                    warm_start = pickle.load(f)
                print(f' Using warm start: {warm_start_path}')
            wts_ws = warm_start['wts']
            B = [wts_ws[cp:nextcp,:] for cp,nextcp in zip(cumsum_p[:-1],cumsum_p[1:])]
            params['randomstart'] = B

        #### Fit iRRR
        if leave_one_out:
            outs = []
            for i in range(ndim):
                print(f' Leaving out {i}')
                sel = list(range(i))+list(range(i+1,ndim))
                params_loo = params.copy()
                params_loo['weight'] = [params['weight'][j] for j in sel]
                if warm_start_path is not None:
                    params_loo['randomstart'] = [params['randomstart'][j] for j in sel]
                out = _run_one([tStim[j] for j in sel],
                               tResp,
                               [vStim[j] for j in sel] if len(vStim)>0 else [],
                               vResp,
                               lam,
                               params_loo)
                out['sample_inds'] = sample_inds
                out['train_inds'] = train_inds
                out['validate_inds'] = validate_inds
                outs.append(out)

            filename = get_iRRR_filename(subject, lam, params.get('irrr_flag',''))
            if nfolds>1:
                # Unusual to do LOO with inner CV (typically use lambda chosen from full model)
                filepath = op.join(outpath,f'{filename}_leaveOneOut_cv{j+1}of{nfolds}.pkl')
            else:
                filepath = op.join(outpath,f'{filename}_leaveOneOut.pkl')

            print(f' Saving {filepath}...')
            with open(filepath,'wb') as f:
                pickle.dump(outs,f)
        else:
            out = _run_one(tStim,tResp,vStim,vResp,lam,params)
            out['sample_inds'] = sample_inds
            out['train_inds'] = train_inds
            out['validate_inds'] = validate_inds

            filename = get_iRRR_filename(subject, lam, params.get('irrr_flag',''))
            if nfolds>1:
                filepath = op.join(outpath,f'{filename}_cv{j+1}of{nfolds}.pkl')
            else:
                filepath = op.join(outpath,f'{filename}.pkl')
            print(f' Saving {filepath}...')
            with open(filepath,'wb') as f:
                pickle.dump(out,f)

def _run_one(tStim, tResp, vStim, vResp, lam, params):
    #% Run regression
    [wts, mu, A, B, Theta, det] = irrr_normal(tResp, tStim, lam,
                                              params,
                                              return_details=True)
    strfs = A

    if len(vStim)>0: # if there is any validation data (CV)
        # Individual electrode R^2
        pred = STRF_utils.compute_pred(np.concatenate(vStim, axis=1), wts, mu.T)
        electrode_R2s = STRF_utils.compute_electrode_rsquared(vResp, pred)

        # Overall R^2
        overall_R2 = STRF_utils.compute_overall_rsquared(vResp, pred)
    else:
        electrode_R2s = []
        overall_R2 = []

    #%
    out = dict(
        lam=lam,
        covmat=None,
        wts=wts,
        electrode_R2s=electrode_R2s,
        overall_R2=overall_R2,
        strfs=strfs,
        mu=mu,
        A=A,
        B=B,
        Theta=Theta,
        details=det,
        params=params
        )

    return out

def get_iRRR_filename(subject, lam, flag=''):
    return f'{subject}_iRRR_Results_lambda{lam:0.1e}{flag}'.replace('+','').replace('.','_')

def run_iRRR_from_config(subject, config_flag, iRRR_config_path, lam_i, outpath=None, **kwargs):
    # Load paths
    if outpath is None:
        with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
            config = json.load(f)
        subjects_dir = config['subjects_dir']
        loadpath = op.join(subjects_dir, subject, config_flag)
    else:
        loadpath = outpath

    # Load iRRR settings
    with open(iRRR_config_path, "r") as f:
        irrr_config = json.load(f)
    weightflag = irrr_config['weightflag']
    lambdas = irrr_config['lambdas']
    warm_start_fname = irrr_config['warm_start_fname']
    params = irrr_config['params']

    irrr_flag = op.splitext(op.basename(iRRR_config_path))[0]
    params['irrr_flag'] = f'_{irrr_flag}'

    run_iRRR(subject, config_flag, lambdas[lam_i], warm_start_path=op.join(loadpath,f'{subject}_{warm_start_fname}'),
             weightflag=weightflag, params=params, outpath=outpath, **kwargs)

def pick_best_lambda(subject, config_flag, iRRR_config_path, nfolds,
                     resultspath=None):
    # Loads results of cross-validation and picks best lambda
    assert(nfolds>1)

    if resultspath is None:
        with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
            config = json.load(f)
        subjects_dir = config['subjects_dir']
        resultspath = op.join(subjects_dir, subject, config_flag)

    # iRRR config
    with open(iRRR_config_path, "r") as f:
        irrr_config = json.load(f)

    lambdas = np.array(irrr_config['lambdas'])
    nlambdas = len(lambdas)
    irrr_flag = op.splitext(op.basename(iRRR_config_path))[0]

    # load CV iRRR results and pick best lambda
    overall_R2_byLambda_cv = np.zeros((nfolds,nlambdas))
    for i,lam in enumerate(lambdas):
        filename = get_iRRR_filename(subject, lam, '_' + irrr_flag)
        for j in range(nfolds):
            irrr_path = op.join(resultspath, f'{filename}_cv{j+1}of{nfolds}.pkl')

            with open(irrr_path,'rb') as f:
                irrr = pickle.load(f)

            overall_R2_byLambda_cv[j,i] = irrr['overall_R2']
    lambda_i =  np.argmax(overall_R2_byLambda_cv.mean(axis=0))

    return lambda_i, lambdas[lambda_i], overall_R2_byLambda_cv

def get_testing_r2(subject, config_flag, iRRR_config_path, lam_i, leave_one_out=False,
                   testing_inds=None, resultspath=None):
    if resultspath is None:
        with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
            config = json.load(f)
        subjects_dir = config['subjects_dir']
        resultspath = op.join(subjects_dir, subject, config_flag)

    # iRRR config
    with open(iRRR_config_path, "r") as f:
        irrr_config = json.load(f)

    lambdas = np.array(irrr_config['lambdas'])
    irrr_flag = op.splitext(op.basename(iRRR_config_path))[0]

    # load testing data
    _, rStim, rResp = load_testing_data(subject, config_flag,
                                               testing_inds,
                                               column_center=False)

    filename = get_iRRR_filename(subject, lambdas[lam_i], '_' + irrr_flag)
    if leave_one_out:
        irrr_path = op.join(resultspath,f'{filename}_leaveOneOut.pkl')
        with open(irrr_path,'rb') as f:
            irrrs = pickle.load(f)
        ndim = len(irrrs)
        for i,irrr in enumerate(irrrs):
            sel = list(range(i))+list(range(i+1,ndim))
            irrr['testing_inds'] = testing_inds
            _add_testing_metrics(irrr,[rStim[j] for j in sel],rResp)
        return irrrs
    else:
        irrr_path = op.join(resultspath, f'{filename}.pkl')
        with open(irrr_path,'rb') as f:
            irrr = pickle.load(f)
        irrr['testing_inds'] = testing_inds
        _add_testing_metrics(irrr,rStim,rResp)
        return irrr

# %%

def main():
    import sys

    subject = sys.argv[1]
    config_flag = sys.argv[2]
    iRRR_config_path = sys.argv[3]
    lam_i = int(sys.argv[4])

    run_iRRR_from_config(subject,config_flag,iRRR_config_path,lam_i)


if __name__ == '__main__':
    main()
