import json
import os.path as op
import pickle
from os import makedirs

import numpy as np

from mtrf_python.scripts.run_irrr_script import run_iRRR_from_config, \
    pick_best_lambda, get_testing_r2
from mtrf_python.scripts.run_ridge_regression_script import \
    run_ridge_regression, get_best_ridge_model


def run_cv_job(subject, config_flag, iRRR_config_path, loo_config_flags,
               outpath, col_ctr_RR = True, alphas = None,
               nfolds_inner = 5,
               train_sample_inds=None, test_sample_inds=None):

    #### Fit OLS + Ridge
    print('Running OLS and Ridge...')
    if alphas is None:
        alphas = np.concatenate([[0],np.logspace(2,7,20)])
    # Run with cross-validation
    run_ridge_regression(subject, config_flag, alphas, column_center=col_ctr_RR,
                         nfolds=nfolds_inner,
                         sample_inds=train_sample_inds, outpath=outpath)
    # Run with entire training data (to get the weights used in the best model)
    run_ridge_regression(subject, config_flag, alphas, column_center=col_ctr_RR,
                         nfolds=1,
                         sample_inds=train_sample_inds, outpath=outpath)
    # Choose alpha based on cv, get best model from the run with entire training data, and compute test R^2
    best_ridge_model, ols_model = get_best_ridge_model(subject, config_flag, alphas, nfolds_inner,
                         column_center=col_ctr_RR, testing_inds=test_sample_inds,
                         resultspath=outpath, return_first=True)

    #### Fit iRRR
    print('Running iRRR...')
    # Run for all lambda with cross-validation
    for lambda_i in range(12):
        run_iRRR_from_config(subject, config_flag, iRRR_config_path,
                             lambda_i, leave_one_out=False, nfolds=nfolds_inner,
                             sample_inds=train_sample_inds, outpath=outpath)
    # Choose lambda based on cross validation and compute testing R^2
    best_lambda_i, _, overall_R2_byLambda_cv = pick_best_lambda(
        subject, config_flag, iRRR_config_path,
        nfolds_inner, resultspath=outpath)
    # Fit model on entire training data using chosen lambda (best model)
    run_iRRR_from_config(subject, config_flag, iRRR_config_path,
                         best_lambda_i, leave_one_out=False, nfolds=1,
                         sample_inds=train_sample_inds, outpath=outpath)
    # Compute testing R^2
    best_irrr = get_testing_r2(subject, config_flag, iRRR_config_path,
                               best_lambda_i, leave_one_out=False,
                               testing_inds=test_sample_inds,
                               resultspath=outpath)

    # Fit LOO iRRRs for best lambda and compute R^2 by comparison to test set
    print('Running LOO models...')
    run_iRRR_from_config(subject, config_flag, iRRR_config_path,
                         best_lambda_i, leave_one_out=True, nfolds=1,
                         sample_inds=train_sample_inds, outpath=outpath)
    loo_irrrs = get_testing_r2(subject, config_flag, iRRR_config_path,
                               best_lambda_i, leave_one_out=True,
                               testing_inds=test_sample_inds,
                               resultspath=outpath)
    group_loo_irrrs = {}
    for loo_config_flag in loo_config_flags:
        loo_outpath = op.join(outpath,loo_config_flag)
        makedirs(loo_outpath)
        # (run OLS as warm start for iRRR)
        run_ridge_regression(subject, loo_config_flag, [0], nfolds=1,
                             column_center=col_ctr_RR, sample_inds=train_sample_inds,
                             outpath=loo_outpath)
        run_iRRR_from_config(subject, loo_config_flag, iRRR_config_path,
                             best_lambda_i, leave_one_out=False, nfolds=1,
                             sample_inds=train_sample_inds, outpath=loo_outpath)
        group_loo_irrrs[loo_config_flag] = \
                get_testing_r2(subject, loo_config_flag, iRRR_config_path,
                               best_lambda_i, leave_one_out=False,
                               testing_inds=test_sample_inds,
                               resultspath=loo_outpath)

    #### Save some summary stats
    # training and test indices
    # OLS, Best Ridge, and Best iRRR models with test R^2
    # LOO models with test R^2
    out = dict(subject=subject,
               config_flag=config_flag,
               iRRR_config_path=iRRR_config_path,
               loo_config_flags=loo_config_flags,
               col_ctr_RR=col_ctr_RR,
               alphas=alphas,
               nfolds_inner=nfolds_inner,
               train_inds=train_sample_inds,
               test_inds=test_sample_inds,
               best_ridge_model=best_ridge_model,
               ols_model=ols_model,
               best_lambda_i=best_lambda_i,
               overall_R2_byLambda_cv=overall_R2_byLambda_cv,
               best_irrr=best_irrr,
               loo_irrrs=loo_irrrs,
               group_loo_irrrs=group_loo_irrrs
               )
    filename = 'cv_summary'
    filepath = op.join(outpath,f'{filename}.pkl')
    print(f'Saving {filepath}...')
    with open(filepath,'wb') as f:
        pickle.dump(out,f)

def run_job_from_config(outpath):
    with open(op.join(outpath,'cv_setup.json'), "r") as f:
        cv_config = json.load(f)

    subject = cv_config['subject']
    config_flag = cv_config['config_flag']
    iRRR_config_path = cv_config['iRRR_config_path']
    loo_config_flags = cv_config['loo_config_flags']
    col_ctr_RR = cv_config.get('col_ctr_RR', True)
    alphas = cv_config.get('alphas', None)
    nfolds_inner = cv_config['nfolds_inner']
    train_sample_inds = cv_config.get('train_sample_inds', None)
    test_sample_inds = cv_config.get('test_sample_inds', None)

    run_cv_job(subject, config_flag, iRRR_config_path, loo_config_flags,
               outpath=outpath, col_ctr_RR = col_ctr_RR, alphas = alphas,
               nfolds_inner=nfolds_inner, train_sample_inds=train_sample_inds,
               test_sample_inds=test_sample_inds)

def main():
    import sys
    outpath = sys.argv[1]
    run_job_from_config(outpath)

if __name__ == '__main__':
    main()
