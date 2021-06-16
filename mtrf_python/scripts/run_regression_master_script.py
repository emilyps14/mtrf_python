import json
import os.path as op
import pickle
import re
from os import makedirs
from subprocess import run, STDOUT

from sklearn.model_selection import GroupKFold

from mtrf_python.scripts.run_cv_job_script import run_job_from_config
from mtrf_python.scripts.script_utils import load_sentence_inds


def run_regression_cv(subject, config_flag, iRRR_config_path,
                      loo_config_flags=[], col_ctr_RR=True, alphas=None,
                      nfolds_outer=10, nfolds_inner=5,
                      bash_path=None):
    # - Assumes that prepare_combined_subject_df has been run
    # - if the regression is speech responsive electrodes only, assumes that the
    # STRF ridge regression has been run for electrode selection
    # - Assumes that regression data have been prepared for config_flag and
    # loo_config_flags
    #
    # bash_path is the path to a script that will submit a job to call
    # run_cv_job_script.py
    # arguments:
    #   path to output folder (containing 'cv_setup.json')
    #   index of cv (e.g. for the output file of the job)
    # Note: bash option seems not to work when called from a pycharm python
    # console (Running python as script from bash terminal works fine)

    with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
        config = json.load(f)
    subjects_dir = config['subjects_dir']

    cv_folder = op.join(subjects_dir, subject, config_flag, f'cv-{nfolds_outer}fold')
    folderi = 1
    while op.exists(cv_folder):
        # don't overwrite a previous run
        cv_folder = op.join(subjects_dir, subject, config_flag, f'cv-{nfolds_outer}fold-{folderi}')
        folderi += 1
    print(f'Target folder: {cv_folder}')
    makedirs(cv_folder)

    # Load sentence indices
    sentence_ind = load_sentence_inds(subject, config_flag)

    # Split sentences into CV folds (deterministic)
    cv_outer = GroupKFold(n_splits=nfolds_outer)

    # loop
    for i, (train_sample_inds, test_sample_inds) in \
            enumerate(cv_outer.split(sentence_ind, groups=sentence_ind)):
        # make a folder for output
        cv_job_path = op.join(cv_folder,f'job_{i}')
        makedirs(cv_job_path)

        # Run everything for the samples
        job_config = dict(subject=subject,
                          config_flag=config_flag,
                          iRRR_config_path=iRRR_config_path,
                          loo_config_flags=loo_config_flags,
                          outpath=cv_job_path,
                          col_ctr_RR = col_ctr_RR,
                          alphas = list(alphas),
                          nfolds_inner = int(nfolds_inner),
                          train_sample_inds=train_sample_inds.tolist(),
                          test_sample_inds=test_sample_inds.tolist())
        with open(op.join(cv_job_path,'cv_setup.json'), "w") as f:
            json.dump(job_config,f)

        if bash_path is None:
            run_job_from_config(cv_job_path)
        else:
            print(f'bash {bash_path} {cv_job_path} {i}')
            run(['bash', bash_path, cv_job_path, str(i)], stderr=STDOUT)

#%%

def collect_cv_results(subjects_dir, subject, config_flag, cv_folder, iRRR_config_path):
    # cv_folder: name of cv folder (e.g. 'cv-10fold-1')
    from mtrf_python.scripts.run_ridge_regression_script import get_ridge_filename
    from mtrf_python.scripts.run_irrr_script import get_iRRR_filename

    nfolds_outer = int(re.search('([0-9]+)fold',cv_folder).group(1))

    # Load paths
    with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
        config = json.load(f)
    subjects_dir = config['subjects_dir']
    loadpath = op.join(subjects_dir, subject, config_flag)

    with open(iRRR_config_path, "r") as f:
        irrr_config = json.load(f)
    lambdas = irrr_config['lambdas']


    ### load CV results
    # Summaries
    cv_summaries = []
    for i in range(nfolds_outer):
        cv_job_path = op.join(loadpath, cv_folder, f'job_{i}', f'cv_summary.pkl')
        with open(cv_job_path,'rb') as f:
            cv_job = pickle.load(f)
        cv_summaries.append(cv_job)

    # For CV fold 1, load all of the iRRR runs
    cv_fold1_resultspath = op.join(loadpath, cv_folder, f'job_1')
    nfolds_inner = cv_summaries[0]['nfolds_inner']
    irrr_flag = cv_summaries[0]['best_irrr']['params']['irrr_flag']

    cv_fold1_irrrs = []
    for i,lam in enumerate(lambdas):
        irrrs_lambda = []
        for j in range(nfolds_inner):
            filename = get_iRRR_filename(subject, lam, irrr_flag)
            irrr_path = op.join(cv_fold1_resultspath, f'{filename}_cv{j+1}of{nfolds_inner}.pkl')
            with open(irrr_path,'rb') as f:
                irrr = pickle.load(f)
            irrrs_lambda.append(irrr)
        cv_fold1_irrrs.append(irrrs_lambda)

    ### Save results outside of the bootstrap folder
    out = dict(cv_summaries=cv_summaries,
               cv_fold1_irrrs=cv_fold1_irrrs)

    filename = f'{cv_folder}_cv_summary.pkl'
    filepath = op.join(loadpath,filename)
    print(f'Saving {filepath}...')
    with open(filepath,'wb') as f:
        pickle.dump(out,f)


#%%

def main():
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
      "iRRR_config_path",  # required
      type=str,
    )
    parser.add_argument(
      "--nfolds_outer",  # optional
      dest='nfolds_outer',
      type=int,
      default=10,  # default if nothing is provided
    )
    parser.add_argument(
      "--nfolds_inner",  # optional
      dest='nfolds_inner',
      type=int,
      default=5,  # default if nothing is provided
    )
    parser.add_argument(
      "--bash_path",  # optional
      dest='bash_path',
      default=None,  # default if nothing is provided
    )
    parser.add_argument(
      "--loo_config_flags",  # optional
      nargs="*",  # 0 or more values expected => creates a list
      type=str,
      default=[],  # default if nothing is provided
    )

    args = parser.parse_args()
    run_regression_cv(args.subject, args.config_flag, args.iRRR_config_path,
                      args.loo_config_flags, nfolds_outer=args.nfolds_outer,
                      nfolds_inner=args.nfolds_inner, bash_path=args.bash_path)


if __name__ == '__main__':
    main()
