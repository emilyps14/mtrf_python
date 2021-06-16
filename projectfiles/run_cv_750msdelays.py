import os.path as op

import numpy as np

from mtrf_python.scripts.prepare_combined_subject_df_script import \
    prepare_combined_subject_df
from mtrf_python.scripts.prepare_regression_data_script import \
    prepare_regression_data
from mtrf_python.scripts.run_regression_master_script import \
    run_regression_cv, collect_cv_results
from mtrf_python.scripts.run_ridge_regression_script import \
    run_ridge_for_electrode_selection

subject = 'Hamilton_Agg_LH_no63_131_143' # 'Hamilton_Agg_LH_9subjects'
strf_config_flag = 'regression_STRF'
config_flag = 'regression_SO1_PR_phnfeats_750msdelays'
loo_config_flags = ['regression_justPhonetic_750msdelays',
                        'regression_SO1_PR_750msdelays']
irrr_flag = 'iRRR_210111'
col_ctr_RR = True
alphas = np.concatenate([[0],np.logspace(2,7,20)])

import json
with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
    config = json.load(f)
subjects_dir = config['subjects_dir']
bash_path = None
iRRR_config_path = op.join('.','config',irrr_flag+'.json') # assumes you'll run the code from the mtrf_python directory

#%% Prep data [creates the file Hamilton_Agg_LH_no63_131_143_HilbAA_70to150_8band_out_resp_log.pkl]
# prepare_combined_subject_df(op.join('.','config', subject + '.json'))

#%% Run STRF model to identify speech responsive electrodes
prepare_regression_data(subject, op.join('.', 'config', strf_config_flag + '.json'))

run_ridge_for_electrode_selection(subject,strf_config_flag,[0],column_center=col_ctr_RR,nfolds=1,r2_threshold=0.05,
                                  savename='strf_r2_threshold_OLS.pkl')
run_ridge_for_electrode_selection(subject,strf_config_flag,alphas,column_center=col_ctr_RR,nfolds=5,r2_threshold=0.05,
                                  savename='strf_r2_threshold_Ridge.pkl')

#%% Use STRF results to prepare feature models with only speech responsive electrodes
prepare_regression_data(subject, op.join('.', 'config', config_flag + '.json'))

for loo_config_flag in loo_config_flags:
    prepare_regression_data(subject,op.join('.','config',loo_config_flag+'.json'))

 #%% Run Bootstrap
run_regression_cv(subject, config_flag, iRRR_config_path,
                  loo_config_flags=loo_config_flags, col_ctr_RR=col_ctr_RR, alphas=alphas,
                  nfolds_outer=10, nfolds_inner=5,
                  bash_path=bash_path)

#%% Collect results (After all jobs are complete)
cv_folder = 'cv-10fold'
collect_cv_results(subjects_dir,subject,config_flag,cv_folder,iRRR_config_path)