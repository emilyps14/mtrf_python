import json
import numpy as np
import os.path as op
import pickle

from mtrf_python import STRF_utils
from sklearn.model_selection import GroupKFold


def load_training_data(subject, config_flag, column_center=False,
                       sample_inds=None, nfolds=1):
    # Generator of training and validation data (stim and response) for each cv fold
    # if nfolds=1 (default), returns the entire data (validation data is empty)
    # sample_inds: the indices to subsample the full data (for the outer CV)

    # Load paths
    with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
        config = json.load(f)
    subjects_dir = config['subjects_dir']

    loadpath = op.join(subjects_dir, subject, config_flag)
    filepath = op.join(loadpath,f'{subject}_RegressionSetup.pkl')
    with open(filepath,'rb') as f:
        data = pickle.load(f)

    dstim = data['dstim']
    resp = data['resp']
    time_in_trial = data['time_in_trial']
    sentence_ind = data['sentence_ind']

    if sample_inds is not None:
        # outer cv
        resp = resp[sample_inds, :]
        dstim = [d[sample_inds, :] for d in dstim]
        time_in_trial = time_in_trial[sample_inds]
        sentence_ind = sentence_ind[sample_inds,:]

    if nfolds>1:
        # inner cv
        cv = GroupKFold(n_splits=nfolds)
        for train_inds,validate_inds in cv.split(sentence_ind, groups=sentence_ind):
            # Create matrices for cross validation
            tStim = [d[train_inds,:] for d in dstim]
            tResp = resp[train_inds,:]
            vStim = [d[validate_inds,:] for d in dstim]
            vResp = resp[validate_inds,:]

            # Column Center if desired
            if column_center:
                print('Centering columns...')
                tStim,tResp = apply_column_center(tStim,tResp)
                vStim,vResp = apply_column_center(vStim,vResp)

            yield loadpath, train_inds, validate_inds, tStim, tResp, vStim, vResp
    else:
        train_inds = np.arange(resp.shape[0])
        validate_inds = []
        tStim = dstim
        tResp = resp
        vStim = []
        vResp = np.array([])

        # Column Center if desired
        if column_center:
            print('Centering columns...')
            tStim,tResp = apply_column_center(tStim,tResp)

        yield loadpath, train_inds, validate_inds, tStim, tResp, vStim, vResp

def load_testing_data(subject, config_flag, testing_inds, column_center=False):
    # Load paths
    with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
        config = json.load(f)
    subjects_dir = config['subjects_dir']

    loadpath = op.join(subjects_dir, subject, config_flag)
    filepath = op.join(loadpath,f'{subject}_RegressionSetup.pkl')
    with open(filepath,'rb') as f:
        data = pickle.load(f)

    dstim = data['dstim']
    resp = data['resp']

    # outer cv
    rResp = resp[testing_inds, :]
    rStim = [d[testing_inds, :] for d in dstim]

    if column_center:
        print('Centering columns...')
        rStim,rResp = apply_column_center(rStim,rResp)

    return loadpath, rStim, rResp

def load_sentence_inds(subject, config_flag):
    # Load paths
    with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
        config = json.load(f)
    subjects_dir = config['subjects_dir']

    loadpath = op.join(subjects_dir, subject, config_flag)
    filepath = op.join(loadpath,f'{subject}_RegressionSetup.pkl')
    with open(filepath,'rb') as f:
        data = pickle.load(f)

    sentence_ind = data['sentence_ind']

    return sentence_ind

def apply_column_center(stim,resp):
    stim_out = [s-s.mean(0,keepdims=True) for s in stim]
    resp_out = resp-resp.mean(0,keepdims=True)
    return stim_out,resp_out

def _add_testing_metrics(model, rStim, rResp):
    if 'mu' in model:
        mu = model['mu'].T
    else:
        mu = 0
    pred = STRF_utils.compute_pred(np.concatenate(rStim, axis=1), model['wts'], mu)
    model['testing_electrode_R2s'] = STRF_utils.compute_electrode_rsquared(rResp, pred)
    model['testing_overall_R2'] = STRF_utils.compute_overall_rsquared(rResp, pred)
