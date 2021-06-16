import numpy as np
from mtrf_python import STRF_utils
#import pytest

def test_compute_pred():
    stim = np.arange(120.).reshape(30,4) # 30 times, 4 features
    wt_array = np.arange(24.).reshape(4,6) # 6 electrodes
    mu = np.arange(6.).reshape(1,6)

    ntimes = stim.shape[0]
    Nk = wt_array.shape[1:]
    pred = np.zeros([ntimes]+list(Nk))
    for kk in np.ndindex(Nk):
        pred[np.s_[:,] + kk] = stim.dot(wt_array[np.s_[:,] + kk])
    pred+=mu

    assert(np.all(np.equal(STRF_utils.compute_pred(stim,wt_array,mu),pred)))


def test_split_training_sentences():
    offset_inds = [2,6,8,12,15]
    splits = [(0.6, 0.2, 0.2),
              (0.8, 0.2)]
    sols = [[[0,1,2,3,4,5,6,7,8],[9,10,11,12],[13,14,15]],
            [[0,1,2,3,4,5,6,7,8,9,10,11,12],[13,14,15]]]

    for split,sol in zip(splits,sols):
        assert(STRF_utils.split_training_sentences(offset_inds,split)==sol)

    offset_inds = [1,3]
    split = [0.5,0.5]
    sol = [[0,1],[2,3]]
    assert(STRF_utils.split_training_sentences(offset_inds,split)==sol)


def test_resample_sentences():
    groups = [6,6,6,2,2,2,2,3,3,4,4,4,4,5,5,5]
    sample = [3,4,4,2]

    order = [7,8,9,10,11,12,9,10,11,12,3,4,5,6]
    new_groups = [3,3,4,4,4,4,4,4,4,4,2,2,2,2]

    assert(STRF_utils.resample_sentences(groups, sample) == (order, new_groups))

    groups = ['2_1','2_1','1_1','1_1','2_2','2_2']
    sample = ['2_2','1_1','2_1','1_1']

    order = [4,5,2,3,0,1,2,3]
    new_groups = ['2_2','2_2','1_1','1_1','2_1','2_1','1_1','1_1']

    assert(STRF_utils.resample_sentences(groups, sample) == (order, new_groups))

def test_get_sentence_offset_inds():
    time_in_trial = [-1,0,1,2,3,-2,-1,0,1,2,3,0,1,2]
    offset_inds = [4,10,13]

    assert(STRF_utils.get_sentence_offset_inds(time_in_trial)==offset_inds)