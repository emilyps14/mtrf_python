import glob
import h5py
import json
import numpy as np
import os.path as op
import pandas as pd
import pickle
from os import makedirs
from scipy.io import loadmat
from shutil import copyfile

from mtrf_python import STRF_utils

def prepare_combined_subject_df(subj_config_path):
    # Load paths
    with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
        config = json.load(f)
    preproc_dir = config['yulia_prcsd_dir']
    subjects_dir = config['subjects_dir']
    imaging_dir = config['imaging_subjects_dir']

    # Load subject config
    with open(subj_config_path, "r") as f:
        subj_config = json.load(f)
    agg_subject = subj_config['agg_subject']
    subjects = subj_config['subjects']
    load_filename = subj_config['load_filename']
    out_filename = subj_config['out_filename']

    # %
    savepath = op.join(subjects_dir, agg_subject)
    if not op.exists(savepath):
        makedirs(savepath)

    # Save a copy of the subject config in the target directory
    copyfile(subj_config_path, op.join(savepath, f'{agg_subject}.json'))

    # %
    nsubjects = len(subjects)

    # Find appropriate files for all subjects
    filepaths = {}
    for subject in subjects:
        outfile = op.join(preproc_dir, subject, 'block_z', load_filename)
        f = glob.glob(outfile)
        if f:
            print(f'{subject}: {f}')
            assert (len(f) == 1)
            filepaths[subject] = f[0]
        else:
            outfile = op.join(preproc_dir, subject, 'timit', 'block_z',
                              load_filename)
            f = glob.glob(outfile)
            if f:
                print(f'{subject} timit dir: {f}')
                assert (len(f) == 1)
                filepaths[subject] = f[0]
            else:
                print(f'{subject}: No data')

    electrode_info = get_all_electrode_info(subjects,imaging_dir)


    # %
    needed_columns = ['name', 'Trials', 'resp', 'dataf', 'befaft',
                      'subject', 'ntrials', 'source_path']

    dfs = {}
    for i, subject in enumerate(subjects):
        source_path = filepaths[subject]

        print(f'{i + 1}/{nsubjects} Loading {source_path}')
        try:
            out = loadmat(source_path, squeeze_me=True)['out']
        except NotImplementedError as e:
            print('   v7.3')
            out = load_out_mat(source_path, squeeze_me=True)

        df = pd.DataFrame(out)
        del out

        df['ntrials'] = df.apply(
            lambda x: x['resp'].shape[2] if len(x['resp'].shape) == 3 else 1,
            axis=1)
        df['subject'] = subject
        df['source_path'] = source_path

        dfs[subject] = df[needed_columns]

    df = pd.concat(dfs.values(), ignore_index=True)

    # % Save
    out_filepath = op.join(savepath, out_filename)
    pickle.dump({'df': df,
                 'electrode_info': electrode_info},
                open(out_filepath, 'wb'))


def parse_dataset(dataset, f, squeeze_me=False):
    # 12/19/19: Matlab -v7.3 saves arrays in F-order, but this doesn't show up
    # in the flags when loaded as an hdf5 file (reports C-order). As a result,
    # we need to transpose all arrays on load.
    if dataset.dtype.kind == 'u':
        data = ''.join(map(chr, dataset[:]))
        type = str
        typename = 'String'
    elif dataset.dtype.kind == 'f':
        data = np.array(dataset).T  # F-order
        type = data.dtype
        typename = 'Float Array'
    elif dataset.dtype.kind == 'O':  # cell array?
        cellarray = np.zeros(dataset.shape, dtype=np.object)
        types = np.zeros(dataset.shape, dtype=np.object)
        for i, column in enumerate(dataset):
            for j in range(len(column)):
                cell, celltype, _ = parse_dataset(f[column[j]], f,
                                                  squeeze_me=squeeze_me)
                cellarray[i, j] = cell
                types[i, j] = celltype
        cellarray = cellarray.T  # F-order
        if all([isinstance(c, str) for c in cellarray.flatten()]):
            data = cellarray.astype(str)
        else:
            data = cellarray
        type = np.ndarray
        typename = 'Cell Array'
    else:
        raise RuntimeError(f'Haven\'t implemented dtype!')
    if squeeze_me:
        data = np.squeeze(data)
    return data, type, typename


def load_out_mat(filepath, squeeze_me=False):
    with h5py.File(filepath, 'r') as f:
        dat = {}
        for k, v in f['out'].items():
            dat[k], type, typename = parse_dataset(v, f, squeeze_me=squeeze_me)
            print(f'{k}: {typename}')
    return dat

def get_all_electrode_info(subjects,imaging_dir):
    electrode_info = []
    for s in subjects:
        einfo = STRF_utils.load_electrode_info(imaging_dir,s)
        einfo['subject'] = s
        electrode_info.append(pd.DataFrame(einfo,index=[f'{s}_{i}' for i in range(len(einfo['rois']))]))
    electrode_info = pd.concat(electrode_info,axis=0)
    return electrode_info


# %%

def main():
    import sys

    subj_config_path = sys.argv[1]

    prepare_combined_subject_df(subj_config_path)


if __name__ == '__main__':
    main()
