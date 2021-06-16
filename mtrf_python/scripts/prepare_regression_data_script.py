import json
import numpy as np
import os.path as op
import pandas as pd
import pickle
from os import makedirs
from shutil import copyfile

from mtrf_python import STRF_utils

def prepare_regression_data(subject, regression_config_path):
    # Load paths
    with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
        config = json.load(f)
    subjects_dir = config['subjects_dir']

    filename = f'{subject}_HilbAA_70to150_8band_out_resp_log.pkl' # Todo make this flexible

    # Load regression config
    with open(regression_config_path, "r") as f:
        reg_config = json.load(f)
    training_features = reg_config['training_features']
    drop_features = reg_config['drop_features']
    time_lag = reg_config['time_lag']
    blnSTGOnly = reg_config.get('blnSTGOnly', False)
    blnDropSilent = reg_config.get('blnDropSilent', False)
    befaft = reg_config.get('befaft', [0, 0])

    # Option to restrict to speech-responsive electrodes
    # Dictionary with keys:
    #     "strf_config_flag": string e.g. "regression_STRF" (name of the config file used for the STRF regression)
    #     "alphas": list of alpha values used for the STRF regression (will pick the best one for each electrode)
    #     "r2_threshold": float Threshold to apply to the r^2 values to decide speech responsive
    #     "col_ctr": whether to column center the stim and response matrices in the ridge
    strf_r2_threshold = reg_config.get('strf_r2_threshold',None)

    if isinstance(time_lag, dict):
        time_lag_dict = time_lag
    else:
        time_lag_dict = {'default': time_lag}

    config_flag = op.splitext(op.basename(regression_config_path))[0]

    #%
    savepath = op.join(subjects_dir, subject, config_flag)
    if not op.exists(savepath):
        makedirs(savepath)

    # Save a copy of the config in the target directory
    copyfile(regression_config_path, op.join(savepath, f'{config_flag}.json'))

    #% Load the stimulus and response data
    df_filepath = op.join(subjects_dir, subject, filename)
    d = pickle.load(open(df_filepath, 'rb'))
    df = d['df']
    electrode_info = d['electrode_info']

    df.dataf = df.dataf.astype(int)
    df.name = df.name.astype(str)

    fs = df.loc[0,'dataf']

    # Check existing padding in the data
    source_befaft = df.loc[0,'befaft']
    if source_befaft[0]<befaft[0] or source_befaft[1]<befaft[1]:
        raise RuntimeError(f"Desired padding ({befaft[0]},{befaft[1]}) is incompatible with input data padding ({source_befaft[0]},{source_befaft[1]})")

    print(f'Using padding: {befaft[0]} s before until {befaft[1]} s after')

    # Figure out how many (if any) samples to drop
    padleft = ((source_befaft[0]-befaft[0])*fs).astype(int)
    padright = ((source_befaft[1]-befaft[1])*fs).astype(int)

    subjects = df.subject.unique()
    Nsubjects = len(subjects)

    #% Load sentence details
    sentdet_df, featurenames, phnnames = STRF_utils.load_sentence_details(op.join(subjects_dir))
    stim_df, phnstimnames = process_sentence_details(sentdet_df, training_features,
                                                        featurenames, phnnames,
                                                        drop_features=drop_features)

    print(f'Using features: {phnstimnames}')

    #% Merge features into df
    df = df.merge(stim_df,on='name',how='left')

    #% Correct response lengths for subjects with non-standard number of samples
    # Known: EC82, EC143
    df['len_resp'] = df.apply(lambda x: x['resp'].shape[1],axis=1)
    df['len_so'] = df.apply(lambda x: x[phnstimnames[0]].shape[1],axis=1)
    print(f'Correcting response lengths for subjects {df.loc[np.logical_not(df.len_resp==df.len_so)].subject.unique()}')

    for i,row in df.iterrows():
        len_resp = row.len_resp
        len_so = row.len_so
        if row.len_resp!=row.len_so:
            diff = (len_resp-len_so)
            assert(diff>0)
            i1 = np.floor(diff/2).astype(int)
            i2 = -np.ceil(diff/2).astype(int)
            # print(f'{row.subject} {row.name}: {row.resp.shape} vs. {len_so}, {i1}:{i2}')
            df.at[i,'resp'] = row.resp[:,i1:i2,]
    df['len_resp'] = df.apply(lambda x: x['resp'].shape[1],axis=1)
    df['nrepeats'] = df.apply(
        lambda x: x['resp'].shape[2] if len(x['resp'].shape)==3 else 1, axis=1)

    #% Time in trial
    df['time_in_trial'] = df.apply(lambda x: np.arange(x['len_resp'])[None,:]/fs - source_befaft[0],axis=1)

    # in the dataframe, sentence_ind will be a np.array of strings '1_2'
    # indicating the sentence index and the repeat number
    # the size is (1,Tsentence,nrepeats)
    # [to play well with STRF_utils.get_training_mat]
    sentence_ind_map = {name:i for i,name in enumerate(df.name.unique())}
    df['sentence_ind'] = df.apply(
        lambda x: np.tile([f'{sentence_ind_map[x["name"]]}_{j}'
                           for j in range(x['nrepeats'])],
                          (1,x['len_resp'],1)), axis=1)

    #% Check columns
    assert(len(df.dataf.unique())==1)
    assert(np.unique(np.stack(df.befaft),axis=0).shape[0]==1)
    assert(len(df.loc[np.logical_not(df.len_resp==df.len_so)])==0)

    for f in phnstimnames:
        assert(all(df.apply(lambda x: x['resp'].shape[1]==x[f].shape[1],axis=1)))

    # Pick sentences that exist for all subjects
    subj_count = df.groupby(['name']).subject.nunique()
    sentence_sel = subj_count[subj_count==Nsubjects].index.to_list()

    print(f'Using sentences: {sentence_sel}')
    df = df.loc[df.name.isin(sentence_sel)]

    # Get stacked data matrices
    phnstim,time_in_trial,sentence_ind,resp,electrodes = \
        get_data_from_df(df,phnstimnames,subjects,
                         padleft,padright)

    # Pick Electrodes
    if blnSTGOnly:
        print(f'Keeping only STG electrodes')
        blnSTG = np.zeros(len(electrodes),dtype=bool)
        for i,ch in enumerate(electrodes):
            if ch in electrode_info.index:
                blnSTG[i] = electrode_info.loc[ch]['rois']=='superiortemporal'
        resp = resp[:,blnSTG]
        electrodes = electrodes[blnSTG]

    if blnDropSilent:
        print(f'Dropping silent electrodes')
        blnSilent = np.all(resp==0,axis=0)
        resp = resp[:,~blnSilent]
        electrodes = electrodes[~blnSilent]

    if strf_r2_threshold is not None:
        print(f'Keeping only speech responsive electrodes')
        blnSpeechResponsive = process_strf_r2_threshold(strf_r2_threshold,subject,subjects_dir,electrodes)
        resp = resp[:,blnSpeechResponsive]
        electrodes = electrodes[blnSpeechResponsive]

    assert(resp.shape[0]==phnstim.shape[0]
           ==df.groupby('subject').len_resp.sum().unique()-padleft*len(sentence_sel)-max(padright-1,0)*len(sentence_sel))

    time_lags = [time_lag_dict[name] if name in time_lag_dict else time_lag_dict['default']
                 for name in phnstimnames]
    dstim, delays = get_delay_matrices(phnstim,phnstimnames,time_lags,fs)

    #% Save regression setup data
    out = dict(
        training_features=training_features,
        fs=fs,
        time_lags=time_lags,
        df=df,
        electrodes=electrodes,
        electrode_info=electrode_info,
        phnstim=phnstim,
        phnstimnames=phnstimnames,
        time_in_trial=time_in_trial,
        sentence_ind=sentence_ind,
        dstim=dstim,
        delays=delays,
        resp=resp,
        sentence_sel=sentence_sel,
        )
    filepath = op.join(savepath,f'{subject}_RegressionSetup.pkl')
    print(f'Saving {filepath}...')
    with open(filepath,'wb') as f:
        pickle.dump(out,f)

def process_sentence_details(sentdet_df, training_features, featurenames,
                             phnnames, drop_features=[]):
    stim_df = sentdet_df[['name','txt']].copy()

    #% Pull out the desired features
    stimnames = []
    for f in training_features:
        # get column names
        if 'feat' in f:
            colnames = featurenames
        elif 'phnmat' in f:
            colnames = phnnames
        elif 'aud' in f:
            nfreqs = sentdet_df.iloc[0].aud.shape[0]
            colnames = [f'f{i}' for i in range(nfreqs)]
        else:
            colnames = [f]

        # expand the data into separate columns
        for i,c in enumerate(colnames):
            if c not in drop_features:
                stim_df[c] = sentdet_df.apply(lambda x: x[f][[i],:],axis=1)

        stimnames.append(colnames)

    phnstimnames = np.concatenate(stimnames)
    keep = np.logical_not(np.in1d(phnstimnames,drop_features))
    phnstimnames = phnstimnames[keep]

    return stim_df, phnstimnames


def process_strf_r2_threshold(strf_r2_threshold,subject,subjects_dir,electrodes=None):
    from mtrf_python.scripts.run_ridge_regression_script import get_ridge_filename

    datapath = op.join(subjects_dir, subject,
                       strf_r2_threshold['strf_config_flag'],
                       strf_r2_threshold['processing_filename'])
    with open(datapath,'rb') as f:
        strf_data = pickle.load(f)
    assert(np.allclose(strf_r2_threshold['alphas'],strf_data['alphas']))
    assert(strf_r2_threshold['r2_threshold']==strf_data['r2_threshold'])
    assert(strf_r2_threshold['col_ctr']==strf_data['column_center'])
    electrodes_all = strf_data['electrodes_all']
    speech_responsive_all = strf_data['speech_responsive_all']

    if electrodes is None:
        return speech_responsive_all
    else:
        # choose only electrodes that are in the electrodes list
        blnSpeechResponsive = np.array([sr for sr,el in zip(speech_responsive_all,electrodes_all) if el in electrodes])
        return blnSpeechResponsive

def get_data_from_df(df,phnstimnames,subjects, padleft=0, padright=0,
                     electrodes_sel=None, repeat_sel=[]):
    # Create stim data
    phnstim = np.concatenate([STRF_utils.get_training_mat(df.loc[df.subject==subjects[0],f],
                                                          padleft,padright,
                                                          blnMean=False)
                              for f in phnstimnames],axis=1)
    assert(len(phnstimnames)==phnstim.shape[1])

    time_in_trial = STRF_utils.get_training_mat(
        df.loc[df.subject==subjects[0]].time_in_trial,
        padleft,padright,blnMean=False)

    sentence_ind = STRF_utils.get_training_mat(
        df.loc[df.subject==subjects[0]].sentence_ind,
        padleft,padright,blnMean=False,sel=repeat_sel)

    # Create resp data for each subject
    resp_list = []
    electrodes = []
    for s in subjects:
        r = STRF_utils.get_training_mat(
            df.loc[df.subject==s].resp,
            padleft,padright,blnMean=False,sel=repeat_sel)
        e = [f'{s}_{i}' for i in range(r.shape[1])]
        resp_list.append(r)
        electrodes.append(e)

    # Concatenate across electrode dimension
    resp = np.concatenate(resp_list,axis=1)
    electrodes = np.concatenate(electrodes)

    if electrodes_sel is not None:
        # pick electrodes
        keep = np.in1d(electrodes,electrodes_sel)
        resp = resp[:,keep,]
        electrodes = electrodes[keep]

    return phnstim,time_in_trial,sentence_ind,resp,electrodes

def get_delay_matrices(phnstim,phnstimnames,time_lags,fs):
    #% Create delay matrices **For each feature separately**
    temp = [STRF_utils.build_delay_matrices(col[:,None],fs,
                                            delay_time=lag,
                                            blnZscore=False) for col,lag in zip(phnstim.T,time_lags)]
    dstim = [t[1] for t in temp]
    delays = [t[0] for t in temp]

    return dstim, delays

# %%

def main():
    import sys

    subject = sys.argv[1]
    regression_config_path = sys.argv[2]

    prepare_regression_data(subject,regression_config_path)


if __name__ == '__main__':
    main()
