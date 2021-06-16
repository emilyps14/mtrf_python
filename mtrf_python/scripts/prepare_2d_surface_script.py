import json
import os.path as op
import pickle
import numpy as np
from es_utils import mayavi_3d_to_2d
from mayavi import mlab
from surfer import Brain


def prepare_2d_surface(agg_subject, config_flag, blnWarped=True,
                       plot_subject='cvs_avg35_inMNI152', hemi='lh'):
    # Load paths
    with open(f"{op.expanduser('~')}/.config/mtrf_config.json", "r") as f:
        config = json.load(f)
    subjects_dir = config['subjects_dir']
    imaging_dir = config['imaging_subjects_dir']

    loadpath = op.join(subjects_dir, agg_subject, config_flag)
    filepath = op.join(loadpath,f'{agg_subject}_RegressionSetup.pkl')
    with open(filepath,'rb') as f:
        data = pickle.load(f)
    electrodes = data['electrodes']
    electrode_info = data['electrode_info']

    # Collect warped electrode coordinates for all subjects
    coordinates = []
    for i,ch in enumerate(electrodes):
        if ch in electrode_info.index:
            if blnWarped:
                coordinates.append(electrode_info.loc[ch,['loc_warped_x','loc_warped_y','loc_warped_z']])
            else:
                coordinates.append(electrode_info.loc[ch,['loc_x','loc_y','loc_z']])
        else:
            coordinates.append([np.nan,np.nan,np.nan])

    coordinates = np.stack(coordinates).astype(float)

    # Set up 2d Image
    mfig = mlab.figure(size=[800,800])
    brain = Brain(plot_subject, hemi, 'pial', subjects_dir=imaging_dir,
                  views='lateral', background='w', figure=mfig)
    img, trans_dict = mayavi_3d_to_2d.make_2d_surface(mfig)

    # Check correspondence
    # brain.add_foci(coords, hemi=hemi, scale_factor=0.2, color='b')
    # mayavi_3d_to_2d.scatter_on_2d_surface(coords, img, trans_dict,
    #                                    scatter_kwargs=dict(s=2,c='b',marker='.'))

    out = dict(
        img=img,
        trans_dict=trans_dict,
        coords_2d=mayavi_3d_to_2d.transform_coordinates(coordinates,trans_dict),
        )

    filepath = op.join(loadpath, f'{agg_subject}_{hemi}_pial_lateral_2dSurf.pkl')
    print(f'Saving {filepath}...')
    with open(filepath,'wb') as f:
        pickle.dump(out,f)


def main():
    import sys

    agg_subject = sys.argv[1]
    config_flag = sys.argv[2]

    prepare_2d_surface(agg_subject, config_flag)


if __name__ == '__main__':
    main()
