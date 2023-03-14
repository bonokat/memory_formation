import mne
import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath('./'))
if not current_dir in sys.path:
    sys.path.append(current_dir)

# mne.set_log_level('CRITICAL')

if __name__ == '__main__':

    root = '.\\'

    parser = argparse.ArgumentParser(description='Script to morph sources to average brain')
    parser.add_argument('-from', type=int, help='id of the first subject', default=1)
    parser.add_argument('-to', type=int, help='id of the last subject', default=None)
    parser.add_argument('-cond', '--condition', type=str, help='condition from stc name to use', default='hits')
    parser.add_argument('-fband', '--frequency_band', type=str, help='fband from stc name to use', default='theta')

    from_,\
        to,\
        cond,\
        fband = vars(parser.parse_args()).values()

    # directories to work with
    data_dir = os.path.join(root, 'data')
    subjects_dir = os.path.join(data_dir, 'subjects')
    sub_inds = [int(subject.replace('sub', '')) for subject in os.listdir(subjects_dir)]
    mris_dir = os.path.join(root, 'data', 'mri')
    fsaverage_src_path = os.path.join(mris_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif')

    if to is None:
        to = max(sub_inds)

    subjects_range = range(from_, to + 1)

    # iterate through subjects
    for sub_ind, subject_name in zip(sub_inds, os.listdir(subjects_dir)):
        if sub_ind not in subjects_range:
            continue

        # directories to work with
        subject_dir = os.path.join(subjects_dir, subject_name)
        print(f'Subject name: {subject_name}')
        print('{0:64s}'.format(f'Reading {subject_name} folder...'), end='')
        stc_dir = os.path.join(subject_dir, 'source_rec')
        stc_path = os.path.join(stc_dir, f'stc_{cond}_{fband}_{subject_name}-stc.h5')
        stc = mne.read_source_estimate(stc_path)
        src_to = mne.read_source_spaces(fsaverage_src_path)
        morph = mne.compute_source_morph(stc, subject_from=subject_name,
                                            subject_to='fsaverage', src_to=src_to,
                                            subjects_dir=mris_dir)
        stc_fsaverage = morph.apply(stc)

        # save morphs and stc_fsaverage
        morph_path = os.path.join(mris_dir, 'morph-maps')
        morph_name = os.path.join(morph_path, fband, f'morph_{subject_name}_{cond}_{fband}')
        stc_fsaverage_path = os.path.join(data_dir, 'group', 'fsaverage_stc', fband)
        stc_fsaverage_name = os.path.join(stc_fsaverage_path, f'fsaverage_stc_{subject_name}_{cond}_{fband}')
        morph.save(morph_name, overwrite=True)
        stc_fsaverage.save(stc_fsaverage_name, ftype='h5', overwrite=True)

