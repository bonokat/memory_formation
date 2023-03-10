import mne
import os
import numpy as np
import argparse
from utils import check_paths
import sys

os.environ['MNE_3D_OPTION_ANTIALIAS']='false' # to make visualization work

current_dir = os.path.dirname(os.path.abspath('./'))
if not current_dir in sys.path:
    sys.path.append(current_dir)

from utils import check_paths
import scipy
from copy import  deepcopy

mne.set_log_level('CRITICAL')

if __name__ == '__main__':

    root = '.\\'

    parser = argparse.ArgumentParser(description='Script to iterate through subjects')
    parser.add_argument('-from', type=int, help='id of the first subject', default=1)
    parser.add_argument('-to', type=int, help='id of the last subject', default=None)
    parser.add_argument('-cond1', '--hits_condition', type=str, help='compose from hits/neg/neu/sure/notsure', default='hits')
    parser.add_argument('-cond2', '--miss_condition', type=str, help='compose from miss/neg/neu/sure/notsure', default='miss')
    parser.add_argument('-bl_start', '--start of the baseline interval', type=float, help='set start of baseline interval', default=-0.5)
    parser.add_argument('-bl_fin', '--finish of the baseline interval', type=float, help='set finish of baseline interval', default=0.)


    from_,\
        to,\
        cond1,\
        cond2,\
        bl_start,\
        bl_fin = vars(parser.parse_args()).values()
    # make correction for prestimulus time
    bl_start += 1.5
    bl_fin += 1.5

    # directories to work with
    subjects_dir = os.path.join(root, 'data', 'subjects')
    sub_inds = [int(subject.replace('sub', '')) for subject in os.listdir(subjects_dir)]
    mris_dir = os.path.join(root, 'data', 'mri')
    fsaverage_src_path = os.path.join(mris_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif')

    if to is None:
        to = max(sub_inds)

    subjects_range = range(from_, to + 1)

    # set frequency bands
    theta = (4., 8.)
    alpha = (8., 15.)
    beta = (15., 30.)
    gamma = (30., 60.)
    freq_bands = [theta, alpha, beta, gamma]

    # # read empty room noise covariance
    noise_cov_path = 'D:\\Ekaterina_Voevodina\\\\memory_formation\\data\\noise_cov'
    noise_cov = mne.read_cov(noise_cov_path, verbose=None)

    # iterate through subjects
    for sub_ind, subject_name in zip(sub_inds, os.listdir(subjects_dir)):
        if sub_ind not in subjects_range:
            continue

        # directories to work with
        print(f'Subject name: {subject_name}')
        print('{0:64s}'.format(f'Reading {subject_name} folder...'), end='')
        subject_dir = os.path.join(subjects_dir, subject_name)
        print('OK')
        epochs_dir = os.path.join(subject_dir, 'epochs')
        trans_path = os.path.join(mris_dir, subject_name, f'{subject_name}_trans')

        # read epochs and trans files
        epochs_path = os.path.join(epochs_dir, 'enc_epochs_prestim_bl.fif')
        epochs = mne.read_epochs(epochs_path, preload=True).pick_types(meg='grad')
        info = epochs.info
        trans = mne.read_trans(trans_path)

        # # compute data covariance using FULL epochs
        data_cov = mne.compute_covariance(epochs, tmin=0.01, tmax=1.5,
                                        method='empirical')

        # regularize noise covariance
        noise_cov = mne.cov.regularize(noise_cov, info, mag=0.1, grad=0.1,
                                    eeg=0.1, rank='info')

        # READING INFO
        # read forward solution
        source_dir = os.path.join(subject_dir, 'source_rec')
        fwd_path = os.path.join(source_dir, f'fwd_{subject_name}')
        fwd = mne.read_forward_solution(fname=fwd_path)

        # # create beamformer
        filters = mne.beamformer.make_lcmv(info, fwd, data_cov, reg=0.05,
                            noise_cov=noise_cov, pick_ori='max-power',
                            weight_norm='unit-noise-gain', rank=None)


        # # SAVING DATA
        # set dir to save source files
        source_prestim_bl_dir = os.path.join(subject_dir, 'source_rec', 'prestim_bl')
        check_paths(source_prestim_bl_dir)

        # # set path and name for each file
        data_cov_path = os.path.join(source_prestim_bl_dir, f'data_cov_{subject_name}')
        noise_cov_path = os.path.join(source_prestim_bl_dir, f'noise_cov_{subject_name}')
        filters_path = os.path.join(source_prestim_bl_dir, f'lcmv_beamformer_{subject_name}.h5')

        # # save files
        mne.write_cov(data_cov_path, data_cov, overwrite=True)
        mne.write_cov(noise_cov_path, noise_cov, overwrite=True)
        filters.save(filters_path, overwrite=True)


        # iterate through freq bands
        for freq_band in freq_bands:
            x, y = freq_band
            epochs_filt = epochs.copy().filter(x, y)

            # stc hits and misses
            print('Computing stc for hits and misses...')
            stc_hits = mne.beamformer.apply_lcmv_epochs(epochs_filt[cond1], filters)
            print('stc hits: OK')
            stc_miss = mne.beamformer.apply_lcmv_epochs(epochs_filt[cond2], filters)
            print('stc miss: OK')

            # envelope hits and misses
            stc_hits_env = deepcopy(stc_hits)
            stc_miss_env = deepcopy(stc_miss)

            print('Computing envelope for hits...')
            for i in range(len(stc_hits_env)):
                stc_hits_env[i].data = np.abs(scipy.signal.hilbert(stc_hits_env[i].data, N=None, axis=-1))
            print('stc_hits_env: OK')

            print('Computing envelope for miss...')
            for i in range(len(stc_miss_env)):
                stc_miss_env[i].data = np.abs(scipy.signal.hilbert(stc_miss_env[i].data, N=None, axis=-1))
            print('stc_miss_env: OK')

            # compute average of the envelope hits and misses
            stc_hits_av = deepcopy(stc_hits_env[0])
            stc_miss_av = deepcopy(stc_miss_env[0])

            print('Computing average for hits...')
            data = np.array([stc.data for stc in stc_hits_env]).mean(0)
            stc_hits_av.data = data
            print('stc_hits_av: OK')

            print('Computing average for miss...')
            data = np.array([stc.data for stc in stc_miss_env]).mean(0)
            stc_miss_av.data = data
            print('stc_miss_av: OK')

            # manually compute baselines for hits and misses
            tmin = int(epochs.info['sfreq']*bl_start)
            tmax = int(epochs.info['sfreq']*bl_fin)
            bl_hits = stc_hits_av.data[:, tmin:tmax].mean(1, keepdims=True)
            bl_miss = stc_miss_av.data[:, tmin:tmax].mean(1, keepdims=True)
            bl_hits_subset = stc_hits_av.data[
                np.random.choice(len(stc_hits_env), len(stc_miss_env), replace=False),
                tmin:tmax
            ].mean(1)
            bl_all = np.concatenate([bl_hits, bl_miss], 0)
            print(
                f'{bl_hits.mean() = : .4f} +- {bl_hits.std() : .4f};\n'
                f'{bl_miss.mean() = : .4f} +- {bl_miss.std() : .4f};\n'
                f'{bl_hits_subset.mean() = : .4f} +- {bl_hits_subset.std() : .4f};\n'
                f'{bl_all.mean() = : .4f} +- {bl_all.std() : .4f}.'
            )

            cond1_ave, cond1_std = np.abs(bl_hits_subset.mean()), np.abs(bl_hits_subset.std())
            cond2_ave, cond2_std = np.abs(bl_miss.mean()), np.abs(bl_miss.std())

            significant = False
            if cond1_ave > cond2_ave:
                if cond1_ave - cond1_std > cond2_ave + cond2_std:
                    significant = True
            else:
                if cond2_ave - cond2_std > cond1_ave + cond1_std:
                    significant = True
            if significant:
                with open("D:\\Ekaterina_Voevodina\\memory_formation\\data\\stc_baseline_prestim.txt", "a+") as file_object:
                    file_object.write("\n")
                    # Append text at the end of file
                    file_object.write(f'!!! WARNING: std is EXCEEDED in !!!\n')
            # log info about baselines
            with open("D:\\Ekaterina_Voevodina\\memory_formation\\data\\stc_baseline_prestim.txt", "a+") as file_object:
                file_object.write("\n")
                file_object.write(
                        f'{subject_name}, {cond1} VS {cond2}, freq {freq_band}:\n'
                        f'{bl_hits.mean() = : .4f} +- {bl_hits.std() : .4f};\n'
                        f'{bl_miss.mean() = : .4f} +- {bl_miss.std() : .4f};\n'
                        f'{bl_hits_subset.mean() = : .4f} +- {bl_hits_subset.std() : .4f};\n'
                        f'{bl_all.mean() = : .4f} +- {bl_all.std() : .4f}.\n'
                        )

            # baselining of stc hits and miss and computing the difference
            print('Applying baseline...')
            stc_hits_av.data -= bl_hits # bl_all
            stc_miss_av.data -= bl_hits
            stc_diff_av = stc_hits_av - stc_miss_av
            print('ALL OK')

            # visualization of the source estimates
            # lims = [0.3, 0.45, 0.6]
            # kwargs = dict(src=src, subject=subject_name, subjects_dir=mris_dir,
            #             initial_time=None, verbose=True)

            # fig = stc_diff_av.plot(hemi='both', views=['dorsal', 'ventral'], clim=dict(kind='value', pos_lims=lims), **kwargs)


            # SAVING DATA
            # save averaged baselined source estimates
            stc_hits_path = os.path.join(source_prestim_bl_dir, f'stc_{cond1}_{freq_band}_{subject_name}')
            stc_miss_path = os.path.join(source_prestim_bl_dir, f'stc_{cond2}_{freq_band}_{subject_name}')
            stc_diff_path = os.path.join(source_prestim_bl_dir, f'stc_diff_{cond1}_VS_{cond2}_{freq_band}_{subject_name}')
            stc_hits_av.save(fname=stc_hits_path, ftype='h5', overwrite=True)
            stc_miss_av.save(fname=stc_miss_path, ftype='h5', overwrite=True)
            stc_diff_av.save(fname=stc_diff_path, ftype='h5', overwrite=True)