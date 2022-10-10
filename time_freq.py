import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_morlet, tfr_multitaper
from utils import check_paths
from utils.visualize import plot_tfr


if __name__ == '__main__':

    root = '.\\'

    parser = argparse.ArgumentParser(description='Script to address subjects')
    parser.add_argument('-from', type=int, help='id of the first subject', default=1)
    parser.add_argument('-to', type=int, help='id of the last subject', default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('-tmin', type=float, help='time to start epoch', default=-1.5)
    parser.add_argument('-tmax', type=float, help='time to end epoch', default=4)
    parser.add_argument('-fmin', type=float, help='minimum frequency', default=3.)
    parser.add_argument('-fmax', type=float, help='maximum frequency', default=90.)
    parser.add_argument(
        '--baseline', '-bl', type=float, nargs='+',
        help='baseline interval', default=(None, 0)
    )
    parser.add_argument('-m', '--method', type=str, help='morlet or multitaper', default='morlet')
    parser.add_argument('-n-tapes', type=int, help='num of tapes for multitaper', default=4)

    subjects_dir = os.path.join(root, 'data', 'subjects')
    sub_inds = [int(subject.replace('sub', '')) for subject in os.listdir(subjects_dir)]

    from_,\
        to,\
        visualize,\
        tmin,\
        tmax,\
        fmin,\
        fmax,\
        bl,\
        m,\
        n_tapes = vars(parser.parse_args()).values()
    bl = tuple(bl) if bl is not None else tuple()

    if to is None:
        to = max(sub_inds)

    subjects_range = range(from_, to + 1)

    for sub_ind, subject_name in zip(sub_inds, os.listdir(subjects_dir)):
        if sub_ind not in subjects_range:
            continue

        print(f'Subject name: {subject_name}')
        print('{0:64s}'.format(f'Reading {subject_name} folder...'), end='')
        subject_dir = os.path.join(subjects_dir, subject_name)
        print('OK')
        epochs_dir = os.path.join(subject_dir, 'epochs')
        tfa_dir = os.path.join(subject_dir, 'tfa')
        pics_dir = os.path.join(subject_dir, 'pics')
        tfa_pics_dir = os.path.join(pics_dir, 'tfa')
        psd_pics_dir = os.path.join(pics_dir, 'psd')
        check_paths(tfa_dir, pics_dir, tfa_pics_dir, psd_pics_dir)

        for epochs_file in os.listdir(epochs_dir):
            print(f'Epochs file: {epochs_file}')
            epochs_path = os.path.join(epochs_dir, epochs_file)
            epochs = mne.read_epochs(epochs_path, preload=True)
            epochs.pick_types(meg='grad')

            cond_list = ['hits', 'miss']
            freqs = np.logspace(*np.log10([fmin, fmax]), num=15)
            n_cycles = 5

            for cond in cond_list:

                fig_psd_epo = epochs[cond].plot_psd(
                    fmin=fmin, fmax=fmax,
                    spatial_colors=True, show=visualize
                )

                fig_psd_epo.savefig(
                    os.path.join(
                        psd_pics_dir,
                        f'enc_psd_{cond}.png'
                    ),
                    dpi=300
                )

                if m == 'morlet':
                    print('Computing morlet transform...')
                    power = tfr_morlet(
                        epochs[cond], freqs=freqs, n_cycles=n_cycles,
                        use_fft=True, return_itc=False,
                        decim=3, n_jobs=5
                    )
                elif m == 'multitaper':
                    print('Computing multitaper transform...')
                    power = tfr_multitaper(
                        epochs['hits'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        decim=3, n_jobs=5, return_itc=False, time_bandwidth=n_tapes + 1
                    )
                else:
                    NotImplementedError(f'Input method {m} does not exist.')

                power.save(
                    os.path.join(
                        tfa_dir,
                        f'tfa_power_{m}_{cond}.h5'
                    ),
                    overwrite=True
                )

                power_plot = plot_tfr(
                    power.copy().apply_baseline(bl)
                )

                if visualize:
                    plt.show()

                power_plot.savefig(
                    os.path.join(
                        tfa_pics_dir,
                        f'tfa_power_{m}_{cond}_bl_{"".join(str(bl)) if bl else "None"}.png'
                    ),
                    dpi=300
                )
