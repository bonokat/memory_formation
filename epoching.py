import os
import argparse
from utils import check_paths

import mne
from utils.preprocessing import read_events
import matplotlib.pyplot as plt
import numpy as np
from typing import *


if __name__ == '__main__': # if we import fuctions from another file, we dont read them straigh away,
                            # but when we initiate them in this script
    root = '.\\'

    parser = argparse.ArgumentParser(description='Script to address subjects')
    parser.add_argument('-from', type=int, help='id of the first subject', default=1)
    parser.add_argument('-to', type=int, help='id of the last subject', default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('-tmin', type=float, help='time to start epoch', default=-1.5)
    parser.add_argument('-tmax', type=float, help='time to end epoch', default=4)
    parser.add_argument(
        '--baseline', '-bl', type=float, nargs='+',
        help='baseline interval', default=None
    )
    parser.add_argument(
        '-sf', '--sfreq', type=float, default=None,
        help='Set sampling frequency for resampled file'
    )


    subjects_dir = os.path.join(root, 'data', 'subjects')
    # in the list of subs folders leave only numbers of subs as integers in list sub_inds
    sub_inds = [int(subject.replace('sub', '')) for subject in os.listdir(subjects_dir)]

    # vars creates a dictionary with parse arguments as values
    from_,\
        to,\
        visualize,\
        tmin,\
        tmax,\
        bl,\
        sfreq = vars(parser.parse_args()).values()
    bl = tuple(bl) if bl is not None else tuple()

    if to is None:
        to = max(sub_inds)

    subjects_range = range(from_, to + 1)

    # zip pairs sub index + sub name (e.g. 2 corresponds to sub2)
    for sub_ind, subject_name in zip(sub_inds, os.listdir(subjects_dir)):
        if sub_ind not in subjects_range:
            continue

        print(f'Subject name: {subject_name}')
        print('{0:64s}'.format(f'Reading {subject_name} folder...'), end='')
        subject_dir = os.path.join(subjects_dir, subject_name)
        print('OK')
        epochs_dir = os.path.join(subject_dir, 'epochs')
        evoked_dir = os.path.join(subject_dir, 'evoked')
        pics_dir = os.path.join(subject_dir, 'pics')
        evoked_pics_dir = os.path.join(pics_dir, 'evoked')
        # check if the folders exist. if not, create them
        check_paths(epochs_dir, evoked_dir, pics_dir, evoked_pics_dir)

        raw_dir = os.path.join(subject_dir, 'raw')
        print('{0:64s}'.format(f'Reading {raw_dir} folder...'), end='')
        print('OK')

        epochs_list = list()
        for raw_name in os.listdir(raw_dir): #for each of raw enc files
            if not '_reconstructed' in raw_name: #take only ica files
                continue

            print(f'Raw file: {raw_name}')

            encoding = raw_name[:9]
            events_name = encoding + '_tsss_mc_trans_all_events.txt'
            events_path = os.path.join(subject_dir, 'events', events_name)

            raw = mne.io.read_raw_fif(os.path.join(raw_dir, raw_name))
            events, event_ids = read_events(raw, events_path)

            event_ids_mod = dict(
                filter(
                    lambda item: len(item[0].split('/')) == 4,
                    event_ids.items()
                )
            )

            selected_event_ids = event_ids_mod.values()

            events_selected = np.array(list(filter(
                lambda row: row[2] in selected_event_ids,
                events
            )))

            epochs_list.append(mne.Epochs(
                raw,
                events_selected,
                event_ids_mod,
                tmin=tmin, tmax=tmax,
                baseline=bl,
                reject_by_annotation=True
            ))

        epochs = mne.concatenate_epochs(epochs_list)

        if sfreq is not None:
            epochs = epochs.resample(sfreq)

        del epochs_list, raw
        evokeds = epochs.load_data()\
            .copy()\
            .pick_types(meg='grad')\
            .apply_baseline(
                baseline=bl if bl else (None, 0)
            )\
            .average(by_event_type=True)

        for evoked in evokeds:
            evo_name = evoked.comment.replace("/", "_")
            fig = evoked.plot(show=False)
            fig.savefig(
                os.path.join(
                    evoked_pics_dir,
                    f'{evo_name}_bl_{"".join(str(bl)) if bl else "None"}.png'
                ),
                dpi=300
            )

            if visualize:
                plt.show()
            plt.close()

            evoked.save(
                os.path.join(
                    evoked_dir,
                    f'enc_evoked_{evo_name}.fif'
                ),
                overwrite=True
            )

        epochs.save(
            os.path.join(
                epochs_dir,
                'enc_epochs.fif'
            ),
            overwrite=True
        )
