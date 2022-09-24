# ICA pipeline
from utils import check_paths
import mne
import os
import logging
import argparse
import matplotlib.pyplot as plt
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)

if __name__ == '__main__': # if we import fuctions from another file, we dont read them straigh away, 
                            # but when we initiate them in this script
    # mne.set_log_level(logging.CRITICAL)
    root = '.\\' # start from the folder with annotate.py

    parser = argparse.ArgumentParser(description='Script to address subjects')
    parser.add_argument('-from', type=int, help='id of the first subject', default=1)
    parser.add_argument('-to', type=int, help='id of the last subject', default=None)
    parser.add_argument('--visualize', action='store_true')

    subjects_dir = os.path.join(root, 'data', 'subjects') # path to the subjects directory
    sub_inds = [int(subject.replace('sub', '')) for subject in os.listdir(subjects_dir)]

    from_,\
        to,\
        visualize = vars(parser.parse_args()).values()
    if to is None:
        to = max(sub_inds)
    subjects_range = range(from_, to + 1)

    for sub_ind, subject_name in zip(sub_inds, os.listdir(subjects_dir)):
        if sub_ind not in subjects_range:
            continue
    
        # for subject_name in os.listdir(subjects_dir): # for every subj in the directory
        print(f'Subject name: {subject_name}')
        print('{0:64s}'.format(f'Reading {subject_name} folder...'), end='')
        subject_dir = os.path.join(subjects_dir, subject_name) # path to each subj one by one
        print('OK')
        ica_dir = os.path.join(subject_dir, 'ica')
        check_paths(ica_dir)
        anno_raw_dir = os.path.join(subject_dir, 'raw') #path to raw file in subj directory
        print('{0:64s}'.format(f'Reading {anno_raw_dir} folder...'), end='')
        print('OK')
        for anno_raw_name in os.listdir(anno_raw_dir): #for each of raw enc files
            if not 'annotated' in anno_raw_name: #skip annotated files
                print(f'Annotated file for subject {subject_name} not found') 
                continue
            print(f'Raw file: {anno_raw_name}')
            
            encoding = anno_raw_name[:9] # give name to anno file enc 1 or 2 
            
            anno_raw_path = os.path.join(anno_raw_dir, anno_raw_name) # go to each raw file
            anno_raw = mne.io.read_raw_fif(anno_raw_path, preload=True) # read raw file
            

            filt_raw = anno_raw.copy()\
                .filter(l_freq=.5, h_freq=90)\
                .notch_filter(50) # filter data

            ica = ICA(n_components=20)
            ica.fit(filt_raw) #ica decomposition on raw filtered data

            ica.exclude = []
            # find which ICs match the ECG pattern
            ecg_indices, ecg_scores = ica.find_bads_ecg(anno_raw, ch_name='ECG063')
            ecg_indices
            # find which ICs match the EOG pattern
            eog_indices, eog_scores = ica.find_bads_eog(anno_raw, ch_name=['EOG061', 'EOG062'])
            eog_indices

            # ica.plot_sources(high_filt_raw)

            ica.exclude = eog_indices + ecg_indices

            #create a copy of raw file and apply ica to it
            reconst_raw = filt_raw.copy()
            ica.apply(reconst_raw)

            if visualize:
                filt_raw.plot()
                plt.show()
                reconst_raw.plot()
                plt.show()

            # save new raw file and ica components file
            reconst_raw.save(
                os.path.join(
                    anno_raw_dir,
                    f'{encoding}_tsss_mc_trans_annotated_filtered_reconstructed.fif'
                ),
                overwrite=True
            ) # save with a name _annotated instead of _BAD
            ica.save(
                os.path.join(
                    ica_dir,
                    f'{encoding}_ica.fif'
                ),
                overwrite=True
            )