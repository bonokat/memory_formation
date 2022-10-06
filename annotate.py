from utils.preprocessing import create_annotations, annotate_raw, create_events  # import our functions for annotations from the folder utils, file preprocessing
import os
import mne
import logging
import argparse

if __name__ == '__main__': # if we import fuctions from another file, we dont read them straigh away, 
                            # but when we initiate them in this script
    mne.set_log_level(logging.CRITICAL)
    root = '.\\' # start from the folder with annotate.py

    parser = argparse.ArgumentParser(description='Script to address subjects')
    parser.add_argument('-from', type=int, help='id of the first subject', default=1)
    parser.add_argument('-to', type=int, help='id of the last subject', default=None)
    subjects_dir = os.path.join(root, 'data', 'subjects') # path to the subjects directory
    sub_inds = [int(subject.replace('sub', '')) for subject in os.listdir(subjects_dir)]

    from_,\
        to = vars(parser.parse_args()).values()

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
        raw_dir = os.path.join(subject_dir, 'raw') #path to raw file in subj directory
        print('{0:64s}'.format(f'Reading {raw_dir} folder...'), end='')
        print('OK')
        for raw_name in os.listdir(raw_dir): #for each of raw enc files
            if 'annotated' in raw_name: #skip annotated files
                continue
            print(f'Raw file: {raw_name}')

            encoding = raw_name[:9] # give name to anno file enc 1 or 2
            anno_name = encoding + '_tsss_mc_trans_BAD.txt' # rest of the file name
            events_name = encoding + '_tsss_mc_trans_all_events.txt' # rest of the file name

            raw_path = os.path.join(raw_dir, raw_name) # go to each raw file
            anno_path = os.path.join(subject_dir, 'annotations', anno_name) # path to each anno file
            events_path = os.path.join(subject_dir, 'events', events_name)
            print('{0:64s}'.format('Reading raw-file ...', end=''))
            raw = mne.io.read_raw_fif(raw_path) # read raw file
            print('OK')
            print('{0:64s}'.format(f'Reading events from {events_name}...'), end='')
            onset = raw._cropped_samp/1000
            
            with open(events_path) as f:
                lines = f.readlines()
            print('OK')

            print('{0:64s}'.format('Creating events...'), end='')
            events, events_ids = create_events(lines) 
            events[:, 0] *= raw.info['sfreq']
            print('OK')

            print('{0:64s}'.format('Creating annotations...'), end='')
            if len(raw.annotations) != 0: # if we dont have annotations, annotate raw file
                raw = annotate_raw(raw, events)
            else:                         # if we have annotations, read file with annotations
                with open(anno_path) as f: # to close file automatically after reading
                    lines = f.readlines()

                annotations = create_annotations(lines, onset) # create annotations from txt file
                raw.set_annotations(annotations) 
                raw = annotate_raw(raw, events) # add annotations to raw file
            print('OK')

            print('Saving annotations...')
            raw.save(raw_path[:-4] + '_annotated.fif', overwrite=True) # save with a name _annotated instead of _BAD
            print(f'SUCCESSFULLY WRITTEN in {raw_dir}')

