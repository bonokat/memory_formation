from utils.preprocessing import create_annotations, annotate_raw
import os
import mne


if __name__ == '__main__':
    root = '.\\'
    subjects_dir = os.path.join(root, 'data', 'subjects')
    for subject_name in os.listdir(subjects_dir):
        subject_dir = os.path.join(subjects_dir, subject_name)
        raw_dir = os.path.join(subject_dir, 'raw')
        for raw_name in os.listdir(raw_dir):
            if 'annotated' in raw_name:
                continue
            
            encoding = raw_name[:9]
            anno_name = encoding + '_tsss_mc_trans_BAD.txt'

            raw_path = os.path.join(raw_dir, raw_name)
            anno_dir = os.path.join(subject_dir, 'annotations')
            anno_path = os.path.join(anno_dir, anno_name)

            raw = mne.io.read_raw_fif(raw_path)
            stim_chs = raw.copy().pick_types(meg=False, stim=True).info['ch_names']
            # stim_chs = 'STI102'
            events = mne.find_events(raw, stim_chs, min_duration=.01)

            if len(raw.annotations) != 0:
                raw = annotate_raw(raw, events)
            else:
                with open(anno_path) as f:
                    lines = f.readlines()

                annotations = create_annotations(lines)
                raw.set_annotations(annotations)
                raw = annotate_raw(raw, events)

            raw.save(raw_path[:-4] + '_annotated.fif', overwrite=True)

