import mne
import os
import re


subjects_mri_dir = 'D:\\Ekaterina_Voevodina\\MRI_FS\\'
subject = 'Sub2_memform'

subject_mri_dir = os.path.join(subjects_mri_dir, 'Sub2_memform')

trans_path = os.path.join(subject_mri_dir,'sub2_trans')

subject_id = int(re.search(r'\d+', subject).group())
root = 'D:\\Ekaterina_Voevodina\\memory_formation\\'
subject_meg_dir = os.path.join(
    root, 'data', 'subjects', f'sub{subject_id}'
)
subject_epochs_dir = os.path.join(subject_meg_dir, 'epochs')
subject_epochs_path = os.path.join(subject_epochs_dir, os.listdir(subject_epochs_dir)[0])

epochs = mne.read_epochs(subject_epochs_path)
info = epochs.info

trans = mne.read_trans(trans_path)

src = mne.setup_source_space(
    subject, spacing='oct6', add_dist='patch',
    subjects_dir=subjects_mri_dir)

conductivity = (0.3,)  # for single layer

model = mne.make_bem_model(subject=subject, ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_mri_dir)
bem = mne.make_bem_solution(model)

fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem,
                                meg=True, eeg=False, mindist=5.0, n_jobs=None,
                                verbose=True)

data_cov = mne.compute_covariance(epochs, tmin=-.5, tmax=1.5,
                                  method='empirical')

empty_room_file =  mne.io.read_raw_fif('D:\\Ekaterina_Voevodina\\memory_formation\\data\\subjects\\empty_room_tsss.fif', preload=True)
empty_room_file = empty_room_file.resample(200)
empty_room_filt = empty_room_file.copy()\
    .filter(l_freq=.5, h_freq=90)\
    .notch_filter(50) # filter data
noise_cov = mne.compute_raw_covariance(
    empty_room_filt, tmin=0, tmax=None)

filters = mne.beamformer.make_lcmv(info, fwd, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='max-power',
                    weight_norm='unit-noise-gain', rank='info', reduce_rank=True)

stc = mne.beamformer.apply_lcmv_epochs(epochs, filters)

lims = [0.3, 0.45, 0.6]
kwargs = dict(src=src, subject=subject, subjects_dir=subjects_mri_dir,
              initial_time=None, verbose=True)

stc[0].plot(hemi='both', clim=dict(kind='value', pos_lims=lims), **kwargs)
