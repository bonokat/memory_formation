import argparse
import os

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Simple script to print your name in terminal')
#     parser.add_argument('--name', type=str, help='Name to print', default='Katya')
#     parser.add_argument('-id', '--identificator', type=int, help='Identificator of the run')
#     parser.add_argument(
#         '--subjects', type=int, nargs='+',
#         help='Subjects numbers to execute script',
#         default=range(10)
#     )
#     parser.add_argument('--save', action='store_true')
#     parser.add_argument(
#         '-dt', '--dtypes', type=str, nargs='+',
#         help='Subjects numbers to execute script',
#         default=['int', 'float']
#     )

#     name,\
#         id,\
#         nsubjects,\
#         save,\
#         datat = vars(parser.parse_args()).values()

#     print(f'Hello {name}!')
#     if id is not None:
#         print(f'Current id is {id}')
#     for s in nsubjects:
#         print(s, end=', ')
#     print()
#     if save:
#         print('I\'m saving it')
#     else:
#         print('I will not save it')
#     print(
#         f'Considering datatypes: {", ".join(datat)}'
#     )

if __name__ == "__main__":
    root = './'
    parser = argparse.ArgumentParser(description='Script to address subjects')
    
    parser.add_argument('-from', type=int, help='id of the first subject', default=1)
    parser.add_argument('-to', type=int, help='id of the last subject', default=None)
    subjects_dir = os.path.join(root, 'data', 'subjects')
    sub_inds = [int(subject.replace('sub', '')) for subject in os.listdir(subjects_dir)]
    
    from_,\
        to = vars(parser.parse_args()).values()

    if to is None:
        to = max(sub_inds)

    subjects_range = range(from_, to + 1)

    for sub_ind, subject_name in zip(sub_inds, os.listdir(subjects_dir)):

        if sub_ind not in subjects_range:
            continue

        print(subject_name)

        subject_dir = os.path.join(subjects_dir, subject_name)
        raw_dir = os.path.join(subject_dir, 'raw')