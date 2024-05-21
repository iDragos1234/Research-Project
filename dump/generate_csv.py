import glob
import os
import pandas as pd
import re
import sys

# python ./image_loader/generate_csv.py > dicom_files_metainfo.csv

# output a CSV list of input files from the OAI and CHECK datasets,
# listing DICOM and respective BoneFinder pointfiles

FILENAME_PATTERNS = {
    'OAI': re.compile('OAI-(?P<subject_id>[0-9]+)-(?P<visit>V[0-9]+)-[0-9]+.dcm'),
    'CHECK': re.compile('(?P<subject_id>[0-9]+)_(?P<visit>T[0-9]+)_APO.dcm'),
}

def parse_filename(filename):
    # Detect subject id and visit
    for dataset_name, filename_regex in FILENAME_PATTERNS.items():
        match = filename_regex.match(filename)
        if match:
            subject_id, visit = match.group('subject_id', 'visit')
            return dataset_name, subject_id, visit
    raise RuntimeError('Filename pattern not recognized.')


tasks = []
for dataset in ('CHECK', 'OAI'):

    for dicom_file_path in glob.glob(f'./data/{dataset}/*.dcm'):
        points_file_path = re.sub('/(CHECK|OAI)/', '/\\1-pointfiles/', dicom_file_path) + '.pts'
        if os.path.exists(points_file_path):
            dataset, subject_id, visit = parse_filename(os.path.basename(dicom_file_path))
            tasks.append({
                'dataset': dataset,
                'subject_id': f'{dataset}-{subject_id}',
                'visit': visit,
                'dicom': dicom_file_path,
                'points': points_file_path,
            })

sys.stdout.write(pd.DataFrame(tasks).to_csv(index=False))