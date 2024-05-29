'''
List the metadata of all DICOM files in the CHECK and OAI datasets.
Metadata consists of: 
    * .dcm DICOM file path, 
    * .dcm.pts points files path, 
    * dataset name (CHECK or OAI), 
    * subject id,
    * dubject visit.
'''
import os, glob, re
from typing import Literal

import constants as ct
import dicom_transformations as dt


DicomMetaData = tuple[str, str, str, str, str]

class ListDicomFiles:

    def __call__(self, data_folder_path: str) -> list[DicomMetaData]:
        return ListDicomFiles._get_dicom_files_metadata(data_folder_path)

    @staticmethod
    def _get_dicom_file_metadata(
        dataset_name: str, 
        visit: str, 
        dicom_file_path: str
    ) -> DicomMetaData:

        # Normalize file path
        dicom_file_path = os.path.normpath(dicom_file_path)
        if not os.path.isfile(dicom_file_path):
            raise dt.PreprocessingException(
                f'There is no file at {dicom_file_path}.'
            )

        # Find the corresponding points file
        points_file_path = re.sub(
            f'\\\\({ct.Dataset.CHECK}|{ct.Dataset.OAI})\\\\',
            '\\\\\\1-pointfiles\\\\', dicom_file_path
        ) + '.pts'

        if not os.path.isfile(points_file_path):
            raise dt.PreprocessingException(
                f'There is no file at {points_file_path}.'
            )

        # Find the filename regex corresponding to the given dataset name
        filename_regex: re.Pattern[str]
        if dataset_name == ct.Dataset.CHECK:
            filename_regex = ct.FilenameRegex.CHECK
        elif dataset_name == ct.Dataset.OAI:
            filename_regex = ct.FilenameRegex.OAI
        else:
            raise dt.PreprocessingException(f'Unknown dataset name. Was: {dataset_name}.')
        
        # Detect subject id and visit
        dicom_file_name = os.path.basename(dicom_file_path)
        match = filename_regex.match(dicom_file_name)
        if match:
            subject_id, subject_visit = match.group('subject_id', 'subject_visit')
            if visit != subject_visit:
                raise dt.PreprocessingException(
                    f'Visit specified in directory name is different from visit specified in file name: {visit} =/= {subject_visit}.'
                )
            return dicom_file_path, points_file_path, dataset_name, subject_id, subject_visit
        raise dt.PreprocessingException('Filename pattern not recognized.')

    @staticmethod
    def _get_dicom_files_metadata(data_folder_path: str) -> list[DicomMetaData]:
        if not os.path.isdir(data_folder_path):
            raise dt.PreprocessingException(
                f'The path {data_folder_path} does not point to a directory.'
            )
        
        dicom_files_metadata: list[DicomMetaData] = []
        for dataset_name in ct.Dataset.items():
            for subject_visit in os.listdir(f'{data_folder_path}/{dataset_name}'):
                for dicom_file_path in glob.glob(f'{data_folder_path}/{dataset_name}/{subject_visit}/*.dcm'):
                    try:
                        dicom_files_metadata.append(
                            ListDicomFiles._get_dicom_file_metadata(dataset_name, subject_visit, dicom_file_path)
                        )
                    except dt.PreprocessingException as e:
                        print(e)

        return dicom_files_metadata
