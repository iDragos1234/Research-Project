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

import constants as ct
import dicom_transforms as dt


DicomFilepath = str
PointsFilepath = str
DatasetName = str
SubjectID = str
SubjectVisit = str


DicomMetadata = tuple[
    DicomFilepath,
    PointsFilepath,
    DatasetName,
    SubjectID,
    SubjectVisit,
]


class ListDicomFiles:
    '''
    The DICOM file metadata consists of:
        - DICOM filepath,
        - BoneFinder points filepath,
        - Dataset name (CHECK or OAI),
        - Subject ID,
        - Subject visit.

    Note that the files are grouped by dataset 
    name and subject visit.

    Each DICOM file is expected to have a 
    corresponding BoneFinder points file.
    '''
    data_dir_path: str

    def __init__(self, data_dir_path: str) -> None:
        self.data_dir_path = data_dir_path

    def __call__(self) -> list[DicomMetadata]:
        return ListDicomFiles._get_dicom_files_metadata(self.data_dir_path)

    @staticmethod
    def _get_dicom_file_metadata(
        dataset_name: str, 
        data_dir_subject_visit: str, 
        dicom_file_path: str
    ) -> DicomMetadata:
        '''
        Get DICOM file metadata for a specified filepath.
        '''
        # Normalize file path.
        dicom_file_path = os.path.normpath(dicom_file_path)

        # Verify that the specified filepath points to an existing file.
        if not os.path.isfile(dicom_file_path):
            raise dt.PreprocessingException(
                f'There is no file at {dicom_file_path}.'
            )

        # Computes the corresponding BoneFinder points filepath.
        points_file_path = re.sub(
            f'\\\\({ct.Dataset.CHECK}|{ct.Dataset.OAI})\\\\',
            '\\\\\\1-pointfiles\\\\', dicom_file_path
        ) + '.pts'

        # Verify that the BoneFinder points filepath points to an existing file.
        if not os.path.isfile(points_file_path):
            raise dt.PreprocessingException(
                f'There is no file at {points_file_path}.'
            )

        # Find the filename regex corresponding to the given dataset name.
        filename_regex: re.Pattern[str]
        if dataset_name == ct.Dataset.CHECK:
            filename_regex = ct.FilenameRegex.CHECK
        elif dataset_name == ct.Dataset.OAI:
            filename_regex = ct.FilenameRegex.OAI
        else:
            raise dt.PreprocessingException(f'Unknown dataset name. Was: {dataset_name}.')

        # Extract DICOM filename from the filepath.
        dicom_file_name = os.path.basename(dicom_file_path)

        # Extract subject id and visit using the filename regex.
        match = filename_regex.match(dicom_file_name)
        if match:
            subject_id, subject_visit = match.group('subject_id', 'subject_visit')
            if data_dir_subject_visit != subject_visit:
                raise dt.PreprocessingException(
                    f'Visit specified in directory name is different from visit specified in file name: {data_dir_subject_visit} =/= {subject_visit}.'
                )
            return dicom_file_path, points_file_path, dataset_name, subject_id, subject_visit
        raise dt.PreprocessingException('Filename pattern not recognized.')

    @staticmethod
    def _get_dicom_files_metadata(data_dir_path: str) -> list[DicomMetadata]:
        '''
        Get the DICOM file metadata for all files located in the specified directory.

        Parameters
        ----------
        data_dir_path (str): The directory path where to look for DICOM files.

        Returns
        -------
        list[DicomMetadata]: A list of `DicomMetadata` for each DICOM file in the data directory.
        '''
        # Verify that the specified directory path is pointing to an existing directory.
        if not os.path.isdir(data_dir_path):
            raise dt.PreprocessingException(
                f'The path {data_dir_path} does not point to a directory.'
            )
        
        dicom_files_metadata: list[DicomMetadata] = []
        for dataset_name in ct.Dataset.items():
            for patient_visit in os.listdir(f'{data_dir_path}/{dataset_name}'):
                for dicom_file_path in glob.glob(f'{data_dir_path}/{dataset_name}/{patient_visit}/*.dcm'):
                    try:
                        dicom_files_metadata.append(
                            ListDicomFiles._get_dicom_file_metadata(
                                dataset_name,
                                patient_visit,
                                dicom_file_path,
                            )
                        )
                    except dt.PreprocessingException as e:
                        print(e)

        return dicom_files_metadata
