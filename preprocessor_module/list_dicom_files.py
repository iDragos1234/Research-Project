'''
List the metadata of all DICOM files in the CHECK and OAI datasets.
Metadata consists of: 
    * .dcm DICOM file path, 
    * .dcm.pts points files path, 
    * dataset name (CHECK or OAI), 
    * subject (patient) ID,
    * subject (patient) visit.
'''
import platform, os, glob, re

import constants as ct
from dicom_transforms import PreprocessingException


DicomFilepath  = str
PointsFilepath = str
DatasetName    = str
SubjectID      = str
SubjectVisit   = str


DicomMetadata = tuple[
    DicomFilepath,
    PointsFilepath,
    DatasetName,
    SubjectID,
    SubjectVisit,
]


class ListDicomFiles:
    '''
    List the metadata of all DICOM files in the CHECK and OAI datasets.
    The DICOM file metadata consists of: 
        * .dcm DICOM filepath, 
        * .dcm.pts BoneFinder points files path, 
        * dataset name (CHECK or OAI), 
        * subject (patient) ID,
        * subject (patient) visit.

    NOTE: in the data folder, the files are grouped by dataset name and subject visit.

    NOTE: DICOM samples that do not have a corresponding points file are discarded.

    Attributes
    ----------
    data_dir_path (str): Path to the data directory
    '''
    data_dir_path: str

    def __init__(self, data_dir_path: str) -> None:
        self.data_dir_path = data_dir_path

    def __call__(self) -> list[DicomMetadata]:
        return ListDicomFiles._get_dicom_files_metadata(self.data_dir_path)
    
    @staticmethod
    def _get_dicom_files_metadata(data_dir_path: str) -> list[DicomMetadata]:
        '''
        Read the DICOM file metadata for all files under the given data directory path.

        Parameters
        ----------
        data_dir_path (str): The data directory path.
        
        Returns
        -------
        A list with the metadata of all DICOM files in the specified directory.
        '''
        # Verify that the specified directory path is pointing to an existing directory.
        if not os.path.isdir(data_dir_path):
            raise PreprocessingException(
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
                    # Discard DICOM samples raising errors.
                    except PreprocessingException as e:
                        print(e)

        return dicom_files_metadata

    @staticmethod
    def _get_dicom_file_metadata(
        dataset_name: str,
        data_dir_subject_visit: str,
        dicom_filepath: str,
    ) -> DicomMetadata:
        '''
        For the specified sample DICOM filepath, read the DICOM file metadata.

        Parameters
        ----------
        dataset_name (str): The dataset (CHECK or OAI) where the current sample belongs to.

        data_dir_subject_visit (str): The subject (patient) visit as specified by the subfolder name where the sample belongs to.

        dicom_filepath (str): DICOM filepath for the current sample.

        Returns
        -------
        DICOM file metadata (.dcm filepath, .pts filepath, dataset name, subject visit, subject ID) for the current sample.
        
        Raises
        ------
        PreprocessingException:
        '''
        # Normalize file path.
        dicom_filepath = os.path.normpath(dicom_filepath)

        # Verify that the specified filepath points to an existing file.
        if not os.path.isfile(dicom_filepath):
            raise PreprocessingException(
                f'There is no file at {dicom_filepath}.'
            )

        # Compute the corresponding BoneFinder points filepath.
        points_file_path: str
        if platform.system() == 'Windows':
            points_file_path = re.sub(
                f'\\\\({ct.Dataset.CHECK}|{ct.Dataset.OAI})\\\\',
                '\\\\\\1-pointfiles\\\\', 
                dicom_filepath,
            ) + '.pts'
        else:
            points_file_path = re.sub(
                f'/({ct.Dataset.CHECK}|{ct.Dataset.OAI})/',
                '/\\1-pointfiles/', 
                dicom_filepath,
            ) + '.pts'

        # Verify that the BoneFinder points filepath points to an existing file.
        if not os.path.isfile(points_file_path):
            raise PreprocessingException(
                f'There is no file at {points_file_path}.'
            )

        # Find the filename regex corresponding to the given dataset name.
        filename_regex: re.Pattern[str]
        if dataset_name == ct.Dataset.CHECK:
            filename_regex = ct.FilenameRegex.CHECK
        elif dataset_name == ct.Dataset.OAI:
            filename_regex = ct.FilenameRegex.OAI
        else:
            raise PreprocessingException(f'Unknown dataset name. Was: {dataset_name}.')

        # Extract DICOM filename from the filepath.
        dicom_file_name = os.path.basename(dicom_filepath)

        # Extract subject id and visit using the filename regex.
        match = filename_regex.match(dicom_file_name)
        if match:
            subject_id, subject_visit = match.group('subject_id', 'subject_visit')
            if data_dir_subject_visit != subject_visit:
                raise PreprocessingException(
                    f'Visit specified in directory name is different from visit specified in file name: {data_dir_subject_visit} =/= {subject_visit}.'
                )
            return dicom_filepath, points_file_path, dataset_name, subject_id, subject_visit
        raise PreprocessingException('Filename pattern not recognized.')
