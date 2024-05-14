"""
List the metadata of all DICOM files in the CHECK and OAI datasets.
Metadata consists of: 
    * .dcm DICOM file path, 
    * .dcm.pts points files path, 
    * dataset name (CHECK or OAI), 
    * subject id,
    * dubject visit.
"""
import os, glob, re


FILENAME_PATTERNS = {
    'OAI': re.compile('OAI-(?P<subject_id>[0-9]+)-(?P<subject_visit>V[0-9]+)-[0-9]+.dcm'),
    'CHECK': re.compile('(?P<subject_id>[0-9]+)_(?P<subject_visit>T[0-9]+)_APO.dcm'),
}


DicomMetaData = tuple[str, str, str, str, str]

class ListDicomFilesMetadata:

    def __call__(self, data_folder_path: str) -> list[DicomMetaData]:
        return ListDicomFilesMetadata._get_dicom_files_metadata(data_folder_path)

    @staticmethod
    def _get_dicom_file_metadata(
        dataset_name: str, 
        visit: str, 
        dicom_file_path: str
    ) -> DicomMetaData:

        # Normalize file path
        dicom_file_path = os.path.normpath(dicom_file_path)
        assert os.path.isfile(dicom_file_path), f'There is no file at {dicom_file_path}.'
        
        # Find the corresponding points file
        points_file_path = re.sub('\\\\(CHECK|OAI)\\\\', '\\\\\\1-pointfiles\\\\', dicom_file_path) + '.pts'
        assert os.path.isfile(points_file_path), f'There is no file at {points_file_path}.'

        # Detect subject id and visit
        filename_regex = FILENAME_PATTERNS[dataset_name]
        dicom_file_name = os.path.basename(dicom_file_path)
        match = filename_regex.match(dicom_file_name)
        if match:
            subject_id, subject_visit = match.group('subject_id', 'subject_visit')
            assert visit == subject_visit, \
                f'Visit specified in directory name different from visit specified in file name: {visit} =/= {subject_visit}.'
            return dicom_file_path, points_file_path, dataset_name, subject_id, subject_visit
        raise RuntimeError('Filename pattern not recognized.')

    @staticmethod
    def _get_dicom_files_metadata(data_folder_path: str) -> list[DicomMetaData]:
        assert os.path.isdir(data_folder_path), 'Given `data_folder_path` does not point to a directory.'
        
        dicom_files_metadata: list[DicomMetaData] = []
        for dataset_name in os.listdir(data_folder_path):
            for subject_visit in os.listdir(f'{data_folder_path}/{dataset_name}'):
                for dicom_file_path in glob.glob(f'{data_folder_path}/{dataset_name}/{subject_visit}/*.dcm'):
                    dicom_files_metadata.append(
                        ListDicomFilesMetadata._get_dicom_file_metadata(dataset_name, subject_visit, dicom_file_path)
                    )

        return dicom_files_metadata
