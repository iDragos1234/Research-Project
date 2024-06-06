from typing import Union
import h5py, tqdm, time

import list_dicom_files
import dicom_transformations as dt
import constants as ct


class DicomPreprocessor:
    '''
    Preprocessor main gateway.

    Attributes
    ----------
    data_folder_path (str): Path to the input data - a folder where 
        the files with the DICOM images and BoneFinder points reside.

    hdf5_file_path (h5py.File): Path to the output file - a file containing
        all the preprocessed images and segmentation masks, intended for model training.

    target_pixel_spacing (tuple[float, float], optional): Spacing between pixels, expressed in mm.

    target_pixel_array_shape (tuple[int, int], optional): Desired image and mask shape
        after applying the `monai.transforms.ResizeWithPadOrCrop` transformation.
        If `None`, no resizing operation is applied.

    samples_limit (float, optional): Limit the number of data 
        samples (images and masks) that are preprocessed. 
        If `None`, all samples are considered.

    verbose (bool): Whether the preprocessor prints debugging messages.
    '''
    data_folder_path: str
    hdf5_file_path: h5py.File
    target_pixel_spacing: Union[tuple[float, float], None]
    target_pixel_array_shape: Union[tuple[int, int], None]
    percentile_normalization: Union[tuple[float, float], None]
    include_background_mask: bool
    samples_limit: Union[float, None]
    verbose: bool

    def __init__(self,
        data_folder_path: str,
        hdf5_file_path: h5py.File,
        target_pixel_spacing: Union[tuple[float, float], None],
        target_pixel_array_shape: Union[tuple[int, int], None],
        percentile_normalization: Union[tuple[float, float], None],
        include_background_mask: bool,
        samples_limit: Union[float, None],
        verbose: bool,
    ) -> None:
        
        self.data_folder_path         = data_folder_path
        self.hdf5_file_path           = hdf5_file_path
        self.target_pixel_spacing     = target_pixel_spacing
        self.target_pixel_array_shape = target_pixel_array_shape
        self.percentile_normalization = percentile_normalization
        self.include_background_mask  = include_background_mask
        self.samples_limit            = samples_limit
        self.verbose                  = verbose

        if self.verbose:
            print(
                f'Initializing DicomPreprocessor...:\n'
                f'  data_folder_path         = {self.data_folder_path}\n'
                f'  hdf5_file_path           = {self.hdf5_file_path}\n'
                f'  target_pixel_spacing     = {self.target_pixel_spacing}\n'
                f'  target_pixel_array_shape = {self.target_pixel_array_shape}\n'
                f'  samples_limit            = {self.samples_limit}'
            )
        return

    def preprocess(self) -> None:
        if self.verbose:
            print('Starting preprocessing...')

        # Keep track of certain stats during preprocessing.
        stats = {
            'start time': time.time(),
            'ellapsed time': None,
            'num erronous files': 0,
            'num preprocessed files': 0,
            'total num files': None
        }

        # List the metadata of all DICOM files
        dicom_files_metadata = list_dicom_files.ListDicomFiles(self.data_folder_path)()
        stats['total num files'] = len(dicom_files_metadata)

        # If `samples_limit` is specified, take the first `samples_limit` number of files.
        # If `samples_limit` is `None`, all files are considered.
        dicom_files_metadata = dicom_files_metadata[:self.samples_limit]

        # Preprocess the DICOM data folder given as input 
        # and output the result to the specified HDF5 file.
        with h5py.File(self.hdf5_file_path, 'w') as hdf5_file_object:
            
            # DICOM transforms:
            # preprocess each input DICOM file
            # and write it to the output HDF5 file.
            dicom_transformations_base = dt.CombineTransformations([
                dt.LoadDicomObject(),
                dt.GetPixelArray(),
                dt.GetSourcePixelSpacing(),
                dt.CheckPhotometricInterpretation(),
                dt.CheckVoilutFunction(),

                dt.RescaleToTargetPixelSpacing(self.target_pixel_spacing),

                dt.PercentilesIntensityNormalization(self.percentile_normalization),
                dt.MinMaxIntensityNormalization(),

                dt.PadSymmetrically(self.target_pixel_array_shape),

                dt.GetBoneFinderPoints(),
                dt.GetSegmentationMasks(self.include_background_mask),
            ])

            for meta in tqdm.tqdm(dicom_files_metadata):
                # Un-wrap DICOM file metadata
                dicom_file_path, points_file_path, dataset, subject_id, subject_visit = meta

                # Create container of DICOM file data on which to apply the transforms
                dicom_container = dt.DicomContainer(
                    dicom_filepath   = dicom_file_path,
                    points_filepath  = points_file_path,
                    hdf5_file_object = hdf5_file_object,
                    dataset          = dataset,
                    subject_id       = subject_id,
                    subject_visit    = subject_visit,
                )

                try:
                    # Apply the transformations base.
                    dicom_transformations_base(dicom_container)

                    # Write the unflipped image and the mask
                    # for segmenting the right hip to the HDF5 file.
                    dt.AppendDicomToHDF5(hip_side=ct.HipSide.RIGHT)(dicom_container)

                    # Flip the image and the segmentation masks.
                    dt.Flip(axis=-1)(dicom_container)

                    # Write the flipped image and the flipped mask 
                    # for segmenting the left hip to the HDF5 file.
                    dt.AppendDicomToHDF5(hip_side=ct.HipSide.LEFT)(dicom_container)

                    stats['num preprocessed files'] += 1

                except dt.PreprocessingException as e:
                    stats['num erronous files'] += 1
                    if self.verbose:
                        print(e)

            hdf5_file_object.close()
        
        stats['ellapsed time'] = time.time() - stats['start time']
        if self.verbose:
            print(
                f'Finished preprocessing.\n'
                f'Stats:\n'
                f'  - Ellapsed time:                {stats["ellapsed time"]:.4f}s;\n'
                f'  - Total number of files:        {stats["total num files"]};\n'
                f'  - Number of preprocessed files: {stats["num preprocessed files"]};\n'
                f'  - Number of erronous files:     {stats["num erronous files"]};'
            )
        return
