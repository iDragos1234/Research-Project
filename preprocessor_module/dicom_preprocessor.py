from typing import Union
import h5py, tqdm, time

import list_dicom_files as ldf
import dicom_transforms as dt
import constants as ct


class Preprocessor:
    '''
    Preprocessor object. Preprocess all DICOM samples located in the specified data directory.
    Write the preprocessing output (metadata, images, segmentation masks) to the specified HDF5 file.

    NOTE: for each DICOM sample, the output consists of 2 images (unflipped and horizontally flipped) 
    and 2 segmentation masks (right and left side of the hip, with the latter horizontally flipped).

    Attributes
    ----------
    data_dir_path (str): 
        Path to the input data directory where the files with 
        the DICOM images and BoneFinder points are located.

    hdf5_filepath (h5py.File): 
        Path to the output HDF5 file containing all the 
        preprocessed images and segmentation masks.

    target_pixel_spacing (tuple[float, float], optional): 
        Isotropic spacing between pixels, expressed in mm.

    target_pixel_array_shape (tuple[int, int], optional): 
        Target image and mask shape after applying the padding transforms.

    percentile_normalization (tuple[float, float], optional):
        Percentile values used by the Pecentiles Normalization transform.

    include_background_mask (bool, optional): 
        Whether to include the background submask.

    verbose (bool): 
        Whether to print debugging messages.
    '''
    data_dir_path: str
    hdf5_filepath: h5py.File
    target_pixel_spacing: Union[tuple[float, float], None]
    target_pixel_array_shape: Union[tuple[int, int], None]
    percentile_normalization: Union[tuple[float, float], None]
    include_background_mask: bool
    verbose: bool

    def __init__(self,
        data_dir_path: str,
        hdf5_filepath: h5py.File,
        target_pixel_spacing: Union[tuple[float, float], None],
        target_pixel_array_shape: Union[tuple[int, int], None],
        percentile_normalization: Union[tuple[float, float], None],
        include_background_mask: bool,
        verbose: bool,
    ) -> None:
        
        self.data_dir_path            = data_dir_path
        self.hdf5_filepath            = hdf5_filepath
        self.target_pixel_spacing     = target_pixel_spacing
        self.target_pixel_array_shape = target_pixel_array_shape
        self.percentile_normalization = percentile_normalization
        self.include_background_mask  = include_background_mask
        self.verbose                  = verbose

        if self.verbose:
            print(
                f'Initializing DicomPreprocessor...:\n'
                f'  data_dir_path            = {self.data_dir_path},\n'
                f'  hdf5_file_path           = {self.hdf5_filepath},\n'
                f'  target_pixel_spacing     = {self.target_pixel_spacing}\n'
                f'  target_pixel_array_shape = {self.target_pixel_array_shape},\n'
                f'  percentile_normalization = {self.percentile_normalization},\n'
                f'  include_background_mask  = {self.include_background_mask}.\n'
            )
        return

    def preprocess(self) -> None:
        if self.verbose:
            print('Starting preprocessing...')

        # Keep track of certain stats during preprocessing.
        stats = {
            'start time': time.time(),
            'ellapsed time': 0,
            'total num files': 0,
            'num preprocessed files': 0,
            'num erronous files': 0,
        }

        # List the metadata of all DICOM files
        dicom_files_metadata = ldf.ListDicomFiles(self.data_dir_path)()
        stats['total num files'] = len(dicom_files_metadata)

        # Preprocess the DICOM data directory given as input 
        # and output the result to the specified HDF5 file.
        with h5py.File(self.hdf5_filepath, 'w') as hdf5_file_object:
            
            # Common base of transforms for each image and right/left mask.
            dicom_transforms_base = dt.CombineTransforms([
                # Load DICOM file data.
                dt.LoadDicomFileObject(),
                dt.ReadPixelArray(),
                dt.ReadSourcePixelSpacing(),

                # Check DICOM file attributes.
                dt.CheckPhotometricInterpretation(),
                dt.CheckVoilutFunction(),

                # Image processing.
                dt.RescaleToTargetPixelSpacing(self.target_pixel_spacing),
                dt.PercentilesIntensityNormalization(self.percentile_normalization),
                dt.MinMaxIntensityNormalization(),
                dt.PadSymmetrically(self.target_pixel_array_shape),

                # Generate right/left segmentation masks.
                dt.ReadBoneFinderPoints(),
                dt.GetSegmentationMasks(self.include_background_mask),
            ])

            for file_meta in tqdm.tqdm(dicom_files_metadata):
                # Un-wrap DICOM file metadata.
                (
                    dicom_file_path,
                    points_file_path,
                    dataset,
                    subject_id,
                    subject_visit,
                ) = \
                    file_meta

                # Create container for DICOM file data 
                # and apply the transforms on this container.
                dicom_container = dt.DicomContainer(
                    dicom_filepath   = dicom_file_path,
                    points_filepath  = points_file_path,
                    dataset          = dataset,
                    subject_id       = subject_id,
                    subject_visit    = subject_visit,
                    hdf5_file_object = hdf5_file_object,
                )

                try:
                    # Apply the transforms base.
                    dicom_transforms_base(dicom_container)

                    # Write the un-flipped image and the mask
                    # for segmenting the right hip to the HDF5 file.
                    dt.WriteToHDF5(hip_side=ct.HipSide.RIGHT)(dicom_container)

                    # Flip the image and the segmentation masks.
                    dt.Flip(axis=-1)(dicom_container)

                    # Write the flipped image and the flipped mask
                    # for segmenting the left hip to the HDF5 file.
                    dt.WriteToHDF5(hip_side=ct.HipSide.LEFT)(dicom_container)

                    stats['num preprocessed files'] += 1

                # Discard samples that raise errors.
                except dt.PreprocessingException as e:
                    if self.verbose:
                        print(e)

            hdf5_file_object.close()
        
        # Print stats.
        stats['ellapsed time'] = time.time() - stats['start time']
        stats['num erronous files'] = stats['total num files'] - stats['num preprocessed files']
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
