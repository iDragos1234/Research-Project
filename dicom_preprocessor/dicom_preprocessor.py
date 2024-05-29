'''
Preprocessor gateway.
How to run:
* open a bash terminal;
* execute the following command in the terminal:
```
python ./dicom_preprocessor/dicom_preprocessor.py \
    --input "./data" \
    --output "./output.h5" \
    --percentile-normalization 5 95 \
    --target-pixel-spacing 0.9 0.9 \
    --target-pixel-array-shape 512 512 \
    --verbose
```
'''
from typing import Union
import h5py, argparse, tqdm, time

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
    samples_limit: float
    verbose: bool

    def __init__(self,
        data_folder_path: str,
        hdf5_file_path: h5py.File,
        target_pixel_spacing: Union[tuple[float, float], None],
        target_pixel_array_shape: Union[tuple[int, int], None],
        percentile_normalization: Union[tuple[float, float], None],
        samples_limit: Union[float, None],
        verbose: bool,
    ) -> None:
        
        self.data_folder_path         = data_folder_path
        self.hdf5_file_path           = hdf5_file_path
        self.target_pixel_spacing     = target_pixel_spacing
        self.target_pixel_array_shape = target_pixel_array_shape
        self.percentile_normalization = percentile_normalization
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

    def __call__(self) -> None:
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

                dt.GetBoneFinderPoints(),
                dt.GetSegmentationMasks(),

                dt.PadSymmetrically(self.target_pixel_array_shape),
            ])

            for meta in tqdm.tqdm(dicom_files_metadata):
                # Un-wrap DICOM file metadata
                dicom_file_path, points_file_path, dataset, subject_id, subject_visit = meta

                # Create container of DICOM file data on which to apply the transforms
                dicom_container = dt.DicomContainer(
                    dicom_file_path          = dicom_file_path,
                    points_file_path         = points_file_path,
                    hdf5_file_object         = hdf5_file_object,
                    dataset                  = dataset,
                    subject_id               = subject_id,
                    subject_visit            = subject_visit,
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
                f'  - Ellapsed time:                {stats['ellapsed time']:.4f}s;\n'
                f'  - Total number of files:        {stats['total num files']};\n'
                f'  - Number of preprocessed files: {stats['num preprocessed files']};\n'
                f'  - Number of erronous files:     {stats['num erronous files']};'
            )
        return


def get_args() -> argparse.Namespace:
    '''
    Parse command-line arguments.

    Returns
    -------
    (argparse.Namespace): An object containing the parsed command line arguments. 
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        required=True,
        type=str,
        help='data folder path as input',
    )

    parser.add_argument(
        '--output',
        required=True,
        type=str,
        help='output HDF5 filepath',
    )

    parser.add_argument(
        '--percentile-normalization',
        nargs=2,
        type=float,
        default=None,
        help='two percentiles for the percentile intensity normalization; must be in the interval [0.0, 100.0]',
    )

    parser.add_argument(
        '--target-pixel-spacing',
        nargs=2,
        type=float,
        default=None,
        help='resample image to target spacing (mm/pixel)',
    )

    parser.add_argument(
        '--target-pixel-array-shape',
        nargs=2,
        type=int,
        default=None,
        help='resize image pixel array to target shape (#rows x #columns)',
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='limit the number of samples to be preprocessed',
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # Set preprocessor parameters
    preprocessor = DicomPreprocessor(
        data_folder_path         = args.input,
        hdf5_file_path           = args.output,
        target_pixel_spacing     = tuple(args.target_pixel_spacing) if args.target_pixel_spacing is not None else None,
        target_pixel_array_shape = tuple(args.target_pixel_array_shape) if args.target_pixel_array_shape is not None else None,
        percentile_normalization = tuple(args.percentile_normalization) if args.percentile_normalization is not None else None,
        samples_limit            = args.limit,
        verbose                  = args.verbose,
    )

    # Perform preprocessing
    preprocessor()
