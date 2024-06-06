'''
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
import argparse

import dicom_preprocessor as dp


def main():
    args = get_args()

    # Set preprocessor parameters
    preprocessor = dp.DicomPreprocessor(
        data_folder_path         = args.input_dir,
        hdf5_file_path           = args.output_hdf5,
        target_pixel_spacing     = tuple(args.target_pixel_spacing)     if args.target_pixel_spacing     is not None else None,
        target_pixel_array_shape = tuple(args.target_pixel_array_shape) if args.target_pixel_array_shape is not None else None,
        percentile_normalization = tuple(args.percentile_normalization) if args.percentile_normalization is not None else None,
        include_background_mask  = args.include_background_mask,
        samples_limit            = args.limit,
        verbose                  = args.verbose,
    )

    preprocessor.preprocess()


def get_args() -> argparse.Namespace:
    '''
    Parse command-line arguments.

    Returns
    -------
    (argparse.Namespace): An object containing the parsed command line arguments. 
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-dir',
        required=True,
        type=str,
        help='data directory path as input',
    )
    parser.add_argument(
        '--output-hdf5',
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
        '--include-background-mask',
        action='store_true',
        help='whether to include the background binary mask in the combined mask',
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
    main()
