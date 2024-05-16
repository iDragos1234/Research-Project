"""
Code adapted from `example-preprocessing-code/convert_for_localization.py`.

How to run: execute the following command in the bash terminal:
```
python ./dicom_preprocessor/dicom_preprocessor.py --input "./data" --output "output.h5" --target-pixel-spacing 1
```
"""
import h5py, argparse, tqdm

from list_dicom_files_metadata import ListDicomFilesMetadata
import dicom_transformations as dt


class DicomPreprocessor:

    data_folder_path: str
    hdf5_file_path: h5py.File
    limit: float
    target_pixel_spacing: float

    def __init__(self,
        data_folder_path: str,
        hdf5_file_path: h5py.File,
        target_pixel_spacing: float,
        limit: float=None,
    ) -> None:
        self.data_folder_path     = data_folder_path
        self.hdf5_file_path       = hdf5_file_path
        self.target_pixel_spacing = target_pixel_spacing
        self.limit                = limit
        return

    def __call__(self) -> None:
        data_folder_path     = self.data_folder_path
        hdf5_file_path       = self.hdf5_file_path
        target_pixel_spacing = self.target_pixel_spacing

        # List the metadata of all DICOM files
        dicom_files_metadata = ListDicomFilesMetadata()(data_folder_path)

        # If `limit` is specified, take the first `limit` number of elements
        # If `limit` is `None`, all files are considered
        dicom_files_metadata = dicom_files_metadata[:self.limit]

        # Preprocess the DICOM data folder given as input 
        # and output the result to the specified HDF5 file
        with h5py.File(hdf5_file_path, 'w') as hdf5_file_object:
            
            # DICOM transforms: preprocess each DICOM file and write it to the hdf5 file
            dicom_transformations_base = dt.SequenceTransformations([
                dt.LoadDicomObject(),
                dt.GetPixelArray(),
                dt.GetSourcePixelSpacing(),
                dt.CheckPhotometricInterpretation(),
                dt.CheckVoilutFunction(),
                dt.ResampleToTargetResolution(),
                dt.NormalizeIntensities(),
                dt.GetBoneFinderPoints(),
                dt.GetSegmentationMasks(),
            ])

            for meta in tqdm.tqdm(dicom_files_metadata):
                # Un-wrap DICOM file metadata
                dicom_file_path, points_file_path, dataset, subject_id, subject_visit = meta

                # Create container of DICOM file data on which to apply the transforms
                dicom_container = dt.DicomContainer(
                    dicom_file_path=dicom_file_path,
                    points_file_path=points_file_path,
                    hdf5_file_object=hdf5_file_object,
                    dataset=dataset,
                    subject_id=subject_id,
                    subject_visit=subject_visit,
                    target_pixel_spacing=target_pixel_spacing,
                )

                try:
                    # Apply the transformations base
                    dicom_transformations_base(dicom_container)

                    # Write the unflipped image and the right hip mask to the HDF5 file
                    dt.AppendDicomToHDF5()(dicom_container)

                    # Flip the image and the segmentation masks
                    dt.FlipHorizontally()(dicom_container)
                    # Write the flipped image and the flipped left hip mask to the HDF5 file
                    dt.AppendDicomToHDF5()(dicom_container)
                except dt.PreprocessingException as e:
                    print(e)

            hdf5_file_object.close()
        return


if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', metavar='DIR', required=True, type=str,
                        help='data folder path as input',)
    parser.add_argument('--output', metavar='HDF5', required=True, type=str,
                        help='output HDF5 file',)
    parser.add_argument('--target-pixel-spacing', metavar='SPACING', type=float,
                        help='resample image to target spacing (mm/pixel)',)
    parser.add_argument('--limit', metavar='LIMIT', type=int,
                        help='limit the number of DICOM files to be preprocessed',)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Set preprocessing parameters
    preprocessor = DicomPreprocessor(
        data_folder_path=args.input,
        hdf5_file_path=args.output,
        target_pixel_spacing=args.target_pixel_spacing,
        limit=args.limit,
    )

    # Perform preprocessing
    preprocessor()
