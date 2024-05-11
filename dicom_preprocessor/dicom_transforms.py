"""
Transforms for the DICOM files.
Code adapted from `example-preprocessing-code/dicom_util.py`.
"""
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from pydicom import dcmread, FileDataset 
from pydicom.dicomdir import DicomDir
from skimage.transform import rescale
import hdf5plugin
import h5py


class Container:
    ...

class Transform:
    
    def __init__(self) -> None:
        ...

    def __call__(self, container: Container):
        ...


type DicomFileObject = FileDataset | DicomDir


class DicomContainer(Container):
    dicom_file_path: str
    points_file_path: str
    hdf5_file_object: h5py.File

    dataset: Literal['CHECK', 'OAI']
    subject_id: str
    subject_visit: str
    
    dicom_file_object: DicomFileObject
    pixel_array: np.ndarray
    
    source_pixel_spacing: list[float]
    target_pixel_spacing: float
    pixel_spacing: list[float]

    intensity_offset: float
    intensity_slope: float

    def __init__(self,
        dicom_file_path: str,
        points_file_path: str,
        hdf5_file_object: h5py.File,
        dataset: Literal['CHECK', 'OAI'],
        subject_id: str,
        subject_visit: str,
        target_pixel_spacing: float = None,
    ) -> None:
        self.dicom_file_path      = dicom_file_path
        self.points_file_path     = points_file_path
        self.hdf5_file_object     = hdf5_file_object

        self.dataset              = dataset
        self.subject_id           = subject_id
        self.subject_visit        = subject_visit

        self.target_pixel_spacing = target_pixel_spacing

        self.intensity_slope      = 1.0
        self.intensity_offset     = 0.0
        return


class DicomTransform(Transform):

    def __init__(self) -> None:
        ...

    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        ...


class CombineTransforms(Transform):

    transforms: list[Transform]
    
    def __init__(self, transforms: list[Transform]) -> None:
        super().__init__()
        self.transforms = transforms
        return

    def __call__(self, container: Container) -> Container:
        for transform in self.transforms:
            container = transform(container)
        return container


class LoadDicomObject(DicomTransform):
    """
    Loads the image from the specified DICOM file into a `DicomContainer`.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        dicom_container.dicom_file_object = dcmread(dicom_container.dicom_file_path)
        return dicom_container
    

class GetPixelArray(DicomTransform):
    """
    Get the pixel array from the `DicomObject` and set it as a field in the `DicomContainer`.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        dicom_container.pixel_array = dicom_container.dicom_file_object.pixel_array
        return dicom_container
    

class GetSourcePixelSpacing(DicomTransform):
    """
    Get the pixel spacing from the `DicomObject` and set it as a field in the `DicomContainer`.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        dicom_image: DicomFileObject = dicom_container.dicom_file_object
        pixel_spacing: list[float]   = dicom_image.get('PixelSpacing') or \
                                       dicom_image.get('ImagerPixelSpacing')

        if pixel_spacing is None:
            raise Exception('No pixel spacing found.')
        if pixel_spacing[0] != pixel_spacing[1]:
            raise Exception('Anisotropic pixel spacing is untested.')

        dicom_container.source_pixel_spacing = pixel_spacing
        dicom_container.pixel_spacing        = pixel_spacing
        
        return dicom_container
        

class CheckPhotometricInterpretation(DicomTransform):
    """
    Check if pixel intensities are stored as `MONOCHROME2` (white = max, black = min).
    DICOM images must be stored with either `MONOCHROME1` or `MONOCHROME2` as their `Photometric Interpretation`.
    If stored as `MONOCHROME1`, intensities are inverted to match `MONOCHROME2`.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        """
        Photometric interpretation of DICOM images must be either MONOCHROME1 or MONOCHROME2.
        If it is MONOCHROME1, invert the pixel intensities to match the MONOCHROME2.
        """
        dicom_image  = dicom_container.dicom_file_object
        pixel_array  = dicom_container.pixel_array
        photo_interp = dicom_image.get('PhotometricInterpretation')
        
        if photo_interp == 'MONOCHROME1':
            # Flip intensities
            dicom_container.pixel_array = np.max(pixel_array) - pixel_array
        elif photo_interp != 'MONOCHROME2':
            raise Exception(f'Photometric interpretation {photo_interp} not supported.')
        return dicom_container
    

class CheckVoilutFunction(DicomTransform):
    """
    The `VOILUTFunction` property of the DICOM image must be `LINEAR`.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        if dicom_container.dicom_file_object.get('VOILUTFunction', 'LINEAR') != 'LINEAR':
            raise Exception('Only supporting VOILUTFunction LINEAR')
        return dicom_container
    

class ResampleToTargetResolution(DicomTransform):
    """
    Resample image to the target resolution, as indicated by the `target_pixel_spacing`.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        source_pixel_spacing = dicom_container.source_pixel_spacing
        target_pixel_spacing = dicom_container.target_pixel_spacing

        # resample to the required resolution
        if target_pixel_spacing is not None:
            scale_factor = source_pixel_spacing[0] / target_pixel_spacing
            dicom_container.pixel_array = rescale(dicom_container.pixel_array, scale_factor)
            dicom_container.pixel_spacing = [target_pixel_spacing, target_pixel_spacing]
        return dicom_container
    

class NormalizeIntensities(DicomTransform):
    """
    Normalize pixel intensities using percentiles.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:

        pixel_array = dicom_container.pixel_array
        pixel_array = pixel_array.astype(float)

        percentile  = np.percentile(pixel_array.flatten(), [5, 95])

        intensity_offset = percentile[0]
        intensity_slope  = percentile[1] - percentile[0]

        pixel_array -= intensity_offset
        pixel_array /= intensity_slope

        dicom_container.pixel_array      = pixel_array
        dicom_container.intensity_offset = intensity_offset
        dicom_container.intensity_slope  = intensity_slope

        return dicom_container


class AppendDicomToHDF5(DicomTransform):
    """
    Write the information contained within the given `DicomContainer` to the HDF5 file indicated by the same `DicomContainer`.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        
        hdf5_file_object     = dicom_container.hdf5_file_object
        dicom_file_path      = dicom_container.dicom_file_path
        # points_file_path     = dicom_container.points_file_path

        dataset              = dicom_container.dataset
        subject_id           = dicom_container.subject_id
        subject_visit        = dicom_container.subject_visit

        source_pixel_spacing = dicom_container.source_pixel_spacing
        # target_pixel_spacing = dicom_container.target_pixel_spacing
        pixel_spacing        = dicom_container.pixel_spacing

        pixel_array          = dicom_container.pixel_array

        intensity_slope      = dicom_container.intensity_slope
        intensity_offset     = dicom_container.intensity_offset

        # Write to hdf5
        group = hdf5_file_object.require_group(f'/scans/{subject_id}/{subject_visit}')
        group.attrs['dataset']    = dataset
        group.attrs['subject_id'] = subject_id
        group.attrs['visit']      = subject_visit

        img_ds = group.create_dataset(
            'image', data=pixel_array,
            **hdf5plugin.Blosc2(cname='blosclz', clevel=9,filters=hdf5plugin.Blosc2.SHUFFLE)
        )
        img_ds.attrs['source']               = dicom_file_path
        img_ds.attrs['pixel_spacing']        = pixel_spacing
        img_ds.attrs['source_pixel_spacing'] = source_pixel_spacing
        img_ds.attrs['intensity_offset']     = intensity_offset
        img_ds.attrs['intensity_slope']      = intensity_slope

        return dicom_container
