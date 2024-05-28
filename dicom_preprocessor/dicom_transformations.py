'''
Transformations for the DICOM files and BoneFinder generated points files.
'''
from typing import Literal, Union
from matplotlib import pyplot as plt
import numpy as np
from pydicom import dcmread, FileDataset 
from pydicom.dicomdir import DicomDir
from skimage.transform import rescale
import hdf5plugin
import h5py
import circle_fit
from skimage.draw import polygon2mask

import constants as ct


TRANSFORMS = (
    'CombineTransforms',
    'LoadDicomObject',
    'GetPixelArray',
    'GetSourcePixelSpacing',
    'CheckPhotometricInterpretation',
    'CheckVoilutFunction',
    'ResampleToTargetResolution',
    'NormalizeIntensities',
    'GetBoneFinderPoints',
    'GetSegmentationMasks',
    'FlipHorizontally',
    'AppendDicomToHDF5',
)


DicomFileObject = Union[FileDataset, DicomDir]


class DicomContainer:
    dicom_file_path: str
    points_file_path: str
    hdf5_file_object: h5py.File

    dataset: Literal['CHECK', 'OAI']
    subject_id: str
    subject_visit: str
    
    dicom_file_object: DicomFileObject
    pixel_array: np.ndarray
    bonefinder_points: np.array
    right_segmentation_mask: np.ndarray
    left_segmentation_mask: np.ndarray
    
    source_pixel_spacing: tuple[float]
    target_pixel_spacing: tuple[float]
    pixel_spacing: tuple[float]

    source_pixel_array_shape: tuple[float, float]
    target_pixel_array_shape: tuple[float, float]
    pixel_array_shape: tuple[float, float]

    intensity_offset: float
    intensity_slope: float

    def __init__(self,
        dicom_file_path: str,
        points_file_path: str,
        hdf5_file_object: h5py.File,
        dataset: Literal['CHECK', 'OAI'],
        subject_id: str,
        subject_visit: str,
        target_pixel_spacing: tuple[float, float],
        target_pixel_array_shape: tuple[float, float], 
        intensity_slope: float = 1.0,
        intensity_offset: float = 0.0,
    ) -> None:
        self.dicom_file_path          = dicom_file_path
        self.points_file_path         = points_file_path
        self.hdf5_file_object         = hdf5_file_object

        self.dataset                  = dataset
        self.subject_id               = subject_id
        self.subject_visit            = subject_visit

        self.target_pixel_spacing     = target_pixel_spacing
        self.target_pixel_array_shape = target_pixel_array_shape

        self.intensity_slope          = intensity_slope
        self.intensity_offset         = intensity_offset

        return


class DicomTransformation:

    def __init__(self) -> None:
        pass

    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        pass


class CombineTransformations(DicomTransformation):

    transformations: list[DicomTransformation]
    
    def __init__(self, transformations: list[DicomTransformation]) -> None:
        super().__init__()
        self.transformations = transformations
        return

    def __call__(self, container: DicomContainer) -> DicomTransformation:
        for t in self.transformations:
            container = t(container)
        return container


class LoadDicomObject(DicomTransformation):
    '''
    Loads the image from the specified DICOM file into a `DicomContainer`.
    '''
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        dicom.dicom_file_object = dcmread(dicom.dicom_file_path)
        return dicom
    

class GetPixelArray(DicomTransformation):
    '''
    Get the pixel array from the `DicomObject` and set it as a field in the `DicomContainer`.
    Note that this transformation adds one more dimension to the original 2D image.
    '''
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        pixel_array = dicom.dicom_file_object.pixel_array

        # Ensure that the 2D image is stored in a 3D pixel array.
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[None]
        else:
            raise PreprocessingException(
                f'Unexpected `pixel_arrray.ndim`.'
                f'Was: {pixel_array.ndim}.'
            )
        dicom.pixel_array              = pixel_array

        # Get the shape of the pixel array
        dicom.source_pixel_array_shape = pixel_array.shape
        dicom.pixel_array_shape        = pixel_array.shape

        return dicom
    

class GetSourcePixelSpacing(DicomTransformation):
    '''
    Get the pixel spacing from the `DicomObject` and set it as a field in the `DicomContainer`.
    '''
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        pixel_spacing: tuple[float, float] = tuple(
            dicom.dicom_file_object.get(ct.DicomAttributes.PIXEL_SPACING.value) or \
            dicom.dicom_file_object.get(ct.DicomAttributes.IMAGER_PIXEL_SPACING.value)
        )
        
        if pixel_spacing is None:
            raise PreprocessingException('No pixel spacing found.')

        if pixel_spacing[0] != pixel_spacing[1]:
            raise PreprocessingException(
                f'Anisotropic pixel spacing is untested.'
                f'Was: {pixel_spacing}.'
            )

        dicom.source_pixel_spacing = pixel_spacing
        dicom.pixel_spacing        = pixel_spacing
        
        return dicom
        

class CheckPhotometricInterpretation(DicomTransformation):
    '''
    Check if pixel intensities are stored as `MONOCHROME2` (white = max, black = min).
    `Photometric Interpretation` of DICOM images must be either `MONOCHROME1` or `MONOCHROME2`.
    If stored as `MONOCHROME1`, intensities are flipped to match `MONOCHROME2`.
    '''
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        dicom_image  = dicom.dicom_file_object
        pixel_array  = dicom.pixel_array
        photo_interp = dicom_image.get(ct.DicomAttributes.PHOTOMETRIC_INTERPRETATION.value)
        
        if photo_interp == ct.PhotometricInterpretation.MONOCHROME1.value:
            # Flip intensities
            dicom.pixel_array = np.max(pixel_array) - pixel_array
        elif photo_interp != ct.PhotometricInterpretation.MONOCHROME2.value:
            raise PreprocessingException(
                f'Photometric interpretation {photo_interp} not supported.'
            )
        return dicom
    

class CheckVoilutFunction(DicomTransformation):
    '''
    The `VOILUTFunction` property of the DICOM image must be `LINEAR`.
    '''
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        voilut_func = dicom.dicom_file_object.get(
            ct.DicomAttributes.VOILUT_FUNCTION.value,
            ct.VoilutFunction.LINEAR.value,
        )
        if voilut_func != ct.VoilutFunction.LINEAR.value:
            raise PreprocessingException(
                f'Only supporting VOILUTFunction LINEAR.'
                f'Was: {voilut_func}.'
            )
        return dicom
    

class RescaleToTargetResolution(DicomTransformation):
    '''
    Resample image to the target resolution, as indicated by the `target_pixel_spacing`.
    '''
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        source_pixel_spacing = dicom.source_pixel_spacing
        target_pixel_spacing = dicom.target_pixel_spacing

        if target_pixel_spacing is not None:
            # Supporting only anisotropic pixel spacing.
            if target_pixel_spacing[0] != target_pixel_spacing[1]:
                raise PreprocessingException(
                    f'Anisotropic pixel spacing is untested.'
                    f'Was: {target_pixel_spacing}.'
                )
            
            # Rescale image pixel array to match target resolution.
            scale_factor        = source_pixel_spacing[0] / target_pixel_spacing[0]
            dicom.pixel_array   = rescale(dicom.pixel_array, scale_factor)
            dicom.pixel_spacing = target_pixel_spacing

        return dicom
    

class NormalizeIntensities(DicomTransformation):
    '''
    Normalize pixel intensities using percentiles (defaults to `[5, 95]` percentiles).
    '''
    percentiles: list[float]

    def __init__(self, percentiles: list[float] = [5, 95]):
        self.percentiles = percentiles

    def __call__(self, dicom: DicomContainer) -> DicomContainer:

        pixel_array = dicom.pixel_array
        pixel_array = pixel_array.astype(float)

        percentiles = np.percentile(
            pixel_array.flatten(),
            self.percentiles,
        )

        intensity_offset = float(percentiles[0])
        intensity_slope  = float(percentiles[1] - percentiles[0])

        pixel_array -= intensity_offset
        pixel_array /= intensity_slope

        dicom.pixel_array      = pixel_array
        dicom.intensity_offset = intensity_offset
        dicom.intensity_slope  = intensity_slope

        return dicom
    

class GetBoneFinderPoints(DicomTransformation):

    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        with open(dicom.points_file_path, 'r') as points_file_object:
            lines = points_file_object.readlines()
            lines = [line.strip() for line in lines]

            # Extract information from file
            version  = int(lines[0].split(' ')[1])
            n_points = int(lines[1].split(' ')[1])
            points   = np.array([
                [float(x) for x in line.split(' ')] 
                    for line in lines[3 : (n_points + 3)]
            ])

            # Verify file structure
            if lines[0]     != f'version: {version}' \
                or lines[1]     != f'n_points: {n_points}' \
                or n_points     != ct.N_BONEFINDER_POINTS \
                or len(lines)   != n_points + 4 \
                or lines[2]     != '{' \
                or lines[163]   != '}' \
                or points.shape != (n_points, 2):
                raise PreprocessingException('Points file structure is invalid.')

            dicom.bonefinder_points = points

        return dicom


class GetSegmentationMasks(DicomTransformation):
    '''
    Compute the RGB pixel array of the segmentation mask,
    highlighting the various components of the hip, 
    for each of the right and left sides of the hip.
    '''
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        pixel_array   = dicom.pixel_array
        pixel_spacing = dicom.pixel_spacing
        points        = dicom.bonefinder_points

        mask_shape    = pixel_array.shape
        mask_shape    = mask_shape[1], mask_shape[2]

        circles = dict()
        for hip_side, offset in ct.HipSideOffset.items():
            for curve_name, curve in ct.HipBoneSubCurve.items():
                xc, yc, r, sigma = circle_fit.taubinSVD(points[np.array(curve) + offset])
                circles[f'{hip_side}_{curve_name}'] = { 'xc': xc, 'yc': yc, 'r': r, 'sigma': sigma }

        for hip_side, offset in ct.HipSideOffset.items():

            # define the bounding box of the segmentation region
            bbox = {
                # top: topmost point of acetabulum curve
                'top':     points[67 + offset][1],
                # medial: most medial point of the sourcil
                'medial':  points[74 + offset][0],
                # lateral:
                'lateral': points[8 + offset][0],
                # bottom: medial bottom of femoral head
                'bottom':  points[27 + offset][1],
            }

            # from most lateral part of the sourcil to center of femoral head
            circle = circles[f'{hip_side}_{ct.HipBoneSubCurve.FEMORAL_HEAD.name}']

            # define the regions
            regions = {
                'femur': np.array([
                    *points[np.array(ct.HipBoneCurve.PROXIMAL_FEMUR.value) + offset],
                ]),
                'acetabulum': np.array([
                    *points[np.array(ct.HipBoneCurve.ACETABULAR_ROOF.value) + offset],
                    [bbox['medial'], bbox['top']],
                ]),
                'joint space': np.array([
                    # note: this polygon is larger than the joint space,
                    # but the excess will be covered by the bone regions
                    # - start from most lateral point of the sourcil
                    points[70 + offset],
                    # - to center of femoral head
                    [circle['xc'], circle['yc']],
                    # - to medial boundary of bbox
                    [bbox['medial'], circle['yc']],
                    # - to medial top
                    [bbox['medial'], bbox['top']],
                    # - to topmost point of acetabulum curve
                    points[67 + offset],
                ]),
            }

            # Extract masks for each region.
            femur_mask = polygon2mask(
                mask_shape,
                (regions['femur'] / pixel_spacing[0])[:, [1, 0]],
            )
            acetabulum_mask = polygon2mask(
                mask_shape,
                (regions['acetabulum'] / pixel_spacing[0])[:, [1, 0]],
            )
            joint_space_mask = polygon2mask(
                mask_shape,
                (regions['joint space'] / pixel_spacing[0])[:, [1, 0]],
            )
            
            # Create combined mask as an ndarray of masks for each region.
            # Note that the joint space mask is larger than the true area,
            # the pixels overlapping with the other regions are set to 0.
            combined_mask = np.zeros(shape=mask_shape, dtype=np.uint8)
            combined_mask = combined_mask[None]
            combined_mask = np.repeat(combined_mask, repeats=len(regions), axis=0)

            combined_mask[0][femur_mask]       = 1
            combined_mask[1][acetabulum_mask]  = 1
            combined_mask[2][joint_space_mask] = 1

            combined_mask[2][femur_mask]       = 0
            combined_mask[2][acetabulum_mask]  = 0

            # Assign the combined mask to the current side of the body being segmented.
            if hip_side == ct.HipSide.RIGHT.name:
                dicom.right_segmentation_mask = combined_mask
            elif hip_side == ct.HipSide.LEFT.name:
                dicom.left_segmentation_mask = combined_mask
            else:
                raise PreprocessingException(
                    f'Side must be either `\'{ct.HipSide.RIGHT.value}\'` or `\'{ct.HipSide.LEFT.value}\'`.'
                    f'Was: `{hip_side}`'
                )
        return dicom
    

class Flip(DicomTransformation):
    '''
    Flip the pixel array and the segmentation masks for both sides.
    By default, the flip occurs horizontally.
    '''
    axis: int

    def __init__(self, axis: int = 2) -> None:
        super().__init__()
        self.axis = axis

    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        dicom.pixel_array = np.flip(
            dicom.pixel_array, 
            axis=self.axis,
        )
        dicom.right_segmentation_mask = np.flip(
            dicom.right_segmentation_mask, 
            axis=self.axis,
        )
        dicom.left_segmentation_mask = np.flip(
            dicom.left_segmentation_mask, 
            axis=self.axis,
        )
        return dicom


class AppendDicomToHDF5(DicomTransformation):
    '''
    Write the information contained within the given `DicomContainer` to the HDF5 file indicated by the same `DicomContainer`.
    '''
    hip_side: ct.HipSide

    def __init__(self, hip_side: ct.HipSide) -> None:
        super().__init__()
        self.hip_side = hip_side
        return

    def __call__(self, dicom: DicomContainer) -> DicomContainer:

        dicom_file_path         = dicom.dicom_file_path
        points_file_path        = dicom.points_file_path
        hdf5_file_object        = dicom.hdf5_file_object

        dataset                 = dicom.dataset
        subject_id              = dicom.subject_id
        subject_visit           = dicom.subject_visit

        dicom_file_object       = dicom.dicom_file_object
        pixel_array             = dicom.pixel_array
        bonefinder_points       = dicom.bonefinder_points
        right_segmentation_mask = dicom.right_segmentation_mask
        left_segmentation_mask  = dicom.left_segmentation_mask

        source_pixel_spacing    = dicom.source_pixel_spacing
        target_pixel_spacing    = dicom.target_pixel_spacing
        pixel_spacing           = dicom.pixel_spacing

        intensity_slope         = dicom.intensity_slope
        intensity_offset        = dicom.intensity_offset

        # Infer which hip side is to be segmented
        segmentation_mask: np.ndarray
        if self.hip_side == ct.HipSide.RIGHT:
            segmentation_mask = right_segmentation_mask
        elif self.hip_side == ct.HipSide.LEFT:
            segmentation_mask = left_segmentation_mask
        else:
            raise PreprocessingException(
                f'Hip side can be either `\'{ct.HipSide.RIGHT.value}\'` or `\'{ct.HipSide.LEFT.value}\'`.'
                f'Was: {self.hip_side.value}.'
            )

        # Write to hdf5:
        group_id = f'/scans/{dataset}/{subject_id}/{subject_visit}/{self.hip_side.value}'
        group    = hdf5_file_object.require_group(group_id)

        # Write meta info
        group.attrs['dicom_file_path']         = dicom_file_path
        group.attrs['points_file_path']        = points_file_path
        # group.attrs['hdf5_file_object'] = hdf5_file_object
        group.attrs['dataset']                 = dataset
        group.attrs['subject_id']              = subject_id
        group.attrs['subject_visit']           = subject_visit
        # group.attrs['dicom_file_object']       = dicom_file_object
        # group.attrs['pixel_array']             = pixel_array
        # group.attrs['bonefinder_points']       = bonefinder_points
        # group.attrs['right_segmentation_mask'] = right_segmentation_mask
        # group.attrs['left_segmentation_mask']  = left_segmentation_mask
        group.attrs['source_pixel_spacing']    = source_pixel_spacing
        group.attrs['target_pixel_spacing']    = target_pixel_spacing
        group.attrs['pixel_spacing']           = pixel_spacing
        group.attrs['intensity_slope']         = intensity_slope
        group.attrs['intensity_offset']        = intensity_offset
        group.attrs['hip_side']                = self.hip_side.value

        # Write image
        img_ds = group.create_dataset(
            'image', data=pixel_array,
            **hdf5plugin.Blosc2(cname='blosclz', clevel=9, filters=hdf5plugin.Blosc2.SHUFFLE)
        )

        # Write mask
        img_ds = group.create_dataset(
            'mask', data=segmentation_mask,
            **hdf5plugin.Blosc2(cname='blosclz', clevel=9,filters=hdf5plugin.Blosc2.SHUFFLE)
        )

        return dicom


class PreprocessingException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
