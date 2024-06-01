'''
Transformations for the DICOM files and BoneFinder generated points files.
'''
from typing import Literal, Union
import numpy as np
from pydicom import dcmread, FileDataset 
from pydicom.dicomdir import DicomDir
from skimage.transform import rescale
import h5py, hdf5plugin
import circle_fit
from skimage.draw import polygon2mask

import constants as ct


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
    pixel_spacing: tuple[float]

    source_pixel_array_shape: tuple[int, int]

    top_pad: int
    bottom_pad: int
    left_pad: int
    right_pad: int

    def __init__(self,
        dicom_file_path: str,
        points_file_path: str,
        hdf5_file_object: h5py.File,
        dataset: Literal['CHECK', 'OAI'],
        subject_id: str,
        subject_visit: str,
    ) -> None:
        self.dicom_file_path  = dicom_file_path
        self.points_file_path = points_file_path
        self.hdf5_file_object = hdf5_file_object
        self.dataset          = dataset
        self.subject_id       = subject_id
        self.subject_visit    = subject_visit

        self.top_pad    = 0
        self.bottom_pad = 0
        self.left_pad   = 0
        self.right_pad  = 0


class DicomTransformation:

    def __init__(self) -> None:
        pass

    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        pass


class CombineTransformations(DicomTransformation):
    '''
    Chain a sequence of `DicomTransformation`s that are applied to a `DicomContainer`.
    '''
    transformations: list[DicomTransformation]
    
    def __init__(self, transformations: list[DicomTransformation]) -> None:
        self.transformations = transformations

    def __call__(self, container: DicomContainer) -> DicomTransformation:
        for t in self.transformations:
            container = t(container)
        return container


class LoadDicomObject(DicomTransformation):
    '''
    Loads the image from the DICOM file into a `DicomContainer`.
    The DICOM filepath is specified as an attribute in the `DicomContainer`.
    '''
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        dicom.dicom_file_object = dcmread(dicom.dicom_file_path)
        return dicom
    

class GetPixelArray(DicomTransformation):
    '''
    Get the pixel array from the `DicomObject` and set it as a field in the `DicomContainer`.
    This transformation ensures that the inital 2D image is wrapped into a 3D array,
    to match the dimensionlity of the combined segmentation mask.
    '''
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        pixel_array = dicom.dicom_file_object.pixel_array

        # Ensure that the initial image is 2D.
        if pixel_array.ndim != 2:
            raise PreprocessingException(
                f'Unexpected `pixel_arrray.ndim`.'
                f'Was: {pixel_array.ndim}.'
            )

        # Get the initial shape of the pixel array
        dicom.source_pixel_array_shape = pixel_array.shape

        # Transform the original 2D image into a 3D one,
        # by adding one more dimension.
        pixel_array = pixel_array[None]
        dicom.pixel_array = pixel_array

        return dicom
    

class GetSourcePixelSpacing(DicomTransformation):
    '''
    Get the source pixel spacing as specified in the DICOM file object
    and set it as a field in the `DicomContainer`.
    '''
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        pixel_spacing: tuple[float, float] = tuple(
            dicom.dicom_file_object.get(ct.DicomAttributes.PIXEL_SPACING) or \
            dicom.dicom_file_object.get(ct.DicomAttributes.IMAGER_PIXEL_SPACING)
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
        photo_interp = dicom_image.get(ct.DicomAttributes.PHOTOMETRIC_INTERPRETATION)
        
        if photo_interp == ct.PhotometricInterpretation.MONOCHROME1:
            # Flip intensities
            dicom.pixel_array = np.max(pixel_array) - pixel_array
        elif photo_interp != ct.PhotometricInterpretation.MONOCHROME2:
            raise PreprocessingException(
                f'Photometric interpretation {photo_interp} not supported.'
            )
        return dicom
    

class CheckVoilutFunction(DicomTransformation):
    '''
    The `VOILUTFunction` attribute of the DICOM file object must be `LINEAR`.
    '''
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        voilut_func = dicom.dicom_file_object.get(
            ct.DicomAttributes.VOILUT_FUNCTION,
            ct.VoilutFunction.LINEAR,
        )
        if voilut_func != ct.VoilutFunction.LINEAR:
            raise PreprocessingException(
                f'Only supporting VOILUTFunction LINEAR.'
                f'Was: {voilut_func}.'
            )
        return dicom
    

class PercentilesIntensityNormalization(DicomTransformation):
    '''
    Normalize image pixel intensities using percentile normalization.

    Attributes
    ----------
    percentiles (tuple[float, float], optional): The percentiles;
        taken from the interval [0.0, 100.0].
        If unspecified, do nothing.
    
    Raises
    ------
    PreprocessingException: 
        If the percentiles are not in the interval `[0.0, 100.0]`.
        If there are not exactly two percentiles.
    '''
    percentiles: tuple[float, float]

    def __init__(self, percentiles: tuple[float, float]) -> None:
        if percentiles is not None:
            if not all(0.0 <= p <= 100.0 for p in percentiles):
                raise PreprocessingException('Percentiles must be in the interval [0.0, 100.0].')
            if len(percentiles) != 2:
                raise PreprocessingException('Accepting exactly two percentiles.')
        self.percentiles = percentiles

    def __call__(self, dicom: DicomContainer) -> DicomContainer:

        if self.percentiles is None:
            return dicom

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

        return dicom
    

class MinMaxIntensityNormalization(DicomTransformation):
    '''
    Normalize image pixel intensities using MinMax normalization
    (i.e., pixels values will be in the interval `[0.0, 1.0]`).
    '''
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        minv = np.min(dicom.pixel_array)
        maxv = np.max(dicom.pixel_array)
        dicom.pixel_array = (dicom.pixel_array - minv) / (maxv - minv)
        return dicom
    

class RescaleToTargetPixelSpacing(DicomTransformation):
    '''
    Rescale image to the target resolution.

    Attributes
    ----------
    target_pixel_spacing (tuple[float, float], optional): 
        The target pixel spacing that the image resolution 
        should match.

    Raises
    ------
    PreprocessingException: If the `target_pixel_spacing` is
        non-equal along the two axes, since anisotropic pixel
        spacing is not supported.
    '''
    target_pixel_spacing: tuple[float, float]

    def __init__(self, target_pixel_spacing: tuple[float, float]=None) -> None:
        # Supporting only anisotropic pixel spacing.
        if target_pixel_spacing is not None \
            and target_pixel_spacing[0] != target_pixel_spacing[1]:
            raise PreprocessingException(
                f'Anisotropic pixel spacing is not supported.'
                f'Was: {target_pixel_spacing}.'
            )
        self.target_pixel_spacing = target_pixel_spacing

    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        source_pixel_spacing = dicom.source_pixel_spacing
        target_pixel_spacing = self.target_pixel_spacing

        if target_pixel_spacing is None:
            return dicom

        # Rescale image pixel array to match the target resolution.
        scale_factor        = source_pixel_spacing[0] / target_pixel_spacing[0]
        dicom.pixel_array   = rescale(dicom.pixel_array, scale_factor)
        dicom.pixel_spacing = target_pixel_spacing

        return dicom


class PadSymmetrically(DicomTransformation):
    '''
    Pad image and segmentation masks symmetrically 
    along each of the vertical and horizontal axes.
    
    Attributes
    ----------
    target_shape (tuple[int, int], optional): If specified,
        pad the image and the segmentation masks to reach
        the target shape.
    '''
    target_shape: tuple[int, int]

    def __init__(self, target_shape: tuple[int, int]=None) -> None:
        self.target_shape = target_shape
    
    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        
        current_shape = dicom.pixel_array.shape[-2:]
        target_shape  = self.target_shape

        # Do not pad if target_shape is unspecified.
        if target_shape is None:
            return dicom

        # Cannot pad if the image shape already exceeds the target shape
        if current_shape[0] > target_shape[0] or \
           current_shape[1] > target_shape[1]:
            raise PreprocessingException(
                f'Cannot pad image with shape {current_shape} to smaller target shape {target_shape}.'
            )
        
        vertical_pad   = target_shape[0] - current_shape[0]
        horizontal_pad = target_shape[1] - current_shape[1]

        dicom.top_pad    = vertical_pad // 2
        dicom.bottom_pad = vertical_pad // 2 + vertical_pad % 2
        dicom.left_pad   = horizontal_pad // 2
        dicom.right_pad  = horizontal_pad // 2 + horizontal_pad % 2

        # Since the image and masks are 3D,
        # and the intention is to pad the 2D image
        # wrapped in the additional dimension,
        # the first dimension is not padded.
        pad = (
            (0, 0),
            (dicom.top_pad, dicom.bottom_pad),
            (dicom.right_pad, dicom.left_pad),
        )

        dicom.pixel_array = np.pad(dicom.pixel_array, pad)
        
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
        mask_shape    = pixel_array.shape[-2:]

        # Account for the amount of pad added to the image.
        pad_offset = np.array([dicom.left_pad, dicom.top_pad])

        circles = dict()
        for hip_side, offset in ct.HipSideOffset.items():
            for curve_name, curve in ct.HipBoneSubCurve.items():
                xc, yc, r, sigma = circle_fit.taubinSVD(points[np.array(curve) + offset])
                circles[f'{hip_side} {curve_name}'] = { 'xc': xc, 'yc': yc, 'r': r, 'sigma': sigma }

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
            circle = circles[f'{hip_side} {ct.HipBoneSubCurve.FEMORAL_HEAD[0]}']

            # define the regions
            regions = {
                'femur': np.array([
                    *points[np.array(ct.HipBoneCurve.PROXIMAL_FEMUR[1]) + offset],
                ]),
                'acetabulum': np.array([
                    *points[np.array(ct.HipBoneCurve.ACETABULAR_ROOF[1]) + offset],
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
                (pad_offset + regions['femur'] / pixel_spacing)[:, [1, 0]],
            )
            acetabulum_mask = polygon2mask(
                mask_shape,
                (pad_offset + regions['acetabulum'] / pixel_spacing)[:, [1, 0]],
            )
            joint_space_mask = polygon2mask(
                mask_shape,
                (pad_offset + regions['joint space'] / pixel_spacing)[:, [1, 0]],
            )
            
            background_mask = ~(femur_mask + acetabulum_mask + joint_space_mask)
            
            # Create combined mask as an ndarray of masks for each region.
            # Note that the joint space mask is larger than the true area,
            # the pixels overlapping with the other regions are set to 0.
            combined_mask = np.zeros(shape=mask_shape, dtype=np.uint8)
            combined_mask = combined_mask[None]
            combined_mask = np.repeat(
                combined_mask,
                repeats = 1 + len(regions),  # <-- Also account for the background
                axis = 0,
            )

            combined_mask[0][background_mask]  = 1
            combined_mask[1][femur_mask]       = 1
            combined_mask[2][acetabulum_mask]  = 1
            combined_mask[3][joint_space_mask] = 1

            # Eliminate additional pixels from the joint space mask
            combined_mask[3][femur_mask]       = 0
            combined_mask[3][acetabulum_mask]  = 0

            # Assign the combined mask to the current side of the body being segmented.
            if hip_side == ct.HipSide.RIGHT:
                dicom.right_segmentation_mask = combined_mask
            elif hip_side == ct.HipSide.LEFT:
                dicom.left_segmentation_mask = combined_mask
            else:
                raise PreprocessingException(
                    f'Side must be either `\'{ct.HipSide.RIGHT}\'` or `\'{ct.HipSide.LEFT}\'`.'
                    f'Was: `{hip_side}`'
                )
        return dicom


class Flip(DicomTransformation):
    '''
    Flip the pixel array and the segmentation masks for both sides.

    Attributes
    ----------
    axis (int): Flip the image and masks around the specified axis.

    Raises
    ------
    PreprocessingException: If the image and segmentation masks
        do not have the same number of dimensions.
    '''
    axis: int

    def __init__(self, axis: int) -> None:
        self.axis = axis

    def __call__(self, dicom: DicomContainer) -> DicomContainer:
        if not (
            dicom.pixel_array.ndim == \
            dicom.right_segmentation_mask.ndim == \
            dicom.left_segmentation_mask.ndim
        ):
            raise PreprocessingException(
                'Image and masks must have the same '
                'number of dimensions to be flipped.'
            )


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
    Write the information contained within the given `DicomContainer`
    to the HDF5 file indicated by the same `DicomContainer`.
    '''
    hip_side: str

    def __init__(self, hip_side: str) -> None:
        self.hip_side = hip_side

    def __call__(self, dicom: DicomContainer) -> DicomContainer:

        dicom_file_path          = dicom.dicom_file_path
        points_file_path         = dicom.points_file_path
        hdf5_file_object         = dicom.hdf5_file_object

        dataset                  = dicom.dataset
        subject_id               = dicom.subject_id
        subject_visit            = dicom.subject_visit

        pixel_array              = dicom.pixel_array
        right_segmentation_mask  = dicom.right_segmentation_mask
        left_segmentation_mask   = dicom.left_segmentation_mask

        source_pixel_spacing     = dicom.source_pixel_spacing
        pixel_spacing            = dicom.pixel_spacing
        source_pixel_array_shape = dicom.source_pixel_array_shape

        # Infer which hip side is to be segmented
        segmentation_mask: np.ndarray
        if self.hip_side == ct.HipSide.RIGHT:
            segmentation_mask = right_segmentation_mask
        elif self.hip_side == ct.HipSide.LEFT:
            segmentation_mask = left_segmentation_mask
        else:
            raise PreprocessingException(
                f'Hip side can be either `\'{ct.HipSide.RIGHT}\'` or `\'{ct.HipSide.LEFT}\'`.'
                f'Was: {self.hip_side}.'
            )

        # Write to hdf5:
        group_id = f'/scans/{dataset}/{subject_id}/{subject_visit}/{self.hip_side}'
        group    = hdf5_file_object.require_group(group_id)

        # Write meta info
        group.attrs['dicom_file_path']          = dicom_file_path
        group.attrs['points_file_path']         = points_file_path
        group.attrs['dataset']                  = dataset
        group.attrs['subject_id']               = subject_id
        group.attrs['subject_visit']            = subject_visit
        group.attrs['source_pixel_spacing']     = source_pixel_spacing
        group.attrs['pixel_spacing']            = pixel_spacing
        group.attrs['source_pixel_array_shape'] = source_pixel_array_shape
        group.attrs['hip_side']                 = self.hip_side

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
