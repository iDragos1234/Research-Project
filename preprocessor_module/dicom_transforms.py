'''
Preprocessing transforms for the DICOM samples.
'''
from typing import Literal, Union
import numpy as np
from pydicom import dcmread, FileDataset
from pydicom.dicomdir import DicomDir
import h5py, hdf5plugin
import circle_fit
from skimage.draw import polygon2mask
from skimage.transform import rescale

import constants as ct


# Type alias for the DICOM file object
DicomFileObject = Union[FileDataset, DicomDir]


class DicomContainer:
    '''
    Data container object for the current sample (extracted from the DICOM file corresponding to this sample).
    Any preprocessing-related data for the current sample is contained in this object.

    Attributes
    ----------
    dicom_filepath (str): 
        DICOM file containing the pelvic X-ray image for the current sample.

    points_filepath (str): 
        Corresponding file containing the BoneFinder-generated points.

    dicom_file_object (DicomFileObject):
        DICOM file object for the current sample.

    hdf5_file_object (h5py.File): 
        HDF5 file object where the preprocessing output for the current sample is written.

    dataset (Literal['CHECK', 'OAI']): 
        Dataset name (either CHECK or OAI cohorts) for the current sample.

    subject_id (str): 
        Subject (patient) ID for the current sample. Represented by a 7-digit number.

    subject_visit (str):
        Subject (patient) visit, expressing the number of years from the initial visit. 
        Represented by a 2-digit number.

    pixel_array (numpy.ndarray):
        X-ray image pixel array.
        NOTE: upon reading, the original 2D image pixel array is wrapped in an additional dimension, producing a 3D array.

    bonefinder_points (numpy.array):
        Array of coordinates for the BoneFinder-generated points.

    right_segmentation_mask (numpy.ndarray):
        Segmentation mask for the left hip (flipped around the vertical axis).

    left_segmentation_mask (numpy.ndarray):
        Segmentation mask for the right hip (un-flipped).

    source_pixel_spacing (tuple[float, float]):
        Pixel spacing for the raw, un-processed X-ray image.
    
    pixel_spacing (tuple[float, float]):
        Pixel spacing for the image pixel array stored in this container (i.e., for the processed image).

    source_pixel_array_shape (tuple[int, int]):
        Pixel array shape for the raw, un-processed X-ray image.

    top_pad, bottom_pad, left_pad, right_pad (int):
        Amount of padding added to the image.
    '''

    dicom_filepath: str
    points_filepath: str

    dicom_file_object: DicomFileObject
    hdf5_file_object: h5py.File

    dataset: Literal['CHECK', 'OAI']
    subject_id: str
    subject_visit: str

    pixel_array: np.ndarray
    bonefinder_points: np.array
    right_segmentation_mask: np.ndarray
    left_segmentation_mask: np.ndarray
    
    source_pixel_spacing: tuple[float, float]
    pixel_spacing: tuple[float, float]

    source_pixel_array_shape: tuple[int, int]

    top_pad: int
    bottom_pad: int
    left_pad: int
    right_pad: int

    def __init__(self,
        dicom_filepath: str,
        points_filepath: str,
        dataset: Literal['CHECK', 'OAI'],
        subject_id: str,
        subject_visit: str,
        hdf5_file_object: h5py.File,
    ) -> None:
        self.dicom_filepath   = dicom_filepath
        self.points_filepath  = points_filepath
        self.hdf5_file_object = hdf5_file_object
        self.dataset          = dataset
        self.subject_id       = subject_id
        self.subject_visit    = subject_visit

        self.top_pad    = 0
        self.bottom_pad = 0
        self.left_pad   = 0
        self.right_pad  = 0


class Transform:
    '''
    Base class for all preprocessing transforms.
    '''
    def __init__(self) -> None:
        pass

    def __call__(self, container: DicomContainer) -> DicomContainer:
        pass


class CombineTransforms(Transform):
    '''
    Chain a sequence of transforms that are applied to a DICOM container.
    '''
    transforms: list[Transform]
    
    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, container: DicomContainer) -> Transform:
        for t in self.transforms:
            container = t(container)
        return container


class LoadDicomFileObject(Transform):
    '''
    Load the file object of the DICOM sample into a `DicomContainer`.
    '''
    def __call__(self, container: DicomContainer) -> DicomContainer:
        container.dicom_file_object = dcmread(container.dicom_filepath)
        return container
    

class ReadPixelArray(Transform):
    '''
    Read the X-ray pixel array.
    The inital 2D image is wrapped within a 3D array, 
    to match the dimensionality of the combined segmentation mask.
    '''
    def __call__(self, container: DicomContainer) -> DicomContainer:
        pixel_array = container.dicom_file_object.pixel_array

        # Check that the initial image is 2D.
        if pixel_array.ndim != 2:
            raise PreprocessingException(
                f'Unexpected `pixel_arrray.ndim`. '
                f'Was: {pixel_array.ndim}. '
                f'File: {container.dicom_filepath}.'
            )

        # Get the initial shape of the pixel array
        container.source_pixel_array_shape = pixel_array.shape

        # Transform the original 2D image into a 3D one, 
        # by wrapping it in an additional dimension.
        container.pixel_array = pixel_array[None]

        return container
    

class ReadSourcePixelSpacing(Transform):
    '''
    Read the source pixel spacing as specified in the DICOM file object
    and set it as a field in the `DicomContainer`.
    '''
    def __call__(self, container: DicomContainer) -> DicomContainer:
        pixel_spacing = \
            container.dicom_file_object.get(ct.DicomAttributes.PIXEL_SPACING       ) or \
            container.dicom_file_object.get(ct.DicomAttributes.IMAGER_PIXEL_SPACING)
        
        if pixel_spacing is None:
            raise PreprocessingException(f'No pixel spacing found. File: {container.dicom_filepath}.')
        
        pixel_spacing: tuple[float, float] = tuple(pixel_spacing)

        if pixel_spacing[0] != pixel_spacing[1]:
            raise PreprocessingException(
                f'Anisotropic pixel spacing is untested.'
                f'Was: {pixel_spacing}.'
                f'File: {container.dicom_filepath}.'
            )

        container.source_pixel_spacing = pixel_spacing
        container.pixel_spacing        = pixel_spacing
        
        return container
        

class CheckPhotometricInterpretation(Transform):
    '''
    Check if pixel intensities are stored as MONOCHROME2 (white = max, black = min).
    Photometric Interpretation of DICOM images must be either MONOCHROME1 or MONOCHROME2.
    If stored as MONOCHROME1, intensities are flipped to match MONOCHROME2.
    '''
    def __call__(self, container: DicomContainer) -> DicomContainer:
        dicom_image  = container.dicom_file_object
        pixel_array  = container.pixel_array
        photo_interp = dicom_image.get(ct.DicomAttributes.PHOTOMETRIC_INTERPRETATION)
        
        if photo_interp == ct.PhotometricInterpretation.MONOCHROME1:
            # Flip intensities
            container.pixel_array = np.max(pixel_array) - pixel_array
        elif photo_interp != ct.PhotometricInterpretation.MONOCHROME2:
            raise PreprocessingException(
                f'Photometric interpretation {photo_interp} not supported. '
                f'File: {container.dicom_filepath}.'
            )
        return container
    

class CheckVoilutFunction(Transform):
    '''
    The `VOILUTFunction` attribute of the DICOM file object must be `LINEAR`.
    '''
    def __call__(self, container: DicomContainer) -> DicomContainer:
        voilut_func = container.dicom_file_object.get(
            ct.DicomAttributes.VOILUT_FUNCTION,
            ct.VoilutFunction.LINEAR,
        )
        if voilut_func != ct.VoilutFunction.LINEAR:
            raise PreprocessingException(
                f'Only supporting VOILUTFunction LINEAR. '
                f'Was: {voilut_func}. '
                f'File: {container.dicom_filepath}.'
            )
        return container


class ZScoreIntensityNormalization(Transform):
    '''
    Normalize image pixel intensities using Z-score normalization.
    '''
    def __call__(self, container: DicomContainer) -> DicomContainer:

        pixel_array = container.pixel_array.astype(float)

        mean    = np.mean(pixel_array)
        std_dev = np.std (pixel_array)

        container.pixel_array = (pixel_array - mean) / std_dev

        return container


class PercentilesIntensityNormalization(Transform):
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
                raise PreprocessingException('Percentiles must be in the interval [0.0, 100.0]. ')
            if len(percentiles) != 2:
                raise PreprocessingException('Accepting only two percentiles.')

        self.percentiles = percentiles

    def __call__(self, container: DicomContainer) -> DicomContainer:

        if self.percentiles is None:
            return container

        pixel_array = container.pixel_array.astype(float)

        percentiles = np.percentile(
            pixel_array.flatten(),
            self.percentiles,
        )

        intensity_offset = float(percentiles[0])
        intensity_slope  = float(percentiles[1] - percentiles[0])

        container.pixel_array = (pixel_array - intensity_offset) / intensity_slope

        return container
    

class MinMaxIntensityNormalization(Transform):
    '''
    Normalize image pixel intensities using MinMax normalization
    (i.e., pixels values will be in the interval `[0.0, 1.0]`).
    '''
    def __call__(self, container: DicomContainer) -> DicomContainer:
        pixel_array = container.pixel_array.astype(float)
        minv = np.min(pixel_array)
        maxv = np.max(pixel_array)
        container.pixel_array = (pixel_array - minv) / (maxv - minv)
        return container
    

class RescaleToTargetPixelSpacing(Transform):
    '''
    Rescale image to the target resolution (or target pixel spacing).

    Attributes
    ----------
    target_pixel_spacing (tuple[float, float], optional): 
        The target pixel spacing that the image resolution should match.

    Raises
    ------
    PreprocessingException: If the `target_pixel_spacing` is anisotropic.
    '''
    target_pixel_spacing: tuple[float, float]

    def __init__(self, target_pixel_spacing: tuple[float, float]=None) -> None:
        # Check that the `target_pixel_spacing` is isotropic.
        if target_pixel_spacing is not None \
            and target_pixel_spacing[0] != target_pixel_spacing[1]:
            raise PreprocessingException(
                f'Anisotropic pixel spacing is not supported. '
                f'Was: {target_pixel_spacing}.'
            )

        self.target_pixel_spacing = target_pixel_spacing

    def __call__(self, container: DicomContainer) -> DicomContainer:
        # Do not rescale if `target_pixel_spacing` is unspecified.
        if self.target_pixel_spacing is None:
            return container

        # Rescale image pixel array to match the target resolution.
        scale_factor            = container.pixel_spacing[0] / self.target_pixel_spacing[0]
        container.pixel_array   = rescale(container.pixel_array, scale_factor)
        container.pixel_spacing = self.target_pixel_spacing

        return container


class PadSymmetrically(Transform):
    '''
    Pad image and segmentation masks symmetrically 
    along each of the vertical and horizontal axes.
    
    Attributes
    ----------
    target_shape (tuple[int, int], optional): 
        Target shape to be reached by padding.

    Raises
    ------
    PreprocessingException: 
        If the current pixel array shape is larger than the target shape.
    '''
    target_shape: tuple[int, int]

    def __init__(self, target_shape: tuple[int, int]=None) -> None:
        self.target_shape = target_shape

    def __call__(self, container: DicomContainer) -> DicomContainer:

        # Do not pad if `target_shape` is unspecified.
        if self.target_shape is None:
            return container

        current_shape = container.pixel_array.shape[-2:]
        target_shape  = self.target_shape

        # Cannot pad if the image shape already exceeds the target shape
        if current_shape[0] > target_shape[0] or \
           current_shape[1] > target_shape[1]:
            raise PreprocessingException(
                f'Cannot pad image with shape {current_shape} to smaller target shape {target_shape}. '
                f'File: {container.dicom_filepath}.'
            )

        vertical_pad   = target_shape[0] - current_shape[0]
        horizontal_pad = target_shape[1] - current_shape[1]

        container.top_pad    = vertical_pad // 2
        container.bottom_pad = vertical_pad // 2 + vertical_pad % 2
        container.left_pad   = horizontal_pad // 2
        container.right_pad  = horizontal_pad // 2 + horizontal_pad % 2

        # Since the image and masks are 3D, and the intention is to pad the 2D image
        # wrapped in one additional dimension (i.e., shape is 1 x height x width), 
        # the first dimension is not padded.
        pad = (
            (0, 0),  # <--- NOTE: no pad on first dimension
            (container.top_pad, container.bottom_pad),
            (container.right_pad, container.left_pad),
        )

        container.pixel_array = np.pad(container.pixel_array, pad)

        return container


class ReadBoneFinderPoints(Transform):

    '''
    Read the array of coordinates from the BoneFinder-generated points file.

    Raises
    ------
    PreprocessingException: If the file data is not structured properly.
    '''
    def __call__(self, container: DicomContainer) -> DicomContainer:
        with open(container.points_filepath, 'r') as points_file_object:
            lines = points_file_object.readlines()
            lines = [line.strip() for line in lines]

            # Read data from the BoneFinder points file.
            version  = int(lines[0].split(' ')[1])
            n_points = int(lines[1].split(' ')[1])
            points   = np.array([
                [float(x) for x in line.split(' ')] 
                    for line in lines[3 : (n_points + 3)]
            ])

            # Verify file structure.
            if lines[0]     != f'version: {version}' \
                or lines[1]     != f'n_points: {n_points}' \
                or n_points     != ct.N_BONEFINDER_POINTS \
                or len(lines)   != n_points + 4 \
                or lines[2]     != '{' \
                or lines[163]   != '}' \
                or points.shape != (n_points, 2):

                raise PreprocessingException(
                    'Points file structure is invalid.'
                    f'File: {container.points_filepath}.'
                )

            container.bonefinder_points = points

        return container


class GetSegmentationMasks(Transform):
    '''
    Compute the one-hot encoded segmentation masks,
    highlighting the various components of the hip,
    for each of the right and left sides of the hip.

    Attributes
    ----------
    include_background_mask (bool): 
        Switch including a background submask in the final combined masks.
    '''
    include_background_mask: bool

    def __init__(self, include_background_mask: bool) -> None:
        super().__init__()
        self.include_background_mask = include_background_mask

    def __call__(self, container: DicomContainer) -> DicomContainer:
        pixel_array   = container.pixel_array
        pixel_spacing = container.pixel_spacing
        points        = container.bonefinder_points
        mask_shape    = pixel_array.shape[-2:]

        # Account for the amount of pad added to the original image when building the masks.
        pad_offset = np.array([container.left_pad, container.top_pad])

        # Fit circles for the femoral head and sourcil (acetabular roof) points.
        circles = dict()
        for hip_side, offset in ct.HipSideOffset.items():
            for curve_name, curve in ct.HipBoneSubCurve.items():
                xc, yc, r, sigma = circle_fit.taubinSVD(points[np.array(curve) + offset])
                circles[f'{hip_side} {curve_name}'] = {
                    'xc': xc,
                    'yc': yc,
                    'r': r,
                    'sigma': sigma,
                }

        # Build the segmentation masks for each hip side.
        for hip_side, offset in ct.HipSideOffset.items():

            # Define the bounding box of the segmentation region.
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

            # From most lateral part of the sourcil to center of femoral head
            femoral_head_circle = circles[f'{hip_side} {ct.HipBoneSubCurve.FEMORAL_HEAD[0]}']

            # Define the regions to be segmented (subsets of the BoneFinder points).
            regions = {
                'femoral head': np.array([
                    *points[np.array(ct.HipBoneCurve.PROXIMAL_FEMUR[1]) + offset],
                ]),
                'acetabulum': np.array([
                    *points[np.array(ct.HipBoneCurve.ACETABULAR_ROOF[1]) + offset],
                    [bbox['medial'], bbox['top']],
                ]),
                'joint space': np.array([
                    # NOTE: this polygon is larger than the joint space,
                    # but the excess will be covered by the bone regions (femur and acetabulum).
                    # - start from most lateral point of the acetabular roof,
                    points[70 + offset],
                    # - to center of femoral head,
                    [femoral_head_circle['xc'], femoral_head_circle['yc']],
                    # - to medial boundary of bbox,
                    [bbox['medial'], femoral_head_circle['yc']],
                    # - to medial top of bbox,
                    [bbox['medial'], bbox['top']],
                    # - to topmost point of acetabular roof curve
                    points[67 + offset],
                ]),
            }

            # Build masks for each region
            # (i.e., draw polygons from the subset of points for each region).
            femur_mask = polygon2mask(
                mask_shape,
                (pad_offset + regions['femoral head'] / pixel_spacing)[:, [1, 0]],
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
            
            # Create the combined mask as an array of masks for each region.
            # NOTE: the joint space mask is larger than the true area,
            # but the pixels overlapping with the other regions are set to 0.
            combined_mask = np.zeros(
                shape = (1 + len(regions), *mask_shape),
                dtype = np.uint8,
            )

            # NOTE: the background mask must be channel 0,
            # in order to not include it in the loss calculation during training.
            combined_mask[0][background_mask ] = 1
            combined_mask[1][femur_mask      ] = 1
            combined_mask[2][acetabulum_mask ] = 1
            combined_mask[3][joint_space_mask] = 1

            # Classify the pixels overlapping with the other
            # regions as background in the joint space mask.
            combined_mask[3][femur_mask      ] = 0
            combined_mask[3][acetabulum_mask ] = 0

            # If specified, remove the background submask.
            if not self.include_background_mask:
                combined_mask = combined_mask[1:]

            # Assign the combined mask to the current side of the body being segmented.
            if hip_side == ct.HipSide.RIGHT:
                container.right_segmentation_mask = combined_mask
            elif hip_side == ct.HipSide.LEFT:
                container.left_segmentation_mask = combined_mask
            else:
                raise PreprocessingException(
                    f'Side must be either `{ct.HipSide.RIGHT}` or `{ct.HipSide.LEFT}`. '
                    f'Was: `{hip_side}`. '
                    f'File: {container.dicom_filepath}.'
                )
        return container


class Flip(Transform):
    '''
    Flip the pixel array and the segmentation masks for both sides.

    Attributes
    ----------
    axis (int): Flip the image and masks around the specified axis.

    Raises
    ------
    PreprocessingException: 
        If the image and segmentation masks do not have the same number of dimensions.
    '''
    axis: int

    def __init__(self, axis: int) -> None:
        self.axis = axis

    def __call__(self, container: DicomContainer) -> DicomContainer:
        if not (
            container.pixel_array.ndim == \
            container.right_segmentation_mask.ndim == \
            container.left_segmentation_mask.ndim
        ):
            raise PreprocessingException(
                'Image and masks must have the same '
                'number of dimensions to be flipped. '
                f'File: {container.dicom_filepath}.'
            )

        container.pixel_array = np.flip(
            container.pixel_array, 
            axis=self.axis,
        )
        container.right_segmentation_mask = np.flip(
            container.right_segmentation_mask, 
            axis=self.axis,
        )
        container.left_segmentation_mask = np.flip(
            container.left_segmentation_mask, 
            axis=self.axis,
        )
        return container


class WriteToHDF5(Transform):
    '''
    Write the processed data contained within the given `DicomContainer`
    to the HDF5 file indicated by the same `DicomContainer`.

    Attributes
    ----------
    hip_side (str):
        Which hip side (right or left) is being segmented in the written sample.
        I.e., it specifies whether the image is flipped and which side of the hip 
        is contained in the segmentation mask (with the left side mask being flipped).
    '''
    hip_side: str

    def __init__(self, hip_side: str) -> None:
        self.hip_side = hip_side

    def __call__(self, container: DicomContainer) -> DicomContainer:

        # Infer which hip side is to be segmented
        segmentation_mask: np.ndarray
        if self.hip_side == ct.HipSide.RIGHT:
            segmentation_mask = container.right_segmentation_mask
        elif self.hip_side == ct.HipSide.LEFT:
            segmentation_mask = container.left_segmentation_mask
        else:
            raise PreprocessingException(
                f'Hip side can be either `\'{ct.HipSide.RIGHT}\'` or `\'{ct.HipSide.LEFT}\'`. '
                f'Was: {self.hip_side}. '
                f'File: {container.dicom_filepath}.'
            )

        # Write to HDF5:

        group_id = f'/scans/{container.dataset}/{container.subject_id}/{container.subject_visit}/{self.hip_side}'
        group    = container.hdf5_file_object.require_group(group_id)

        # Write meta info.
        group.attrs['dicom_filepath']           = container.dicom_filepath
        group.attrs['points_filepath']          = container.points_filepath
        group.attrs['dataset']                  = container.dataset
        group.attrs['subject_id']               = container.subject_id
        group.attrs['subject_visit']            = container.subject_visit
        group.attrs['hip_side']                 = self.hip_side

        group.attrs['source_pixel_spacing']     = container.source_pixel_spacing
        group.attrs['pixel_spacing']            = container.pixel_spacing
        group.attrs['source_pixel_array_shape'] = container.source_pixel_array_shape

        group.attrs['top_pad']                  = container.top_pad
        group.attrs['bottom_pad']               = container.bottom_pad
        group.attrs['left_pad']                 = container.left_pad
        group.attrs['right_pad']                = container.right_pad

        # Write the processed image.
        img_ds = group.create_dataset(
            'image', data=container.pixel_array,
            **hdf5plugin.Blosc2(cname='blosclz', clevel=9, filters=hdf5plugin.Blosc2.SHUFFLE)
        )

        # Write the (right/left) mask.
        img_ds = group.create_dataset(
            'mask', data=segmentation_mask,
            **hdf5plugin.Blosc2(cname='blosclz', clevel=9,filters=hdf5plugin.Blosc2.SHUFFLE)
        )

        return container


class PreprocessingException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
