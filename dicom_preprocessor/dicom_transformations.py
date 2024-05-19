"""
Transforms for the DICOM files.
Code adapted from `example-preprocessing-code/dicom_util.py`.
"""
from typing import Literal, Union
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


class Container:
    ...

class Transformation:
    
    def __init__(self) -> None:
        ...

    def __call__(self, container: Container):
        ...


DicomFileObject = Union[FileDataset, DicomDir]

Point = tuple[float, float]


class DicomContainer(Container):
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
    
    source_pixel_spacing: list[float]
    target_pixel_spacing: float
    pixel_spacing: list[float]

    intensity_offset: float
    intensity_slope: float
    is_flipped_horizontally: bool

    def __init__(self,
        dicom_file_path: str,
        points_file_path: str,
        hdf5_file_object: h5py.File,
        dataset: Literal['CHECK', 'OAI'],
        subject_id: str,
        subject_visit: str,
        target_pixel_spacing: float = None,
        intensity_slope: float = 1.0,
        intensity_offset: float = 0.0,
        is_flipped_horizontally: bool = False
    ) -> None:
        self.dicom_file_path         = dicom_file_path
        self.points_file_path        = points_file_path
        self.hdf5_file_object        = hdf5_file_object

        self.dataset                 = dataset
        self.subject_id              = subject_id
        self.subject_visit           = subject_visit

        self.target_pixel_spacing    = target_pixel_spacing

        self.intensity_slope         = intensity_slope
        self.intensity_offset        = intensity_offset

        self.is_flipped_horizontally = is_flipped_horizontally
        return


class DicomTransformation(Transformation):

    def __init__(self) -> None:
        ...

    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        ...


class SequenceTransformations(Transformation):

    transforms: list[Transformation]
    
    def __init__(self, transforms: list[Transformation]) -> None:
        super().__init__()
        self.transforms = transforms
        return

    def __call__(self, container: Container) -> Container:
        for transform in self.transforms:
            container = transform(container)
        return container


class LoadDicomObject(DicomTransformation):
    """
    Loads the image from the specified DICOM file into a `DicomContainer`.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        dicom_container.dicom_file_object = dcmread(dicom_container.dicom_file_path)
        return dicom_container
    

class GetPixelArray(DicomTransformation):
    """
    Get the pixel array from the `DicomObject` and set it as a field in the `DicomContainer`.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        dicom_container.pixel_array = dicom_container.dicom_file_object.pixel_array
        return dicom_container
    

class GetSourcePixelSpacing(DicomTransformation):
    """
    Get the pixel spacing from the `DicomObject` and set it as a field in the `DicomContainer`.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        dicom_image: DicomFileObject = dicom_container.dicom_file_object
        pixel_spacing: list[float]   = dicom_image.get('PixelSpacing') or \
                                       dicom_image.get('ImagerPixelSpacing')
        
        if pixel_spacing is None:
            raise PreprocessingException('No pixel spacing found.')
        if pixel_spacing[0] != pixel_spacing[1]:
            raise PreprocessingException('Anisotropic pixel spacing is untested.')

        dicom_container.source_pixel_spacing = pixel_spacing
        dicom_container.pixel_spacing        = pixel_spacing
        
        return dicom_container
        

class CheckPhotometricInterpretation(DicomTransformation):
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
            raise PreprocessingException(
                f'Photometric interpretation {photo_interp} not supported.'
            )
        return dicom_container
    

class CheckVoilutFunction(DicomTransformation):
    """
    The `VOILUTFunction` property of the DICOM image must be `LINEAR`.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        voilut_func = dicom_container.dicom_file_object.get('VOILUTFunction', 'LINEAR')
        if voilut_func != 'LINEAR':
            raise PreprocessingException(f'Only supporting VOILUTFunction LINEAR. Was {voilut_func}.')
        return dicom_container
    

class ResampleToTargetResolution(DicomTransformation):
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
    

class NormalizeIntensities(DicomTransformation):
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
    

class GetBoneFinderPoints(DicomTransformation):
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        points_file_path = dicom_container.points_file_path

        with open(points_file_path, 'r') as points_file_object:

            lines = points_file_object.readlines()
            lines = [line.strip() for line in lines]

            # Extract information from file
            version  = int(lines[0].split(' ')[1])
            n_points = int(lines[1].split(' ')[1])
            points   = np.array([
                [float(x) for x in line.split(' ')] for line in lines[3 : (n_points + 3)]
            ])

            # Verify file structure
            if lines[0]     != f'version: {version}' \
            or lines[1]     != f'n_points: {n_points}' \
            or n_points     != ct.N_POINTS \
            or len(lines)   != n_points + 4 \
            or lines[2]     != '{' \
            or lines[163]   != '}' \
            or points.shape != (n_points, 2):
                raise PreprocessingException('Points file structure is invalid.')

            dicom_container.bonefinder_points = points

        return dicom_container


class GetSegmentationMasks(DicomTransformation):
    """
    Compute the RGB pixel array of the segmentation mask,
    highlighting the various components of the hip.
    In the `DicomContainer` it is specified whether to compute 
    the mask for the left, right or both sides of the hip.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        pixel_array   = dicom_container.pixel_array
        pixel_spacing = dicom_container.pixel_spacing
        points        = dicom_container.bonefinder_points

        circles = dict()
        for side, offset in ct.SIDES.items():
            for name, curve in ct.SUB_CURVES.items():
                xc, yc, r, sigma = circle_fit.taubinSVD(points[np.array(curve) + offset])
                circles[f'{side} {name}'] = { 'xc': xc, 'yc': yc, 'r': r, 'sigma': sigma }

        js_bbox = {}
        for side, offset in ct.SIDES.items():

            combined_mask = np.zeros(shape=pixel_array.shape, dtype=np.uint8)
            fg_mask       = np.zeros_like(combined_mask, dtype=bool)

            # background label inside the bounding box
            combined_mask[:] = ct.LABELS['background']
            
            # define the bounding box of the segmentation region
            js_bbox[side] = bbox = {
                # top: topmost point of acetabulum curve
                'top':     points[67 + offset][1],
                # medial: most medial point of the sourcil
                'medial':  points[74 + offset][0],
                # lateral:
                'lateral': points[8 + offset][0],
                # bottom: medial bottom of femoral head
                'bottom':  points[27 + offset][1],
            }

            # include bbox in foreground/background mask
            fg_mask[polygon2mask(
                fg_mask.shape,
                np.array([
                    [bbox['top'], bbox['lateral']],
                    [bbox['bottom'], bbox['lateral']],
                    [bbox['bottom'], bbox['medial']],
                    [bbox['top'], bbox['medial']],
                ]) / np.array(pixel_spacing)[[1, 0]]
            )] = True

            # from most lateral part of the sourcil to center of femoral head
            circle = circles[f'{side} femoral head']

            # define the regions
            regions = {
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
                'acetabulum': np.array([
                    *points[np.array(ct.CURVES['acetabular roof']) + offset],
                    [bbox['medial'], bbox['top']],
                ]),
                'femur': np.array([
                    *points[np.array(ct.CURVES['proximal femur']) + offset],
                ]),
            }

            # add regions to mask
            for idx, (name, region) in enumerate(regions.items()):
                mask = polygon2mask(
                    combined_mask.shape,
                    (region / pixel_spacing)[:, [1, 0]]
                )
                combined_mask[mask] = idx + 2

            # set background outside bounding box
            combined_mask[~fg_mask] = ct.LABELS['ignore']

            # rgb_img = np.repeat(pixel_array[:, :, None], repeats=3, axis=2).astype(float)
            # rgb_img = rgb_img - rgb_img.min()
            # rgb_img = rgb_img / rgb_img.max()
            # rgb_seg = ct.COLORS[combined_mask, :]
            # rgb     = np.clip(rgb_seg, 0, 1)

            if side == 'left':
                dicom_container.left_segmentation_mask = combined_mask
            elif side == 'right':
                dicom_container.right_segmentation_mask = combined_mask
            else:
                raise RuntimeError(f'Side must be either `left` or `right`. Was `{side}`')
        
        return dicom_container
    

class FlipHorizontally(DicomTransformation):
    """
    Flip the pixel array and the segmentation masks for both sides horizontally.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:
        dicom_container.pixel_array = np.flip(
            dicom_container.pixel_array, 
            axis=1,
        )
        dicom_container.right_segmentation_mask = np.flip(
            dicom_container.right_segmentation_mask, 
            axis=1,
        )
        dicom_container.left_segmentation_mask = np.flip(
            dicom_container.left_segmentation_mask, 
            axis=1,
        )
        dicom_container.is_flipped_horizontally = \
            not dicom_container.is_flipped_horizontally
        return dicom_container


class AppendDicomToHDF5(DicomTransformation):
    """
    Write the information contained within the given `DicomContainer` to the HDF5 file indicated by the same `DicomContainer`.
    """
    def __call__(self, dicom_container: DicomContainer) -> DicomContainer:

        dicom_file_path         = dicom_container.dicom_file_path
        points_file_path        = dicom_container.points_file_path
        hdf5_file_object        = dicom_container.hdf5_file_object

        dataset                 = dicom_container.dataset
        subject_id              = dicom_container.subject_id
        subject_visit           = dicom_container.subject_visit

        dicom_file_object       = dicom_container.dicom_file_object
        pixel_array             = dicom_container.pixel_array
        bonefinder_points       = dicom_container.bonefinder_points
        right_segmentation_mask = dicom_container.right_segmentation_mask
        left_segmentation_mask  = dicom_container.left_segmentation_mask

        source_pixel_spacing    = dicom_container.source_pixel_spacing
        target_pixel_spacing    = dicom_container.target_pixel_spacing
        pixel_spacing           = dicom_container.pixel_spacing

        intensity_slope         = dicom_container.intensity_slope
        intensity_offset        = dicom_container.intensity_offset
        is_flipped_horizontally = dicom_container.is_flipped_horizontally

        # Infer which hip side is to be segmented
        hip_side: Literal['right', 'left']
        segmentation_mask: np.ndarray
        if is_flipped_horizontally:
            hip_side = 'left'
            segmentation_mask = left_segmentation_mask
        else:
            hip_side = 'right'
            segmentation_mask = right_segmentation_mask

        # Write to hdf5
        group_id = f'/scans/{dataset}/{subject_id}/{subject_visit}/{hip_side}'
        group = hdf5_file_object.require_group(group_id)

        group.attrs['dicom_file_path'] = dicom_file_path
        group.attrs['points_file_path'] = points_file_path
        # group.attrs['hdf5_file_object'] = hdf5_file_object
        group.attrs['dataset'] = dataset
        group.attrs['subject_id'] = subject_id
        group.attrs['subject_visit'] = subject_visit
        # group.attrs['dicom_file_object'] = dicom_file_object
        # group.attrs['pixel_array'] = pixel_array
        # group.attrs['bonefinder_points'] = bonefinder_points
        # group.attrs['right_segmentation_mask'] = right_segmentation_mask
        # group.attrs['left_segmentation_mask'] = left_segmentation_mask
        group.attrs['source_pixel_spacing'] = source_pixel_spacing
        group.attrs['target_pixel_spacing'] = target_pixel_spacing
        group.attrs['pixel_spacing'] = pixel_spacing
        group.attrs['intensity_slope'] = intensity_slope
        group.attrs['intensity_offset'] = intensity_offset
        group.attrs['is_flipped_horizontally'] = is_flipped_horizontally

        img_ds = group.create_dataset(
            'image', data=pixel_array,
            **hdf5plugin.Blosc2(cname='blosclz', clevel=9, filters=hdf5plugin.Blosc2.SHUFFLE)
        )

        img_ds = group.create_dataset(
            'segmentation_mask', data=segmentation_mask,
            **hdf5plugin.Blosc2(cname='blosclz', clevel=9,filters=hdf5plugin.Blosc2.SHUFFLE)
        )

        return dicom_container


class PreprocessingException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
