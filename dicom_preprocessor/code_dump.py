# from typing import Literal

# import numpy as np


# import dicom_transforms as dt
# import constants as ct


# def get_segmentation_mask(dicom_container: dt.DicomContainer) -> dt.DicomContainer:
#     pixel_array   = dicom_container.pixel_array
#     pixel_spacing = dicom_container.pixel_spacing
#     points        = dicom_container.bonefinder_points

#     circles = dict()
#     for side, offset in ct.SIDES.items():
#         for name, curve in ct.SUB_CURVES.items():
#             xc, yc, r, sigma = circle_fit.taubinSVD(points[np.array(curve) + offset])
#             circles[f'{side} {name}'] = { 'xc': xc, 'yc': yc, 'r': r, 'sigma': sigma }

#     js_bbox = {}
#     for side, offset in ct.SIDES.items():

#         combined_mask = np.zeros(shape=pixel_array.shape, dtype=np.uint8)
#         fg_mask       = np.zeros_like(combined_mask, dtype=bool)

#         # background label inside the bounding box
#         combined_mask[:] = ct.LABELS['background']
        
#         # define the bounding box of the segmentation region
#         js_bbox[side] = bbox = {
#             # top: topmost point of acetabulum curve
#             'top':     points[67 + offset][1],
#             # medial: most medial point of the sourcil
#             'medial':  points[74 + offset][0],
#             # lateral:
#             'lateral': points[8 + offset][0],
#             # bottom: medial bottom of femoral head
#             'bottom':  points[27 + offset][1],
#         }

#         # include bbox in foreground/background mask
#         fg_mask[polygon2mask(
#             fg_mask.shape,
#             np.array([
#                 [bbox['top'], bbox['lateral']],
#                 [bbox['bottom'], bbox['lateral']],
#                 [bbox['bottom'], bbox['medial']],
#                 [bbox['top'], bbox['medial']],
#             ]) / np.array(pixel_spacing)[[1, 0]]
#         )] = True

#         # from most lateral part of the sourcil to center of femoral head
#         circle = circles[f'{side} femoral head']

#         # define the regions
#         regions = {
#             'joint space': np.array([
#                 # note: this polygon is larger than the joint space,
#                 # but the excess will be covered by the bone regions
#                 # - start from most lateral point of the sourcil
#                 points[70 + offset],
#                 # - to center of femoral head
#                 [circle['xc'], circle['yc']],
#                 # - to medial boundary of bbox
#                 [bbox['medial'], circle['yc']],
#                 # - to medial top
#                 [bbox['medial'], bbox['top']],
#                 # - to topmost point of acetabulum curve
#                 points[67 + offset],
#             ]),
#             'acetabulum': np.array([
#                 *points[np.array(ct.CURVES['acetabular roof']) + offset],
#                 [bbox['medial'], bbox['top']],
#             ]),
#             'femur': np.array([
#                 *points[np.array(ct.CURVES['proximal femur']) + offset],
#             ]),
#         }

#         # add regions to mask
#         for idx, (name, region) in enumerate(regions.items()):
#             mask = polygon2mask(
#                 combined_mask.shape,
#                 (region / pixel_spacing)[:, [1, 0]]
#             )
#             combined_mask[mask] = idx + 2

#         # set background outside bounding box
#         combined_mask[~fg_mask] = ct.LABELS['ignore']

#         rgb_img = np.repeat(pixel_array[:, :, None], repeats=3, axis=2).astype(float)
#         rgb_img = rgb_img - rgb_img.min()
#         rgb_img = rgb_img / rgb_img.max()
#         rgb_seg = ct.COLORS[combined_mask, :]
#         rgb     = np.clip(rgb_seg, 0, 1)

#         if side == 'left':
#             dicom_container.left_segmentation_mask = rgb
#         elif side == 'right':
#             dicom_container.right_segmentation_mask = rgb
#         else:
#             raise RuntimeError(f'Side must be either `left` or `right`, but was: `{side}`')
    
#     return dicom_container

# #======================================================================================================

# # def plot_points(dicom_container: dt.DicomContainer) -> None:

# #     pixel_array   = dicom_container.pixel_array
# #     pixel_spacing = dicom_container.pixel_spacing
# #     points        = np.array(get_points(dicom_container))

# #     # plot the image with superimposed curves
# #     plt.imshow(pixel_array, cmap='gray')
# #     plt.colorbar()

# #     # plot curves for right and left
# #     for side, offset in SIDES.items():
# #         for idx, (name, curve) in enumerate(CURVES.items()):
# #             color = plt.rcParams['axes.prop_cycle'].by_key()['color'][idx]
# #             plt.plot(
# #                 *(points[np.array(curve) + offset] / pixel_spacing).transpose(),
# #                 marker='o', color=color,
# #                 label=f'{side} {name}',
# #             )
# #     plt.tight_layout()
# #     plt.show()
# #     return




# # def save_segmentation_mask():

# #     # set background outside bounding box
# #     combined_mask[~fg_mask] = LABELS['ignore']

# #     if img.pixel_array.shape != img_pixels.shape:
# #         print('Pixel spacing changed; DICOM-SEG not saved.')
# #     else:
# #         algorithm = hd.content.AlgorithmIdentificationSequence(
# #             name='BoneFinder ground truth',
# #             family=codes.cid7162.ArtificialIntelligence,
# #             version='1.0',
# #         )

# #         segment_descriptions = []
# #         for label, label_idx in LABELS.items():
# #             if label_idx > 0:
# #                 segment_descriptions.append(hd.seg.SegmentDescription(
# #                     segment_number=label_idx,
# #                     segment_label=label,
# #                     segmented_property_category=codes.SCT.Bone,
# #                     segmented_property_type=codes.SCT.Bone,
# #                     algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
# #                     algorithm_identification=algorithm,
# #                 ))

# #         seg_dataset = hd.seg.Segmentation(
# #             source_images=[img],
# #             pixel_array=combined_mask,
# #             segmentation_type=hd.seg.SegmentationTypeValues.FRACTIONAL,
# #             # BINARY doesn't work in Weavis, but FRACTIONAL is similar and does
# #             # segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
# #             segment_descriptions=segment_descriptions,
# #             series_instance_uid=hd.UID(),
# #             series_number=2,
# #             sop_instance_uid=hd.UID(),
# #             instance_number=1,
# #             manufacturer='Example',
# #             manufacturer_model_name='Example',
# #             software_versions='v1',
# #             device_serial_number='N/A',
# #         )
            
# #         seg_dataset.save_as(image_file_segmentation)

# #======================================================================================================

# # dicom_container = dt.DicomContainer(
# #     dicom_file_path='C:/Users/drago/Desktop/Research-Project/data/OAI/OAI-9003175-V06-20090723.dcm',
# #     points_file_path='C:/Users/drago/Desktop/Research-Project/data/OAI-pointfiles/OAI-9003175-V06-20090723.dcm.pts',
# #     hdf5_file_object=None,
# #     dataset=None,
# #     subject_id=None,
# #     subject_visit=None,
# #     target_pixel_spacing=0.5,
# # )

# # dicom_transforms = dt.CombineTransforms([
# #     dt.LoadDicomObject(),
# #     dt.GetPixelArray(),
# #     dt.GetSourcePixelSpacing(),
# #     dt.CheckPhotometricInterpretation(),
# #     dt.CheckVoilutFunction(),
# #     dt.ResampleToTargetResolution(),
# #     dt.NormalizeIntensities(),
# #     # dt.AppendDicomToHDF5(),
# # ])

# # dicom_transforms(dicom_container)

# # # plot_points(dicom_container)

# # get_rgb_segmentation_mask(dicom_container)











# # data_folder_path = './data'

# # dicom_files_metadata = ListDicomFilesMetadata()(data_folder_path)

# # for meta in tqdm.tqdm(dicom_files_metadata):
# #     dicom_file_path, points_file_path, dataset, subject_id, subject_visit = meta



