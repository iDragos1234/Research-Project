import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
import dtw

from dicom_dataset import DicomDatasetBuilder


datasets = DicomDatasetBuilder(
    './all_no_bg.h5',
    'data_split.csv',
).build()
dataset = datasets[0]


if False:
    for idx, sample_id in enumerate(dataset.data):

        sample = dataset[idx]
        image, mask = sample['image'], sample['mask']

        print('Shapes:', image.shape, mask.shape)

        plt.subplot(1, 2, 1)
        # plt.title(f'Plot #{idx + 1} - sample {sample_id}')
        plt.imshow(image[0], 'grey')
        # plt.colorbar()

        plt.subplot(1, 2, 2)
        # plt.title(f'Plot #{idx + 1} - sample {sample_id}')
        plt.imshow(sum((idx + 1) * submask for idx, submask in enumerate(mask)), 'grey')
        # plt.colorbar()
        plt.show()




def foo(
    image,
    mask,
    source_pixel_spacing,
    pixel_spacing,
):
    # borders = torch.zeros(image[0].shape)

    # image = image[:, 800:1000, 500:700]
    # mask  = mask [:, 800:1000, 500:700]
    
    femur_head_lower = mask[0] * np.roll(mask[2], +1, 0)
    femur_head_upper = mask[2] * np.roll(mask[0], -1, 0)
    acetabulum_upper = mask[1] * np.roll(mask[2], -1, 0)
    acetabulum_lower = mask[2] * np.roll(mask[1], +1, 0)

    femur_head_lower_pts = np.argwhere(femur_head_lower)
    femur_head_upper_pts = np.argwhere(femur_head_upper)
    acetabulum_upper_pts = np.argwhere(acetabulum_upper)
    acetabulum_lower_pts = np.argwhere(acetabulum_lower)

    femur_head_pts = (femur_head_lower_pts + femur_head_upper_pts) / 2
    acetabulum_pts = (acetabulum_lower_pts + acetabulum_upper_pts) / 2

    # Create combined mask
    combined = 2 * (femur_head_lower + acetabulum_upper) + \
               3 * (femur_head_upper + acetabulum_lower)
    combined += mask[0] + mask[1] + mask[2]

    # DTW
    seqA = femur_head_pts.T
    seqB = acetabulum_pts.T

    print(seqA.shape, seqA.dtype, seqB.shape, seqB.dtype)

    # Sort sequences by the x-axis
    seqA = np.array(sorted(seqA, key=lambda p: p[1]))
    seqB = np.array(sorted(seqB, key=lambda p: p[1]))

    # Optional: swap sequences
    seqA, seqB = seqB, seqA

    # Optional: select elements
    # seqB = seqB[1:]

    # Get DTW alignment
    alignment = dtw.dtw(seqA, seqB, step_pattern='symmetric2')

    # Get array of distances
    distances = np.sqrt(np.sum(
        (seqA[alignment.index1] - seqB[alignment.index2]) ** 2,
        axis=-1,
    ))

    # Scale distances by pixel spacing
    distances *= (source_pixel_spacing / pixel_spacing)

    # Plot DTW alignments
    plt.figure('DTW in action')
    plt.subplot(1, 2, 1)
    for i, j in zip(alignment.index1, alignment.index2):
        plt.plot(
            [seqA[i, 1], seqB[j, 1]],
            [seqA[i, 0], seqB[j, 0]],
            'c',
        )

    # Plot combined mask
    plt.imshow(combined)

    # Plot distances
    plt.subplot(1, 2, 2)
    plt.plot(list(range(len(distances))), distances, 'o-')
    plt.show()


def bar(
    image,
    mask,
    source_pixel_spacing,
    pixel_spacing,
):
    
    image = image[:, 200:300, 50:200]
    mask  = mask [:, 200:300, 50:200]

    femur_head_lower = mask[0] * np.roll(mask[2], +1, 0)
    femur_head_upper = mask[2] * np.roll(mask[0], -1, 0)
    acetabulum_upper = mask[1] * np.roll(mask[2], -1, 0)
    acetabulum_lower = mask[2] * np.roll(mask[1], +1, 0)

    femur_head_lower_pts = np.argwhere(femur_head_lower)
    femur_head_upper_pts = np.argwhere(femur_head_upper)
    acetabulum_upper_pts = np.argwhere(acetabulum_upper)
    acetabulum_lower_pts = np.argwhere(acetabulum_lower)

    femur_head_pts = (femur_head_lower_pts + femur_head_upper_pts) / 2
    acetabulum_pts = (acetabulum_lower_pts + acetabulum_upper_pts) / 2

    # Create combined mask
    combined = 2 * (femur_head_lower + acetabulum_upper) + \
               3 * (femur_head_upper + acetabulum_lower)
    combined += mask[0] + mask[1] + mask[2]

    # Sequences
    seqA = femur_head_pts.T
    seqB = acetabulum_pts.T

    seqA = np.array(sorted(seqA, key=lambda p: p[1]))
    seqB = np.array(sorted(seqB, key=lambda p: p[1]))

    print(seqA.shape, seqA.dtype, seqB.shape, seqB.dtype)

    # Optional: swap sequences
    # seqA, seqB = seqB, seqA

    # Compute distance matrix
    matrix = np.array([
        np.sqrt(((p - seqB) ** 2).sum(axis=-1)) for p in seqA
    ])

    # Scale distances to real (source) pixel spacing
    matrix *= (source_pixel_spacing / pixel_spacing)

    # Compute alignment and distances
    alignment = np.argmin(matrix, axis=-1)
    alignment = list(zip(range(len(alignment)), alignment))
    
    distances = np.min(matrix, axis=-1)

    # Get minJSW:
    m, n = matrix.shape
    idx = np.argmin(matrix)
    minX, minY = idx // n, idx % n
    mJSW = matrix[minX, minY]
    print(mJSW)

    # Plot combined mask
    plt.subplot(1, 2, 1)
    for i, j in alignment:
        plt.plot(
            [seqA[i, 1], seqB[j, 1]],
            [seqA[i, 0], seqB[j, 0]],
        )
    plt.imshow(combined)

    # Plot distances
    plt.subplot(1, 2, 2)
    plt.plot(list(range(len(distances))), distances, 'o-')
    plt.show()



for idx in range(len(dataset)):

    sample = dataset[idx]

    image = sample['image']
    mask  = sample['mask' ]

    meta  = dataset.get_item_meta(idx)

    # foo(
    #     image,
    #     mask,
    #     meta['group attributes']['source_pixel_spacing'][0],
    #     meta['group attributes']['pixel_spacing'       ][0],
    # )

    bar(
        image,
        mask,
        meta['group attributes']['source_pixel_spacing'][0],
        meta['group attributes']['pixel_spacing'       ][0],
    )