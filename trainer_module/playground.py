import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from monai.transforms import (
    Compose,
    Resized,
    ScaleIntensityd,
    SpatialPadd,
    GaussianSharpen,
    ImageFilter,
    ResizeWithPadOrCropd,
)
import dtw

from dicom_dataset import DicomDatasetBuilder


# keys = ['image', 'mask']
# transform = Compose([
#     # Resized(keys, spatial_size=(256, 256)),
#     # SpatialPadd(keys, (512, 512)),
#     # ScaleIntensityd(keys),
#     # ResizeWithPadOrCropd(keys, spatial_size=512),
# ])
transformations = None


datasets = DicomDatasetBuilder(
    './output.h5',
    'data_split.csv',
).build()
dataset = datasets[0]




if True:
    for idx, sample_id in enumerate(dataset.data):

        sample = dataset[idx]
        image, mask = sample['image'], sample['mask']

        print('Shapes:', image.shape, mask.shape)

        plt.subplot(1, 2, 1)
        plt.title(f'Plot #{idx + 1} - sample {sample_id}')
        plt.imshow(image[0])
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title(f'Plot #{idx + 1} - sample {sample_id}')
        plt.imshow(sum((idx + 1) * submask for idx, submask in enumerate(mask)))
        plt.colorbar()
        plt.show()





test_sample = dataset[0]

image = test_sample['image']
mask  = test_sample['mask']


from joint_space_width.joint_space_width import joint_space_width
min_jsw, jsw_array = joint_space_width(mask, 0.2)
plt.title('JSW')
plt.plot(range(len(jsw_array)), jsw_array, 'o-')
plt.show()




def foo(image, mask):
    # borders = torch.zeros(image[0].shape)

    image = image[:, 800:1000, 500:700]
    mask  = mask [:, 800:1000, 500:700]
    
    femur_head_lower = mask[0] * mask[2].roll(+1, 0)
    femur_head_upper = mask[0].roll(-1, 0) * mask[2]
    acetabulum_upper = mask[1] * mask[2].roll(-1, 0)
    acetabulum_lower = mask[1].roll(+1, 0) * mask[2]

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
    seqA = femur_head_pts
    seqB = acetabulum_pts

    # Sort sequences by the x-axis
    seqA = np.array(sorted(seqA, key=lambda p: p[1]))
    seqB = np.array(sorted(seqB, key=lambda p: p[1]))

    # Optional: swap sequences
    seqA, seqB = seqB, seqA

    # Optional: select elements
    # seqB = seqB[1:]

    # Get DTW alignment
    alignment = dtw.dtw(seqA, seqB, step_pattern='asymmetric')

    # Get array of distances
    distances = np.sqrt(np.sum(
        (seqA[alignment.index1] - seqB[alignment.index2]) ** 2,
        axis=-1,
    ))

    # Scale distances by pixel spacing
    distances *= 0.2

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


foo(image, mask)


def bar(image, mask):
    # borders = torch.zeros(image[0].shape)

    image = image[:, 880:940, 530:700]
    mask  = mask [:, 880:940, 530:700]
    
    femur_head_lower = mask[0] * mask[2].roll(+1, 0)
    femur_head_upper = mask[0].roll(-1, 0) * mask[2]
    acetabulum_upper = mask[1] * mask[2].roll(-1, 0)
    acetabulum_lower = mask[1].roll(+1, 0) * mask[2]

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
    seqA = femur_head_pts
    seqB = acetabulum_pts

    # Optional: swap sequences
    seqA, seqB = seqB, seqA

    # Compute distance matrix
    matrix = np.array([
        np.sqrt(np.sum((p - seqB) ** 2, axis=-1)) for p in seqA
    ])
    matrix *= 0.2

    # Compute alignment and distances
    alignment = np.argmin(matrix, axis=-1)
    alignment = list(zip(range(len(alignment)), alignment))
    
    distances = np.min(matrix, axis=-1)
    print(distances[70:80])

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

bar(image, mask)
