import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, 
    Resized, 
    ScaleIntensityd, 
    GaussianSharpen, 
    ImageFilter,
)
import numpy as np

from dicom_dataset import DicomDatasetBuilder


keys = ['image', 'mask']
transform = Compose([
    Resized(keys, (256, 256)),
    ScaleIntensityd(keys),
])

datasets = DicomDatasetBuilder()\
    .set_hdf5_source('C:\\Users\\drago\\Desktop\\Research-Project\\output.h5')\
    .set_transform(transform)\
    .load_data()\
    .build()
dataset = datasets[0]


for idx, sample_id in enumerate(dataset.data[:2]):

    sample = dataset[idx]
    image, mask = sample['image'], sample['mask']

    print(image.shape, mask.shape)

    plt.title(f'Plot #{idx + 1} - sample {sample_id}')
    plt.imshow(image[0])
    plt.colorbar()
    plt.show()

    plt.title(f'Plot #{idx + 1} - sample {sample_id}')
    plt.imshow(mask[0] + 2 * mask[1] + 3 * mask[2])
    plt.colorbar()
    plt.show()
