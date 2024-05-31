import logging
import os
import sys
import tempfile
from glob import glob

from matplotlib import pyplot as plt
import torch
from PIL import Image

import monai
from monai.data import decollate_batch, DataLoader
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    SpatialPadd,
    ScaleIntensityd,
)

from dicom_dataset import DicomDatasetBuilder


def main():

    '''
    Set-up preprocessing tranformations.
    '''
    keys = ['image', 'mask']
    transform = Compose([
        SpatialPadd(keys, spatial_size=(512, 512)),
        ScaleIntensityd(keys),
    ])

    #====================================================================================

    '''
    Set-up datasets and data-loaders.
    '''
    datasets = DicomDatasetBuilder()\
        .set_hdf5_source('C:/Users/drago/Desktop/research-project/output.h5')\
        .set_transform(transform)\
        .load_data()\
        .build()
    dataset = datasets[0]

    print('Dataset size:', len(dataset))

    data_loader  = DataLoader(dataset)

    #====================================================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    dice_metric = DiceMetric(include_background=True, reduction='mean', get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    model.load_state_dict(torch.load('C:/Users/drago/Desktop/best_metric_model.pth'))

    model.eval()
    with torch.no_grad():
        for data in data_loader:
            #-----------------------------------
            inputs = data['image'].to(device)
            labels = data['mask'].to(device)
            outputs = model(inputs)
            #-----------------------------------
            inputs = decollate_batch(inputs)
            labels = decollate_batch(labels)
            outputs = [post_trans(i) for i in decollate_batch(outputs)]
            #-----------------------------------
            # TODO: DELETE >>>
            if True:
                '''
                Plot overlap of predicted masks.
                '''
                plt.figure('Overlap of predicted masks')
                plt.subplot(1, 2, 1)
                plt.subplot(1, 2, 1)
                plt.title('Input image')
                plt.imshow(inputs[0][0])
                plt.subplot(1, 2, 2)
                plt.title('Overlap of predicted masks')
                plt.imshow(outputs[0][0] + outputs[0][1] + outputs[0][2])
                plt.show()

                '''
                Plot symmetric difference between real masks and predicted masks.
                '''
                plt.figure('Symmetric difference between real and predicted masks')
                plt.subplot(2, 2, 1)
                plt.title('Input image')
                plt.imshow(inputs[0][0])

                plt.subplot(2, 2, 2)
                plt.title('Femur head')
                plt.imshow(labels[0][0] - outputs[0][0])

                plt.subplot(2, 2, 3)
                plt.title('Acetabular roof')
                plt.imshow(labels[0][1] - outputs[0][1])

                plt.subplot(2, 2, 4)
                plt.title('Joint space')
                plt.imshow(labels[0][2] - outputs[0][2])

                plt.show()
                # plt.subplot(2, 4, 1)
                # for i in range(3):
                #     plt.subplot(2, 4, i + 1)
                #     plt.title(f'y channel {i}')
                #     plt.imshow(outputs[0][i] + 2 * labels[0][i])

                # plt.subplot(2, 4, 4)
                # plt.title(f'input image')
                # plt.imshow(inputs[0][0])

                # for i in range(3):
                #     plt.subplot(2, 4, i + 5)
                #     plt.title(f'y_hat channel {i}')
                #     plt.imshow(outputs[0][i] + labels[0][i])
                plt.show()
            # <<< TODO: DELETE
            #-----------------------------------
            dice_metric(y_pred=outputs, y=labels)
            print('evaluation metric:', dice_metric.aggregate().item())
            dice_metric.reset()
            #-----------------------------------

if __name__ == '__main__':
    main()
