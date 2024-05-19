"""
```
py model.py --input ./../output.h5 --model-dir ./results --test-size 0.1 --validation-size 0.1 --train-size 0.8 --device cpu --max-epochs 20 --validation-interval 1
```
"""
import argparse
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets.unet import UNet
from monai.networks.nets.basic_unet import BasicUNet
from monai.data import DataLoader, list_data_collate
import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ToTensord, 
    Resized, 
    ScaleIntensityd, 
    ImageFilterd,
)

from train import train
from dicom_dataset import DicomDataset, DicomDatasetBuilder


def main(
    input_hdf5_filepath: str,
    model_dir: str,
    test_size: float, 
    validation_size: float,
    train_size: float,
    device_name: str,
    max_epochs: int,
    validation_interval: int,
):
    #====================================================================================

    keys = ['image', 'mask']
    transform = Compose([
        # EnsureChannelFirstd(keys, channel_dim=1),
        Resized(keys, (256, 256)),
        ScaleIntensityd(keys),
    ])
    #====================================================================================

    datasets = DicomDatasetBuilder()\
        .set_hdf5_source(input_hdf5_filepath)\
        .set_transform(transform)\
        .load_samples()\
        .train_validation_test_split(
            test_size=test_size, 
            validation_size=validation_size, 
            train_size=train_size, 
        )\
        .build()
    test_dataset, validation_dataset, train_dataset = datasets

    print('Dataset sizes:', len(test_dataset), len(validation_dataset), len(train_dataset))

    test_data_loader       = DataLoader(test_dataset)
    validation_data_loader = DataLoader(validation_dataset)
    # train_data_loader      = DataLoader(train_dataset)
    train_data_loader      = DataLoader(train_dataset, batch_size=2, collate=list_data_collate)

    data_in = test_data_loader, validation_data_loader, train_data_loader

    #====================================================================================

    device = torch.device(device_name)

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        # norm=Norm.BATCH,
    ).to(device)

    #loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
    # loss_function = DiceLoss()
    loss_function = DiceLoss(sigmoid=True, include_background=False)

    # optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    train(
        model, 
        data_in, 
        loss_function, 
        optimizer, 
        max_epochs, 
        model_dir, 
        validation_interval, 
        device
    )

    return

    #====================================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', metavar='HDF5', required=True, type=str, help='input HDF5 file',)
    parser.add_argument('--model-dir', metavar='DIR', required=True, type=str, help='model output directory',)

    parser.add_argument('--test-size', required=True, type=float, help='size of the test dataset',)
    parser.add_argument('--validation-size', required=True, type=float, help='size of the validation dataset',)
    parser.add_argument('--train-size', required=True, type=float, help='size of the train dataset',)
   
    parser.add_argument('--device', required=True, type=str, help='device on which to run the model',)

    parser.add_argument('--max-epochs', required=True, type=int, help='maximum number of epochs')
    parser.add_argument('--validation-interval', required=True, type=int, help='time interval at which to periodically validate the model')

    args = parser.parse_args()

    main(
        input_hdf5_filepath = args.input,
        model_dir           = args.model_dir,
        test_size           = args.test_size,
        validation_size     = args.validation_size,
        train_size          = args.train_size,
        device_name         = args.device,
        max_epochs          = args.max_epochs,
        validation_interval = args.validation_interval,
    )
