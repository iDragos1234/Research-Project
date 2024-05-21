"""
```
py model.py --input ./../output.h5 --model-dir ./results --test-size 0.1 --validation-size 0.1 --train-size 0.8 --device cpu --learning-rate 1e-3 --weight-decay 0 --max-epochs 20 --batch-size 2 --num-workers 0 --validation-interval 1 --seed 42 --verbose
```
"""
import argparse
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets.unet import UNet
from monai.data import DataLoader, list_data_collate
from monai.transforms import (
    Compose,
    Resized,
    ScaleIntensityd,
    Activations,
    AsDiscrete,
)
from monai.utils import set_determinism

from train import train
from dicom_dataset import DicomDatasetBuilder


def main(
    input_hdf5_filepath: str,
    model_dir: str,

    test_size: float, 
    valid_size: float,
    train_size: float,

    device_name: str,

    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    batch_size: int,
    num_workers: int,
    validation_interval: int,
    verbose: bool,
    seed: int,
):
    
    '''
    Set seed.
    '''
    set_determinism(seed)

    #====================================================================================

    '''
    Set-up preprocessing tranformations.
    '''
    keys = ['image', 'mask']
    transform = Compose([
        Resized(keys, (256, 256)),
        ScaleIntensityd(keys),
    ])

    #====================================================================================

    '''
    Set-up datasets and data-loaders.
    '''
    datasets = DicomDatasetBuilder()\
        .set_hdf5_source(input_hdf5_filepath)\
        .set_transform(transform)\
        .load_data()\
        .split_data([test_size, valid_size, train_size])\
        .build()
    test_dataset, valid_dataset, train_dataset = tuple(datasets)

    if verbose:
        print('Dataset sizes:', len(test_dataset), len(valid_dataset), len(train_dataset))

    test_data_loader  = DataLoader(
        test_dataset,
        num_workers=num_workers,    
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        num_workers=num_workers,  
    )
    train_data_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    data_in = test_data_loader, valid_data_loader, train_data_loader

    #====================================================================================

    '''
    Initialize U-Net model, Dice loss function, Dice metric and optimizer.
    '''
    device = torch.device(device_name)

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    dice_loss = DiceLoss(sigmoid=True)
    dice_metric = DiceMetric()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    '''
    Add post-processing transformations
    '''
    post_transformations = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5),
    ])

    #====================================================================================

    '''
    Start the training process.
    '''
    train(
        model                = model, 
        data_in              = data_in, 
        loss_function        = dice_loss, 
        metric_function      = dice_metric,
        optimizer            = optimizer, 
        max_epochs           = max_epochs, 
        model_directory_path = model_dir, 
        validation_interval  = validation_interval, 
        device               = device,
        post_transf          = post_transformations,
        verbose              = verbose,
    )

    return

    #====================================================================================


if __name__ == '__main__':

    '''
    Parse command-line arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', metavar='HDF5', required=True, type=str, help='input HDF5 file',)
    parser.add_argument('--model-dir', metavar='DIR', required=True, type=str, help='model output directory',)

    parser.add_argument('--test-size', required=True, type=float, help='size of the test dataset',)
    parser.add_argument('--validation-size', required=True, type=float, help='size of the validation dataset',)
    parser.add_argument('--train-size', required=True, type=float, help='size of the train dataset',)
   
    parser.add_argument('--device', required=True, type=str, help='device on which to run the model',)

    parser.add_argument('--learning-rate', required=True, type=float, help='learning rate')
    parser.add_argument('--weight-decay', required=True, type=float, help='learning rate')
    parser.add_argument('--max-epochs', required=True, type=int, help='maximum number of epochs')
    parser.add_argument('--batch-size', required=True, type=int, help='training batch size')
    parser.add_argument('--num-workers', required=True, type=int, help='`num_workers` parameter of `DataLoader`')
    parser.add_argument('--validation-interval', required=True, type=int, help='time interval at which to periodically validate the model')
    parser.add_argument('--seed', required=False, type=int, help='seed')

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    

    '''
    Create datasets and model, and start training
    '''
    main(
        input_hdf5_filepath = args.input,
        model_dir           = args.model_dir,

        test_size           = args.test_size,
        valid_size          = args.validation_size,
        train_size          = args.train_size,

        device_name         = args.device,
        
        learning_rate       = args.learning_rate,
        weight_decay        = args.weight_decay,
        max_epochs          = args.max_epochs,
        batch_size          = args.batch_size,
        num_workers         = args.num_workers,
        validation_interval = args.validation_interval,

        verbose             = args.verbose,
        seed                = args.seed,
    )
