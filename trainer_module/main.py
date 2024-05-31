'''
```
python ./data_loader/main.py \
    --input-hdf5 ./output.h5 \
    --input-data-split-csv ./data_split.csv \
    --output-model-dir ./results \
    --device cpu \
    --learning-rate 1e-3 \
    --weight-decay 1e-5 \
    --max-epochs 20 \
    --batch-size 2 \
    --num-workers 0 \
    --validation-interval 1 \
    --seed 42 \
    --verbose
```
'''
import argparse
import torch
from monai.utils import set_determinism

import data_loader_builder
import models
import trainer


def main():
    args = get_args()

    train(
        hdf5_filepath           = args.input_hdf5,
        data_split_csv_filepath = args.input_data_split_csv,
        model_dir_path          = args.output_model_dir,
        model_id                = args.model,

        device_name             = args.device,

        learning_rate           = args.learning_rate,
        weight_decay            = args.weight_decay,
        max_epochs              = args.max_epochs,
        batch_size              = args.batch_size,
        num_workers             = args.num_workers,
        validation_interval     = args.validation_interval,

        seed                    = args.seed,
        verbose                 = args.verbose,
    )



def train(
    hdf5_filepath: str,
    data_split_csv_filepath: str,
    model_dir_path: str,
    model_id: str,

    device_name: str,

    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    batch_size: int,
    num_workers: int,
    validation_interval: int,

    seed: int,
    verbose: bool,
):
    # Set seed for reproducibility purposes.
    set_determinism(seed)


    # Build datasets (training, validation and testing)
    # using the split specified in the data split CSV file.
    (
        train_data_loader,
        valid_data_loader,
        test_data_loader,  # <--- Not used for training
    ) = data_loader_builder.DataLoaderBuilder(
        hdf5_filepath,
        data_split_csv_filepath,
        batch_size,
        num_workers,
        verbose,
    ).build()

    # Get the specified device (`'cpu'` or `'cuda'`).
    device = torch.device(device_name)

    # Fetch the selected model setting to be trained.
    model_setting = models.MODELS[model_id](
        learning_rate,
        weight_decay,
    )

    # Initialize model trainer with the selected model setting.
    model_trainer = trainer.Trainer(
        model_setting,
        train_data_loader,
        valid_data_loader,
        device,
        max_epochs,
        model_dir_path,
        validation_interval,
        verbose,
    )

    # Train the model.
    model_trainer.train()

    return


def get_args() -> argparse.Namespace:
    '''
    Parse command-line arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-hdf5',
        required=True,
        type=str,
        help='input HDF5 (.h5) filepath',
    )
    parser.add_argument(
        '--input-data-split-csv',
        required=True,
        type=str,
        help='csv filepath where data split is specified',
    )
    parser.add_argument(
        '--output-model-dir',
        required=True,
        type=str,
        help='model training output directory',
    )
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        choices=list(models.MODELS.keys()),
        help='select the model to train',
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='device on which to train the model',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='learning rate',
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0,
        help='weight decay',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=100,
        help='maximum number of epochs',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='data loaders batch size',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='data loaders number of workers parameter',
    )
    parser.add_argument(
        '--validation-interval',
        type=int,
        default=1,
        help='epoch interval at which to periodically validate the model',
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='seed',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
