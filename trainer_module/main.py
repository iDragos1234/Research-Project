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

import models, trainer


def main() -> None:
    args = get_args()

    model_trainer = trainer.TrainerBuilder(
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
    ).build()

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
