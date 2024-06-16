import argparse

import models, trainer


def main() -> None:
    args = get_args()

    model_trainer = trainer.Trainer(
        hdf5_filepath               = args.input_hdf5,
        data_split_csv_filepath     = args.input_data_split_csv,
        input_model_state_filepath  = args.input_model_state_filepath,
        output_model_state_filepath = args.output_model_state_filepath,

        model_id         = args.model,
        device_name      = args.device,
        max_epochs       = args.max_epochs,
        batch_size       = args.batch_size,
        num_workers      = args.num_workers,
        valid_interval   = args.validation_interval,
        output_stats_dir = args.output_stats_dir,
        seed             = args.seed,
        verbose          = args.verbose,
    )

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
        '--input-model-state-filepath',
        type=str,
        default=None,
        help='filepath for the initial model state; optional',
    )
    parser.add_argument(
        '--output-model-state-filepath',
        required=True,
        type=str,
        help='model state output file after training',
    )
    parser.add_argument(
        '--output-stats-dir',
        required=True,
        type=str,
        help='stats output directory path',
    )
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        choices=list(models.MODELS.keys()),
        help='id of the model to train',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='device on which to train the model',
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
