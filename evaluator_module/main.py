import sys, argparse
sys.path.append('./../research-project')

from evaluator import Evaluator
from trainer_module import models


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-hdf5',
        required=True,
        type=str,
        help='input HDF5 (.h5) filepath containing the preprocessed data',
    )
    parser.add_argument(
        '--input-data-split-csv',
        required=True,
        type=str,
        help='csv filepath where data split is specified',
    )
    parser.add_argument(
        '--input-model-state-filepath',
        required=True,
        type=str,
        help='filepath for the model state',
    )
    parser.add_argument(
        '--output-stats-dir',
        required=True,
        type=str,
        help='stats output directory path',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='device on which to test the model',
    )
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        choices=list(models.MODELS.keys()),
        help='id of the model to test',
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
        '--seed',
        type=int,
        help='seed',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
    )

    return parser.parse_args()


def main():

    args = get_args()

    model_evaluator = Evaluator(
        hdf5_filepath           = args.input_hdf5,
        data_split_csv_filepath = args.input_data_split_csv,
        model_state_filepath    = args.input_model_state_filepath,
        output_stats_dir        = args.output_stats_dir,
        device_name             = args.device,
        model_id                = args.model,
        batch_size              = args.batch_size,
        num_workers             = args.num_workers,
        seed                    = args.seed,
        verbose                 = args.verbose,
    )

    model_evaluator.evaluate()


if __name__ == '__main__':
    main()
