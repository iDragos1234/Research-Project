import sys, argparse
sys.path.append('./../research-project')
sys.path.append('./../research-project/trainer_module')

from trainer_module import models
from jsw_calculator import JSWCalculator


def get_args():
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
        '--output-dir',
        required=True,
        type=str,
        help='directory path to output the JSW results',
    )
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        choices=list(models.MODELS.keys()),
        help='id of the model to train',
    )
    parser.add_argument(
        '--input-model-state-filepath',
        type=str,
        default=None,
        help='filepath for the initial model state; optional',
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

    jsw_calculator = JSWCalculator(
        hdf5_filepath           = args.input_hdf5,
        data_split_csv_filepath = args.input_data_split_csv,
        model_id                = args.model,
        model_state_filepath    = args.input_model_state_filepath,
        output_dir              = args.output_dir,
        seed                    = args.seed,
        verbose                 = args.verbose,
    )

    jsw_calculator.calculate_jsw()

if __name__ == '__main__':
    main()