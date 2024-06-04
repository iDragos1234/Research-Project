import argparse

from data_splitter import DataSplitter


def main():
    args = get_args()

    data_splitter = DataSplitter(
        hdf5_filepath = args.input_hdf5,
        csv_filepath  = args.output_csv,
        ratios        = args.ratios,
        seed          = args.seed,
        verbose       = args.verbose,
    )

    data_splitter()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-hdf5',
        required=True,
        type=str,
        help='input HDF5 filepath',
    )
    parser.add_argument(
        '--output-csv',
        required=True,
        type=str,
        help='output CSV filepath',
    )
    parser.add_argument(
        '--ratios',
        type=float,
        nargs='*',
        default=[1],
        help='data split ratios',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='set seed for data shuffling',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
