'''
Data splitter module
--------------------

* Splits the data residing in the input HDF5 file
    into training, validation and testing datasets.

* How to run:
    * open a bash terminal;
    * run the following command:
    ```
    python ./data_splitter.py \
        --input output.h5 \
        --output data_split.csv \
        --ratios 0.8 0.1 0.1 \
        --seed 42 \
        --verbose
    ```
'''
from typing import Union
import time, argparse, csv
import h5py, hdf5plugin  # <--- DO NOT REMOVE!
from torch import Generator
from torch.utils.data import random_split


class DataSplitter:
    '''
    TODO
    '''
    hdf5_filepath: str
    csv_filepath: str
    ratios: list[float]
    seed: Union[int, None]
    verbose: bool

    def __init__(self,
        hdf5_filepath: str,
        csv_filepath: str,
        ratios: list[float],
        seed: Union[int, None],
        verbose: bool,
    ) -> None:
        if not all(0.0 <= r <= 1.0 for r in ratios):
            raise DataSplitException('Ratios must be in the interval [0.0, 1.0].')
        if not (1.0 - 1e-5 <= abs(sum(ratios)) <= 1.0 + 1e-5):
            raise DataSplitException('Ratios must sum up to 1.')

        self.hdf5_filepath = hdf5_filepath
        self.csv_filepath  = csv_filepath
        self.ratios        = ratios
        self.seed          = seed
        self.verbose       = verbose

    def __call__(self):
        if self.verbose:
            print('Starting data splitting...')

        start_time = time.time()

        samples    = self._find_samples(self.hdf5_filepath)
        data_split = self._split_data(samples, self.ratios, self.seed)
        _          = self._write_data_split_to_csv_file(data_split, self.csv_filepath)

        if self.verbose:
            print(
                f'Finished data splitting.\n'
                f'  - Ellapsed time:           {(time.time() - start_time):.4f}s;\n'
                f'  - Total number of samples: {len(samples)};\n'
                f'  - Datasets sizes:          {[len(s) for s in data_split]}.'
            )

    @staticmethod
    def _find_samples(hdf5_filepath: str) -> list[str]:
        samples: list[str] = list()
        with h5py.File(hdf5_filepath, 'r') as hdf5_file_object:
            for dataset_name in hdf5_file_object['/scans']:
                dataset_path = f'/scans/{dataset_name}'
                for subject_id in hdf5_file_object[dataset_path]:
                    subject_id_path = f'{dataset_path}/{subject_id}'
                    for subject_visit in hdf5_file_object[subject_id_path]:
                        subject_visit_path = f'{subject_id_path}/{subject_visit}'
                        for hip_side in hdf5_file_object[subject_visit_path]:
                            hip_side_path = f'{subject_visit_path}/{hip_side}'
                            samples.append(hip_side_path)
        return sorted(samples)
    
    @staticmethod
    def _split_data(
        samples: list[str],
        ratios: list[float],
        seed: Union[int, None],
    ):
        if seed is None:
            return random_split(samples, ratios)

        generator = Generator().manual_seed(seed)
        return random_split(samples, ratios, generator)
    
    @staticmethod
    def _write_data_split_to_csv_file(
        data_split,
        csv_filepath: str
    ) -> None:
        header = ['sample', 'dataset']
        
        rows = []
        for dataset_id, dataset in enumerate(data_split):
            rows.extend([sample, dataset_id] for sample in dataset)

        # Write to CSV file.    
        with open(csv_filepath, 'w') as csv_file_object:
            csv_writer = csv.writer(csv_file_object)
            csv_writer.writerow(header)
            csv_writer.writerows(rows)
        return
    

class DataSplitException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        required=True,
        type=str,
        help='input HDF5 filepath',
    )
    parser.add_argument(
        '--output',
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
    args = get_args()

    data_splitter = DataSplitter(
        hdf5_filepath = args.input,
        csv_filepath  = args.output,
        ratios        = args.ratios,
        seed          = args.seed,
        verbose       = args.verbose,
    )

    data_splitter()
