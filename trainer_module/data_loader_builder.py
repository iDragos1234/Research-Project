import torch
from monai.data import DataLoader, list_data_collate

import dicom_dataset as dicom_dataset

class DataLoaderBuilder:

    hdf5_filepath: str
    data_split_csv_filepath: str
    
    batch_size: int
    num_workers: int
    verbose: bool

    def __init__(self,
        hdf5_filepath: str,
        data_split_csv_filepath: str,
        batch_size: int,
        num_workers: int,
        verbose: bool,
    ) -> None:
        self.hdf5_filepath           = hdf5_filepath
        self.data_split_csv_filepath = data_split_csv_filepath
        self.batch_size              = batch_size
        self.num_workers             = num_workers
        self.verbose                 = verbose

    def build(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset, valid_dataset, test_dataset = tuple(
            dicom_dataset.DicomDatasetBuilder(
                hdf5_filepath           = self.hdf5_filepath,
                data_split_csv_filepath = self.data_split_csv_filepath,
            ).build()
        )

        if self.verbose:
            print(
                f'Dataset sizes:\n'
                f'   - training:   {len(train_dataset)},\n'
                f'   - validation: {len(valid_dataset)},\n'
                f'   - testing:    {len(test_dataset )}.'
            )

        # Build data loaders.
        train_data_loader = DataLoader(
            dataset     = train_dataset,
            shuffle     = True,
            batch_size  = self.batch_size,
            num_workers = self.num_workers,
            collate_fn  = list_data_collate,
            pin_memory  = torch.cuda.is_available(),
        )
        valid_data_loader = DataLoader(
            dataset     = valid_dataset,
            num_workers = self.num_workers,
        )
        test_data_loader  = DataLoader(
            dataset     = test_dataset,
            num_workers = self.num_workers,
        )

        return (
            train_data_loader,
            valid_data_loader,
            test_data_loader,
        )
