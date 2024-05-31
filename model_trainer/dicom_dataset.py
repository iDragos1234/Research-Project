from itertools import groupby
from typing import Union
import numpy as np
import h5py, hdf5plugin  # <--- DO NOT REMOVE!
from torch import float32, tensor, Tensor
from torch.utils.data import Dataset
from monai.transforms import Transform


class DicomDataset(Dataset):

    hdf5_filepath: str
    data: list[str]
    transform: Transform
    img_dtype: type

    def __init__(self, 
        hdf5_filepath: str,
        data: list[str],
        transform: Union[Transform, None]=None,
        img_dtype: type = float32,
    ) -> None:
        super().__init__()
        self.hdf5_filepath = hdf5_filepath
        self.data          = data
        self.transform     = transform
        self.img_dtype     = img_dtype
        return
    
    def __len__(self):
        return len(self.data)
    
    def get_item_meta(self, index) -> dict:
        with h5py.File(self.hdf5_filepath, 'r') as hdf5_file_object:
            sample = self.data[index]
            group  = hdf5_file_object[sample]
            image  = group['image']
            mask   = group['segmentation_mask']

            # collect metadata
            meta = {
                'group_attributes': group.attrs,
                'image_attributes': image.attrs,
            }
        
        return meta

    def __getitem__(self, index) -> dict[str, Tensor]:
        with h5py.File(self.hdf5_filepath, 'r') as hdf5_file_object:
            # find subject, visit, image
            sample = self.data[index]
            group  = hdf5_file_object[sample]
            image  = group['image']
            mask   = group['mask']

            # collect metadata
            meta = {
                'group attributes': group.attrs,
                'image attributes': image.attrs,
            }

            # correct output format and dtype
            image = tensor(np.array(image), dtype=self.img_dtype)
            mask  = tensor(np.array(mask),  dtype=self.img_dtype)

            item = {
                'image': image,
                'mask' : mask,
            }

            return item if self.transform is None \
                else self.transform(item)
                    

class DicomDatasetBuilder:

    hdf5_filepath: str
    data_split_csv_filepath: str
    transform: Transform
    img_dtype: type

    data: list[str]

    def __init__(self,
        hdf5_filepath: str,
        data_split_csv_filepath: str,
        transform: Union[Transform, None] = None,
        img_dtype: type = float32,
    ) -> None:
        self.hdf5_filepath           = hdf5_filepath
        self.data_split_csv_filepath = data_split_csv_filepath
        self.transform               = transform
        self.img_dtype               = img_dtype

    def build(self) -> list[DicomDataset]:
        with open(self.data_split_csv_filepath, 'r') as file:
            lines = [line.strip().split(',') for line in file.readlines()[1:]]

            data_splits = [
                [sample[0] for sample in group]
                    for _, group in groupby(lines, key=lambda row: row[1])
            ]

            datasets = [
                DicomDataset(
                    hdf5_filepath = self.hdf5_filepath,
                    data          = data,
                    transform     = self.transform,
                    img_dtype     = self.img_dtype,
                )
                for data in data_splits
            ]

            return datasets
