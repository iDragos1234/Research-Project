import numpy as np
import h5py
import hdf5plugin  # <--- DO NOT REMOVE!
from torch import float32, tensor, Tensor
from torch.utils.data import Dataset
from monai.transforms.transform import Transform
from monai.data import partition_dataset


class DicomDataset(Dataset):

    hdf5_filename: str
    data: list[str]
    transform: Transform
    samples: list[str]
    img_dtype: type

    def __init__(self, 
        hdf5_filename: str,
        data: list[str],
        transform: Transform=None,
        img_dtype: type = float32,
    ) -> None:
        super().__init__()
        self.hdf5_filename = hdf5_filename
        self.data          = data
        self.transform     = transform
        self.img_dtype     = img_dtype
        return
    
    def __len__(self):
        return len(self.data)
    
    def get_item_meta(self, index) -> dict:
        with h5py.File(self.hdf5_filename, 'r') as hdf5_file_object:
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
        with h5py.File(self.hdf5_filename, 'r') as hdf5_file_object:
            # find subject, visit, image
            sample = self.data[index]
            group  = hdf5_file_object[sample]
            image  = group['image']
            mask   = group['segmentation_mask']

            # collect metadata
            meta = {
                'group_attributes': group.attrs,
                'image_attributes': image.attrs,
            }

            # correct output format and dtype
            image = tensor(np.array(image)[None], dtype=self.img_dtype)
            mask  = tensor(np.array(mask), dtype=self.img_dtype)

        return {'image': image, 'mask': mask} if self.transform is None \
            else self.transform({'image': image, 'mask': mask})
            

class DicomDatasetBuilder:

    hdf5_filename: str
    img_dtype: type
    transform: Transform
    data: list[str]
    data_splits: list[list[str]]

    def __init__(self) -> None:
        self.transform = None
        self.img_dtype = float32
        return

    def set_hdf5_source(self, hdf5_filename: str):
        self.hdf5_filename = hdf5_filename
        return self
    
    def set_img_dtype(self, img_dtype: type):
        self.img_dtype = img_dtype
        return self
    
    def set_transform(self, transform: Transform):
        self.transform = transform
        return self
    
    def load_data(self):
        data: list[str] = []
        with h5py.File(self.hdf5_filename, 'r') as hdf5_file_object:
            for dataset in hdf5_file_object['/scans']:
                dataset_path = f'/scans/{dataset}'
                for subject_id in hdf5_file_object[dataset_path]:
                    subject_id_path = f'{dataset_path}/{subject_id}'
                    for subject_visit in hdf5_file_object[subject_id_path]:
                        subject_visit_path = f'{subject_id_path}/{subject_visit}'
                        for hip_side in hdf5_file_object[subject_visit_path]:
                            hip_side_path = f'{subject_visit_path}/{hip_side}'
                            data.append(hip_side_path)
        self.data        = data
        self.data_splits = [data]
        return self
    
    def split_data(self,
        ratios: list[float],
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.data_splits = partition_dataset(
            data=self.data, 
            ratios=ratios,
            shuffle=shuffle,
            seed=seed, 
        )
        return self

    def build(self) -> list[DicomDataset]:
        return [
            DicomDataset(
                hdf5_filename=self.hdf5_filename,
                data=data,
                transform=self.transform,
                img_dtype=self.img_dtype,
            ) 
            for data in self.data_splits
        ]
