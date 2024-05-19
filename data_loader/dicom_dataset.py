import numpy as np
import h5py
import hdf5plugin  # <--- DO NOT REMOVE!
from torch import float32, tensor, is_tensor, Tensor
from torch.utils.data import Dataset
from monai.transforms.transform import Transform
from monai.data import partition_dataset

from sklearn.model_selection import train_test_split


class DicomDataset(Dataset):

    hdf5_filename: str
    data: list[str]
    transform: Transform
    samples: list[str]
    img_dtype: type

    def __init__(self, 
        hdf5_filename: str,
        data: list[str],
        transform: Transform,
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
        # if is_tensor(index):
        #     index = index.tolist()

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
            mask  = tensor(np.array(mask)[None], dtype=self.img_dtype)

        return self.transform({'image': image, 'mask': mask})
            

class DicomDatasetBuilder:

    hdf5_filename: str
    data: list[str]
    data_splits: list[list[str]]
    transform: Transform

    def set_hdf5_source(self, hdf5_filename: str):
        self.hdf5_filename = hdf5_filename
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
        self.data = data
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
    
    def train_test_split(self, 
        test_size=None, 
        train_size=None, 
        random_state=None, 
        shuffle=True,
        stratify=None,
    ):
        """
        Splits the samples into test and train samples.
        Refer to https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
        """
        temp = train_test_split(
            self.data,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        self.test_samples  = temp[1]
        self.train_samples = temp[0]
        return self

    def train_validation_test_split(self, 
        test_size=None, 
        validation_size=None,
        train_size=None, 
        random_state=None, 
        shuffle=True,
        stratify=None,
    ):
        """
        Splits the data into test, validation and train sets.
        Refer to https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
        """
        temp = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        self.test_samples = temp[1]

        temp = train_test_split(
            temp[0],
            test_size=validation_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        self.validation_samples = temp[1]
        self.train_samples      = temp[0]
        return self

    def build(self):
        if self.test_samples is not None \
            and self.validation_samples is not None \
            and self.train_samples is not None:
            return DicomDataset(self.hdf5_filename, self.test_samples, self.transform), \
                   DicomDataset(self.hdf5_filename, self.validation_samples, self.transform), \
                   DicomDataset(self.hdf5_filename, self.train_samples, self.transform)
        elif self.test_samples is not None \
            and self.validation_samples is None \
            and self.train_samples is not None:
            return DicomDataset(self.hdf5_filename, self.test_samples, self.transform), \
                   DicomDataset(self.hdf5_filename, self.train_samples, self.transform)
        elif self.test_samples is None \
            and self.validation_samples is None \
            and self.train_samples is None:
            return DicomDataset(self.hdf5_filename, self.data, self.transform)
        else:
            raise RuntimeError()
