from typing import Literal
import h5py
import hdf5plugin  # <--- DO NOT REMOVE!
import torch
import torch.utils.data


Dataset = Literal['CHECK', 'OAI']
SubjectID = str
SubjectVisit = str
HipSide = Literal['left', 'right']


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, 
        hdf5_filename: str, 
        img_dtype: type = torch.float32
    ) -> None:
        super().__init__()
        self.hdf5_filename: str = hdf5_filename
        
        self.samples = MyDataset._load_samples(hdf5_filename)

        self.img_dtype: type = img_dtype
        return

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with h5py.File(self.hdf5_filename, 'r') as hdf5_file_object:
            # find subject, visit, image
            group = hdf5_file_object[sample]

            image     = group['image']
            segm_mask = group['segmentation_mask']

            image_attrs = dict(image.attrs)

            # load image data and segmentation mask
            img       = image[:]
            segm_mask = segm_mask[:]

            # correct output format and dtype
            img       = torch.tensor(img[None, :, :], dtype=self.img_dtype)
            segm_mask = torch.tensor(segm_mask[None, :, :, :], dtype=self.img_dtype)

            # collect metadata
            meta = {}
            meta.update(group.attrs)
            meta.update(image_attrs)

        return img, segm_mask, meta
    
    @staticmethod
    def _load_samples(hdf5_filename: str) -> list[str]:
        samples: list[str] = []
        with h5py.File(hdf5_filename, 'r') as hdf5_file_object:
            for dataset in hdf5_file_object['/scans']:
                dataset_path = f'/scans/{dataset}'
                for subject_id in hdf5_file_object[dataset_path]:
                    subject_id_path = f'{dataset_path}/{subject_id}'
                    for subject_visit in hdf5_file_object[subject_id_path]:
                        subject_visit_path = f'{subject_id_path}/{subject_visit}'
                        for hip_side in hdf5_file_object[subject_visit_path]:
                            hip_side_path = f'{subject_visit_path}/{hip_side}'
                            samples.append(hip_side_path)
        return samples
