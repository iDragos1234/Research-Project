import h5py
import hdf5plugin  # <--- DO NOT REMOVE!
import torch
import torch.utils.data


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, 
        hdf5_filename: str, 
        subjects: list[str], 
        img_dtype: type = torch.float32
    ) -> None:
        super().__init__()
        self.hdf5_filename: str = hdf5_filename
        self.subjects: list[str] = subjects
        
        self.samples = MyDataset._load_samples(hdf5_filename, subjects)

        self.img_dtype: type = img_dtype
        return

    def __getitem__(self, index: int):
        subject_id, visit = self.samples[index]
        with h5py.File(self.hdf5_filename, 'r') as hdf5_file_object:
            # find subject, visit, image
            group = hdf5_file_object[f'/scans/{subject_id}/{visit}']
            image = group['image']
            image_attrs = dict(image.attrs)

            # load image data
            img = image[:]

            # correct output format and dtype
            img = torch.tensor(img[None, :, :], dtype=self.img_dtype)

            # collect metadata
            meta = {}
            meta.update(group.attrs)
            meta.update(image_attrs)

        return img, meta
    
    @staticmethod
    def _load_samples(hdf5_filename: str, subjects: str) -> list[tuple[str, str]]:
        samples: list[tuple[str, str]] = []
        with h5py.File(hdf5_filename, 'r') as hdf5_file_object:
            for subject_id in subjects:
                subject_path = f'/scans/{subject_id}'
                for visit in hdf5_file_object[subject_path]:
                    samples.append((subject_id, visit))
        return samples
