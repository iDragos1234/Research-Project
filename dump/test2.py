import os

import matplotlib.pyplot as plt

from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    ScaleIntensity,
    ToTensor,
    Rotate90,
    Flip,
    Resize,
)


class ImageLoader:
    pass


class Preprocessor:

    def __init__(self, images_folder: str) -> None:
        self.images_folder = images_folder

        self.dicom_files = [
            os.path.join(images_folder, x) for x in os.listdir(self.images_folder) if x.endswith('.dcm')
        ]

        self.train_transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize(spatial_size=(400, 300)),
            ScaleIntensity(),
            Rotate90(),
            Flip(spatial_axis=0),
            ToTensor(),
        ])

        return

    def __call__(self):

        bad_images = []

        print(self.dicom_files.__len__())

        for dicom_file in self.dicom_files:

            try:
                img = LoadImage()(dicom_file)
                print(img.meta.keys())
                # img = self.train_transforms(dicom_file)

                # plt.imshow(img[0], cmap='gray')
                # plt.colorbar()
                # plt.tight_layout()
                # plt.show()
            except:
                bad_images.append(dicom_file)

            # output = self.train_transforms(dicom_file)

        print(bad_images)

        return
    

def run():
    images_folder = './data/'
    Preprocessor(images_folder)()
    return

run()
