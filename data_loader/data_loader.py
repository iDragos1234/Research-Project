from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from monai.transforms import Resize

from my_dataset import MyDataset


dataset = MyDataset('C:\\Users\\drago\\Desktop\\Research-Project\\output.h5')

dataloader = DataLoader(dataset)

for idx, sample in enumerate(dataset.samples):

    img, segm_mask, meta = dataset[idx]
    # segm_mask = Resize([300, 300, 3])(segm_mask)

    print(img.shape, segm_mask.shape)

    plt.title(f'Plot #{idx + 1} - sample {sample}')
    plt.imshow(img)
    plt.colorbar()
    plt.show()

    plt.title(f'Plot #{idx + 1} - sample {sample}')
    plt.imshow(segm_mask)
    plt.colorbar()
    plt.show()
