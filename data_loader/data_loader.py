from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from my_dataset import MyDataset


dataset = MyDataset(
    'C:\\Users\\drago\\Desktop\\Research-Project\\output.h5', 
    ['9000099', '9000296', '9000798', '9001695', '9003175']
)

dataloader = DataLoader(
    dataset,
)

img, meta = dataset[2]

plt.imshow(img[0])
plt.colorbar()
plt.show()