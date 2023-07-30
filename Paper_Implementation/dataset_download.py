import os
from torchvision import datasets
import torchvision.transforms as transforms

path2data = "./data"

if not os.path.exists(path2data):
    os.mkdir(path2data)

train_data = datasets.STL10(path2data, split='train', download=True, transform=transforms.ToTensor())
val_data = datasets.STL10(path2data, split='test', download=True, transform=transforms.ToTensor())

print(len(train_data))
print(len(val_data))