import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import Food101
from PIL import Image

class Food101Dataset(Food101):
    def __init__(self, root='/scratch/network/ssd4/jianhaoy/Food101/', split="test", target_transform=None, download=True, transform=None):
        super(Food101Dataset, self).__init__(root=root, split=split, target_transform=transform, download=download, transform=transform)
        
    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        print(type(image),type(label))

        return image, label