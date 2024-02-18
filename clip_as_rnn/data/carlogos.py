import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class CarLogosDataset(Dataset):
    def __init__(self, root='/datasets/jianhaoy/CarLogos/', img_transform=None, target_transform=None):
        self.root_dir = os.path.join(root, "vehicle-logos-dataset")
        self.dataframe = pd.read_csv(os.path.join(self.root_dir, 'structure.csv'))
        self.img_transform = img_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 2])
        target_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 3])
        class_name = self.dataframe.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")
        target = Image.open(target_path).convert('L')
        
        # print(img_path, target_path, class_name)
        # print(np.array(target).shape, np.unique(np.array(target).flatten()))
        if self.img_transform:
            image = self.img_transform(image)

        if self.target_transform:
            target = self.target_transform(target)
        # print(target.size())

        return image, img_path, target, class_name
