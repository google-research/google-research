import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation
from  PIL import Image
from xml.etree.ElementTree import Element as ET_Element
from xml.etree.ElementTree import parse as ET_parse
import collections
import pickle

CLASS2ID = {
    'Background': 0,
    'Aero plane': 1,
    'Bicycle': 2,
    'Bird': 3,
    'Boat': 4,
    'Bottle': 5,
    'Bus': 6,
    'Car': 7,
    'Cat': 8,
    'Chair': 9,
    'Cow': 10,
    'Dining table': 11,
    'Dog': 12,
    'Horse': 13,
    'Motorbike': 14,
    'Person': 15,
    'Potted plant': 16,
    'Sheep': 17,
    'Sofa': 18,
    'Train': 19,
    'Tv/Monitor': 20,
    # ... add more entries as needed
    'Border': 255
}


VOC_CLASSES = ['aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair seat', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
                   ]


BACKGROUND_CATEGORY = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','keyboard','helmet',
                        'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign',
                        ]


class VOCDataset(VOCSegmentation):
    def __init__(self, root='/datasets/jianhaoy/PASCAL/', year='2012', split="val", target_transform=None, download=False, transform=None):
        super(VOCDataset, self).__init__(root=root, image_set=split, year=year, target_transform=transform, download=download, transform=transform)
        self.idx_to_class = {val: key for (key, val) in CLASS2ID.items()}
                
    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        target = np.asarray(Image.open(self.masks[index]), dtype=np.int32)

        _, unique_values = self.process_target(np.array(target))
        classnames = [self.idx_to_class[idx] for idx in unique_values]

        if self.transforms:
            image = self.transform(image)

        
        return image, str(image_path), target, classnames
    
    def process_target(self,arr):
        # Set values 0 and 255 to 1
        arr[(arr == 0) | (arr == 255)] = 0

        # Find unique values (excluding 0 and 255)
        unique_values = np.unique(arr[(arr != 0) & (arr != 255)])

        # Create separate masks for each unique value
        masks = [arr == value for value in unique_values]
        masks = [Image.fromarray(arr) for arr in masks]
        masks = [self.target_transform(arr) for arr in masks]

        return masks, unique_values


