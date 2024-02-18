import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from  PIL import Image

class OxfordpetsDataset(OxfordIIITPet):
    def __init__(self, root='/datasets/jianhaoy/OxfordPET/', split="test", target_types='segmentation', download=False, transform=None):
        super(OxfordpetsDataset, self).__init__(root=root, split=split, target_types=target_types, download=download, transform=transform)
        self.transform = transform
        self.target_transform = transform
        self.idx_to_class = {val: key for (key, val) in self.class_to_idx.items()}
        
    def __getitem__(self, idx):
        image_path = self._images[idx]
        image = Image.open(image_path).convert("RGB")
        target = Image.open(self._segs[idx])
        target = process_target(target)
        # print(np.array(target), np.unique(np.array(target).flatten()))
        class_label = self._labels[idx]
        class_name = self.idx_to_class[int(class_label)]

        if self.transforms:
            image = self.transform(image)
            target = self.target_transform(target)
            
        # print(type(image),type(target),image_path, class_name)
            
        return image, str(image_path), target, class_name
    
def process_target(image):
    # Convert PIL image to NumPy array
    array_data = np.array(image)
    # Create a new array where only 1s are preserved
    new_array = np.where(array_data == 1, 1, 0)
    # Convert the NumPy array back to PIL image
    new_image = Image.fromarray(new_array.astype(np.uint8))

    return new_image
        