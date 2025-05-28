# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""COCO Stuff Dataset."""

import os
import numpy as np
from PIL import Image
import torch


COCO_OBJECT_CLASSES = [
    'person with clothes,people,human',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird avian',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack,bag',
    'umbrella,parasol',
    'handbag,purse',
    'necktie',
    'suitcase',
    'frisbee',
    'skis',
    'sknowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'dessertspoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair seat',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor screen',
    'laptop',
    'mouse',
    'remote control',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hairdrier,blowdrier',
    'toothbrush',
]


class COCODataset(torch.utils.data.Dataset):
  """COCO Object Dataset."""

  def __init__(self, root, split='val', transform=None):
    """Construct COCO Object Dataset.

    Args:
        root (string): Root directory where images are downloaded.
        split (string): Path to the annotation file.
        transform (callable, optional): Optional transform to be applied on a
        sample.
    """
    self.root = root
    self.image_dir = os.path.join(root, 'images', f'{split}2017')
    self.ann_dir = os.path.join(root, 'annotations', f'{split}2017')
    self.images = os.listdir(self.image_dir)
    self.transform = transform

  def __getitem__(self, index):
    img_path = os.path.join(self.image_dir, self.images[index])
    img = Image.open(img_path).convert('RGB')
    img = np.asarray(img)
    idx = self.images[index].split('.')[0]
    ann_path = os.path.join(self.ann_dir, f'{idx}_instanceTrainIds.png')
    ann = np.asarray(Image.open(ann_path), dtype=np.int32)

    return img, img_path, ann, idx

  def __len__(self):
    return len(self.images)
