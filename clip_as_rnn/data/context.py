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

"""Pascal Context Dataset."""

from typing import Any, List, Tuple

import numpy as np
from PIL import Image
# pylint: disable=g-importing-member
from torchvision.datasets.voc import _VOCBase


PASCAL_CONTEXT_CLASSES = [
    'airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat',
    'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
    'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door',
    'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard',
    'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform',
    'plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky',
    'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'monitor',
    'wall', 'water', 'window', 'wood']

PASCAL_CONTEXT_STUFF_CLASS = [
    'bedclothes', 'ceiling', 'cloth', 'curtain', 'floor', 'grass', 'ground',
    'light', 'mountain', 'platform', 'road', 'sidewalk', 'sky', 'snow', 'wall',
    'water', 'window', 'wood', 'door', 'fence', 'rock']

PASCAL_CONTEXT_THING_CLASS = [
    'airplane', 'bag', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book',
    'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'chair', 'computer',
    'cow', 'cup', 'dog', 'flower', 'food', 'horse', 'keyboard', 'motorbike',
    'mouse', 'person', 'plate', 'plant', 'sheep', 'shelves', 'sign', 'sofa',
    'table', 'track', 'train', 'tree', 'truck', 'monitor']

PASCAL_CONTEXT_STUFF_CLASS_ID = [
    3, 15, 17, 21, 25, 28, 29, 32, 34, 38, 40, 44, 46, 47, 55, 56, 57, 58, 23,
    24, 41]

PASCAL_CONTEXT_THING_CLASS_ID = [
    0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 26, 27,
    30, 31, 33, 35, 36, 37, 39, 42, 43, 45, 48, 49, 50, 51, 52, 53, 54]


class CONTEXTSegmentation(_VOCBase):
  """Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/> Segmentation Dataset.

  Attributes:
      root (string): Root directory of the VOC Dataset.
      year (string, optional): The dataset year, supports years ``"2007"`` to
        ``"2012"``.
      image_set (string, optional): Select the image_set to use, ``"train"``,
        ``"trainval"`` or ``"val"``. If ``year=="2007"``, can also be
        ``"test"``.
      download (bool, optional): If true, downloads the dataset from the
        internet and puts it in root directory. If dataset is already
        downloaded, it is not downloaded again.
      transform (callable, optional): A function/transform that  takes in an PIL
        image and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in
        the target and transforms it.
      transforms (callable, optional): A function/transform that takes input
        sample and its target as entry and returns a transformed version.
  """

  _SPLITS_DIR = 'SegmentationContext'
  _TARGET_DIR = 'SegmentationClassContext'
  _TARGET_FILE_EXT = '.png'

  @property
  def masks(self):
    return self.targets

  def __getitem__(self, index):
    """Get a sample of image and segmentation.

    Args:
      index (int): Index
    Returns:
      tuple: (image, target) where target is the image segmentation.
    """
    img = Image.open(self.images[index]).convert('RGB')
    target = Image.open(self.masks[index])

    if self.transforms is not None:
      img, target = self.transforms(img, target)

    return img, target


class CONTEXTDataset(CONTEXTSegmentation):
  """Pascal Context Dataset."""

  def __init__(self, root, year='2012', split='val', transform=None):
    super(CONTEXTDataset, self).__init__(
        root=root,
        image_set=split,
        year=year,
        transform=transform,
        download=False,
    )
    # self.idx_to_class = {val: key for (key, val) in CLASS2ID.items()}

  def __getitem__(self, index):
    image_path = self.images[index]
    image = Image.open(image_path).convert('RGB')
    target = np.asarray(Image.open(self.masks[index]), dtype=np.int32)
    # transpose the target width and height
    # target = target.transpose(1, 0)

    if self.transforms:
      image = self.transform(image)

    return image, str(image_path), target, index
