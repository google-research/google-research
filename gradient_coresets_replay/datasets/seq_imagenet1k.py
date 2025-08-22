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

"""Sequential ImageNet-1k dataset."""

import glob
import math
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from gradient_coresets_replay.backbone.resnet18 import resnet18
from gradient_coresets_replay.datasets.transforms import DeNormalize
from gradient_coresets_replay.datasets.utils.continual_dataset import ContinualDataset
from gradient_coresets_replay.datasets.utils.continual_dataset import get_previous_train_loader
from gradient_coresets_replay.datasets.utils.continual_dataset import store_masked_loaders
from gradient_coresets_replay.datasets.utils.validation import get_train_val
from gradient_coresets_replay.utils.conf import base_path


class Imagenet1k(Dataset):
  """Defines Imagenet1k as for the others pytorch datasets."""

  def __init__(
      self,
      root,
      train = True,
      transform = None,
      target_transform = None,
      download = False,
  ):
    self.not_aug_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    self.root = root
    self.train = train
    self.transform = transform
    self.target_transform = target_transform
    self.download = download

    if download:
      if os.path.isdir(root) and len(os.listdir(root)):
        print('Download not needed, files already on disk.')
      else:
        # https://drive.google.com/file/d/1bY8--UVhySGO8xbPv2HG4YtJWCzY_65E/view?usp=sharing
        print('Downloading dataset')
        gdd.download_file_from_google_drive(
            file_id='1bY8--UVhySGO8xbPv2HG4YtJWCzY_65E',
            dest_path=os.path.join(root, 'imagenet-1000.zip'),
            unzip=True,
        )
    dirlist = os.listdir(os.path.join(root, 'imagenet-mini/val'))
    targets_dict = {j: i for i, j in enumerate(sorted(dirlist))}

    self.data = np.array(
        glob.glob(
            os.path.join(
                root,
                'imagenet-mini/%s/*/*' % ('train' if self.train else 'val'),
            )
        )
    )
    self.targets = np.array([targets_dict[i.split('/')[-2]] for i in self.data])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    imagepath, target = self.data[index], self.targets[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.open(imagepath).convert('RGB')
    original_img = img.copy()

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    if hasattr(self, 'logits'):
      return img, target, original_img, self.logits[index]

    return img, target


class MyImagenet1k(Imagenet1k):
  """Defines Imagenet1k as for the others pytorch datasets."""

  def __init__(
      self,
      root,
      train = True,
      transform = None,
      target_transform = None,
      download = False,
  ):
    self.root = root
    super().__init__(root, train, transform, target_transform, download)

  def __getitem__(self, index):
    imagepath, target = self.data[index], self.targets[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.open(imagepath).convert('RGB')
    original_img = img.copy()

    not_aug_img = self.not_aug_transform(original_img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    if hasattr(self, 'logits'):
      return img, target, not_aug_img, self.logits[index]

    return img, target, not_aug_img


class SequentialImagenet1k(ContinualDataset):
  """Sequential ImageNet-1k dataset."""

  name = 'seq-imagenet1k'
  setting = 'class-il'
  n_classes_per_task = 200
  n_tasks = 5
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomCrop(224, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])

  def __init__(self, args):
    super().__init__(args)
    self.train_dataset = MyImagenet1k(
        base_path() + 'IMAGENET1k',
        train=True,
        download=True,
        transform=self.transform,
    )
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        self.get_normalization_transform(),
    ])
    if self.args.validation:
      self.train_dataset, self.test_dataset = get_train_val(
          self.train_dataset, test_transform, self.name
      )
    else:
      self.test_dataset = MyImagenet1k(
          base_path() + 'IMAGENET1k',
          train=False,
          download=True,
          transform=test_transform,
      )

    if self.args.streaming:
      self.current_pos = 0
      self.stream_train_indices = self.stream_indices()
      self.num_streams = math.ceil(
          len(self.stream_train_indices) / self.args.stream_batch_size
      )
      for _ in range(self.n_tasks):
        _ = self.get_data_loaders()  # to store self.test_dataloaders

  def get_data_loaders(self):
    transform = self.transform

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        self.get_normalization_transform(),
    ])

    train_dataset = MyImagenet1k(
        base_path() + 'IMAGENET1k',
        train=True,
        download=True,
        transform=transform,
    )
    if self.args.validation:
      train_dataset, test_dataset = get_train_val(
          train_dataset, test_transform, self.name
      )
    else:
      test_dataset = Imagenet1k(
          base_path() + 'IMAGENET1k',
          train=False,
          download=True,
          transform=test_transform,
      )

    train, test = store_masked_loaders(train_dataset, test_dataset, self)
    return train, test

  def not_aug_dataloader(self, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), self.get_denormalization_transform()]
    )

    train_dataset = MyImagenet1k(
        base_path() + 'IMAGENET1k',
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = get_previous_train_loader(train_dataset, batch_size, self)

    return train_loader

  @staticmethod
  def get_backbone():
    return resnet18(
        SequentialImagenet1k.n_classes_per_task * SequentialImagenet1k.n_tasks
    )

  @staticmethod
  def get_loss():
    return F.cross_entropy

  def get_transform(self):
    transform = transforms.Compose([transforms.ToPILImage(), self.transform])
    return transform

  @staticmethod
  def get_barlow_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_prime = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return (transform, transform_prime)

  @staticmethod
  def get_normalization_transform():
    transform = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    return transform

  @staticmethod
  def get_denormalization_transform():
    transform = DeNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return transform
