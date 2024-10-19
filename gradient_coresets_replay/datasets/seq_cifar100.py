# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Sequential CIFAR100 dataset."""

import math
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR100
from gradient_coresets_replay.backbone.resnet18 import resnet18
from gradient_coresets_replay.datasets.transforms import DeNormalize
from gradient_coresets_replay.datasets.utils.continual_dataset import ContinualDataset
from gradient_coresets_replay.datasets.utils.continual_dataset import get_previous_train_loader
from gradient_coresets_replay.datasets.utils.continual_dataset import store_masked_loaders
from gradient_coresets_replay.datasets.utils.validation import get_train_val
from gradient_coresets_replay.utils.conf import base_path


class MyCIFAR100(CIFAR100):
  """Overrides the CIFAR100 dataset to change the getitem function."""

  def __init__(
      self,
      root,
      train=True,
      transform=None,
      target_transform=None,
      download=False,
  ):
    self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
    super().__init__(
        root, train, transform, target_transform, download
    )

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]

    # to return a PIL Image
    img = Image.fromarray(img, mode='RGB')
    original_img = img.copy()

    not_aug_img = self.not_aug_transform(original_img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    if hasattr(self, 'logits'):
      return img, target, not_aug_img, self.logits[index]

    return img, target, not_aug_img


class SequentialCIFAR100(ContinualDataset):
  """Sequential CIFAR100 dataset."""

  name = 'seq-cifar100'
  setting = 'class-il'
  n_classes_per_task = 20
  n_tasks = 5
  transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
  ])

  def __init__(self, args):
    super().__init__(args)
    self.train_dataset = MyCIFAR100(
        base_path() + 'CIFAR100',
        train=True,
        download=True,
        transform=self.transform,
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), self.get_normalization_transform()]
    )
    if self.args.validation:
      self.train_dataset, self.test_dataset = get_train_val(
          self.train_dataset, test_transform, self.name
      )
    else:
      self.test_dataset = CIFAR100(
          base_path() + 'CIFAR100',
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

    test_transform = transforms.Compose(
        [transforms.ToTensor(), self.get_normalization_transform()]
    )

    train_dataset = MyCIFAR100(
        base_path() + 'CIFAR100', train=True, download=True, transform=transform
    )
    if self.args.validation:
      train_dataset, test_dataset = get_train_val(
          train_dataset, test_transform, self.name
      )
    else:
      test_dataset = CIFAR100(
          base_path() + 'CIFAR100',
          train=False,
          download=True,
          transform=test_transform,
      )

    train, test = store_masked_loaders(train_dataset, test_dataset, self)
    return train, test

  def not_aug_dataloader(self, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), self.get_normalization_transform()]
    )

    train_dataset = MyCIFAR100(
        base_path() + 'CIFAR100', train=True, download=True, transform=transform
    )
    train_loader = get_previous_train_loader(train_dataset, batch_size, self)

    return train_loader

  @staticmethod
  def get_transform():
    transform = transforms.Compose(
        [transforms.ToPILImage(), SequentialCIFAR100.transform]
    )
    return transform

  @staticmethod
  def get_barlow_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ),
    ])
    transform_prime = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ),
    ])
    return (transform, transform_prime)

  @staticmethod
  def get_backbone():
    return resnet18(
        SequentialCIFAR100.n_classes_per_task * SequentialCIFAR100.n_tasks
    )

  @staticmethod
  def get_loss():
    return F.cross_entropy

  @staticmethod
  def get_normalization_transform():
    transform = transforms.Normalize(
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    )
    return transform

  @staticmethod
  def get_denormalization_transform():
    transform = DeNormalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    return transform
