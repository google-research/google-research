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

"""Base class for continual datasets."""

import abc
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import datasets
from torchvision.transforms import transforms

nn = torch.nn
abstractmethod = abc.abstractmethod


class ContinualDataset:
  """Continual learning evaluation setting."""

  name = None
  setting = None
  n_classes_per_task = None
  n_tasks = None
  transform = None

  def __init__(self, args):
    """Initializes the train and test lists of dataloaders."""
    self.train_loader = None
    self.test_loaders = []
    self.i = 0
    self.args = args
    self.imbalanced = args.imbalanced
    self.limit_per_task = args.limit_per_task
    self.current_pos = 0

  def stream_indices(self):
    """helps in preparing streaming dataloader.

    Returns:
      list of training indices
    """
    stream_train_indices = []
    limit = self.limit_per_task
    for i in range(
        0, self.n_tasks * self.n_classes_per_task, self.n_classes_per_task
    ):
      ind = np.where(
          np.logical_and(
              np.array(self.train_dataset.targets) >= i,
              np.array(self.train_dataset.targets)
              < i + self.n_classes_per_task,
          )
      )[0]

      if self.imbalanced and i == (self.n_tasks - 1) * self.n_classes_per_task:
        limit = self.args.task_imbalance
      if limit > len(ind):
        limit = len(ind)
      ind = np.random.choice(ind, limit, replace=False)
      stream_train_indices.extend(ind)
    return stream_train_indices

  def get_stream_dataloader(self):
    assert self.current_pos < self.num_streams * self.args.stream_batch_size
    inds = self.stream_train_indices[
        self.current_pos : min(
            self.current_pos + self.args.stream_batch_size,
            len(self.train_dataset),
        )
    ]
    self.current_pos = min(
        self.current_pos + self.args.stream_batch_size, len(self.train_dataset)
    )
    return DataLoader(
        Subset(self.train_dataset, inds),
        batch_size=self.args.batch_size,
        shuffle=False,
        num_workers=8,
    )

  @abstractmethod
  def get_data_loaders(self):
    """Creates and returns the training and test loaders for the current task."""
    pass

  @abstractmethod
  def not_aug_dataloader(self, batch_size):
    """Returns the dataloader of the current task, not applying data augmentation."""
    pass

  @staticmethod
  @abstractmethod
  def get_backbone():
    """Returns the backbone to be used for to the current dataset."""
    pass

  @staticmethod
  @abstractmethod
  def get_transform():
    """Returns the transform to be used for to the current dataset."""
    pass

  @staticmethod
  @abstractmethod
  def get_barlow_transform():
    """Returns the transform to be used for supervised contrastive loss."""
    pass

  @staticmethod
  @abstractmethod
  def get_loss():
    """Returns the loss to be used for to the current dataset."""
    pass

  @staticmethod
  @abstractmethod
  def get_normalization_transform():
    """Returns the transform used for normalizing the current dataset."""
    pass

  @staticmethod
  @abstractmethod
  def get_denormalization_transform():
    """Returns the transform used for denormalizing the current dataset."""
    pass


def store_masked_loaders(
    train_dataset, test_dataset, setting
):
  """Divides the dataset into tasks.

  Args:
    train_dataset: train dataset
    test_dataset: test dataset
    setting: continual learning setting

  Returns:
    train and test loaders
  """
  train_mask = np.logical_and(
      np.array(train_dataset.targets) >= setting.i,
      np.array(train_dataset.targets) < setting.i + setting.n_classes_per_task,
  )
  test_mask = np.logical_and(
      np.array(test_dataset.targets) >= setting.i,
      np.array(test_dataset.targets) < setting.i + setting.n_classes_per_task,
  )

  train_dataset.data = train_dataset.data[train_mask]
  test_dataset.data = test_dataset.data[test_mask]

  train_dataset.targets = np.array(train_dataset.targets)[train_mask]
  test_dataset.targets = np.array(test_dataset.targets)[test_mask]

  train_loader = DataLoader(
      train_dataset, batch_size=setting.args.batch_size, shuffle=True
  )
  test_loader = DataLoader(
      test_dataset, batch_size=setting.args.batch_size, shuffle=False
  )
  setting.test_loaders.append(test_loader)
  setting.train_loader = train_loader

  setting.i += setting.n_classes_per_task
  return train_loader, test_loader


def get_previous_train_loader(
    train_dataset, batch_size, setting
):
  """Creates a dataloader for the previous task.

  Args:
    train_dataset: the entire training set
    batch_size: the desired batch size
    setting: the continual dataset at hand

  Returns:
    dataloader
  """
  train_mask = np.logical_and(
      np.array(train_dataset.targets) >= setting.i - setting.n_classes_per_task,
      np.array(train_dataset.targets)
      < setting.i - setting.n_classes_per_task + setting.n_classes_per_task,
  )

  train_dataset.data = train_dataset.data[train_mask]
  train_dataset.targets = np.array(train_dataset.targets)[train_mask]

  return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
