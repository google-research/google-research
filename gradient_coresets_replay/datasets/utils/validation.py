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

"""Train val splliting."""

import os
from continual_learning_rishabh.utils import create_if_not_exists
import numpy as np
from PIL import Image
import torch
from torchvision import datasets
from torchvision import transforms


class ValidationDataset(torch.utils.data.Dataset):
  """Validation dataset."""

  def __init__(
      self,
      data,
      targets,
      transform = None,
      target_transform = None,
  ):
    self.data = data
    self.targets = targets
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    if isinstance(img, np.ndarray):
      if np.max(img) < 2:
        img = Image.fromarray(np.uint8(img * 255))
      else:
        img = Image.fromarray(img)
    else:
      img = Image.fromarray(img.numpy())

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


def get_train_val(
    train,
    test_transform,
    dataset,
    val_perc = 0.1,
):
  """Extract val_perc% of the training set as the validation set.

  Args:
    train: training dataset
    test_transform: transformation of the test dataset
    dataset: dataset name
    val_perc: percentage of the training set to be extracted

  Returns:
    the training set and the validation set
  """
  dataset_length = train.data.shape[0]
  directory = '/workdir/continual_learning_rishabh/datasets/val_permutations/'
  create_if_not_exists(directory)
  file_name = dataset + '.pt'
  if os.path.exists(directory + file_name):
    print('using predefined splits')
    perm = torch.load(directory + file_name)
  else:
    perm = torch.randperm(dataset_length)
    torch.save(perm, directory + file_name)
  train.data = train.data[perm]
  train.targets = np.array(train.targets)[perm]
  test_dataset = ValidationDataset(
      train.data[: int(val_perc * dataset_length)],
      train.targets[: int(val_perc * dataset_length)],
      transform=test_transform,
  )
  train.data = train.data[int(val_perc * dataset_length) :]
  train.targets = train.targets[int(val_perc * dataset_length) :]

  return train, test_dataset
