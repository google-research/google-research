# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Dataset Handler."""
import os
import torch
from cascaded_networks.datasets import cifar_handler


class DataHandler:
  """Handler for datasets."""

  def __init__(self,
               dataset_name,
               data_root,
               val_split=0.1,
               split_idxs_root='/tmp/split_idxs',
               noise_type=None,
               ood_mode=None,
               load_previous_splits=True):
    """Initialize dataset handler."""
    self.dataset_name = dataset_name
    self.data_root = data_root
    self.val_split = val_split
    self.noise_type = noise_type
    self._ood_mode = ood_mode
    self.load_previous_splits = load_previous_splits

    # Set idx with dataset_name
    split_idxs_root = os.path.join(split_idxs_root, dataset_name)
    if not os.path.exists(split_idxs_root):
      os.makedirs(split_idxs_root)

    if split_idxs_root and val_split:
      self.split_idxs_root = self._build_split_idx_root(split_idxs_root,
                                                        dataset_name)
    else:
      self.split_idxs_root = None

    self.datasets = self._build_datasets()
    self._set_num_classes(dataset_name)

  def _set_num_classes(self, dataset_name):
    """Set number of classes in dataset."""
    if dataset_name == 'CIFAR10':
      self.num_classes = 10
    elif dataset_name == 'CIFAR100':
      self.num_classes = 100
    elif dataset_name == 'TinyImageNet':
      self.num_classes = 200

  def get_transform(self, dataset_key=None):
    """Build dataset transform."""
    if dataset_key is None:
      dataset_key = list(self.datasets.keys())[0]

    normalize_transform = None
    # Grab transforms - location varies depending on base dataset.
    try:
      transforms = self.datasets[dataset_key].transform.transforms
      found = True
    except AttributeError:
      found = False

    if not found:
      try:
        transforms = self.datasets[dataset_key].dataset.transform.transforms
        found = True
      except AttributeError:
        found = False

    if not found:
      print('Transform list not found!')
    else:
      found = False
      for xform in transforms:
        if 'normalize' in str(xform).lower():
          normalize_transform = xform
          found = True
          break

    if not found:
      print('Normalization transform not found!')
    return normalize_transform

  def _build_split_idx_root(self, split_idxs_root, dataset_name):
    """Build directory for split idxs."""
    if '.json' in split_idxs_root and not os.path.exists(split_idxs_root):
      split_idxs_root = os.path.join(split_idxs_root, dataset_name)
    print(f'Setting split idxs root to {split_idxs_root}')
    if not os.path.exists(split_idxs_root):
      print(f'{split_idxs_root} does not exist!')
      os.makedirs(split_idxs_root)
      print('Complete.')
    return split_idxs_root

  def _build_datasets(self):
    """Build dataset."""
    if 'cifar' in self.dataset_name.lower():
      dataset_dict = cifar_handler.create_datasets(
          self.data_root,
          dataset_name=self.dataset_name,
          val_split=self.val_split,
          split_idxs_root=self.split_idxs_root,
          noise_type=self.noise_type,
          load_previous_splits=self.load_previous_splits)
    elif self.dataset_name.lower() == 'tinyimagenet':
      assert False, 'Not Implemented.'
    return dataset_dict

  def build_loader(self,
                   dataset_key,
                   flags,
                   dont_shuffle_train=False):
    """Build dataset loader."""

    # Get dataset source
    dataset_src = self.datasets[dataset_key]

    # Specify shuffling
    if dont_shuffle_train:
      shuffle = False
    else:
      shuffle = dataset_key == 'train'

    # Creates dataloaders, which load data in batches
    loader = torch.utils.data.DataLoader(
        dataset=dataset_src,
        batch_size=flags.batch_size,
        shuffle=shuffle,
        num_workers=flags.num_workers,
        drop_last=flags.drop_last,
        pin_memory=True)

    return loader
