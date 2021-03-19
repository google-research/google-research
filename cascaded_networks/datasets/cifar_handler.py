# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""CIFAR10/100 loader."""
import copy
import json
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
from cascaded_networks.datasets import noise


class CIFAR10Handler(torchvision.datasets.CIFAR10):
  """CIFAR10 dataset handler."""

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


class CIFAR100Handler(torchvision.datasets.CIFAR100):
  """CIFAR100 dataset handler."""

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


def get_transforms(dataset_key,
                   mean,
                   std,
                   noise_type=None,
                   noise_transform_all=False):
  """Create dataset transform list."""
  if dataset_key == 'train':
    transforms_list = [
        T.RandomCrop(32, padding=4, padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]
  else:
    transforms_list = [
        T.ToTensor(),
        T.Normalize(mean, std),
    ]

  if (noise_type is not None
      and (dataset_key == 'train' or noise_transform_all)):
    transforms_list.append(noise.NoiseHandler(noise_type))

  transforms = T.Compose(transforms_list)

  return transforms


def split_dev_set(dataset_src, mean, std, dataset_len,
                  val_split, split_idxs_root, load_previous_splits):
  """Load or create (and save) train/val split from dev set."""
  # Compute number of train / val splits
  n_val_samples = int(dataset_len * val_split)
  n_sample_splits = [dataset_len - n_val_samples, n_val_samples]

  # Split data
  train_set, val_set = torch.utils.data.random_split(dataset_src,
                                                     n_sample_splits)

  train_set.dataset = copy.copy(dataset_src)
  val_set.dataset.transform = get_transforms('test', mean, std)

  # Set indices save/load path
  val_percent = int(val_split * 100)
  if '.json' not in split_idxs_root:
    idx_filepath = os.path.join(
        split_idxs_root, f'{val_percent}-{100-val_percent}_val_split.json')
  else:
    idx_filepath = split_idxs_root

  # Check load indices
  if load_previous_splits and os.path.exists(idx_filepath):
    print(f'Loading previous splits from {idx_filepath}')
    with open(idx_filepath, 'r') as infile:
      loaded_idxs = json.load(infile)

    # Set indices
    train_set.indices = loaded_idxs['train']
    val_set.indices = loaded_idxs['val']

  # Save idxs
  else:
    print(f'Saving split idxs to {idx_filepath}...')
    save_idxs = {
        'train': list(train_set.indices),
        'val': list(val_set.indices),
    }

    # Dump to json
    with open(idx_filepath, 'w') as outfile:
      json.dump(save_idxs, outfile)

  # Print
  print(f'{len(train_set):,} train examples loaded.')
  print(f'{len(val_set):,} val examples loaded.')

  return train_set, val_set


def set_dataset_stats(dataset_name):
  """Set dataset stats for normalization given dataset."""
  if dataset_name.lower() == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

  elif dataset_name.lower() == 'cifar100':
    mean = (0.5071, 0.4866, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
  return mean, std


def build_dataset(root,
                  dataset_name,
                  dataset_key,
                  mean,
                  std,
                  val_split=None,
                  split_idxs_root=None,
                  load_previous_splits=True,
                  noise_type=None,
                  noise_transform_all=False):
  """Build dataset."""
  print(f'Loading {dataset_name} {dataset_key} data...')

  # Datsaet
  if dataset_name.lower() == 'cifar10':
    dataset_op = CIFAR10Handler
  elif dataset_name.lower() == 'cifar100':
    dataset_op = CIFAR100Handler
  else:
    assert False, f'{dataset_name} wrapper not implemented!'

  # Transforms
  transforms = get_transforms(dataset_key, mean, std,
                              noise_type, noise_transform_all)

  # Build dataset source
  dataset_src = dataset_op(root=root,
                           train=dataset_key == 'train',
                           transform=transforms,
                           target_transform=None,
                           download=True)

  # Get number samples in dataset
  dataset_len = dataset_src.data.shape[0]

  # Split
  if dataset_key == 'train':
    if val_split:
      dataset_src = split_dev_set(dataset_src,
                                  mean,
                                  std,
                                  dataset_len,
                                  val_split,
                                  split_idxs_root,
                                  load_previous_splits)
    else:
      dataset_src = dataset_src, None

  # Stdout out
  print((f'{dataset_len:,} '
         f'{"dev" if dataset_key=="train" else dataset_key} '
         f'examples loaded.'))

  return dataset_src


def create_datasets(root,
                    dataset_name,
                    val_split,
                    load_previous_splits=False,
                    split_idxs_root=None,
                    noise_type=None):
  """Create train, val, test datasets."""

  # Set stats
  mean, std = set_dataset_stats(dataset_name)

  # Build datasets
  train_dataset, val_dataset = build_dataset(
      root,
      dataset_name,
      dataset_key='train',
      mean=mean,
      std=std,
      val_split=val_split,
      split_idxs_root=split_idxs_root,
      load_previous_splits=load_previous_splits,
      noise_type=noise_type)

  test_dataset = build_dataset(root,
                               dataset_name,
                               dataset_key='test',
                               mean=mean,
                               std=std,
                               noise_type=noise_type,
                               load_previous_splits=load_previous_splits)

  # Package
  dataset_dict = {
      'train': train_dataset,
      'val': val_dataset,
      'test': test_dataset,
  }

  return dataset_dict
