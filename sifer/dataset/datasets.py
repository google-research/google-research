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

"""Datasets."""

import os
from typing import Any, Optional

import pandas as pd
from PIL import Image
from PIL import ImageFile
import torch
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


class SubpopDataset:
  """Parent class for subpopulation/biased datasets."""

  N_STEPS = 5001  # Default, subclasses may override
  CHECKPOINT_FREQ = 100  # Default, subclasses may override
  N_WORKERS = 4  # Default, subclasses may override
  INPUT_SHAPE = None  # Subclasses should override
  SPLITS = {'tr': 0, 'va': 1, 'te': 2}  # Default, subclasses may override
  EVAL_SPLITS = ['te']  # Default, subclasses may override

  def __init__(
      self,
      root,
      split,
      metadata,
      transform,
      train_attr = 'yes',
      subsample_type = None,
  ):
    df = pd.read_csv(metadata)
    df = df[df['split'] == (self.SPLITS[split])]

    self.idx = list(range(len(df)))
    self.x = (
        df['filename'].astype(str).map(lambda x: os.path.join(root, x)).tolist()
    )
    self.y = df['y'].tolist()
    self.a = (
        df['a'].tolist() if train_attr == 'yes' else [0] * len(df['a'].tolist())
    )
    self.transform_ = transform
    self._count_groups()

    if subsample_type is not None:
      self.subsample(subsample_type)

  def _count_groups(self):
    """Calculates weights for groups according to their size."""
    self.weights_g, self.weights_y = [], []
    self.num_attributes = len(set(self.a))
    self.num_labels = len(set(self.y))
    self.group_sizes = [0] * self.num_attributes * self.num_labels
    self.class_sizes = [0] * self.num_labels

    for i in self.idx:
      self.group_sizes[self.num_attributes * self.y[i] + self.a[i]] += 1
      self.class_sizes[self.y[i]] += 1

    for i in self.idx:
      self.weights_g.append(
          len(self)
          / self.group_sizes[self.num_attributes * self.y[i] + self.a[i]]
      )
      self.weights_y.append(len(self) / self.class_sizes[self.y[i]])

  def subsample(self, subsample_type):
    """Subsamples the dataset according to the given subsample type."""
    assert subsample_type in {'group', 'class'}
    perm = torch.randperm(len(self)).tolist()
    min_size = (
        min(list(self.group_sizes))
        if subsample_type == 'group'
        else min(list(self.class_sizes))
    )

    counts_g = [0] * self.num_attributes * self.num_labels
    counts_y = [0] * self.num_labels
    new_idx = []
    for p in perm:
      y, a = self.y[self.idx[p]], self.a[self.idx[p]]
      if (
          subsample_type == 'group'
          and counts_g[self.num_attributes * int(y) + int(a)] < min_size
      ) or (subsample_type == 'class' and counts_y[int(y)] < min_size):
        counts_g[self.num_attributes * int(y) + int(a)] += 1
        counts_y[int(y)] += 1
        new_idx.append(self.idx[p])

    self.idx = new_idx
    self._count_groups()

  def transform(self, x):
    return self.transform_(Image.open(x).convert('RGB'))

  def __getitem__(
      self, index
  ):
    i = self.idx[index]
    x = self.transform(self.x[i])
    y = torch.tensor(self.y[i], dtype=torch.long)
    a = torch.tensor(self.a[i], dtype=torch.long)
    return i, x, y, a

  def __len__(self):
    return len(self.idx)


class Waterbirds(SubpopDataset):
  """Waterbirds dataset."""
  N_STEPS = 10001
  CHECKPOINT_FREQ = 300
  INPUT_SHAPE = (
      3,
      224,
      224,
  )

  def __init__(
      self,
      data_path,
      split,
      train_attr = 'yes',
      subsample_type = None,
  ):
    root = os.path.join(
        data_path, 'waterbirds', 'waterbird_complete95_forest2water2'
    )
    metadata = os.path.join(data_path, 'waterbirds', 'metadata_waterbirds.csv')
    transform = transforms.Compose([
        transforms.Resize((
            int(224 * (256 / 224)),
            int(224 * (256 / 224)),
        )),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    self.data_type = 'images'
    super().__init__(
        root, split, metadata, transform, train_attr, subsample_type
    )


class CelebA(SubpopDataset):
  """CelebA dataset."""
  N_STEPS = 30001
  CHECKPOINT_FREQ = 1000
  INPUT_SHAPE = (
      3,
      224,
      224,
  )

  def __init__(
      self,
      data_path,
      split,
      train_attr = 'yes',
      subsample_type = None,
  ):
    root = os.path.join(data_path, 'celeba', 'img_align_celeba')
    metadata = os.path.join(data_path, 'celeba', 'metadata_celeba.csv')
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    self.data_type = 'images'
    super().__init__(
        root, split, metadata, transform, train_attr, subsample_type
    )
