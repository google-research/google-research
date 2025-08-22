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

"""Data generation script and dataset module."""

from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_LAUNCH_BLOCKING = 1


class Normalize(object):
  """Min-Max normalization of the samples."""

  def __call__(self, inp_seq):
    min_val, max_val = inp_seq.min(), inp_seq.max()
    new_inp = (inp_seq - min_val) / (max_val - min_val)
    return new_inp


class TSDataset(Dataset):
  """Time series dataset class.

  """

  def __init__(self, x, args,
               transform=None, target_transform=None):
    self.x = torch.as_tensor(x, dtype=torch.float)
    self.input_len = args.input_len
    self.output_len = args.output_len
    self.input_dim = args.input_dim
    self.output_dim = args.output_dim
    self.window_len = self.input_len + self.output_len

    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    input_seq = self.x[index:index + self.input_len]
    input_seq = input_seq[Ellipsis, :self.input_dim]
    output_seq = self.x[index + self.input_len:index + self.window_len]
    output_seq = output_seq[Ellipsis, :self.output_dim]

    if self.transform:
      input_seq = self.transform(input_seq)
    if self.target_transform:
      output_seq = self.target_transform(output_seq)

    return (input_seq, output_seq)

  def __len__(self):
    num_samples = self.x.shape[0] - self.window_len
    return num_samples


class DataModule(LightningDataModule):
  """Lightning data module for time series.

  """

  def __init__(self, data, args):
    super().__init__()
    self.data = data  # [seq_len, num_nodes, num_features]

    self.args = args
    self.batch_size = args.batch_size
    self.num_workers = 12

  def setup(self, stage):
    # seq x batch x feature
    # generate x and y vectors

    # define split ratio for train, val and test
    num_samples = self.data.shape[0]

    num_train = round(num_samples * 0.7)
    num_test = round(num_samples * 0.2)
    num_val = num_samples - num_train - num_test

    self.x_train = self.data[:num_train]  # train series
    self.x_val = self.data[num_train:num_train + num_val]  # val_series
    self.x_test = self.data[-num_test:]  # test_series

    # print('Total number of samples: {}'.format(num_samples))
    # print('Percentage for train: {:.2f}'.format(100*num_train/num_samples))
    # print('Percentage for val: {:.2f}'.format(100*num_val/num_samples))
    # print('Percentage for test: {:.2f}'.format(100*num_test/num_samples))

  # drop the last batch for size reason
  def train_dataloader(self):
    loader = DataLoader(
        TSDataset(self.x_train, self.args, transform=Normalize()),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=True,
        drop_last=True)
    return loader

  def val_dataloader(self):
    loader = DataLoader(
        TSDataset(self.x_val, self.args, transform=Normalize()),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
        drop_last=True)
    return loader

  def test_dataloader(self):
    loader = DataLoader(
        TSDataset(self.x_test, self.args, transform=Normalize()),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
        drop_last=True)
    return loader

  def predict_dataloader(self):
    loader = DataLoader(
        TSDataset(self.x_test, self.args, transform=Normalize()),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False,
        drop_last=True)
    return loader
