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
"""Data processing and loading classes for time series datasets.

Defines data-related classes for data preprocessing, data splitting and
dataset loading.
"""

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class StandardScaler():
  """Applies standardization transformation to data."""

  def __init__(self, mean, std):
    """Instantiates the data transformation.

    Args:
      mean: mean to be subtracted from the data.
      std: standard deviation to be divided.
    """
    self.mean = torch.as_tensor(mean, dtype=torch.float)
    self.std = torch.as_tensor(std, dtype=torch.float)

  def __call__(self, inp_seq):
    """Applies the forward standardization transformation to input data.

    Args:
      inp_seq: input data, must have the same shape as self.mean and self.std.

    Returns:
      new_inp: transformed data.
    """
    new_inp = (inp_seq - self.mean) / self.std
    return new_inp

  def inverse_transform(self, inp_seq):
    """Applies the inverse standardization transformation to input data.

    Usually used to transform the predicted time series from the neural
    network model to real time series.

    Args:
      inp_seq: input data, must have the same shape as self.mean and self.std.

    Returns:
      new_inp: transformed data.
    """
    new_inp = inp_seq * self.std + self.mean
    return new_inp


class TSDataset(Dataset):
  """PyTorch dataset interface class for time series data."""

  def __init__(self, x, args, transform=None):
    """Instantiates the dataset interface.

    Args:
      x: the numpy array of time series data, with the shape (seq_len x
        num_nodes x num_features).
      args: python argparse.ArgumentParser class, we only use data-related
        arguments here.
      transform: the transformation applied to the data, we use StandardScaler
        here.
    """
    self.x = torch.as_tensor(x, dtype=torch.float)
    self.input_len = args.input_len
    self.output_len = args.output_len
    self.input_dim = args.input_dim
    self.output_dim = args.output_dim
    self.window_len = self.input_len + self.output_len
    self.transform = transform

  def __getitem__(self, index):
    """Gets the time series data sample corresponding to the index.

    Args:
      index: data sample index.

    Returns:
      Tuple of (input time series window, ground truth of output time
        series window).
    """
    input_seq = self.x[index:index + self.input_len]
    input_seq = input_seq[Ellipsis, :self.input_dim]
    output_seq = self.x[index + self.input_len:index + self.window_len]
    output_seq = output_seq[Ellipsis, :self.output_dim]

    if self.transform:
      input_seq = self.transform(input_seq)
      output_seq = self.transform(output_seq)

    return (input_seq, output_seq)

  def __len__(self):
    """Returns the total number of data samples in the dataset."""
    num_samples = self.x.shape[0] - self.window_len
    return num_samples


class DataModule(LightningDataModule):
  """PyTorch Lightning data module class for time series data."""

  def __init__(self, data, args):
    """Instantiates the data module.

    Args:
      data: the numpy array of time series data, with the shape (seq_len x
        num_nodes x num_features).
      args: python argparse.ArgumentParser class.
    """
    super().__init__()
    self.data = data  # [seq_len, num_nodes, num_features]
    self.args = args
    self.batch_size = args.batch_size
    self.num_workers = args.num_workers

  def setup(self):
    """Splits the data and defines the preprocessing transformation.

    Splits the data to train/test/val set with specified ratios in
    self.args.splits, and defines data preprocessing transformation.
    """
    num_samples = self.data.shape[0]

    num_train = round(num_samples * self.args.splits[0])
    num_test = round(num_samples * self.args.splits[1])
    num_val = num_samples - num_train - num_test

    self.x_train = self.data[:num_train]  # train series
    self.x_val = self.data[num_train:num_train + num_val]  # val_series
    self.x_test = self.data[-num_test:]  # test_series

    self.min_vals = self.x_train.min(axis=(0, 1), keepdims=True)
    self.max_vals = self.x_train.max(axis=(0, 1), keepdims=True)
    self.mean = self.x_train.mean(axis=(0, 1), keepdims=True)
    self.std = self.x_train.std(axis=(0, 1), keepdims=True)
    self.transform = StandardScaler(self.mean, self.std)

  def train_dataloader(self):
    """Returns the training data loader."""
    dataset = TSDataset(self.x_train, self.args, transform=self.transform)
    loader = DataLoader(
        dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=True)
    return loader

  def val_dataloader(self):
    """Returns the validation data loader."""
    dataset = TSDataset(self.x_val, self.args, transform=self.transform)
    loader = DataLoader(
        dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False)
    return loader

  def test_dataloader(self):
    """Returns the testing data loader."""
    dataset = TSDataset(self.x_test, self.args, transform=self.transform)
    loader = DataLoader(
        dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False)
    return loader
