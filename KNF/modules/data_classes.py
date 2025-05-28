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

"""dataset classes."""
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(torch.utils.data.Dataset):
  """Dataset class for M4 mini dataset or new dataset."""

  def __init__(
      self,
      input_length,  # num of input steps
      output_length,  # forecasting horizon
      freq,  # The frequency of time series
      direc,  # path to numpy data files
      direc_test=None,  # for testing mode, we also need to load test data.
      mode="train",  # train, validation or test
      jumps=1  # The number of skipped steps when generating samples
  ):

    self.input_length = input_length
    self.output_length = output_length
    self.mode = mode

    # Load training set
    self.data_lsts = np.load(direc, allow_pickle=True)

    # First do global standardization
    self.ts_means, self.ts_stds = [], []
    for i, item in enumerate(self.data_lsts):
      avg, std = np.mean(item), np.std(item)
      self.ts_means.append(avg)
      self.ts_stds.append(std)
      self.data_lsts[i] = (self.data_lsts[i] - avg) / std

    self.ts_means = np.array(self.ts_means)
    self.ts_stds = np.array(self.ts_stds)

    if mode == "test":
      self.test_lsts = np.load(direc_test, allow_pickle=True)
      for i, item in enumerate(self.test_lsts):
        self.test_lsts[i] = (item - self.ts_means[i]) / self.ts_stds[i]
      self.ts_indices = list(range(len(self.test_lsts)))

    elif mode == "train" or "valid":

      # shuffle slices before split
      np.random.seed(123)
      self.ts_indices = []
      for i, item in enumerate(self.data_lsts):
        for j in range(0,
                       len(item) - input_length - output_length,
                       jumps):
          self.ts_indices.append((i, j))

      np.random.shuffle(self.ts_indices)

      # 90%-10% train-validation split
      train_valid_split = int(len(self.ts_indices) * 0.9)
      if mode == "train":
        self.ts_indices = self.ts_indices[:train_valid_split]
      elif mode == "valid":
        self.ts_indices = self.ts_indices[train_valid_split:]
    else:
      raise ValueError("Mode can only be one of train, valid, test")

  def __len__(self):
    return len(self.ts_indices)

  def __getitem__(self, index):
    if self.mode == "test":
      x = self.data_lsts[index][-self.input_length:]
      y = self.test_lsts[index]

    else:
      i, j = self.ts_indices[index]
      x = self.data_lsts[i][j:j + self.input_length]
      y = self.data_lsts[i][j + self.input_length:j + self.input_length +
                            self.output_length]

    return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class M4Dataset(torch.utils.data.Dataset):
  """Dataset class for M4 dataset."""

  def __init__(
      self,
      input_length,  # num of input steps
      output_length,  # forecasting horizon
      freq,  # The frequency of time series
      direc,  # path to numpy data files
      direc_test=None,  # for testing mode, we also need to load test data.
      mode="train",  # train, validation or test
      jumps=1  # The number of skipped steps when generating sliced samples
  ):

    self.input_length = input_length
    self.output_length = output_length
    self.mode = mode

    # Load training set
    self.train_data = np.load(direc, allow_pickle=True)
    self.data_lsts = self.train_data.item().get(freq)

    # First do global standardization
    self.ts_means, self.ts_stds = [], []
    for i, item in enumerate(self.data_lsts):
      avg, std = np.mean(item), np.std(item)
      self.ts_means.append(avg)
      self.ts_stds.append(std)
      self.data_lsts[i] = (self.data_lsts[i] - avg) / std

    self.ts_means = np.array(self.ts_means)
    self.ts_stds = np.array(self.ts_stds)

    if mode == "test":
      self.test_lsts = np.load(direc_test, allow_pickle=True).item().get(freq)
      for i, item in enumerate(self.test_lsts):
        self.test_lsts[i] = (item - self.ts_means[i]) / self.ts_stds[i]
      self.ts_indices = list(range(len(self.test_lsts)))

    elif mode == "train" or "valid":

      # shuffle slices before split
      np.random.seed(123)
      self.ts_indices = []
      for i, item in enumerate(self.data_lsts):
        for j in range(0, len(item)-input_length-output_length, jumps):
          self.ts_indices.append((i, j))
      np.random.shuffle(self.ts_indices)

      # 90%-10% train-validation split
      train_valid_split = int(len(self.ts_indices) * 0.9)
      if mode == "train":
        self.ts_indices = self.ts_indices[:train_valid_split]
      elif mode == "valid":
        self.ts_indices = self.ts_indices[train_valid_split:]
    else:
      raise ValueError("Mode can only be one of train, valid, test")

  def __len__(self):
    return len(self.ts_indices)

  def __getitem__(self, index):
    if self.mode == "test":
      x = self.data_lsts[index][-self.input_length:]
      y = self.test_lsts[index]

    else:
      i, j = self.ts_indices[index]
      x = self.data_lsts[i][j:j + self.input_length]
      y = self.data_lsts[i][j + self.input_length:j + self.input_length +
                            self.output_length]

    return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class CrytosDataset(torch.utils.data.Dataset):
  """Dataset class for Cryptos data."""

  def __init__(
      self,
      input_length,  # num of input steps
      output_length,  # forecasting horizon
      direc,  # path to numpy data files
      direc_test=None,  # # for testing mode, we also need to load test data.
      mode="train",  # train, validation or test
      jumps=10,  # number of skipped steps between two sliding windows
      freq=None):

    self.input_length = input_length
    self.output_length = output_length
    self.mode = mode

    # Load training set
    self.train_data = np.load(direc, allow_pickle=True)

    # First do global standardization
    self.ts_means, self.ts_stds = [], []
    for i, item in enumerate(self.train_data):
      avg, std = np.mean(
          item, axis=0, keepdims=True), np.std(
              item, axis=0, keepdims=True)
      self.ts_means.append(avg)
      self.ts_stds.append(std)
      self.train_data[i] = (self.train_data[i] - avg) / std

    self.ts_means = np.concatenate(self.ts_means, axis=0)
    self.ts_stds = np.concatenate(self.ts_stds, axis=0)

    if mode == "test":
      self.test_lsts = np.load(direc_test, allow_pickle=True)
      for i, item in enumerate(self.test_lsts):
        self.test_lsts[i] = (self.test_lsts[i] -
                             self.ts_means[i]) / self.ts_stds[i]

      # change the input length (< 100) will not affect the target output
      self.ts_indices = []
      for i, item in enumerate(self.test_lsts):
        for j in range(100, len(item) - output_length, output_length):
          self.ts_indices.append((i, j))

    elif mode == "train" or "valid":
      # shuffle slices before split
      np.random.seed(123)
      self.ts_indices = []
      for i, item in enumerate(self.train_data):
        for j in range(0, len(item)-input_length-output_length, jumps):
          self.ts_indices.append((i, j))

      np.random.shuffle(self.ts_indices)

      # 90%-10% train-validation split
      train_valid_split = int(len(self.ts_indices) * 0.9)
      if mode == "train":
        self.ts_indices = self.ts_indices[:train_valid_split]
      elif mode == "valid":
        self.ts_indices = self.ts_indices[train_valid_split:]
    else:
      raise ValueError("Mode can only be one of train, valid, test")

  def __len__(self):
    return len(self.ts_indices)

  def __getitem__(self, index):
    if self.mode == "test":
      i, j = self.ts_indices[index]
      x = self.test_lsts[i][j - self.input_length:j]
      y = self.test_lsts[i][j:j + self.output_length]
    else:
      i, j = self.ts_indices[index]
      x = self.train_data[i][j:j + self.input_length]
      y = self.train_data[i][j + self.input_length:j + self.input_length +
                             self.output_length]
    return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class TrajDataset(torch.utils.data.Dataset):
  """Dataset class for NBA player trajectory data."""

  def __init__(
      self,
      input_length,  # num of input steps
      output_length,  # forecasting horizon
      direc,  # path to numpy data files
      direc_test=None,  # for testing mode, we also need to load test data.
      mode="train",  # train, validation or test
      jumps=10,  # number of skipped steps between two sliding windows
      freq=None):

    self.input_length = input_length
    self.output_length = output_length
    self.mode = mode

    # Load training set
    self.train_data = np.load(direc, allow_pickle=True)

    # First do global standardization
    self.ts_means, self.ts_stds = [], []
    for i, item in enumerate(self.train_data):
      avg = np.mean(item, axis=0, keepdims=True)
      std = np.std(item, axis=0, keepdims=True)
      self.ts_means.append(avg)
      self.ts_stds.append(std)
      self.train_data[i] = (item - avg) / std

    if mode == "test":
      self.ts_means, self.ts_stds = [], []
      self.test_lsts = np.load(direc_test, allow_pickle=True)
      for i, item in enumerate(self.test_lsts):
        avg = np.mean(item, axis=0, keepdims=True)
        std = np.std(item, axis=0, keepdims=True)
        self.ts_means.append(avg)
        self.ts_stds.append(std)
        self.test_lsts[i] = (self.test_lsts[i] - avg) / std

      # change the input length (<100) will not affect the target output
      self.ts_indices = []
      for i in range(len(self.test_lsts)):
        for j in range(50,
                       300 - output_length,
                       50):
          self.ts_indices.append((i, j))
    elif mode == "train" or "valid":
      # shuffle slices before split
      np.random.seed(123)
      self.ts_indices = []
      for i, item in enumerate(self.train_data):
        for j in range(0, len(item)-input_length-output_length, jumps):
          self.ts_indices.append((i, j))
      np.random.shuffle(self.ts_indices)

      # 90%-10% train-validation split
      train_valid_split = int(len(self.ts_indices) * 0.9)
      if mode == "train":
        self.ts_indices = self.ts_indices[:train_valid_split]
      elif mode == "valid":
        self.ts_indices = self.ts_indices[train_valid_split:]
    else:
      raise ValueError("Mode can only be one of train, valid, test")

    self.ts_means = np.concatenate(self.ts_means, axis=0)
    self.ts_stds = np.concatenate(self.ts_stds, axis=0)

  def __len__(self):
    return len(self.ts_indices)

  def __getitem__(self, index):
    if self.mode == "test":
      i, j = self.ts_indices[index]
      x = self.test_lsts[i][j - self.input_length:j]
      y = self.test_lsts[i][j:j + self.output_length]
    else:
      i, j = self.ts_indices[index]
      x = self.train_data[i][j:j + self.input_length]
      y = self.train_data[i][j + self.input_length:j + self.input_length +
                             self.output_length]
    return torch.from_numpy(x).float(), torch.from_numpy(y).float()
