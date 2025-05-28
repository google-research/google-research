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

"""RollingEvaluationDatasetFactory for roll-forward training and evaluation."""

from lib.windowed_sequence_dataset import WindowedSequenceDataset
import numpy as np
import torch.utils.data


class RollingEvaluationDatasetFactory:
  """RollingEvaluationDatasetFactory class.

  Produces datasets up to each time point, so training and prediction can
  proceed by rolling forward one timepoint at a time.
  """

  def __init__(
      self,
      x,
      lengths,
      forecasting_horizon,
      minmax_scaling,
      input_window_size,
      default_nan_value=1e15,
      per_series_scaling=True,
  ):
    self.x = x
    self.lengths = lengths
    self.forecasting_horizon = forecasting_horizon
    self.minmax_scaling = minmax_scaling
    self.input_window_size = input_window_size
    self.default_nan_value = default_nan_value
    self.per_series_scaling = per_series_scaling

    # entire dataset
    self.complete_dataset = WindowedSequenceDataset(
        x,
        lengths,
        forecasting_horizon,
        minmax_scaling,
        input_window_size,
        default_nan_value=default_nan_value,
        per_series_scaling=per_series_scaling,
    )

  def get_data_for_timepoint(self, t, test_per_series=True):
    """Gets data up to a certain time point.

    Args:
      t: time point to get data up to
      test_per_series: whether to return test data along with the
        training data in each batch

    Returns:
      dataset corresponding to data up to a certain time point
    """

    assert t > 0  # 1-indexed
    xt = self.x[:, :t]  # training data: data up to t
    train_dataset = WindowedSequenceDataset(
        xt,
        np.minimum(self.lengths, t),
        self.forecasting_horizon,
        self.minmax_scaling,
        self.input_window_size,
        default_nan_value=self.default_nan_value,
        per_series_scaling=self.per_series_scaling,
    )
    dataset = RollingEvaluationDataset(
        train_dataset,
        self.complete_dataset,
        t,
        self.input_window_size,
        test_per_series=test_per_series,
    )  # train_dataset with 'test_' keys for test data
    return dataset

  def __len__(self):
    return len(self.x)


class RollingEvaluationDataset(torch.utils.data.dataset.Dataset):
  """RollingEvaluationDataset class.

  Training data goes up to time t, and test data is the next time point for
  which predictions should be made.
  """

  def __init__(
      self,
      train_dataset,
      complete_dataset,
      t,
      input_window_size,
      test_per_series=True,
  ):
    super().__init__()
    self.train_dataset = train_dataset
    time_idx = t - input_window_size
    self.test_dataset = TimeIndexedDataset(
        complete_dataset, time_idx, prefix='test_'
    )
    self.t = t
    self.input_window_size = input_window_size
    self.test_per_series = test_per_series

  def __getitem__(self, index):
    train_d = self.train_dataset[index]
    d = train_d
    if self.test_per_series:
      test_d = {k: v for (k, v) in self.test_dataset[index].items()}
      d.update(test_d)
    return d

  def __len__(self):
    return len(self.train_dataset)

  def get_test_data(self):
    return self.test_dataset


class TimeIndexedDataset(torch.utils.data.dataset.Dataset):
  """TimeIndexedDataset class.

  Gets data corresponding to a certain time index.
  """

  def __init__(self, dataset, time_idx, prefix=''):
    self.dataset = dataset
    self.time_idx = time_idx
    self.prefix = prefix

  def __getitem__(self, index):
    datapt = self.dataset[index]
    d = {}
    for k, v in datapt.items():
      if len(v.shape) >= 3:
        d[self.prefix + k] = v[self.time_idx].unsqueeze(0)
      else:
        d[self.prefix + k] = v
    return d

  def __len__(self):
    return len(self.dataset)
