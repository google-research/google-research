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

"""Defines the WindowedSequenceDataset."""

# allow capital letter names for dimensions to improve clarity (e.g. N, T, D)
# pylint: disable=invalid-name

import numpy as np
import torch
import torch.utils.data
from utils.preprocessing_utils import compute_windows
from utils.preprocessing_utils import preprocess_numpy_data


class WindowedSequenceDataset(torch.utils.data.dataset.Dataset):
  """WindowedSequenceDataset object.

  Pytorch dataset object which formats sequences to prepare them as input
  to neural sequence lib (e.g. LSTM).
  """

  def __init__(
      self,
      x,
      lengths,
      forecasting_horizon,
      minmax_scaling,
      input_window_size,
      default_nan_value = 1e15,
      per_series_scaling=True,
      target_dims=(0,),
  ):
    """Initializes the WindowedSequenceDataset object.

    Takes in a numpy array of sequences, and pre-processes this data into a
    tensors
    ready for pytorch machine learning lib to learn from and predict on.

    Args:
      x: numpy array (N x T) of all sequences
      lengths: numpy array (N) containing the length of each sequence
      forecasting_horizon: desired number of timesteps into the future for
        lib to predict
      minmax_scaling: whether to scale the time series to be between 0 and 1
      input_window_size: number of past timesteps for model to use as input
      default_nan_value: value to impute nans with
      per_series_scaling: whether to scale differently for each series
      target_dims: dimensions containing target values
    """
    super().__init__()
    self.forecasting_horizon = forecasting_horizon
    self.minmax_scaling = minmax_scaling
    self.input_window_size = input_window_size
    self.default_nan_value = default_nan_value
    self.x, self.x_offset, self.x_scale = preprocess_numpy_data(
        x,
        minmax_scaling,
        default_nan_value=default_nan_value,
        per_series_scaling=per_series_scaling,
        ignore_dims=[
            i for i in range(x.shape[-1]) if i not in list(target_dims)
        ],
    )

    self.default_nan_value = default_nan_value
    self.lengths = torch.from_numpy(lengths)

    assert len(self.x_scale.shape) == 2
    assert len(self.x_offset.shape) == 2

  def _compute_inputs_and_targets(
      self, x, lengths, default_nan_value
  ):
    """Format x tensor into tensor of inputs and targets.

    Args:
      x: tensor (N x T x D) containing all time series
      lengths: tensor (N) containing the length of each series
      default_nan_value: value to impute nans with

    Returns:
      inputs: tensor, N x T_sliding (< T) x input_window_size x D, containing
      inputs
      targets: tensor, N x T_sliding (< T) x forecasting_horizon x
      len(target_dims), containing targets
      input_times: tensor of timestamps corresponding to each entry of inputs
      target_times: tensor of timestamps corresponding to each entry of targets
      target_times_mask: binary tensor, N x T_sliding (< T) x
      forecasting_horizon x T x D,
        where each entry (i,j,k,l,m) indicates whether the (i,j,k,l) entry in
        target_times is less than or equal to l + 1  (where l is zero-indexed).
        This is useful when trying to aggregate across all previous forecast
        errors (i.e.) all errors before each timepoint.
    """
    N, T, D = x.shape

    window = self.input_window_size
    forecasting_horizon = self.forecasting_horizon

    inputs, targets = compute_windows(x, window, forecasting_horizon)
    assert inputs.shape[0] == N
    assert inputs.shape[1] == T - window + 1
    assert inputs.shape[2] == window
    assert inputs.shape[3] == D

    assert targets.shape[0] == N
    assert targets.shape[1] == T - window + 1
    assert targets.shape[2] == forecasting_horizon
    assert targets.shape[3] == D

    # compute timesteps associated with each unrolled input and target
    times = np.ones((N, T)) * default_nan_value
    for i, l in enumerate(lengths):
      times[i, :l] = np.arange(1, l + 1)
    times = torch.from_numpy(times)
    input_times, target_times = compute_windows(
        times, window, forecasting_horizon
    )

    # compute masks associated with each timestep
    repeat_ts = target_times.unsqueeze(-1).repeat(1, 1, 1, 1, T + 1)
    target_times_mask = torch.zeros(repeat_ts.shape)
    for t in range(1, T + 2):
      idx = t - 1
      target_times_mask[:, :, :, :, idx] = (
          repeat_ts[:, :, :, :, idx] <= t
      ).float()
    target_times_mask = target_times_mask[
        :, :, :, :, window:
    ]  # starting e.g. timepoint 37, whether time is passed
    return inputs, targets, input_times, target_times, target_times_mask

  def __getitem__(self, index):
    inputs, targets, input_times, target_times, target_times_mask = (
        self._compute_inputs_and_targets(
            self.x[index : index + 1],
            self.lengths[index : index + 1],
            default_nan_value=self.default_nan_value,
        )
    )

    assert inputs.shape[0] == 1
    assert targets.shape[0] == 1
    assert input_times.shape[0] == 1
    assert target_times.shape[0] == 1

    d = {
        'x': self.x[index],
        'x_offset': self.x_offset,
        'x_scale': self.x_scale,
        'inputs': inputs[0],
        'targets': targets[0],
        'lengths': self.lengths[index],
        'input_times': input_times[0],
        'target_times': target_times[0],
        'target_times_mask': target_times_mask[0],
    }
    if self.x_offset.shape[0] > 1:
      d['x_offset'] = self.x_offset[index]
      d['x_scale'] = self.x_scale[index]
    return d

  def __len__(self):
    return len(self.x)
