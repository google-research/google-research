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

"""Utility functions.

Operations are compatible with pytorch.
Includes computation of averages and variances on variable-length sequences.
"""

import torch


def get_ragged_sum(
    arr,
    lens,
    axis = -1,
    device=torch.device('cpu'),
):
  """Compute sum along the given axis.

  Assumes that non-nans of arr are grouped at the beginning of the sequence.

  Args:
    arr: Tensor containing sequences
    lens: Tensor containing actual length of each sequence
    axis: axis along which to take the mean
    device: which device to perform computations on

  Returns:
    Tensor with averages for each sequence
  """
  # add zero as first dimension so that when there are zero non-nan values,
  # it selects zero as the value
  zeros_shape = list(arr.shape)
  zeros_shape[axis] = 1
  zero = torch.zeros(zeros_shape).to(device)
  arr = torch.cat([zero, arr], dim=axis)
  arr = torch.cumsum(arr, axis)

  sums = torch.gather(arr, axis, lens)

  mask = (lens > 0).float()
  sums = sums * mask
  arr = sums.squeeze(axis)
  return arr


def get_ragged_mean(
    arr,
    lens,
    axis = -1,
    device=torch.device('cpu'),
):
  """Compute average value along the given axis.

  Assumes that non-nans of arr are grouped at the beginning of the sequence.

  Args:
    arr: Tensor containing sequences
    lens: Tensor containing actual length of each sequence
    axis: axis along which to take the mean
    device: which device to perform computations on

  Returns:
    Tensor with averages for each sequence
  """
  # add zero as first dimension so that when there are zero non-nan values,
  # it selects zero as the value
  zeros_shape = list(arr.shape)
  zeros_shape[axis] = 1
  zero = torch.zeros(zeros_shape).to(device)
  arr = torch.cat([zero, arr], dim=axis)
  arr = torch.cumsum(arr, axis)

  sums = torch.gather(arr, axis, lens)
  mask = (lens > 0).float()
  sums = sums * mask
  assert mask.max() <= 1.0
  soft_lens = lens.float() + (1 - mask)  # replace 0's with 1's to avoid nans
  arr = sums / soft_lens
  arr = arr.squeeze(axis)
  return arr


def get_ragged_var(
    arr,
    lens,
    axis = -1,
    device=torch.device('cpu'),
):
  """Compute variance along the given axis.

  Assumes non-nans of arr are grouped at the beginning of the sequence.

  Args:
    arr: Tensor containing sequences
    lens: Tensor containing actual length of each sequence
    axis: axis along which to take the variance
    device: which device to perform computations on

  Returns:
    Tensor of variances for each sequence
  """
  # subtract mean from each value
  repeat_shape = [1] * len(arr.shape)
  repeat_shape[axis] = arr.shape[axis]
  squared_dev = (
      arr
      - get_ragged_mean(arr, lens=lens, axis=axis, device=device)
      .unsqueeze(axis)
      .repeat(*repeat_shape)
  ) ** 2

  # add zero as first dimension so that when there are zero non-nan values,
  # it selects zero as the value
  arr = squared_dev
  zeros_shape = list(arr.shape)
  zeros_shape[axis] = 1
  zero = torch.zeros(zeros_shape).to(device)
  arr = torch.cat([zero, arr], dim=axis)
  arr = torch.cumsum(arr, axis)

  sums = torch.gather(arr, axis, lens)
  mask = (lens > 0).float()
  sums = sums * mask
  assert mask.max() <= 1.0
  soft_lens = lens.float() + (1 - mask)  # replace 0's with 1's to avoid nans
  arr = sums / soft_lens
  arr = arr.squeeze(axis)
  return arr
