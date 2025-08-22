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

"""Utility functions for data pre-processing."""

# allow capital letter names for dimensions to improve clarity (e.g. N, T, D)
# pylint: disable=invalid-name

import numpy as np
import torch
import torch.utils.data


def get_dataloaders(
    dataset,
    train_prop = 0.7,
    val_prop = 0.1,
    batch_size = 20,
):
  """Get train and test dataloaders for a dataset.

  Args:
    dataset: pytorch Dataset
    train_prop: proportion of dataset to be used for training
    val_prop: proportion of dataset to be used for validation
    batch_size: batch size

  Returns:
    train_loader: training set dataloader
    val_loader: validation set dataloader
    test_loader: test set dataloader
  """

  cutoff1 = int(train_prop * len(dataset))
  cutoff2 = int(val_prop * len(dataset)) + cutoff1

  indices = torch.randperm(len(dataset))
  train_dataset = torch.utils.data.dataset.Subset(dataset, indices[:cutoff1])
  val_dataset = torch.utils.data.dataset.Subset(
      dataset, indices[cutoff1:cutoff2]
  )
  test_dataset = torch.utils.data.dataset.Subset(dataset, indices[cutoff2:])

  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True
  )

  val_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size=batch_size, shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
      test_dataset, batch_size=batch_size, shuffle=False
  )

  return train_loader, val_loader, test_loader


def preprocess_numpy_data(
    x,
    minmax_scaling,
    default_nan_value = 1e15,
    per_series_scaling=False,
    ignore_dims=None,
):
  """Pre-processes a numpy data into a tensor, imputing and scaling if desired.

  Pre-processing includes imputing with a given default value for nans, and
  scaling all series to be between 0 and 1 if desired.

  Args:
    x: numpy array (N series x T max timepoints) to be preprocessed, with nans
      padding sequences that are not length T
    minmax_scaling: whether to scale each series to be between 0 and 1
    default_nan_value: value to replace nans with
    per_series_scaling: whether to scale each series separately
    ignore_dims: any dimensions to ignore for scaling

  Returns:
    x_imputed: imputed (and scaled) x tensor
    x_offset: scaling additive offset tensor
    x_scale: scaling multiplicative factor tensor
  """
  ignored_x = None
  processed_dims = None
  if ignore_dims:
    ignored_x = x[:, :, ignore_dims]
    processed_dims = [i for i in range(x.shape[2]) if i not in ignore_dims]
    x = x[:, :, processed_dims]

  if per_series_scaling:
    x_offset = np.nanmin(x, axis=1)
    x_scale = np.nanmax(x, axis=1) - np.nanmin(x, axis=1)
    x_offset = torch.from_numpy(x_offset)
    x_scale = torch.from_numpy(x_scale)

    x_imputed = np.nan_to_num(x, nan=default_nan_value)
    offset = np.tile(np.array(x_offset)[:, np.newaxis, :], (1, x.shape[1], 1))
    scale = np.tile(np.array(x_scale)[:, np.newaxis, :], (1, x.shape[1], 1))
    offset = torch.from_numpy(offset)
    scale = torch.from_numpy(scale)
  else:
    x_offset = np.nanmin(np.nanmin(x, axis=1), axis=0)
    x_offset = np.tile(x_offset[np.newaxis, :], (x.shape[0], 1))
    x_scale = np.nanmax(np.nanmax(x, axis=1), axis=0) - np.nanmin(
        np.nanmin(x, axis=1), axis=0
    )
    x_scale = np.tile(x_scale[np.newaxis, :], (x.shape[0], 1))
    x_offset = torch.from_numpy(x_offset)
    x_scale = torch.from_numpy(x_scale)

    offset = np.tile(np.array(x_offset)[:, np.newaxis, :], (1, x.shape[1], 1))
    scale = np.tile(np.array(x_scale)[:, np.newaxis, :], (1, x.shape[1], 1))
    offset = torch.from_numpy(offset)
    scale = torch.from_numpy(scale)
  x_imputed = np.nan_to_num(x, nan=default_nan_value)
  x_imputed = torch.from_numpy(x_imputed).float()

  if minmax_scaling:
    x_imputed = (x_imputed - offset) / scale

  if ignore_dims:
    if processed_dims != [0]:
      # for now, assume that ignored dimensions come after processed ones
      raise NotImplementedError()
    ignored_x = torch.from_numpy(np.nan_to_num(ignored_x,
                                               nan=default_nan_value))
    x_imputed = torch.cat([x_imputed, ignored_x], dim=2)
  x_imputed.requires_grad = False
  return x_imputed, x_offset, x_scale


def get_rollout(
    tensor,
    dimension,
    rollout_size,
    keep_dim_size=True,
    impute_val=1e15,
    device=torch.device('cpu'),
):
  """Rolls out the tensor along the specified dimension.

  Args:
    tensor: tensor to roll out
    dimension: dimension along which to roll out
    rollout_size: how much to roll out
    keep_dim_size: whether to add padding to maintain the same tensor shape
    impute_val: value to use for padding
    device: device to perform computations on

  Returns:
    rolled out tensor
  """
  if keep_dim_size:
    padded_size = list(tensor.shape)
    padded_size[dimension] = rollout_size - 1
    nans = torch.ones(padded_size).to(device) * impute_val
    tensor = torch.cat([tensor, nans], dim=dimension)

  rollout = tensor.unfold(dimension=dimension, size=rollout_size, step=1)
  permutation = list(range(len(rollout.shape)))  # [0, 1, 2, 3, 4]
  permutation = (
      permutation[: dimension + 1]
      + [permutation[-1]]
      + permutation[dimension + 1 : -1]
  )  # [0, 1, 2, 4, 3]
  rollout = rollout.permute(*permutation)
  return rollout


def compute_windows(
    x,
    input_window_size,
    forecasting_horizon,
    impute_val = 1e15,
):
  """Decomposes the original series into input chunks and target chunks.

  Args:
    x: N x T x D tensor containing original series
    input_window_size: size of input window
    forecasting_horizon: how many timesteps to predict into future
    impute_val: what to replace nan timestamps with

  Returns:
    inputs: tensor, N x T_sliding (< T) x input_window_size x D
    outputs: tensor, N x T_sliding (< T) x input_window_size x D
  """
  if len(x.shape) == 3:
    N, _, D = x.shape
  else:
    assert len(x.shape) == 2
    N, _ = x.shape
    D = 1
    x = x.unsqueeze(-1)

  # computed windowed input (N x T - window + 1 x window)
  inputs = x.unfold(1, input_window_size, 1).permute(0, 1, 3, 2)

  # compute corresponding targets
  nans = np.ones((N, forecasting_horizon, D)) * impute_val
  nans = torch.from_numpy(nans).float()
  targets = torch.cat([x[:, input_window_size:], nans], dim=1)
  targets = targets.unfold(
      1, forecasting_horizon, 1
  )  # N  x T - window - L + 1 x L
  targets = targets.permute(0, 1, 3, 2)

  assert inputs.shape[:2] == targets.shape[:2]
  assert (targets[:, -1, :] < 1e10).sum() == 0  # last has no targets

  if len(x.shape) == 2:
    inputs = inputs.squeeze(-1)
    targets = targets.squeeze(-1)

  return inputs, targets


def collapse_first2dims(tensor):
  s = tensor.shape
  return tensor.view(s[0] * s[1], *s[2:])


def get_mask_from_target_times(target_times, device):
  """Given a tensor of target_times, retrieves a corresponding mask.

  If the target_times tensor is tensor of times of shape
  T1 (num timepoints) x T2 (num unrolled timepoints) x L (lead-time) x D,
  then the target_times_mask will be of shape T1 x T2 x L x D x T2,
  where each entry of the mask at slice [:, :, :, :, t] indicates whether
  the time is less than or equal to the time corresponding to t

  Args:
    target_times: a tensor of the times corresponding to each entry
    device: device to perform computations on

  Returns:
    a binary tensor mask
  """

  num_timepts, num_unrolled_timepts, _, _ = target_times.shape
  repeat_ts = target_times.unsqueeze(-1).repeat(
      1, 1, 1, 1, num_unrolled_timepts
  )
  target_times_mask = torch.zeros(repeat_ts.shape).to(device)
  for idx1 in range(num_timepts):
    start_t = int(min(1e10, target_times[idx1, 0, 0].item()))
    for idx4, t_cutoff in enumerate(
        range(start_t, start_t + num_unrolled_timepts)
    ):
      target_times_mask[idx1, :, :, :, idx4] = (
          repeat_ts[idx1, :, :, :, idx4] <= t_cutoff
      ).float()
  return target_times_mask
