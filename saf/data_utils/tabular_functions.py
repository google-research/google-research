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

"""Misc. tabular processing functions."""

import numpy as np


# pylint: disable=redefined-builtin
def time_series_normalization(time_sequences_data,
                              target_index,
                              num_items=1,
                              counterfactual_index=None,
                              counterfactual_rel_index=None,
                              type='standard',
                              per_item=False,
                              epsilon=1e-6):
  """Normalizes time-varying datasets.

  Args:
    time_sequences_data: Time-series data of size (num_items,
      len_labeled_timesteps, num_features).
    target_index: Index of the target variable.
    num_items: Number of items.
    counterfactual_index: Index of the counterfactual variable.
    counterfactual_rel_index: Index of the relative counterfactual variable.
    type: Type of normalization: standard or min_max.
    per_item: Per item/entity/location or entire time-series normalization.
    epsilon: A small number against 0 division.

  Returns:
    time_sequences_data: Normalized ts data.
    output_shift: The tensor of shifts, of size [num_items, 1, 1].
    output_scale: The tensor of scales, of size [num_items, 1, 1].
  """

  if per_item:
    normalization_axes = (1)
  else:
    normalization_axes = (0, 1)

  if type == 'standard':
    time_varying_shift = np.mean(
        time_sequences_data, axis=normalization_axes, keepdims=True)
    time_varying_scale = np.std(
        time_sequences_data, axis=normalization_axes, keepdims=True)
  elif type == 'min_max':
    time_varying_shift = np.amin(
        time_sequences_data, axis=normalization_axes, keepdims=True)
    time_varying_scale = (
        np.amax(time_sequences_data, axis=normalization_axes, keepdims=True) -
        time_varying_shift)
  else:
    time_varying_shift = 0.0
    time_varying_scale = 1.0
    print('Normalization type not supported')

  # pylint: disable=g-bool-id-comparison
  if type is not None and per_item is False:
    time_varying_shift = np.tile(time_varying_shift, [num_items, 1, 1])
    time_varying_scale = np.tile(time_varying_scale, [num_items, 1, 1])

  time_sequences_data = (time_sequences_data - time_varying_shift) / (
      epsilon + time_varying_scale)

  output_shift = time_varying_shift[:, 0, target_index]
  output_scale = time_varying_scale[:, 0, target_index]

  if counterfactual_index and counterfactual_rel_index:
    counterfactual_shift = time_varying_shift[:, 0, counterfactual_index]
    counterfactual_scale = time_varying_scale[:, 0, counterfactual_index]
    counterfactual_rel_shift = time_varying_shift[:, 0,
                                                  counterfactual_rel_index]
    counterfactual_rel_scale = time_varying_scale[:, 0,
                                                  counterfactual_rel_index]
    return (time_sequences_data, output_shift, output_scale,
            counterfactual_shift, counterfactual_scale,
            counterfactual_rel_shift, counterfactual_rel_scale)
  else:
    return time_sequences_data, output_shift, output_scale


def static_normalization(static_data, type='standard', epsilon=1e-6):
  """Normalizes static datasets.

  Args:
    static_data: Static data.
    type: Type of normalization: standard or min_max.
    epsilon: A small number against 0 division.

  Returns:
    Normalized data.
  """

  if type == 'standard':
    static_shift = np.mean(static_data, axis=0, keepdims=True)
    static_scale = np.std(static_data, axis=0, keepdims=True)
  elif type == 'min_max':
    static_shift = np.amin(static_data, axis=0, keepdims=True)
    static_scale = (np.amax(static_data, axis=0, keepdims=True) - static_shift)
  else:
    static_shift = 0.0
    static_scale = 1.0
    print('Normalization type not supported')

  static_data = (static_data - static_shift) / (epsilon + static_scale)

  return static_data
