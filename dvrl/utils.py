# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
"""

# Necessary packages and function call
import numpy as np


def label_corruption(y_train, noise_rate):
  """Label corruptions on training labels.

  Args:
    y_train: training labels
    noise_rate: input noise ratio

  Returns:
    corrupted_y_train: corrupted training labels
    orig_idx: not corrupted index
    noise_idx: corrupted index
  """

  # Possible Y
  y_set = list(set(y_train))

  # Set orig_idx, noise_idx
  temp_idx = np.random.permutation(len(y_train))

  noise_idx = temp_idx[:int(len(y_train) * noise_rate)]
  orig_idx = temp_idx[int(len(y_train) * noise_rate):]

  # Corrupt label
  corrupted_y_train = y_train.copy()

  for itt in noise_idx:
    temp_y_set = y_set.copy()
    del temp_y_set[y_train[itt]]
    rand_idx = np.random.randint(len(y_set) - 1)
    corrupted_y_train[itt] = temp_y_set[rand_idx]

  return corrupted_y_train, orig_idx, noise_idx
