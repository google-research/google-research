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

"""Three synthetic data generations.
"""

# Necessary functions and packages call
import numpy as np


def synthetic_data_loading(data_name='Syn1', data_no=1000, seed=0):
  """Generates synthetic datasets.

  Args:
    data_name: Syn1, Syn2, Syn3
    data_no: number of training and testing sets
    seed: random seed

  Returns:
    x_train: training features
    y_train: training labels
    x_test: testing features
    y_test: testing labels
    c_test: ground truth weights
    test_idx: order of testing set index based on the distance from the boundary
  """

  # X generation (X ~ N(0,I))
  np.random.seed(seed)
  data_x = np.random.normal(0, 1, [2 * data_no, 11])

  # Y and ground truth local dynamics (C) initialization
  data_y = np.zeros([2 * data_no,])
  data_c = np.zeros([2 * data_no, 11])

  # Boundary definition
  if data_name == 'Syn1':
    idx0 = np.where(data_x[:, 9] < 0)[0]
    idx1 = np.where(data_x[:, 9] >= 0)[0]

  elif data_name == 'Syn2':
    idx0 = np.where(data_x[:, 9] + np.exp(data_x[:, 10]) < 1)[0]
    idx1 = np.where(data_x[:, 9] + np.exp(data_x[:, 10]) >= 1)[0]

  elif data_name == 'Syn3':
    idx0 = np.where(data_x[:, 9] + np.power(data_x[:, 10], 3) < 0)[0]
    idx1 = np.where(data_x[:, 9] + np.power(data_x[:, 10], 3) >= 0)[0]

  # Y generation
  data_y[idx0] = data_x[idx0, 0] + 2 * data_x[idx0, 1]
  data_y[idx1] = 0.5 * data_x[idx1, 2] + 1 * data_x[idx1, 3] + \
       1 * data_x[idx1, 4] + 0.5 * data_x[idx1, 5]

  # Ground truth local dynamics (C) generation
  data_c[idx0, 0] = 1.0
  data_c[idx0, 1] = 2.0

  data_c[idx1, 2] = 0.5
  data_c[idx1, 3] = 1.0
  data_c[idx1, 4] = 1.0
  data_c[idx1, 5] = 0.5

  # Splits train/test sets
  x_train = data_x[:data_no, :]
  x_test = data_x[data_no:, :]

  y_train = data_y[:data_no]
  y_test = data_y[data_no:]

  c_test = data_c[data_no:, :]

  # Order of testing set index based on the distance from the boundary
  if data_name == 'Syn1':
    test_idx = np.argsort(np.abs(x_test[:, 9]))

  elif data_name == 'Syn2':
    test_idx = np.argsort(np.abs(x_test[:, 9] + np.exp(x_test[:, 10]) - 1))

  elif data_name == 'Syn3':
    test_idx = np.argsort(np.abs(x_test[:, 9] + np.power(x_test[:, 10], 3)))

  # Returns datasets
  return x_train, y_train, x_test, y_test, c_test, test_idx
