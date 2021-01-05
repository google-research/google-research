# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Generates a small-scale subset of SVHN dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio

np.random.seed(seed=0)


def sample(input_path,
           output_path,
           is_test=False,
           num_classes=5,
           n_train_per_class=60,
           n_test_per_class=1200):
  """Samples from the given input path and saves the sampled dataset."""

  train_data = sio.loadmat(input_path)

  new_data = []
  new_data_y = []

  new_data_1 = []
  new_data_y_1 = []

  for i in range(num_classes):
    label_id = i + 1
    ori_index = np.array(np.where(train_data['y'] == label_id)[0])
    np.random.shuffle(ori_index)
    index = ori_index[:n_train_per_class]
    label_data = np.array(train_data['X'][:, :, :, index])
    new_data.append(label_data)
    new_data_y.append(np.array(train_data['y'][index, :]))
    if is_test:
      index = ori_index[n_train_per_class:n_train_per_class + n_test_per_class]
      label_data = np.array(train_data['X'][:, :, :, index])
      new_data_1.append(label_data)
      new_data_y_1.append(np.array(train_data['y'][index, :]))
  new_data = np.concatenate(new_data, 3)
  new_data_y = np.concatenate(new_data_y, 0)

  sio.savemat(
      open(output_path, 'wb'),
      {
          'X': new_data,
          'y': new_data_y
      },
  )
  if is_test:
    new_data = np.concatenate(new_data_1, 3)
    new_data_y = np.concatenate(new_data_y_1, 0)
    sio.savemat(
        open('test_small_32x32.mat', 'wb'),
        {
            'X': new_data,
            'y': new_data_y
        },
    )


sample(
    'train_32x32.mat',
    'train_small_32x32.mat',
    n_train_per_class=60,
)
sample(
    'test_32x32.mat',
    'val_small_32x32.mat',
    n_train_per_class=600,
    n_test_per_class=1200,
    is_test=True)
