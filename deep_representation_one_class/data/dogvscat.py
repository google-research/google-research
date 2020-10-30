# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""DogVsCat data loader."""

import os
import numpy as np
import tensorflow as tf

from deep_representation_one_class.data.cifar import CIFAR


class DogVsCat(CIFAR):
  """DogVsCat data loader."""

  def __init__(self, root, dataset='dogvscat', input_shape=(64, 64, 3)):
    assert input_shape[0] == input_shape[1]
    np_path = os.path.join(
        root, 'dogs-vs-cats', 'train_{}x{}.npz'.format(input_shape[0],
                                                       input_shape[1]))
    with tf.io.gfile.GFile(np_path, 'rb') as f:
      np_data = np.load(f)
      image_array = np_data['image']
      label_array = np_data['label']

    # split into train and test
    image_pos = image_array[label_array[:, 0] == 0]
    image_neg = image_array[label_array[:, 0] == 1]
    label_pos = label_array[label_array[:, 0] == 0]
    label_neg = label_array[label_array[:, 0] == 1]
    np.random.seed(0)
    randidx_pos = np.random.permutation(image_pos.shape[0])
    train_image_pos = image_pos[randidx_pos[:10000]]
    train_label_pos = label_pos[randidx_pos[:10000]]
    test_image_pos = image_pos[randidx_pos[10000:]]
    test_label_pos = label_pos[randidx_pos[10000:]]
    randidx_neg = np.random.permutation(image_neg.shape[0])
    train_image_neg = image_neg[randidx_neg[:10000]]
    train_label_neg = label_neg[randidx_neg[:10000]]
    test_image_neg = image_neg[randidx_neg[10000:]]
    test_label_neg = label_neg[randidx_neg[10000:]]
    x_train = np.concatenate((train_image_pos, train_image_neg), axis=0)
    x_test = np.concatenate((test_image_pos, test_image_neg), axis=0)
    y_train = np.concatenate((train_label_pos, train_label_neg), axis=0)
    y_test = np.concatenate((test_label_pos, test_label_neg), axis=0)
    self.trainval_data = [
        x_train, y_train,
        np.expand_dims(np.arange(len(y_train)), axis=1)
    ]
    self.test_data = [
        x_test, y_test,
        np.expand_dims(np.arange(len(y_test)), axis=1)
    ]
    self.dataset = dataset
    self.input_shape = input_shape


class DogVsCatOOD(DogVsCat):
  """DogVsCat for OOD."""

  def __init__(self,
               root,
               dataset='dogvscat',
               input_shape=(64, 64, 3),
               category=0):
    super(DogVsCatOOD, self).__init__(
        root=root, dataset=dataset, input_shape=input_shape)
    if isinstance(category, str):
      try:
        category = int(float(category))
      except ValueError:
        msg = f'category {category} must be integer convertible.'
        raise ValueError(msg)
    self.category = category
    self.process_for_ood(category=self.category)
