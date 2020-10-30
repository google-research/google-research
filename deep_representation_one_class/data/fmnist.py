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
"""FashionMNIST data loader."""

import numpy as np
import tensorflow_datasets as tfds

from deep_representation_one_class.data.cifar import CIFAR


class FashionMNIST(CIFAR):
  """FashionMNIST data loader."""

  def __init__(self, root, dataset='fmnist', input_shape=(32, 32, 3)):
    builder = tfds.builder(name='fashion_mnist', data_dir=root)
    if not root:
      builder.download_and_prepare()
    ds_train = builder.as_dataset(split='train')
    ds_test = builder.as_dataset(split='test')
    x_train_raw, y_train = self.get_numpy_dataset(ds_train)
    x_test_raw, y_test = self.get_numpy_dataset(ds_test)
    x_train = np.zeros((x_train_raw.shape[0], 32, 32, 1), dtype=np.uint8)
    x_train[:, 2:-2, 2:-2, :] = x_train_raw
    x_train = np.concatenate([x_train] * 3, axis=-1)
    x_test = np.zeros((x_test_raw.shape[0], 32, 32, 1), dtype=np.uint8)
    x_test[:, 2:-2, 2:-2, :] = x_test_raw
    x_test = np.concatenate([x_test] * 3, axis=-1)
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

  def get_numpy_dataset(self, ds):
    image_array = np.stack([d['image'].numpy() for d in ds])
    label_array = np.stack([d['label'].numpy() for d in ds])[:, None]
    label_array = np.uint8(label_array)
    return image_array, label_array


class FashionMNISTOOD(FashionMNIST):
  """FashionMNIST for OOD."""

  def __init__(self,
               root,
               dataset='fmnist',
               input_shape=(32, 32, 3),
               category=0):
    super(FashionMNISTOOD, self).__init__(
        root=root, dataset=dataset, input_shape=input_shape)
    if isinstance(category, str):
      try:
        category = int(float(category))
      except ValueError:
        msg = f'category {category} must be integer convertible.'
        raise ValueError(msg)
    self.category = category
    self.process_for_ood(category=self.category)
