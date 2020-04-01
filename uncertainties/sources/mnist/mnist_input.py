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

"""MNIST handwritten digits dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


def load_data(data_path_mnist):
  """Loads the MNIST dataset.

  Args:
      data_path_mnist: path where mnist.npz is.
  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  nb_classes = 10
  with tf.gfile.Open(data_path_mnist, 'r') as f:
    tp = np.load(f)
    x_train, y_train = tp['x_train'], tp['y_train']
    x_test, y_test = tp['x_test'], tp['y_test']

  # Preprocessing
  x_train = x_train.reshape(60000, 784)
  x_test = x_test.reshape(10000, 784)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255

  y_train = tf.keras.utils.to_categorical(y_train, num_classes=nb_classes)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=nb_classes)

  return (x_train, y_train), (x_test, y_test)





