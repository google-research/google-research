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

# Lint as: python2, python3
"""CIFAR python-version dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import six.moves.cPickle
import tensorflow.compat.v1 as tf

NUM_TRAIN = 50000
NUM_TEST = 10000
IMAGE_SIZE = 24


def load_file(filename, data_path):
  with tf.gfile.Open(os.path.join(data_path, filename), 'rb') as fo:
    dic = six.moves.cPickle.load(fo)
  return dic


def preprocess(images, distorted):
  """Preprocess the images."""
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  n = images.shape[0]
  images = tf.cast(images, tf.float32)

  if distorted:
    # Image processing for training the network. Note the many random
    # distortions applied to the image.
    # Randomly crop a [height, width] section of the image.
    images = tf.random_crop(images, [n, height, width, 3])
    # Randomly flip the image horizontally.
    images = tf.image.random_flip_left_right(images)
  else:
    images = tf.image.resize_image_with_crop_or_pad(images, height, width)

  # Subtract off the mean and divide by the variance of the pixels.
  images = tf.map_fn(tf.image.per_image_standardization, images)
  # Set the shapes of tensors.
  images.set_shape([n, height, width, 3])
  return images


def load_data(distorted, data_path, dataset):
  """Loads the CIFAR10/100 datasets.

  Args:
    distorted: boolean, True if the train images are distorted.
    data_path: path of the data directory
    dataset: string, cifar10 or cifar100

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  x_train, y_train = [], []
  if dataset == 'cifar10':
    num_classes = 10
    for filename in ['data_batch_' + str(i) for i in np.arange(1, 6)]:
      dic = load_file(filename, data_path)
      x_train.append(dic['data'])
      y_train.append(np.asarray(dic['labels']))

    x_train = np.vstack(tuple(x_train))
    y_train = np.concatenate(tuple(y_train))

    dic = load_file('test_batch', data_path)
    x_test = dic['data']
    y_test = np.asarray(dic['labels'])
  elif dataset == 'cifar100':
    num_classes = 100
    dic = load_file('train', data_path)
    x_train = dic['data']
    y_train = np.asarray(dic['fine_labels'])
    dic = load_file('test', data_path)
    x_test = dic['data']
    y_test = np.asarray(dic['fine_labels'])
  else:
    raise NotImplementedError('Dataset should be cifar10 or cifar100')

  # Preprocessing
  x_train = np.reshape(x_train, [x_train.shape[0], 3, 32, 32])
  x_test = np.reshape(x_test, [x_test.shape[0], 3, 32, 32])
  x_train = np.transpose(x_train, [0, 2, 3, 1])
  x_test = np.transpose(x_test, [0, 2, 3, 1])

  y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

  x_train = preprocess(x_train, distorted=distorted)
  x_test = preprocess(x_test, distorted=False)

  # Convert from tf Tensor to numpy array
  sess = tf.Session()
  with sess.as_default():
    x_train_np, x_test_np = sess.run([x_train, x_test])

  return (x_train_np, y_train), (x_test_np, y_test)
