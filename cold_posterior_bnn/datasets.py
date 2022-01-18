# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
""""Datasets prepared for SG-MCMC methods.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from cold_posterior_bnn.imdb import imdb_data


IMDB_NUM_WORDS = 20000
IMDB_SEQUENCE_LENGTH = 100


def load_imdb(with_info=False, subsample_n=0):
  """Load IMDB dataset.

  Args:
    with_info: bool, whether to return info dictionary.
    subsample_n: int, if >0, subsample training set to this size.

  Returns:
    Tuple of (dataset dict, info dict) if with_info else only
    the dataset.
  """
  (x_train, y_train), (x_val, y_val), (x_test, y_test) = imdb_data.load_data(
      num_words=IMDB_NUM_WORDS, maxlen=IMDB_SEQUENCE_LENGTH)

  original_train_size = x_train.shape[0]
  if subsample_n > 0:
    x_train = x_train[0:subsample_n, :]
    y_train = y_train[0:subsample_n]

  dataset = {
      'x_train': x_train,
      'y_train': y_train,
      'x_val': x_val,
      'y_val': y_val,
      'x_test': x_test,
      'y_test': y_test
  }

  if with_info:
    info = {
        'input_shape': x_train.shape[1:],
        'train_num_examples': x_train.shape[0],
        'train_num_examples_orig': original_train_size,
        'test_num_examples': x_test.shape[0],
        'num_classes': 2,
        'num_words': IMDB_NUM_WORDS,
        'sequence_length': IMDB_SEQUENCE_LENGTH,
    }
    return dataset, info

  return dataset


def load_cifar10(split, with_info=False, data_augmentation=True,
                 subsample_n=0):
  """This is a fork of edward2.utils.load_dataset.

  Returns a tf.data.Dataset with <image, label> pairs.

  Args:
    split: tfds.Split.
    with_info: bool.
    data_augmentation: bool, if True perform simple data augmentation on the
      TRAIN split with random left/right flips and random cropping.  If False,
      do not perform any data augmentation.
    subsample_n: int, if >0, subsample training set to this size.

  Returns:
    Tuple of (tf.data.Dataset, tf.data.DatasetInfo) if with_info else only
    the dataset.
  """
  dataset, ds_info = tfds.load('cifar10',
                               split=split,
                               with_info=True,
                               batch_size=-1)
  image_shape = ds_info.features['image'].shape

  numpy_ds = tfds.as_numpy(dataset)
  numpy_images = numpy_ds['image']
  numpy_labels = numpy_ds['label']

  # Perform subsampling if requested
  original_train_size = numpy_images.shape[0]
  if subsample_n > 0:
    subsample_n = min(numpy_images.shape[0], subsample_n)
    numpy_images = numpy_images[0:subsample_n, :, :, :]
    numpy_labels = numpy_labels[0:subsample_n]

  dataset = tf.data.Dataset.from_tensor_slices((numpy_images, numpy_labels))

  def preprocess(image, label):
    """Image preprocessing function."""
    if data_augmentation and split == tfds.Split.TRAIN:
      image = tf.image.random_flip_left_right(image)
      image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
      image = tf.image.random_crop(image, image_shape)

    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

  dataset = dataset.map(preprocess)

  if with_info:
    info = {
        'train_num_examples': ds_info.splits['train'].num_examples,
        'train_num_examples_orig': original_train_size,
        'test_num_examples': ds_info.splits['test'].num_examples,
        'input_shape': ds_info.features['image'].shape,
        'num_classes': ds_info.features['label'].num_classes,
    }
    return dataset, info
  return dataset


def get_generators_from_ds(dataset):
  """Returns generators for efficient training.

  Args:
    dataset: dataset dictionary.

  Returns:
    tfds generators for training and test data.
  """
  data_train = tf.data.Dataset.from_tensor_slices(
      (dataset['x_train'], dataset['y_train']))
  data_test = tf.data.Dataset.from_tensor_slices(
      (dataset['x_test'], dataset['y_test']))

  return data_train, data_test




