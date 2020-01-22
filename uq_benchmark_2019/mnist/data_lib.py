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

# Lint as: python2, python3
"""Library for constructing MNIST and distorted MNIST datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import flags
from absl import logging

import attr
import numpy as np
import scipy.ndimage
from six.moves import range
from six.moves import zip

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

NUM_TRAIN_EXAMPLES = 50 * 1000
DUMMY_DATA_SIZE = 99
MNIST_IMAGE_SHAPE = (28, 28, 1)

DATA_OPTS_ROLL = [dict(split='test', roll_pixels=k) for k in range(2, 28, 2)]
DATA_OPTS_ROTATE = [dict(split='test', rotate_degs=k)
                    for k in range(15, 181, 15)]
DATA_OPTS_OOD = [dict(split='test', dataset_name='fashion_mnist'),
                 dict(split='test', dataset_name='not_mnist')]

DATA_OPTIONS_LIST = [
    dict(split='train'),
    dict(split='valid'),
    dict(split='test')] + DATA_OPTS_ROLL + DATA_OPTS_ROTATE + DATA_OPTS_OOD


FLAGS = flags.FLAGS
flags.DEFINE_string('mnist_path_tmpl', None,
                    'Template path to MNIST data tables.')
flags.DEFINE_string('not_mnist_path_tmpl', None,
                    'Template path to NotMNIST data tables.')


@attr.s
class MnistDataOptions(object):
  split = attr.ib()
  dataset_name = attr.ib('mnist')
  roll_pixels = attr.ib(0)
  rotate_degs = attr.ib(0)


def _crop_center(images, size):
  height, width = images.shape[1:3]
  i0 = height // 2 - size // 2
  j0 = width // 2 - size // 2
  return images[:, i0:i0 + size, j0:j0 + size]


def _tfr_parse_fn(serialized, img_bytes_key='image/encoded'):
  features = {'image/class/label': tf.io.FixedLenFeature((), tf.int64),
              img_bytes_key: tf.io.FixedLenFeature([1], tf.string)}
  parsed = tf.io.parse_single_example(serialized, features)
  image = tf.io.decode_raw(parsed[img_bytes_key], tf.uint8)
  image = tf.reshape(image, [28, 28, 1])
  return image, parsed['image/class/label']


def _mnist_dataset_from_tfr(split_name):
  # train_small contains the first 50K rows the train set; valid is the last 10K
  split_key = 'train_small' if split_name == 'train' else split_name
  path = FLAGS.mnist_path_tmpl % split_key
  logging.info('Reading dataset from %s', path)
  parse_fn = functools.partial(_tfr_parse_fn, img_bytes_key='image/encoded')
  return tf.data.TFRecordDataset(path).map(parse_fn)


def _not_mnist_dataset_from_tfr(split_name):
  if split_name != 'test':
    raise ValueError('We should only use NotMNIST test data.')
  path = FLAGS.not_mnist_path_tmpl % split_name
  logging.info('Reading dataset from %s', path)
  parse_fn = functools.partial(_tfr_parse_fn, img_bytes_key='image/raw')
  return tf.data.TFRecordDataset(path).map(parse_fn)


def _dataset_from_tfds(dataset_name, split):
  if split != 'test':
    raise ValueError('We should only use split=test from tfds.')
  return tfds.load(dataset_name, split=split, as_supervised=True)


def build_dataset(opts, fake_data=False):
  """Returns an <images, labels> dataset pair."""
  opts = MnistDataOptions(**opts)
  logging.info('Building dataset with options: %s', opts)

  if fake_data:
    images = np.random.rand(DUMMY_DATA_SIZE, *MNIST_IMAGE_SHAPE)
    labels = np.random.randint(0, 10, DUMMY_DATA_SIZE)
    return images, labels

  # We can't use in-distribution data from tfds due to inconsistent orderings.
  if opts.dataset_name == 'mnist':
    dataset = _mnist_dataset_from_tfr(opts.split)
  elif opts.dataset_name == 'not_mnist':
    dataset = _not_mnist_dataset_from_tfr(opts.split)
  else:
    dataset = _dataset_from_tfds(opts.dataset_name, opts.split)

  # Download dataset to memory.
  images, labels = list(zip(*tfds.as_numpy(dataset.batch(10**4))))
  images = np.concatenate(images, axis=0).astype(np.float32)
  labels = np.concatenate(labels, axis=0)

  images /= 255
  if opts.rotate_degs:
    images = scipy.ndimage.rotate(images, opts.rotate_degs, axes=[-2, -3])
    images = _crop_center(images, 28)
  if opts.roll_pixels:
    images = np.roll(images, opts.roll_pixels, axis=-2)

  return images, labels
