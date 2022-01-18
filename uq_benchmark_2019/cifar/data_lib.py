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

# Lint as: python2, python3
"""Library for constructing data for CIFAR experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os.path
from absl import flags
from absl import logging

import attr
import numpy as np
import robustness_dhtd
from six.moves import range
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uq_benchmark_2019 import experiment_utils
from uq_benchmark_2019 import image_data_utils

flags.DEFINE_string('cifar_dir', None, 'Path to CIFAR-10 data tables.')
flags.DEFINE_string('cifar_c_dir', None, 'Path to CIFAR-C data tables.')
FLAGS = flags.FLAGS
gfile = tf.io.gfile

_TINY_DATA_SIZE = 99

CIFAR_SHAPE = (32, 32, 3)
CIFAR_NUM_CLASSES = 10
CIFAR_NUM_TRAIN_EXAMPLES = int(4e4)  # Used for data-splitting and SVI training.

CIFAR_TEST_SIZE = 10000

_LABEL_BYTES = 1  # 2 for Cifar100.
_IMAGE_BYTES = 32 * 32 * 3
_RECORD_BYTES = _LABEL_BYTES + _IMAGE_BYTES
_CIFAR_FILENAMES = {'train': ['data_batch_%d.bin' % i for i in range(1, 5)],
                    'valid': ['data_batch_5.bin'], 'test': ['test_batch.bin']}


def _parse_cifar(record):
  """Parses a CIFAR10 example."""
  record_as_bytes = tf.io.decode_raw(record, tf.uint8)
  # The first bytes represent the label, which we convert from
  # uint8->int32.
  label = tf.cast(tf.strided_slice(record_as_bytes, [0], [_LABEL_BYTES]),
                  tf.int32)
  # The remaining bytes after the label represent the image, which
  # we reshape from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_as_bytes, [_LABEL_BYTES], [_RECORD_BYTES]),
      [3, 32, 32])
  # Convert from [depth, height, width] to [height, width, depth].
  image = tf.transpose(depth_major, [1, 2, 0])
  return (image, label)


def _download_cifar_split(split, is_training):
  """Build a CIFAR-10 dataset from TFDS (as opposed to a custom TFR table)."""
  del is_training
  filenames = [os.path.join(FLAGS.cifar_dir, f)
               for f in _CIFAR_FILENAMES[split]]
  dataset = (tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
             .map(_parse_cifar))
  return experiment_utils.download_dataset(dataset)


def _download_alt_dataset(config):
  dataset_to_dl = tfds.load(config.alt_dataset_name,
                            split=config.split,
                            as_supervised=True)
  return experiment_utils.download_dataset(dataset_to_dl)


def build_dataset(config, is_training=False, fake_data=False):
  """Returns a tf.data.Dataset with <image, label> pairs.

  Args:
    config: DataConfig instance.
    is_training: Whether to build a dataset for training
        (with shuffling and image distortions).
    fake_data: If true, use randomly generated data.
  Returns:
    tf.data.Dataset
  """
  if fake_data:
    logging.info('Generating fake data for config: %s', config)
    return image_data_utils.make_fake_data(CIFAR_SHAPE)

  logging.info('Building dataset for config:\n%s', attr.asdict(config))
  if config.corruption_type and config.corruption_static:
    return image_data_utils.make_static_dataset(config, _get_static_cifar_c)

  if config.alt_dataset_name:
    all_images, all_labels = _download_alt_dataset(config)
  else:
    all_images, all_labels = _download_cifar_split(config.split, is_training)

  if config.corruption_type:
    assert (config.corruption_value is not None) != (
        config.corruption_level > 0)
    # NOTE: dhtd corruptions expect to be applied before float32 conversion.
    apply_corruption = functools.partial(
        robustness_dhtd.corrupt,
        severity=config.corruption_level,
        severity_value=config.corruption_value, dim=32,
        corruption_name=config.corruption_type, dataset_name='cifar')
    all_images = np.stack([apply_corruption(im) for im in all_images])

  dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))

  def prep_fn(image, label):
    """Image preprocessing function."""
    if config.roll_pixels:
      image = tf.roll(image, config.roll_pixels, -2)
    if is_training:
      image = tf.image.random_flip_left_right(image)
      image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
      image = tf.image.random_crop(image, CIFAR_SHAPE)

    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

  return dataset.map(prep_fn)


def _get_static_cifar_c(corruption_type, corruption_level):
  """Load static CIFAR-C images to tf.dataset."""

  start_ind = (corruption_level - 1) * CIFAR_TEST_SIZE
  cifar_c_images = os.path.join(
      FLAGS.cifar_c_dir, '{}.npy'.format(corruption_type))
  cifar_c_labels = os.path.join(FLAGS.cifar_c_dir, 'labels.npy')

  with gfile.GFile(cifar_c_images, 'rb') as f:
    im = np.load(f)

  with gfile.GFile(cifar_c_labels, 'rb') as f:
    lab = np.load(f)

  return tf.data.Dataset.from_tensor_slices(
      (im[start_ind:start_ind+CIFAR_TEST_SIZE],
       lab[start_ind:start_ind+CIFAR_TEST_SIZE]))
