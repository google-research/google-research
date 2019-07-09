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

"""Common utilities for CIFAR and ImageNet datasets."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import attr
import numpy as np
import tensorflow as tf

CORRUPTION_TYPES = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
    'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
    'static_compression', 'pixelate', 'saturate', 'shot_noise', 'spatter',
    'speckle_noise', 'zoom_blur']

_TINY_DATA_SIZE = 99


@attr.s
class DataConfig(object):
  """Define config for (optionally) corrupted ImageNet and CIFAR data.

  Attributes:
    split: String, dataset split ('train' or 'test').
    roll_pixels: Int, number of pixels by which to roll the image.
    corruption_type: String, the name of the corruption function to apply
      (must be one of CORRUPTION_TYPES).
    corruption_static: Bool. If True, use the corrupted images provided by
      Hendrycks and Dietterich (2019) as static imagess. If False, apply
      corruption functions to standard images.
    corruption_level: Int, level (from 1 to 5) of the corruption values
      defined by Hendrycks and Dietterich (2019). If 0, then use
      `corruption_value` instead.
    corruption_value: Float or tuple, corruption value to apply to the image
      data. If None, then use `corruption_level` instead.
    alt_dataset_name: Optional name of an alternate dataset (e.g. SVHN for
      OOD CIFAR experiments).
  """
  split = attr.ib()
  roll_pixels = attr.ib(0)
  corruption_type = attr.ib(
      None, validator=attr.validators.in_(CORRUPTION_TYPES + [None]))
  corruption_static = attr.ib(False)
  corruption_level = attr.ib(0, validator=attr.validators.in_(range(6)))
  corruption_value = attr.ib(None)
  alt_dataset_name = attr.ib(None)


DATA_CONFIG_TRAIN = DataConfig('train')
DATA_CONFIG_VALID = DataConfig('valid')
DATA_CONFIG_TEST = DataConfig('test')


def make_fake_data(image_shape, num_examples=_TINY_DATA_SIZE):
  images = np.random.rand(num_examples, *image_shape)
  labels = np.random.randint(0, 10, num_examples)
  return tf.data.Dataset.from_tensor_slices((images, labels))


def make_static_dataset(config, data_reader_fn):
  """Make a tf.Dataset of corrupted images read from disk."""
  if config.corruption_level not in range(1, 6):
    raise ValueError('Corruption level of the static images must be between 1'
                     ' and 5.')
  if config.split != 'test':
    raise ValueError('Split must be `test` for corrupted images.')

  if config.corruption_value is not None:
    raise ValueError('`corruption_value` must be `None` for static images.')

  dataset = data_reader_fn(config.corruption_type, config.corruption_level)

  def convert_image_dtype(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label
  return dataset.map(convert_image_dtype)


def get_data_config(name):
  """Parse data-config name into a DataConfig.

  Args:
    name: String of form "{corruption-family}-{options}" or "train", "test".
        For example: roll-24, corrupt-static_brightness_1
  Returns:
    DataConfig instance.
  """
  base_configs = {'train': DATA_CONFIG_TRAIN,
                  'valid': DATA_CONFIG_VALID, 'test': DATA_CONFIG_TEST}
  if name in base_configs:
    return base_configs[name]

  parsed = name.split('-', 1)
  if parsed[0] == 'roll':
    return DataConfig('test', roll_pixels=int(parsed[1]))
  elif parsed[0] == 'corrupt':
    corruption_src, corruption_type, corruption_x = parsed[1].split('-', 2)
    if corruption_src in ('static', 'array'):
      return DataConfig('test',
                        corruption_type=corruption_type,
                        corruption_static=corruption_src == 'static',
                        corruption_level=int(corruption_x))
    elif corruption_src == 'value':
      return DataConfig('test',
                        corruption_type=corruption_type,
                        corruption_value=float(corruption_x))
  elif parsed[0] == 'svhn':
    return DataConfig('test', alt_dataset_name='svhn_cropped')
  elif parsed[0] == 'celeb_a':
    return DataConfig('test', alt_dataset_name='celeb_a')
  raise ValueError('Data config name not recognized: %s' % name)
