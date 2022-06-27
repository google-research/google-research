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

"""Data generators for the Cifar10 dataset."""

import tensorflow as tf

from gift.data import base_dataset
from gift.data import dataset_utils


class Cifar10(base_dataset.ImageDataset):
  """Data loader for the Cifar10 dataset.

  Original CIFAR10 with no perturbation.
  """

  @property
  def name(self):
    return 'cifar10'

  def set_static_dataset_configs(self):
    self._channels = 3
    # TODO(samiraabnar): What should we do when we don't have a validation set.
    self._splits_dict = {'train': 'train', 'test': 'test', 'validation': 'test'}
    self._crop_padding = 8
    self._mean_rgb = [0.4914, 0.4822, 0.4465]
    self._stddev_rgb = [0.2470, 0.2435, 0.2616]
    self.resolution = self.resolution or 32
    if self.resolution != 32:
      self.resize_mode = self.resize_mode or 'resize'
    else:
      self.resize_mode = None
    self.eval_augmentations = None


class TranslatedCifar10(Cifar10):
  """Translated Cifar10.

  CIFAR10 images shifted in various degrees in both x and y directions.
  """

  def set_static_dataset_configs(self):
    super().set_static_dataset_configs()
    self.resolution = self.resolution or 96
    self.resize_mode = self.resize_mode or 'resize'

  def preprocess_example(self, example, env_name=''):
    """Preprocesses the given image.

    Args:
      example: dict: Example that has an 'image' and a 'label'. wise.)
      env_name: str; Unused variable (Used in multi env setup).

    Returns:
      A preprocessed image `Tensor`.
    """
    example['image'] = dataset_utils.perturb_image(
        example['image'], {
            'scale_factor': 1.0,
            'translate_factor': tf.random.uniform((), minval=-1, maxval=1),
            'base_size': 32,
            'final_size': 32 * 3
        })

    return super().preprocess_example(example, env_name)


class ScaledTranslatedCifar10(Cifar10):
  """Translated Cifar10.

  CIFAR10 images shifted in various degrees in both x and y directions and also
  scaled with various degrees from 1x up to 2x.
  """

  def set_static_dataset_configs(self):
    super().set_static_dataset_configs()
    self.resolution = self.resolution or 96
    self.resize_mode = self.resize_mode or 'resize'

  def preprocess_example(self, example, env_name=''):
    """Preprocesses the given image.

    Args:
      example: dict: Example that has an 'image' and a 'label'. wise.)
      env_name: str; Unused variable (Used in multi env setup).

    Returns:
      A preprocessed image `Tensor`.
    """
    example['image'] = dataset_utils.perturb_image(
        example['image'], {
            'scale_factor': tf.random.uniform((), minval=1.0, maxval=2),
            'translate_factor': tf.random.uniform((), minval=-1, maxval=1),
            'base_size': 32,
            'final_size': 32 * 3
        })

    return super().preprocess_example(example, env_name)


class MultiCifar10(base_dataset.MutliEnvironmentImageDataset):
  """Data loader for the CIFAR10 dataset.

  Multi environment CIFAR10, includes original CIFAR10 environment and other
  arbitrary perturbations (scale, translate, blur).
  """

  _ALL_ENVIRONMENTS = [
      'cifar', 'translated', 'scaled', 'blurred', 'scaled_translated',
      'translated_gap'
  ]

  @property
  def name(self):
    return 'cifar10'

  def set_static_dataset_configs(self):
    self._channels = 3
    self._splits_dict = {'train': 'train', 'test': 'test', 'validation': 'test'}
    self._crop_padding = 8
    self._mean_rgb = [0.4914, 0.4822, 0.4465]
    self._stddev_rgb = [0.2470, 0.2435, 0.2616]
    self.resolution = self.resolution or 96
    self.resize_mode = None
    self.eval_augmentations = None

    # Get train splits
    train_splits = []
    train_splits_size = int(100 / len(self.train_environments))
    for i in range(len(self.train_environments)):
      start = train_splits_size * i
      end = train_splits_size * (i + 1)
      train_splits.append(f'train[{start}%:{end}%]')

    # Get test and validation splits
    test_splits = []
    validation_splits = []
    for i in range(len(self.eval_environments)):
      test_splits.append('test')
      validation_splits.append('test')

    self._splits_dict = {
        'train': dict(zip(self.train_environments, train_splits)),
        'test': dict(zip(self.eval_environments, test_splits)),
        'validation': dict(zip(self.eval_environments, validation_splits))
    }

  def preprocess_example(self, example, env_name):
    """Preprocesses the given image.

    Args:
      example: dict: Example that has an 'image' and a 'label'.
      env_name: float; How much the image should be rotate (counter clock wise.)

    Returns:
      A preprocessed image `Tensor`.
    """
    translate_factor = tf.random.uniform(
        (), minval=0.5 if 'gap' in env_name else 0.0,
        maxval=1) if 'translated' in env_name else 0.0
    scale_factor = tf.random.uniform(
        (), minval=1.5 if 'gap' in env_name else 1.0,
        maxval=2.0) if 'scaled' in env_name else 1.0

    example['image'] = dataset_utils.perturb_image(
        example['image'], {
            'scale_factor': scale_factor,
            'translate_factor': translate_factor,
            'blur_factor': 1.0 if 'blurred' in env_name else 0.0,
            'base_size': 32,
            'final_size': 32 * 3
        })

    example = super().preprocess_example(example, env_name)
    example['translate_factor'] = translate_factor
    example['scale_factor'] = scale_factor

    return example

  def get_tfds_env_name(self, env_name):
    """Environment name used to load tfds data."""
    return self.name


class MultiCifar10Rotated(base_dataset.MutliEnvironmentImageDataset):
  """Data loader for the CIFAR10 dataset.

  Multiple environments, each correspond to different degrees of rotation.
  """

  _ALL_ENVIRONMENTS = ['0_5', '5_55', '55_60']
  _ENV_RANGES = [':50%', '50%:75%', '75%:100%']

  @property
  def name(self):
    return 'cifar10'

  def set_static_dataset_configs(self):
    self._channels = 3
    self._crop_padding = 8
    self._mean_rgb = [0.4914, 0.4822, 0.4465]
    self._stddev_rgb = [0.2470, 0.2435, 0.2616]
    self.resolution = self.resolution or 32
    self.resize_mode = None
    self.eval_augmentations = None

    # Get train splits
    train_splits = []
    for i in range(len(self.train_environments)):
      train_splits.append(f'train[{self._ENV_RANGES[i]}]')

    # Get test and validation splits
    test_splits = []
    validation_splits = []
    for i in range(len(self.eval_environments)):
      test_splits.append('test')
      validation_splits.append('test')

    self._splits_dict = {
        'train': dict(zip(self.train_environments, train_splits)),
        'test': dict(zip(self.eval_environments, test_splits)),
        'validation': dict(zip(self.eval_environments, validation_splits))
    }

  def preprocess_example(self, example, env_name, rotate=True):
    """Preprocesses the given image.

    Args:
      example: dict: Example that has an 'image' and a 'label'.
      env_name: str; Environment name, used to determine how much the image
        should be rotate (counter clock wise.)
      rotate: bool; Determines whether we should apply rotation or not (this is
        to avoid duplicated rotation with the parents class is called from a
        child class).

    Returns:
      A preprocessed image `Tensor`.
    """
    example = super().preprocess_example(example, env_name)
    if rotate:
      example['inputs'] = dataset_utils.rotated_data_builder(
          example['inputs'], env_name)
    return example

  def get_tfds_env_name(self, env_name):
    """Environment name used to load tfds data."""
    return self.name


class MultiCifar10Scaled(base_dataset.MutliEnvironmentImageDataset):
  """Data loader for the CIFAR10 dataset.

    CIFAR10 images scaled based on fixed (constant) scale factors based on
    environment name.
  """

  _ALL_ENVIRONMENTS = ['1', '2', '3']

  @property
  def name(self):
    return 'cifar10'

  def set_static_dataset_configs(self):
    self._channels = 3
    self._crop_padding = 8
    self._mean_rgb = [0.4914, 0.4822, 0.4465]
    self._stddev_rgb = [0.2470, 0.2435, 0.2616]
    self.resolution = self.resolution or 32 * 3
    self.resize_mode = None
    self.eval_augmentations = None

    # Get train splits
    train_splits = []
    train_splits_size = int(100 / len(self.train_environments))
    for i in range(len(self.train_environments)):
      start = train_splits_size * i
      end = train_splits_size * (i + 1)
      train_splits.append(f'train[{start}%:{end}%]')

    # Get test and validation splits
    test_splits = []
    validation_splits = []
    for i in range(len(self.eval_environments)):
      test_splits.append('test')
      validation_splits.append('test')

    self._splits_dict = {
        'train': dict(zip(self.train_environments, train_splits)),
        'test': dict(zip(self.eval_environments, test_splits)),
        'validation': dict(zip(self.eval_environments, validation_splits))
    }

  def preprocess_example(self, example, env_name, rotate=True):
    """Preprocesses the given image.

    Args:
      example: dict: Example that has an 'image' and a 'label'.
      env_name: str; Environment name, used to determine how much the image
        should be rotate (counter clock wise.)
      rotate: bool; Determines whether we should apply rotation or not (this is
        to avoid duplicated rotation with the parents class is called from a
        child class).

    Returns:
      A preprocessed image `Tensor`.
    """

    example['image'] = dataset_utils.perturb_image(
        example['image'], {
            'scale_factor': float(env_name),
            'translate_factor': 0.0,
            'base_size': 32,
            'final_size': 32 * 3
        })

    return super().preprocess_example(example, env_name)

  def get_tfds_env_name(self, env_name):
    """Environment name used to load tfds data."""
    return self.name
