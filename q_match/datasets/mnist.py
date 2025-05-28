# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""MNIST daataset.
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from q_match.datasets import dataset_splits
from q_match.datasets.dataset import Dataset

_NAME = 'MNIST'


class MNIST1PDataset(Dataset):
  """MNIST dataset with 1% as labeled."""

  def __init__(self, dataset_path, batch_size=512, num_parallel_calls=60):
    self.pretext_ds = tfds.load(
        _NAME,
        data_dir=dataset_path,
        split=self.get_pretext_split())

    self.train_ds = tfds.load(
        _NAME,
        data_dir=dataset_path,
        split=self.get_train_split())

    self.validation_ds = tfds.load(
        _NAME,
        data_dir=dataset_path,
        split=self.get_validation_split())

    self.test_ds = tfds.load(
        _NAME,
        data_dir=dataset_path,
        split=self.get_test_split())

    if self.get_pretext_validation_split() is None:
      self.pretext_validation_ds = None
    else:
      self.pretext_validation_ds = tfds.load(
          _NAME,
          data_dir=dataset_path,
          split=self.get_pretext_validation_split())

    self.batch_size = batch_size
    self.num_parallel_calls = num_parallel_calls

    self.num_classes = dataset_splits.DATASET_TO_NUM_CLASSES['mnist']

  def get_pretext_split(self):
    """Pretext split."""
    return 'train[:95%]'

  def get_train_split(self):
    """Train split."""
    return 'train[:1%]'

  def get_validation_split(self):
    """Validation split."""
    return 'train[1%:11%]'

  def get_test_split(self):
    """Test split."""
    return 'test'

  def get_pretext_validation_split(self):
    """Pretext validation split."""
    return 'train[95%:]'

  def _format_data(self, original_data):
    return {
        'features':
            tf.cast(tf.reshape(original_data['image'], [-1]), tf.float32),
        'target':
            original_data['label']
    }

  def get_train_ds(self, cache=True, shuffle=True):
    train_ds = self.train_ds.map(self._format_data,
                                 num_parallel_calls=self.num_parallel_calls)
    if cache:
      train_ds = train_ds.cache()
    if shuffle:
      train_ds = train_ds.shuffle(10000)
    return train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_validation_ds(self):
    validation_ds = self.validation_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    return validation_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_test_epoch_iterator(self):
    test_ds = self.test_ds.map(self._format_data,
                               num_parallel_calls=self.num_parallel_calls)
    return test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_example_features(self):
    train_iterator = iter(self.get_train_ds())
    example_features = train_iterator.get_next()['features']
    del train_iterator
    return example_features

  def get_pretext_ds(self, cache=True, shuffle=True):
    pretext_ds = self.pretext_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    if cache:
      pretext_ds = pretext_ds.cache()
    if shuffle:
      pretext_ds = pretext_ds.shuffle(10000)
    return pretext_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_pretext_validation_ds(self):
    if self.pretext_validation_ds is None:
      return None
    validation_ds = self.pretext_validation_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    return validation_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_num_classes(self):
    return 10


class MNIST10PDataset(MNIST1PDataset):
  """10% labeled data."""

  def get_pretext_split(self):
    """Pretext split."""
    return 'train'

  def get_train_split(self):
    """Train split."""
    return 'train[:10%]'

  def get_validation_split(self):
    """Validation split."""
    return 'train[10%:20%]'

  def get_test_split(self):
    """Test split."""
    return 'test'

  def get_pretext_validation_split(self):
    """Pretext validation split."""
    return None


class MNISTPre100Dataset(MNIST1PDataset):
  """10% labeled data."""

  def get_pretext_split(self):
    """Pretext split."""
    return 'train'

  def get_train_split(self):
    """Train split."""
    return 'train[:1%]'

  def get_validation_split(self):
    """Validation split."""
    return 'train[10%:20%]'

  def get_test_split(self):
    """Test split."""
    return 'test'

  def get_pretext_validation_split(self):
    """Pretext validation split."""
    return None


class MNISTPre10Dataset(MNIST1PDataset):
  """10% labeled data."""

  def get_pretext_split(self):
    """Pretext split."""
    return 'train[:10%]'

  def get_train_split(self):
    """Train split."""
    return 'train[:1%]'

  def get_validation_split(self):
    """Validation split."""
    return 'train[10%:20%]'

  def get_test_split(self):
    """Test split."""
    return 'test'

  def get_pretext_validation_split(self):
    """Pretext validation split."""
    return None
