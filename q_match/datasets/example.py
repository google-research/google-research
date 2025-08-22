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

"""A minimal example dataset.
"""
import jax
import tensorflow as tf

from q_match.datasets.dataset import Dataset

_ROWS = 2 ** 9
_COLS = 2 ** 2


def _create_example_data():
  key = jax.random.PRNGKey(0)
  features = jax.random.normal(key=key, shape=(_ROWS, _COLS))
  targets = jax.random.bernoulli(key=key, shape=(_ROWS,)
                                 ).astype(jax.numpy.int32)
  return {'features': features, 'target': targets}


class ExampleDataset(Dataset):
  """Example dataset with minimal structure."""

  def __init__(self, dataset_path=None, batch_size=32):
    data = _create_example_data()

    self.batch_size = batch_size

    self.pretext_ds = tf.data.Dataset.from_tensor_slices(data)
    self.train_ds = tf.data.Dataset.from_tensor_slices(data)
    self.validation_ds = tf.data.Dataset.from_tensor_slices(data)
    self.test_ds = tf.data.Dataset.from_tensor_slices(data)

  def _chain_ops(self, ds):
    return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_pretext_ds(self):
    return self._chain_ops(self.pretext_ds)

  def get_train_ds(self):
    return self._chain_ops(self.train_ds)

  def get_validation_ds(self):
    return self._chain_ops(self.validation_ds)

  def get_test_epoch_iterator(self):
    return self._chain_ops(self.test_ds)

  def get_example_features(self):
    train_iterator = iter(self.get_train_ds())
    example_features = train_iterator.get_next()['features']
    del train_iterator
    return example_features

  def get_num_classes(self):
    return 2
