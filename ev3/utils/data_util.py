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

"""Data utility methods."""

from collections.abc import Mapping, Sequence
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from ev3 import base


class TestDataIterator(base.DataIterator):
  """An iterator class generating batches of test data."""

  def __init__(
      self,
      all_data,
      batch_size,
      n_all_data,
      rand_key,
      batch_axis = 0,
  ):
    """Constructor.

    Args:
      all_data: All the data that is to be used to generate batches.
      batch_size: Number of samples in each batch.
      n_all_data: Number of samples in all_data.
      rand_key: Random seed determining how the batches are sampled from
        all_data.
      batch_axis: The axis along with the samples are enumerated.
    """
    assert batch_size <= n_all_data, (
        f'Batch size (={batch_size}) cannot be larger than the total number of '
        f'samples in the dataset (i.e. {n_all_data}).'
    )
    self.all_data = all_data
    self.batch_axis = batch_axis
    self.batch_size = batch_size
    self.n_all_data = n_all_data
    self.rand_key = rand_key
    self._start_pos = 0

  def __next__(self):
    if self._start_pos + self.batch_size > self.n_all_data:
      self._start_pos = 0
      self.rand_key, perm_key = jax.random.split(self.rand_key, 2)
      self.all_data = jax.tree.map(
          lambda arr: jax.random.permutation(  # pylint: disable=g-long-lambda
              perm_key, arr, axis=self.batch_axis
          ),
          self.all_data,
      )
    batch = jax.tree.map(
        lambda arr: jax.lax.dynamic_slice_in_dim(  # pylint: disable=g-long-lambda
            arr,
            start_index=self._start_pos,
            slice_size=self.batch_size,
            axis=self.batch_axis,
        ),
        self.all_data,
    )
    self._start_pos += self.batch_size
    return batch

  def concat_batches(self, batches, batch_axis=0):
    def concat(*arrays):
      return jnp.concatenate(arrays, axis=batch_axis)

    return jax.tree.map(concat, *batches)

  def next(self, num_batches=1):
    """Get a batch with num_batches x batch_size samples.

    Args:
      num_batches: Number of batches to merge together.

    Returns:
      A batch of data of size num_batches x batch_size.
    """
    return self.concat_batches([self.__next__() for _ in range(num_batches)])


class NumpyDataIterator(base.DataIterator):
  """An iterator class generating batches of data from a large NumPy dataset."""

  def __init__(
      self,
      all_data,
      batch_size,
      n_all_data,
      rand_key = 42,
      batch_axis = 0,
  ):
    """Constructor.

    Args:
      all_data: All the data that is to be used to generate batches.
      batch_size: Number of samples in each batch.
      n_all_data: Number of samples in all_data.
      rand_key: Random seed determining how the batches are sampled from
        all_data.
      batch_axis: The axis along with the samples are enumerated.
    """
    assert batch_size <= n_all_data, (
        f'Batch size (={batch_size}) cannot be larger than the total number of '
        f'samples in the dataset (i.e. {n_all_data}).'
    )
    self.all_data = all_data
    if batch_axis:
      raise NotImplementedError(
          'NumpyDataIterator can only deal with data where the batch axis is 0.'
      )
    self.batch_axis = batch_axis
    self.batch_size = batch_size
    self.n_all_data = n_all_data
    self.rng = np.random.default_rng(seed=rand_key)

  def __next__(self):
    start_pos = self.rng.integers(0, self.n_all_data - self.batch_size)
    end_pos = start_pos + self.batch_size
    return jax.tree.map(
        lambda arr: jnp.array(arr[start_pos:end_pos]),
        self.all_data,
    )

  def concat_batches(self, batches, batch_axis=0):
    def concat(*arrays):
      return jnp.concatenate(arrays, axis=batch_axis)

    return jax.tree.map(concat, *batches)

  def next(self, num_batches=1):
    """Get a batch with num_batches x batch_size samples.

    Args:
      num_batches: Number of batches to merge together.

    Returns:
      A batch of data of size num_batches x batch_size.
    """
    return self.concat_batches([self.__next__() for _ in range(num_batches)])


def normalize_img(
    image, label
):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255.0, label


class TFDataIterator(base.DataIterator):
  """An iterator class generating batches of TFDS data."""

  def __init__(
      self,
      ds_name,
      batch_size,
      process_fn = normalize_img,
      decoders=None,
      split='train',
      shuffle=True,
  ):
    """Data iterator for Tensorflow datasets.

    Args:
      ds_name: The name of the TF dataset that is to be loaded.
      batch_size: The number of samples in each batch.
      process_fn: A function to be applied to all training samples in the
        dataset.
      decoders: TFDS decoders.
      split: The name of the split.
      shuffle: A boolean indicating whether the dataset should be shuffled.
    """
    ds, self.ds_info = tfds.load(
        ds_name,
        split=split,
        as_supervised=True,
        with_info=True,
        decoders=decoders,
    )

    num_examples = self.ds_info.splits[split].num_examples
    ds = ds.cache()
    ds = ds.repeat()
    if shuffle:
      ds = ds.shuffle(num_examples, reshuffle_each_iteration=True)

    auto = tf.data.experimental.AUTOTUNE
    if process_fn is not None:
      ds = ds.map(process_fn, num_parallel_calls=auto)

    ds = ds.batch(batch_size, drop_remainder=True)
    self.ds = ds.prefetch(auto)
    self.ds_iter = self.ds.__iter__()

  def get_next_tf_batch(self):
    """Returns the features and labels of the next batch."""
    try:
      return self.ds_iter.next()
    except StopIteration:
      self.ds_iter = self.ds.__iter__()
      return self.ds_iter.next()

  def __next__(self):
    features, labels = self.get_next_tf_batch()
    return {'feature': jnp.float32(features), 'label': jnp.float32(labels)}
