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

"""Utils for handling camera intrinsics in TensorFlow."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow.compat.v1 as tf

from depth_and_motion_learning import maybe_summary


def invert_intrinsics_matrix(intrinsics_mat):
  """Inverts an intrinsics matrix.

  Inverting matrices in not supported on TPU. The intrinsics matrix has however
  a closed form expression for its inverse, and this function invokes it.

  Args:
    intrinsics_mat: A tensor of shape [.... 3, 3], representing an intrinsics
      matrix `(in the last two dimensions).

  Returns:
    A tensor of the same shape containing the inverse of intrinsics_mat
  """
  with tf.name_scope('invert_intrinsics_matrix'):
    intrinsics_mat = tf.convert_to_tensor(intrinsics_mat)
    intrinsics_mat_cols = tf.unstack(intrinsics_mat, axis=-1)
    if len(intrinsics_mat_cols) != 3:
      raise ValueError('The last dimension of intrinsics_mat should be 3, not '
                       '%d.' % len(intrinsics_mat_cols))

    fx, _, _ = tf.unstack(intrinsics_mat_cols[0], axis=-1)
    _, fy, _ = tf.unstack(intrinsics_mat_cols[1], axis=-1)
    x0, y0, _ = tf.unstack(intrinsics_mat_cols[2], axis=-1)

    zeros = tf.zeros_like(fx)
    ones = tf.ones_like(fx)

    row1 = tf.stack([1.0 / fx, zeros, zeros], axis=-1)
    row2 = tf.stack([zeros, 1.0 / fy, zeros], axis=-1)
    row3 = tf.stack([-x0 / fx, -y0 / fy, ones], axis=-1)

    return tf.stack([row1, row2, row3], axis=-1)


class HashTableIndexer(object):
  """Using a hash table, maps sparse keys to dense integer indices.

  The main method is get_or_create_index(self, key), which allocates (or fetches
  if exists) an integer index, in [0, max_index), to each `key`. The indices are
  created sequentially: The first key would receive the index 0, the second 1,
  etc.
  """

  def __init__(self,
               max_index,
               key_dtype=tf.string,
               empty_key='',
               deleted_key='DELETED',
               name=None):
    """Creates an instance.

    Args:
      max_index: An integer, all keys will be mapped to indices will be in the
        interval [0, max_index). Trying to insert more keys will raise an
        exception.
      key_dtype: Type of the key.
      empty_key: A key that denotes "no key".
      deleted_key: A key that denotes a deleted key.
      name: A string, name scope for the operations.
    """
    self._scope = ('%s/' % name if name else '') + 'HashTableIndexer'
    with tf.name_scope(self._scope):
      self._key_to_index = tf.lookup.experimental.DenseHashTable(
          key_dtype=key_dtype,
          value_dtype=tf.int32,
          default_value=-1,
          empty_key=empty_key,
          deleted_key=deleted_key)
      self._max_index = max_index

  def get_or_create_index(self, key):
    """Creates (or returns existing) index (in [0, max_index)) for `key`."""
    with tf.name_scope(self._scope):
      index = self._key_to_index.lookup(key)

      def insert_and_return_index():
        new_index = tf.cast(self._key_to_index.size(), tf.int32)
        with tf.control_dependencies([new_index]):
          add_key = self._key_to_index.insert(key, new_index)
        with tf.control_dependencies([add_key]):
          return tf.identity(new_index)

      index = tf.cond(tf.less(index, 0), insert_and_return_index, lambda: index)

      assertion = tf.debugging.assert_less(
          index, self._max_index, ['Number of keys exceeds the maximum.'])

      with tf.control_dependencies([assertion]):
        return tf.identity(index)


class VariableDenseHashTable(object):
  """A wrapper of DenseHashTable, but with variable values.

  tf.lookup.experimental.DenseHashTable does not support pushing gradients into
  its values. This class uses DenseHashTable to map keys to indices (starting
  from 0) and these indices are used to select a respective slice of a variable.
  This way a loss can depend on the values of the keys, and these values can
  receive gradients.
  """

  def __init__(self,
               default_value,
               max_values,
               key_dtype=tf.string,
               empty_key='',
               deleted_key='DELETED',
               name=None):
    """Creates an instance.

    Args:
      default_value: A tf.Tensor. Newly-intesred keys will have it as the value.
      max_values: max_values: An integer, the maximum number of entries in the
        table. Trying to insert more will raise an exception.
      key_dtype: Type of the key.
      empty_key: A key that denotes "no key".
      deleted_key: A key that denotes a deleted key.
      name: A string, name scope for the operations.
    """
    self._scope = ('%s/' % name if name else '') + 'VariableDenseHashTable'
    with tf.name_scope(self._scope):
      self._indexer = HashTableIndexer(max_values, key_dtype, empty_key,
                                       deleted_key)
      default_value = tf.convert_to_tensor(default_value)
      tile_multiple = [max_values] + [1] * default_value.shape.rank
      default_value = tf.expand_dims(default_value, 0)
      values_initializer = tf.tile(default_value, tile_multiple)
      self._values = tf.compat.v1.get_variable(
          'values', initializer=values_initializer)

  def lookup_or_insert(self, key):
    """Looks up (or inserts) `key` and returns the value (or the default)."""
    return self.lookup_by_index(self._indexer.get_or_create_index(key))

  def lookup_by_index(self, index):
    with tf.name_scope(self._scope):
      return tf.gather(self._values, index)


def _get_intrinsics_from_coefficients(coefficients, height, width):
  fx_factor, fy_factor, x0_factor, y0_factor = tf.unstack(coefficients, axis=1)
  fx = fx_factor * 0.5 * (height + width)
  fy = fy_factor * 0.5 * (height + width)
  x0 = x0_factor * width
  y0 = y0_factor * height
  return fx, fy, x0, y0


def create_and_fetch_intrinsics_per_video_index(video_index, height, width,
                                                max_video_index=1000,
                                                num_summaries=10):
  """Fetches the intrinsic mcatrix of a batch of video index.

  Args:
    video_index: A batch of scalars (int32-s) representing video indices, must
      be in [0, max_video_index).
    height: Image height in pixels.
    width: Image width in pixels.
    max_video_index: Maximum video_index (video_index < max_video_index).
    num_summaries: Number of video_indices for which intrinsics will be
      displayed on TensorBoard.

  Returns:
    A batch of intrinsics matrices (shape: [B, 3, 3], where B is the length of
    `video_index`
  """
  intrin_initializer = tf.tile([[1.0, 1.0, 0.5, 0.5]], [max_video_index, 1])
  intrin_factors = tf.compat.v1.get_variable(
      'all_intrin', initializer=intrin_initializer)

  batch_factors = tf.gather(intrin_factors, video_index)
  fx, fy, x0, y0 = _get_intrinsics_from_coefficients(
      batch_factors, height, width)
  zero = tf.zeros_like(fx)
  one = tf.ones_like(fx)
  int_mat = [[fx, zero, x0], [zero, fy, y0], [zero, zero, one]]
  int_mat = tf.transpose(int_mat, [2, 0, 1])

  if num_summaries > 0:
    fx, fy, x0, y0 = _get_intrinsics_from_coefficients(
        intrin_factors, height, width)
    for i in range(num_summaries):
      maybe_summary.scalar('intrinsics/0%d/fx' % i, fx[i])
      maybe_summary.scalar('intrinsics/0%d/fy' % i, fy[i])
      maybe_summary.scalar('intrinsics/0%d/x0' % i, x0[i])
      maybe_summary.scalar('intrinsics/0%d/y0' % i, y0[i])

  maybe_summary.histogram('intrinsics/fx', fx)
  maybe_summary.histogram('intrinsics/fy', fy)
  maybe_summary.histogram('intrinsics/x0', x0)
  maybe_summary.histogram('intrinsics/y0', y0)

  return int_mat
