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

# Lint as: python3
"""Implements the IndexStore class."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import enum

from typing import Tuple, Union
import numpy as np
import six

import tensorflow.compat.v1 as tf


class IndexStoreType(enum.Enum):
  basic = 'basic index store'
  cyclic = 'cyclic index store'
  padding = 'index store constant padding'


def get_index_store(index_store_type):
  """Returns an IndexStore class of appropriate type."""
  index_stores = {
      IndexStoreType.basic: IndexStore,
      IndexStoreType.cyclic: CyclicIndexStore,
      IndexStoreType.padding: PaddingIndexStore
  }

  if index_store_type not in index_stores:
    raise ValueError('Supported index store types are: %s. Got: %s' %
                     (index_stores.keys(), index_store_type))
  return index_stores[index_store_type]


@six.add_metaclass(abc.ABCMeta)
class IndexStoreBase(object):
  """Base class for index stores."""

  @property
  def type(self):
    """Returns the maximum size of the index store."""
    return self._type

  @property
  def size(self):
    """Returns the maximum size of the index store."""
    return self._max_size

  def current(self):
    """Returns the current index allocated for the index request."""
    return self._current

  @abc.abstractmethod
  def allocate_variables_from_pool(self, pool,
                                   size):
    """Allocates variables from the given tensor.

    Args:
      pool: A `tf.Tensor` object that serves as the variable pool.
      size: int, number of elements to allocate from tensor.

    Returns:
      A `tf.Tensor` object of given size.
    """
    pass

  def _validate_pool(self, pool):
    """Validates the given `pool` tensor.

    The given `pool` tensor must be a rank 1 tensor whose size should be equal
    to self._max_size.

    Args:
      pool: An `tf.Tensor` object.
    """
    if pool.shape.rank != 1:
      raise ValueError('The rank of pool tensor must be 1. Got: %d' %
                       pool.shape.rank)
    if pool.shape.num_elements() != self._max_size:
      raise ValueError(
          'The size of pool tensor does not match the size of this index '
          'store. Size of pool tensor: %d, size of index store: %d' %
          (pool.shape.num_elements(), self._max_size))


class IndexStore(IndexStoreBase):
  """Manages indices.

  IndexStore allows to get non-overlapping sequencial indices.
  """

  def __init__(self, max_size = None):
    """Create an IndexStore.

    Args:
      max_size: An integer, the maximal size of the index store.
    """
    self._max_size = max_size
    self._current = 0
    self._type = IndexStoreType.basic

  @property
  def max_size(self):
    """Returns the maxium size of the index store.

    THIS METHOD IS DEPRECATED. Use `self.size` instead.
    """
    return self.size

  def _size_ok(self, size):
    return self._max_size is None or (size > 0 and size <= self._max_size)

  def allocate_indices(self, length):
    """Consumes indices from the store.

    THIS METHOD IS DEPRECATED. Use `self.allocate_variables_from_pool` instead.

    Args:
      length: Number of elements to consume.

    Returns:
      A tuple with the begin, end of the indices to be consumed.
    """
    if isinstance(length, tf.Dimension):
      length = length.value
    current = self._current
    if not self._size_ok(self._current + length):
      raise ValueError(
          'Required length unavailable size = %d < used = %d + requested = %d' %
          (self._max_size, self._current, length))
    self._current = self._current + length
    return (current, current + length)

  def allocate_variables_from_pool(self, pool,
                                   size):
    """Allocates variables from the given pool."""
    self._validate_pool(pool)
    begin, end = self.allocate_indices(size)
    tf.logging.info('Allocating %d : %d slice.', begin, end)
    return tf.identity(pool[begin:end])


class CyclicIndexStore(IndexStoreBase):
  """Index store that allow index reuse.

  For CyclicIndexStore class, when the requested indices exceed the originally
  specified size, it will go back and start to allocate indices from the very
  beginning.
  """

  def __init__(self, max_size):
    """Creates an CyclicIndexStore.

    Args:
      max_size: An integer, the maximal size for the CyclicIndexStore.
    """
    self._max_size = max_size
    self._current = 0
    self._type = IndexStoreType.cyclic

  def _allocate_indices(
      self, length):
    """Allocates incies from the store.

    Args:
      length: Number of indices to allocate.

    Returns:
      A tuple of (prefix_begin, prefix_end, repeats, suffix). `prefix_begin` and
      `prefix_end` indicate the begin and end indices before the allocated index
      exceeds the allowed size (self.size). `repeats` indicate how many times
      the entire index range are reused for the requested index length. `suffix`
      represents the last portion of the allocated indices. When the requested
      indcies result exceeding the allowed size, and it starts to reuse the
      indices from very beginning (index 0). The actual allocated elements from
      the pool should be: pool[prefix_begin:prefix_end], pool[0:self.size] *
      repeats, and pool[0:suffix].
    """
    if isinstance(length, tf.Dimension):
      length = length.value

    prefix_begin = self._current
    self._current = (self._current + length) % self._max_size

    repeats = int(np.floor(length / self._max_size))
    if prefix_begin + length <= self._max_size:
      prefix_end = prefix_begin + length
      suffix = 0
    else:
      prefix_end = self._max_size
      suffix = (prefix_begin + length) % self._max_size
      if suffix >= prefix_begin:
        # When suffix passes the prefix_begin (original starting index),
        # pool[prefix_begin:prefix_end] + pool[0:suffix] has already contributed
        # more than self.size elements. Thus, we should minus this from the
        # repeats.
        repeats = repeats - 1

    return (prefix_begin, prefix_end, repeats, suffix)

  def allocate_variables_from_pool(self, pool,
                                   size):
    """Allocates variables from the given pool."""
    self._validate_pool(pool)
    prefix_begin, prefix_end, repeats, suffix = self._allocate_indices(size)
    tf.logging.info(
        'Allocating %d : %d as prefix slice, %d repeats, and '
        '0 : %d as suffix slice', prefix_begin, prefix_end, repeats, suffix)
    tensor_list = ([pool[prefix_begin:prefix_end]] + [pool] * repeats +
                   [pool[0:suffix]])
    return tf.concat(tensor_list, axis=0)


class PaddingIndexStore(IndexStoreBase):
  """Index store that pads constant when requested index is out of bound."""

  def __init__(self, max_size, padding = 0.01):
    """Creates a PaddingIndexStore instance.

    Args:
      max_size: An integer, the maximal size for the PaddingIndexStore.
      padding: an float, the constant to used to pad when the requested index is
        out of bound (larger than max_size).
    """
    self._max_size = max_size
    self._padding = padding
    self._current = 0
    self._type = IndexStoreType.padding

  def _allocate_indices(
      self, length):
    """Allocates indices from store.

    Args:
      length: number of indices to allocate.

    Returns:
      A tuple of `(begin, end, num_padding)`. `begin` and `end` are the start
      and end indices allocated, `num_padding` represents the number of
      constants to pad when the requested indices exceed the `self.size`.
      If the requested indices are less than `self.size`, `num_padding` will be
      zero.
    """
    if isinstance(length, tf.Dimension):
      length = length.value

    begin = self._current
    curr_pos = self._current + length
    end = curr_pos if curr_pos <= self._max_size else self._max_size
    self._current = end
    num_padding = begin + length - end

    return begin, end, num_padding

  def allocate_variables_from_pool(self, pool,
                                   size):
    """Allocates variables from the given pool."""
    self._validate_pool(pool)
    begin, end, num_padding = self._allocate_indices(size)
    tf.logging.info('Allocating %d : %d slice with %d elements padded.', begin,
                    end, num_padding)

    if num_padding == 0:
      return pool[begin:end]

    tensor_list = [
        pool[begin:end],
        tf.constant(self._padding, shape=(num_padding,), dtype=pool.dtype)
    ]
    return tf.concat(tensor_list, axis=0)
