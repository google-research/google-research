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

# Lint as: python3
"""Code for managing an in-memory cache of negative examples."""
import collections
from typing import Dict

import tensorflow.compat.v2 as tf

NegativeCache = collections.namedtuple('NegativeCache', ['data', 'age'])


class CacheManager(object):
  """Manages an in-memory FIFO or LRU cache of tensor data.

  To use with a distributed strategy, create this object in a strategy scope.
  When the instance methods are called with strategy.run, the inputs and outputs
  are PerReplica tensors, and the data is kept in the memory of the replica.
  Thus each replica has its own cache and receive their own updates.

  The age of items is tracked by the number of update steps since the item was
  added. The oldest items are removed when new items are added. If use_lru is
  set to True, then an items age is reset when its data is updated in the cache.

  Example with a distributed strategy:

  with strategy.scope():
    cache_manager = CacheManager(specs, cache_size)
    init_cache = tf.function(cache_manager.init_cache)
    update_cache = tf.function(cache_manager.update_cache)
    cache = strategy.run(init_cache)
    updates = ...
    cache = strategy.run(update_cache, args=(cache, updates))
  """

  def __init__(self,
               specs,
               cache_size,
               use_lru = True):
    self.specs = specs
    self.cache_size = cache_size
    self.use_lru = use_lru

  def init_cache(self):
    """Creates a zero-initialized cache using the specs."""
    data = {
        k: tf.zeros(shape=[self.cache_size] + spec.shape, dtype=spec.dtype)
        for k, spec in self.specs.items()
    }
    age = tf.zeros(shape=[self.cache_size], dtype=tf.int32)
    return NegativeCache(data=data, age=age)

  def _get_new_item_indices(self, age, updates, mask=None):
    any_update = list(updates.values())[0]
    num_updates = tf.shape(any_update)[0]
    _, new_item_indices = tf.math.top_k(age, num_updates)
    if mask is not None:
      mask = tf.cast(mask, dtype=tf.int32)
      unmasked_indices = (tf.cumsum(mask) - 1) * mask
      unmasked_indices = tf.expand_dims(unmasked_indices, axis=1)
      new_item_indices = tf.gather_nd(new_item_indices, unmasked_indices)
    return new_item_indices

  def _update_indices(self, data, updates, indices, mask=None):
    indices = tf.expand_dims(indices, axis=1)
    updated_data = {}
    for k in self.specs.keys():
      if k in updates:
        updated_data[k] = _masked_tensor_scatter_nd_update(
            data[k], indices, updates[k], mask)
      else:
        updated_data[k] = data[k]
    return updated_data

  def _set_age(self, age, indices, value, mask=None):
    indices = tf.expand_dims(indices, axis=1)
    num_updates = tf.shape(indices)[0]
    values = value * tf.ones([num_updates], dtype=tf.int32)
    updated_age = _masked_tensor_scatter_nd_update(age, indices, values, mask)
    return updated_age

  def update_cache(self,
                   cache,
                   *,
                   new_items=None,
                   new_items_mask=None,
                   updated_item_data=None,
                   updated_item_indices=None,
                   updated_item_mask=None):
    """Updates a cache with new items and updated items.

    First the updates specified by `updated_item_data`, `updated_item_indices`
    are done. Then `new_items` are added by removing the items with the highest
    age. The updated items' and new items' ages are set to zero while all other
    items' ages are incremented by one.

    When the oldest items are selected, the age of the updated items is
    considered to be smaller than all other elements. If there is a tie in ages,
    the item with the smallest index is removed first.

    Args:
      cache: The current state of the cache.
      new_items: A dictionary with new items to the cache. It must have all keys
        as defined by the specs.
      new_items_mask: An optional boolean mask such that only items with True
        are added to the cache.
      updated_item_data: A dictionary with the updates to apply to existing
        elements. The keys must be a subset of the keys defined by the specs.
      updated_item_indices: The location of the items to update.
      updated_item_mask: An optional boolean mask that will mask the updated
        items, preventing them from changing the cache.

    Returns:
      The new state of the cache.
    """
    if new_items and set(new_items.keys()) != set(cache.data.keys()):
      raise ValueError(
          'Value `new_items` must have the same keys ({}) as `cache` ({}).'
          .format(set(new_items.keys()), set(cache.data.keys())))

    if updated_item_data and not set(updated_item_data.keys()).issubset(
        set(cache.data.keys())):
      raise ValueError(
          'Value `updated_item_data` keys ({}) must be contained in same keys as `cache` ({}).'
          .format(set(updated_item_data.keys()), set(cache.data.keys())))

    data = cache.data
    age = cache.age
    if updated_item_data is not None or updated_item_indices is not None:
      data = self._update_indices(data, updated_item_data, updated_item_indices,
                                  updated_item_mask)
      if self.use_lru:
        age = self._set_age(
            age, updated_item_indices, value=-1, mask=updated_item_mask)
    if new_items is not None:
      new_item_indices = self._get_new_item_indices(age, new_items,
                                                    new_items_mask)
      data = self._update_indices(data, new_items, new_item_indices,
                                  new_items_mask)
      age = self._set_age(age, new_item_indices, value=-1, mask=new_items_mask)
    age = age + 1
    return NegativeCache(data=data, age=age)


def _get_broadcastable_mask(mask, target):
  mask_shape = tf.concat(
      [tf.shape(mask),
       tf.ones(tf.rank(target) - 1, dtype=tf.int32)], axis=0)
  return tf.cast(tf.reshape(mask, mask_shape), dtype=target.dtype)


def _masked_tensor_scatter_nd_update(tensor, indices, updates, mask=None):
  """Performs tensor_scatter_nd_update with masked updates."""
  if mask is None:
    return tf.tensor_scatter_nd_update(tensor, indices, updates)
  # We have to handle two cases: (1) all updates are masked and (2) there is
  # at least one update not masked. We do not want to unnessesarily recreate the
  # cache and to support TPU we cannot use condtional statements.
  pred = tf.reduce_any(tf.cast(mask, tf.bool))
  pred_indices = tf.cast(pred, indices.dtype)
  pred_updates = tf.cast(pred, updates.dtype)
  indices_mask = tf.expand_dims(tf.cast(mask, dtype=indices.dtype), axis=1)
  updates_mask = _get_broadcastable_mask(mask, updates)
  # If there is at least one unmasked element we will get it here. We will
  # replace all masked indices and updates with this element.
  _, not_masked = tf.math.top_k(indices_mask[:, 0])
  not_masked = not_masked[0]
  indices = indices * indices_mask + indices[not_masked] * (1 - indices_mask)
  updates = updates * updates_mask + updates[not_masked] * (1 - updates_mask)
  # If all elements are masked, then indices will become all zero and we
  # "update" the tensor with the value at the zeroth index, effectively not
  # changing the tensor.
  indices = pred_indices * indices
  updates = pred_updates * updates + (1 - pred_updates) * tensor[0]
  return tf.tensor_scatter_nd_update(tensor, indices, updates)
