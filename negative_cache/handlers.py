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

"""Handlers for calculating cache losses and updating the cache.
"""
from typing import Callable, Dict, Tuple

import tensorflow.compat.v2 as tf

from negative_cache import losses
from negative_cache import negative_cache
from negative_cache import util


class _MultiCacheLossHandler(tf.Module):
  """Handles cache loss calculation and cache updates for a multi-cache."""

  def __init__(self,
               cache_managers,
               cache_loss,
               embedding_key,
               data_keys,
               use_cross_replica = False):
    super(_MultiCacheLossHandler, self).__init__(name='MultiCacheLossHandler')
    self.cache_managers = cache_managers
    self.cache_loss = cache_loss
    self.embedding_key = embedding_key
    self.data_keys = data_keys
    self.use_cross_replica = use_cross_replica
    self.cache = self._initialize_cache()

  def _initialize_cache(self):
    cache = {}
    for key in self.cache_managers:
      initial_cache = self.cache_managers[key].init_cache()
      initial_cache_age = _make_cache_variable(initial_cache.age)
      initial_cache_data = {
          k: _make_cache_variable(initial_cache.data[k])
          for k in initial_cache.data
      }
      cache[key] = negative_cache.NegativeCache(
          data=initial_cache_data, age=initial_cache_age)
    return cache

  def _update_cache(self, new_item_embeddings, new_item_features,
                    cache_loss_return):
    is_single_cache = len(self.cache) == 1
    for cache_key in self.cache:
      new_items = {self.embedding_key: new_item_embeddings[cache_key]}
      for data_key in self.data_keys:
        new_items[data_key] = new_item_features[cache_key][data_key]
      updated_item_data = cache_loss_return.updated_item_data[cache_key]
      updated_item_indices = cache_loss_return.updated_item_indices[cache_key]
      if is_single_cache:
        updated_item_mask = None
      else:
        updated_item_mask = cache_loss_return.updated_item_mask[cache_key]
      if self.use_cross_replica:
        new_items = _cross_replica_concat_dict(new_items)
        updated_item_data = _cross_replica_concat_dict(updated_item_data)
        updated_item_indices = util.cross_replica_concat(
            updated_item_indices, axis=0)
        if updated_item_mask is not None:
          updated_item_mask = util.cross_replica_concat(
              updated_item_mask, axis=0)
      new_cache = self.cache_managers[cache_key].update_cache(
          self.cache[cache_key],
          new_items=new_items,
          updated_item_data=updated_item_data,
          updated_item_indices=updated_item_indices,
          updated_item_mask=updated_item_mask)
      for data_key, variable in self.cache[cache_key].data.items():
        variable.assign(new_cache.data[data_key])
      self.cache[cache_key].age.assign(new_cache.age)

  def update_cache_and_compute_loss(
      self, item_network, query_embeddings,
      pos_item_embeddings, new_item_embeddings,
      new_item_features):
    cache_loss_return = self.cache_loss(
        doc_network=item_network,
        query_embeddings=query_embeddings,
        pos_doc_embeddings=pos_item_embeddings,
        cache=self.cache)
    tf.summary.scalar('cache/interpretable_prediction_loss',
                      cache_loss_return.interpretable_loss)
    tf.summary.scalar('cache/staleness', cache_loss_return.staleness)
    self._update_cache(new_item_embeddings, new_item_features,
                       cache_loss_return)
    return cache_loss_return.training_loss


class CacheLossHandler(tf.Module):
  """Handles cache loss calculation and cache updates for a single cache.

  This class handles calculating the loss based on the given CacheLoss object,
  recording tensorboard summaries with cache statistics, and updating the cache.

  Properties:
    cache_manager: A CacheManager object that manages the cache.
    cache_loss: The CacheLoss object that is used to calculate the loss.
    embedding_key: The key in the cache that contains the item embedding data.
    data_keys: A tuple of keys that contain the data used to calculate the item
        embedding.
    use_cross_replica: If true, then cache updates are shared across all
        replicas.

  """

  def __init__(self,
               cache_manager,
               cache_loss,
               embedding_key,
               data_keys,
               use_cross_replica = False):
    super(CacheLossHandler, self).__init__()
    self.cache_manager = cache_manager
    self.cache_loss = cache_loss
    self.embedding_key = embedding_key
    self.data_keys = data_keys
    self.use_cross_replica = use_cross_replica
    self._cache_key = 'cache'
    self._cache_loss_handler = _MultiCacheLossHandler(
        cache_managers={self._cache_key: self.cache_manager},
        cache_loss=self.cache_loss,
        embedding_key=self.embedding_key,
        data_keys=self.data_keys,
        use_cross_replica=self.use_cross_replica)

  @property
  def cache(self):
    return self._cache_loss_handler.cache[self._cache_key]

  def update_cache_and_compute_loss(
      self, item_network, query_embeddings,
      pos_item_embeddings, features):
    """Calculate the loss and perform all cache updates.

    Args:
      item_network: The network used to embed the items.
      query_embeddings: The embeddings of the queries.
      pos_item_embeddings: The corresponding embeddings of the items that are
        positive for the queries.
      features: The features needed to generate the embeddings of the positive
        items.

    Returns:
      The training loss.
    """
    return self._cache_loss_handler.update_cache_and_compute_loss(
        item_network,
        query_embeddings,
        pos_item_embeddings,
        new_item_embeddings={self._cache_key: pos_item_embeddings},
        new_item_features={self._cache_key: features})


def _make_cache_variable(initial_value):
  return tf.Variable(
      initial_value,
      aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
      synchronization=tf.VariableSynchronization.ON_READ,
      trainable=False)


def _cross_replica_concat_dict(tensor_dict):
  cross_replica_dict = {}
  for key, tensor in tensor_dict.items():
    cross_replica_dict[key] = util.cross_replica_concat(tensor, axis=0)
  return cross_replica_dict
