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

"""Implements loss functions for dual encoder training with a cache."""

import abc
import collections
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import tensorflow.compat.v2 as tf

from negative_cache import negative_cache
from negative_cache import retrieval_fns
from negative_cache import util

CacheLossReturn = collections.namedtuple('CacheLossReturn', [
    'training_loss',
    'interpretable_loss',
    'updated_item_data',
    'updated_item_indices',
    'updated_item_mask',
    'staleness',
])


class CacheLoss(object, metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def __call__(
      self, doc_network,
      query_embeddings, pos_doc_embeddings,
      cache):
    pass


_RetrievalReturn = collections.namedtuple('_RetrievalReturn', [
    'retrieved_data', 'scores', 'retrieved_indices',
    'retrieved_cache_embeddings'
])


def _score_documents(query_embeddings,
                     doc_embeddings,
                     score_transform = None,
                     all_pairs = False):
  """Calculates the dot product of query, document embedding pairs."""
  if all_pairs:
    scores = tf.matmul(query_embeddings, doc_embeddings, transpose_b=True)
  else:
    scores = tf.reduce_sum(query_embeddings * doc_embeddings, axis=1)
  if score_transform is not None:
    scores = score_transform(scores)
  return scores


def _batch_concat_with_no_op(tensors):
  """If there is only one tensor to concatenate, this is a no-op."""
  if len(tensors) == 1:
    return tensors[0]
  else:
    return tf.concat(tensors, axis=0)


def _retrieve_from_caches(query_embeddings,
                          cache,
                          retrieval_fn,
                          embedding_key,
                          data_keys,
                          sorted_data_sources,
                          score_transform=None,
                          top_k = None):
  """Retrieve elements from a cache with the given retrieval function."""
  all_embeddings = _batch_concat_with_no_op([
      cache[data_source].data[embedding_key]
      for data_source in sorted_data_sources
  ])
  all_data = {}
  for key in data_keys:
    all_data[key] = _batch_concat_with_no_op(
        [cache[data_source].data[key] for data_source in sorted_data_sources])
  scores = _score_documents(
      query_embeddings,
      all_embeddings,
      score_transform=score_transform,
      all_pairs=True)
  if top_k:
    scores, top_k_indices = util.approximate_top_k_with_indices(scores, top_k)
    top_k_indices = tf.cast(top_k_indices, dtype=tf.int64)
    retrieved_indices = retrieval_fn(scores)
    batch_index = tf.expand_dims(
        tf.range(tf.shape(retrieved_indices)[0], dtype=tf.int64), axis=1)
    retrieved_indices_with_batch_index = tf.concat(
        [batch_index, retrieved_indices], axis=1)
    retrieved_indices = tf.gather_nd(top_k_indices,
                                     retrieved_indices_with_batch_index)
    retrieved_indices = tf.expand_dims(retrieved_indices, axis=1)
  else:
    retrieved_indices = retrieval_fn(scores)
  retrieved_indices = tf.stop_gradient(retrieved_indices)
  retrieved_data = {
      k: tf.gather_nd(v, retrieved_indices) for k, v in all_data.items()
  }
  retrieved_cache_embeddings = tf.gather_nd(all_embeddings, retrieved_indices)
  return _RetrievalReturn(retrieved_data, scores, retrieved_indices,
                          retrieved_cache_embeddings)


def _get_data_sorce_start_position_and_cache_sizes(
    cache, embedding_key,
    sorted_data_sources
):
  """Gets the first index and size per data sources in the concatenated data."""
  curr_position = tf.constant(0, dtype=tf.int64)
  start_positions = {}
  cache_sizes = {}
  for data_source in sorted_data_sources:
    start_positions[data_source] = curr_position
    cache_sizes[data_source] = tf.shape(
        cache[data_source].data[embedding_key], out_type=tf.int64)[0]
    curr_position = curr_position + cache_sizes[data_source]
  return start_positions, cache_sizes


def _get_retrieved_embedding_updates(
    cache, embedding_key,
    sorted_data_sources, retrieved_indices,
    retrieved_embeddings
):
  """Gets the updates for the retrieved data."""
  updated_item_indices = {}
  updated_item_data = {}
  updated_item_mask = {}
  start_positions, cache_sizes = _get_data_sorce_start_position_and_cache_sizes(
      cache, embedding_key, sorted_data_sources)
  for data_source in sorted_data_sources:
    updated_item_indices[
        data_source] = retrieved_indices - start_positions[data_source]
    updated_item_data[data_source] = {embedding_key: retrieved_embeddings}
    updated_item_mask[data_source] = (
        retrieved_indices >= start_positions[data_source]) & (
            retrieved_indices <
            start_positions[data_source] + cache_sizes[data_source])
    updated_item_indices[data_source] = tf.squeeze(
        updated_item_indices[data_source], axis=1)
    updated_item_mask[data_source] = tf.squeeze(
        updated_item_mask[data_source], axis=1)
  return updated_item_data, updated_item_indices, updated_item_mask


def _get_staleness(cache_embeddings,
                   updated_embeddings):
  error = cache_embeddings - updated_embeddings
  mse = tf.reduce_sum(error**2, axis=1)
  normalized_mse = mse / tf.reduce_sum(updated_embeddings**2, axis=1)
  return normalized_mse


_LossCalculationReturn = collections.namedtuple('_LossCalculationReturn', [
    'training_loss', 'interpretable_loss', 'staleness', 'retrieval_return',
    'retrieved_negative_embeddings'
])


class AbstractCacheClassificationLoss(CacheLoss, metaclass=abc.ABCMeta):
  """Abstract method for cache classification losses.

  Inherit from this object and override `_retrieve_from_cache` and
  `_score_documents` to implement a cache classification loss based on the
  specified retrieval and scoring approaches.
  """

  @abc.abstractmethod
  def _retrieve_from_cache(self, query_embeddings, cache):
    pass

  @abc.abstractmethod
  def _score_documents(self, query_embeddings, doc_embeddings):
    pass

  def _calculate_training_loss_and_summaries(
      self,
      doc_network,
      query_embeddings,
      pos_doc_embeddings,
      cache,
      reducer=tf.math.reduce_mean):
    """Calculates the cache classification loss and associated summaries."""
    positive_scores = self._score_documents(query_embeddings,
                                            pos_doc_embeddings)
    retrieval_return = self._retrieve_from_cache(query_embeddings, cache)
    retrieved_negative_embeddings = doc_network(retrieval_return.retrieved_data)
    retrieved_negative_scores = self._score_documents(
        query_embeddings, retrieved_negative_embeddings)
    cache_and_pos_scores = tf.concat(
        [tf.expand_dims(positive_scores, axis=1), retrieval_return.scores],
        axis=1)
    prob_pos = tf.nn.softmax(cache_and_pos_scores, axis=1)[:, 0]
    prob_pos = tf.stop_gradient(prob_pos)
    training_loss = (1.0 - prob_pos) * (
        retrieved_negative_scores - positive_scores)
    interpretable_loss = -tf.math.log(prob_pos)
    staleness = _get_staleness(retrieval_return.retrieved_cache_embeddings,
                               retrieved_negative_embeddings)
    if reducer is not None:
      training_loss = reducer(training_loss)
      interpretable_loss = reducer(interpretable_loss)
      staleness = reducer(staleness)
    return _LossCalculationReturn(
        training_loss=training_loss,
        interpretable_loss=interpretable_loss,
        staleness=staleness,
        retrieval_return=retrieval_return,
        retrieved_negative_embeddings=retrieved_negative_embeddings)


class CacheClassificationLoss(AbstractCacheClassificationLoss):
  """Implements an efficient way to train with a cache classification loss.

  The cache classification loss is the negative log probability of the positive
  document when the distribution is the softmax of all documents. This object
  allows calculating:
    (1) An efficient stochastic loss function whose gradient is approximately
        the same as the cache classification loss in expectation. This gradient
        can be calculated by feeding only O(batch_size) documents through the
        document network, rather than O(cache_size) for the standard
        implementation.
    (2) An approximation of the value cache classification loss using the cached
        embeddings. The loss described above is not interpretable. This loss is
        a direct approximation of the cache classification loss, however we
        cannot calculate a gradient of this loss.

  Calling the CacheClassificationLoss return a CacheLossReturn object, which
  has the following fields:
    training_loss: Use this to calculate gradients.
    interpretable_loss: An interpretable number for the CacheClassificationLoss
        to use as a Tensorboard summary.
    updated_item_data, updated_item_indices, updated_item_mask: Use these in
        the negative cache updates. These describe the cache elements that were
        retrieved and current embedding calculated.
    staleness: This is the square error between the retrieved cache embeddings
        and the retrieved embeddings as defined by the current state of the
        model. Create a summary of this value as a proxy for the error due to
        cache staleness.
  """

  def __init__(self,
               embedding_key,
               data_keys,
               score_transform = None,
               top_k = None,
               reducer = tf.math.reduce_mean):
    """Initializes the CacheClassificationLoss object.

    Args:
      embedding_key: The key containing the embedding in the cache.
      data_keys: The keys containing the document data in the cache.
      score_transform: Scores are transformed by this function before use.
        Specifically we have scores(i, j) = score_transform(dot(query_embed_i,
        doc_embed_j))
      top_k: If set, the top k scoring negative elements will be mined and the
        rest of the elements masked before calculating the loss.
      reducer: Function that reduces the losses to a single scaler. If None,
        then the elementwise losses are returned.
    """
    self.embedding_key = embedding_key
    self.data_keys = data_keys
    self.score_transform = score_transform
    self.top_k = top_k
    self.reducer = reducer
    self._retrieval_fn = retrieval_fns.GumbelMaxRetrievalFn()

  def _retrieve_from_cache(
      self, query_embeddings,
      cache):
    sorted_data_sources = sorted(cache.keys())
    return _retrieve_from_caches(query_embeddings, cache, self._retrieval_fn,
                                 self.embedding_key, self.data_keys,
                                 sorted_data_sources, self.score_transform,
                                 self.top_k)

  def _score_documents(self, query_embeddings,
                       doc_embeddings):
    return _score_documents(
        query_embeddings, doc_embeddings, score_transform=self.score_transform)

  def __call__(
      self, doc_network,
      query_embeddings, pos_doc_embeddings,
      cache):
    """Calculates the cache classification losses.

    Args:
      doc_network: The network that embeds the document data.
      query_embeddings: Embeddings for the queries.
      pos_doc_embeddings: Embeddings for the documents that are positive for the
        given queries.
      cache: The cache of document data and embeddings.

    Returns:
      A CacheLossReturn object with the training loss, interpretable loss, and
      data needed to update the cache element embeddings that were retrieved and
      recalculated.
    """
    loss_calculation_return = self._calculate_training_loss_and_summaries(
        doc_network, query_embeddings, pos_doc_embeddings, cache, self.reducer)
    training_loss = loss_calculation_return.training_loss
    interpretable_loss = loss_calculation_return.interpretable_loss
    staleness = loss_calculation_return.staleness
    retrieval_return = loss_calculation_return.retrieval_return
    retrieved_negative_embeddings = loss_calculation_return.retrieved_negative_embeddings
    sorted_data_sources = sorted(cache.keys())
    updated_item_data, updated_item_indices, updated_item_mask = _get_retrieved_embedding_updates(
        cache, self.embedding_key, sorted_data_sources,
        retrieval_return.retrieved_indices, retrieved_negative_embeddings)
    return CacheLossReturn(
        training_loss=training_loss,
        interpretable_loss=interpretable_loss,
        updated_item_data=updated_item_data,
        updated_item_indices=updated_item_indices,
        updated_item_mask=updated_item_mask,
        staleness=staleness)


def _get_local_elements_global_data(all_elements_local_data, num_replicas):
  all_elements_local_data = tf.expand_dims(all_elements_local_data, axis=1)
  return tf.raw_ops.AllToAll(
      input=all_elements_local_data,
      group_assignment=[list(range(num_replicas))],
      concat_dimension=1,
      split_dimension=0,
      split_count=num_replicas)


class DistributedCacheClassificationLoss(AbstractCacheClassificationLoss):
  """Implements a cache classification loss with a sharded cache.

  This object implements a cache classification loss when the cache is sharded
  onto multiple replicas. This code calculates the loss treating the sharded
  cache as one unit, so all queries are affected by all cache elements in every
  replica.

  Currently, the updated_item_* fields (i.e., the embedding updates for items
  already in the cache) in the CacheLossReturn are empty. This does not affect
  new items introduced to the cache.
  """

  def __init__(self,
               embedding_key,
               data_keys,
               score_transform = None,
               top_k = None,
               reducer = tf.math.reduce_mean):
    self.embedding_key = embedding_key
    self.data_keys = data_keys
    self.score_transform = score_transform
    self.top_k = top_k
    self.reducer = reducer
    self._retrieval_fn = retrieval_fns.GumbelMaxRetrievalFn()

  def _score_documents(self, query_embeddings,
                       doc_embeddings):
    return _score_documents(
        query_embeddings, doc_embeddings, score_transform=self.score_transform)

  def _retrieve_from_cache(
      self, query_embeddings,
      cache):
    sorted_data_sources = sorted(cache.keys())
    all_query_embeddings = util.cross_replica_concat(query_embeddings, axis=0)
    num_replicas = tf.distribute.get_replica_context().num_replicas_in_sync
    # Performs approximate top k across replicas.
    if self.top_k:
      top_k_per_replica = self.top_k // num_replicas
    else:
      top_k_per_replica = self.top_k
    retrieval_return = _retrieve_from_caches(all_query_embeddings, cache,
                                             self._retrieval_fn,
                                             self.embedding_key, self.data_keys,
                                             sorted_data_sources,
                                             self.score_transform,
                                             top_k_per_replica)
    # We transfer all queries to all replica and retrieve from every shard.
    all_queries_local_weight = tf.math.reduce_logsumexp(
        retrieval_return.scores, axis=1)
    local_queries_global_weights = _get_local_elements_global_data(
        all_queries_local_weight, num_replicas)
    local_queries_all_retrieved_data = {}
    for key in retrieval_return.retrieved_data:
      local_queries_all_retrieved_data[key] = _get_local_elements_global_data(
          retrieval_return.retrieved_data[key], num_replicas)
    local_queries_all_retrieved_embeddings = _get_local_elements_global_data(
        retrieval_return.retrieved_cache_embeddings, num_replicas)
    # We then sample a shard index proportional to its total weight.
    # This allows us to do Gumbel-Max sampling without modifying APIs.
    selected_replica = self._retrieval_fn(local_queries_global_weights)
    selected_replica = tf.stop_gradient(selected_replica)
    num_elements = tf.shape(selected_replica)[0]
    batch_indices = tf.range(num_elements)
    batch_indices = tf.cast(batch_indices, tf.int64)
    batch_indices = tf.expand_dims(batch_indices, axis=1)
    selected_replica_with_batch = tf.concat([batch_indices, selected_replica],
                                            axis=1)
    retrieved_data = {
        k: tf.gather_nd(v, selected_replica_with_batch)
        for k, v in local_queries_all_retrieved_data.items()
    }
    retrieved_cache_embeddings = tf.gather_nd(
        local_queries_all_retrieved_embeddings, selected_replica_with_batch)
    return _RetrievalReturn(
        retrieved_data=retrieved_data,
        scores=local_queries_global_weights,
        retrieved_indices=None,
        retrieved_cache_embeddings=retrieved_cache_embeddings)

  def __call__(
      self, doc_network,
      query_embeddings, pos_doc_embeddings,
      cache):
    loss_calculation_return = self._calculate_training_loss_and_summaries(
        doc_network, query_embeddings, pos_doc_embeddings, cache, self.reducer)
    training_loss = loss_calculation_return.training_loss
    interpretable_loss = loss_calculation_return.interpretable_loss
    staleness = loss_calculation_return.staleness
    return CacheLossReturn(
        training_loss=training_loss,
        interpretable_loss=interpretable_loss,
        updated_item_data={k: None for k in cache},
        updated_item_indices={k: None for k in cache},
        updated_item_mask={k: None for k in cache},
        staleness=staleness)
