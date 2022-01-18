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

# python3
"""Defines attention mechanisms."""

import abc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cache_replacement.policy_learning.cache_model import utils


class Attention(nn.Module):
  """Attends over memory cells to compute a context vector.

  Given:
      - memory keys k_1, ..., k_n (n = num_cells)
      - memory values v_1, ..., v_n
      - a query q

  Computes:
    attention weights a_i = softmax(score(q, k)_i)
    context = sum_i a_i v_i
  """
  __metaclass__ = abc.ABCMeta

  def forward(self, memory_keys, memory_values, queries, masks=None):
    """Computes attention weights and the context vector.

    Args:
      memory_keys (torch.FloatTensor): batch of keys of shape
        (batch_size, num_cells, key_dim)
      memory_values (torch.FloatTensor): batch of values of shape
        (batch_size, num_cells, value_dim)
      queries (torch.FloatTensor): batch of queries of shape
        (batch_size, key_dim)
      masks (torch.ByteTensor | None): batch of masks of shape
        (batch_size, num_cells). Masks out cells where the value is 0. Defaults
        to no masking.

    Returns:
      attention_weights (torch.FloatTensor): shape (batch_size, num_cells)
      context (torch.FloatTensor): shape (batch_size, value_dim)
    """
    if masks is None:
      masks = torch.ones(memory_keys.shape[0], memory_keys.shape[1])
    masks = masks.unsqueeze(1)

    # (batch_size, 1, num_cells)
    attention_weights = F.softmax(self._score(queries, memory_keys), -1)
    masked_attention_weights = utils.mask_renormalize(attention_weights, masks)

    # (batch_size, 1, value_dim)
    context = torch.bmm(masked_attention_weights, memory_values)
    return masked_attention_weights.squeeze(1), context.squeeze(1)

  @abc.abstractmethod
  def _score(self, queries, memory_keys):
    """Computes the score function between queries and memory keys.

    Args:
      queries (torch.FloatTensor): see forward.
      memory_keys (torch.FloatTensor): see forward.

    Returns:
      scores (torch.FloatTensor): score(q, k) of shape
        (batch_size, 1, num_cells)
    """
    raise NotImplementedError()


class ScaledDotProductAttention(Attention):
  """Score(q, k) = <q, k> / sqrt(dim(q)) (Vaswani et. al., 2017)."""

  def __init__(self, memory_key_dim):
    """Constructs.

    Args:
      memory_key_dim (int): dimensionality of memory keys. Future calls to
        forward should have memory_keys of this dimensionality.
    """
    super().__init__()
    self._scale = 1. / np.sqrt(memory_key_dim)

  def _score(self, queries, memory_keys):
    # (batch_size, 1, key_dim)
    queries = queries.unsqueeze(1)
    # (batch_size, key_dim, num_cells)
    memory_keys = memory_keys.transpose(1, 2)

    # (batch_size, 1, num_cells)
    return torch.bmm(queries, memory_keys) * self._scale


class GeneralAttention(Attention):
  """Score(q, k) = q.T W k. W is learned. (Luong et. al., 2015)."""

  def __init__(self, query_dim, memory_key_dim, weight_initializer=None):
    """Constructs.

    Args:
      query_dim (int): dimensionality of queries.
      memory_key_dim (int): dimensionality of memory keys.
      weight_initializer (Callable): takes tensor as input and initializes its
        weight in place (initializes the W matrix). Default is Xavier uniform.
    """
    super().__init__()
    if weight_initializer is None:
      weight_initializer = nn.init.xavier_uniform_

    w = torch.zeros(query_dim, memory_key_dim)
    weight_initializer(w)
    self._w = nn.Parameter(w)

  def _score(self, queries, memory_keys):
    # (batch_size, 1, query_dim)
    queries = queries.unsqueeze(1)

    # (batch_size, key_dim, num_cells)
    memory_keys = memory_keys.transpose(1, 2)

    # (batch_size, 1, key_dim)
    transformed_queries = torch.matmul(queries, self._w)
    # (batch_size, 1, num_cells)
    scores = torch.bmm(transformed_queries, memory_keys)
    return scores


class MultiQueryAttention(nn.Module):
  """Attention with num_queries queries per batch.

  Vectorized version of:
    queries = ... # (batch_size, num_queries, query_dim)
    values = ... # (batch_size, num_cells, value_dim)
    keys = ... # (batch_size, num_cells, key_dim)
    masks = ... # (batch_size, num_cells)

    attention_weights = []
    contexts = []
    for query in queries:
      attention_weight, context = base_attention(keys, values, query, masks)
      attention_weights.append(attention_weight)
      contexts.append(context)
    # (batch_size, num_queries, num_cells), (batch_size, num_queries, value_dim)
    return torch.stack(attention_weights, 1), torch.stack(contexts, 1)
  """

  def __init__(self, base_attention):
    """Constructs.

    Args:
      base_attention (Attention): attention mechanism to perform per query.
    """
    super().__init__()
    self._base_attention = base_attention

  def forward(self, memory_keys, memory_values, queries, masks=None):
    """Computes attention weights and the context vector for multiple queries.

    Args:
      memory_keys (torch.FloatTensor): batch of keys of shape
        (batch_size, num_cells, key_dim)
      memory_values (torch.FloatTensor): batch of values of shape
        (batch_size, num_cells, value_dim)
      queries (torch.FloatTensor): batch of queries of shape
        (batch_size, num_queries, query_dim) with num_queries queries per batch.
      masks (torch.ByteTensor | None): see Attention.

    Returns:
      attention_weights (torch.FloatTensor): shape
        (batch_size, num_queries, num_cells) where attention_weights[:, i]
        corresponds to queries[:, i]
      contexts (torch.FloatTensor): shape (batch_size, num_queries, value_dim)
        where context[:, i] corresponds to queries[:, i].
    """
    batch_size = queries.shape[0]
    num_queries = queries.shape[1]
    num_cells = memory_keys.shape[1]

    if masks is None:
      masks = torch.ones(batch_size, num_cells)

    memory_keys = memory_keys.repeat(1, num_queries, 1).view(
        batch_size * num_queries, num_cells, -1)
    memory_values = memory_values.repeat(1, num_queries, 1).view(
        batch_size * num_queries, num_cells, -1)
    masks = masks.repeat(1, num_queries).view(
        batch_size * num_queries, num_cells)

    # attention_weights: (batch_size * num_queries, num_cells)
    # contexts: (batch_size * num_queries, value_dim)
    attention_weights, contexts = self._base_attention(
        memory_keys, memory_values, queries.view(-1, queries.shape[-1]), masks)

    # Reshape
    attention_weights = attention_weights.view(batch_size, num_queries, -1)
    contexts = contexts.view(batch_size, num_queries, -1)
    return attention_weights, contexts
