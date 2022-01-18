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

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Abstract class for Collaborative Filtering models."""
# pytype: skip-file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import tensorflow.compat.v2 as tf

from hyperbolic.learning import regularizers


class CFModel(tf.keras.Model, abc.ABC):
  """Abstract CF embedding model class.

  Module to define basic operations in CF embedding models, including embedding
  initialization, computing embeddings and pairs' scores.
  Attributes:
    sizes: Pair of size 2 containing (n_users, n_items).
    rank: Integer, embeddings dimension.
    bias: String indicating if the bias is learned ('learn'),
      constant ('constant'), or zero (anything else).
    initializer: tf.keras.initializers class indicating which initializer
      to use.
    item_regularizer: tf.keras.regularizers.Regularizer class from regularizers,
      indicating which regularizer to use for item embeddings.
    user_regularizer: tf.keras.regularizers.Regularizer class from regularizers,
    indicating which regularizer to use for user embeddings.
    user: Tensorflow tf.keras.layers.Embedding class, holding user
      embeddings.
    item: Tensorflow tf.keras.layers.Embedding class, holding item
      embeddings.
    bu: Tensorflow tf.keras.layers.Embedding class, holding user biases.
    bi: Tensorflow tf.keras.layers.Embedding class, holding item biases.
    gamma: non trainable tf.Variable representing the margin for
      distance-based losses.
    rhs_dep_lhs: Bool indicating if in the distance comparisons, the
      right hand side of the distance function (item embeddings) depends on
      the left hand side (user embeddings).
  """

  def __init__(self, sizes, args):
    """Initialize CF embedding model.

    Args:
      sizes: pair of size 2 containing (n_users, n_items).
      args: Namespace with config arguments (see config.py for detailed overview
        of arguments supported).
    """
    super(CFModel, self).__init__()
    self.sizes = sizes
    self.rank = args.rank
    self.bias = args.bias
    self.initializer = getattr(tf.keras.initializers, args.initializer)
    self.item_regularizer = getattr(regularizers, args.regularizer)(
        args.item_reg)
    self.user_regularizer = getattr(regularizers, args.regularizer)(
        args.user_reg)
    self.user = tf.keras.layers.Embedding(
        input_dim=sizes[0],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.user_regularizer,
        name='user_embeddings')
    self.item = tf.keras.layers.Embedding(
        input_dim=sizes[1],
        output_dim=self.rank,
        embeddings_initializer=self.initializer,
        embeddings_regularizer=self.item_regularizer,
        name='item_embeddings')
    train_biases = self.bias == 'learn'
    self.bu = tf.keras.layers.Embedding(
        input_dim=sizes[0],
        output_dim=1,
        embeddings_initializer='zeros',
        name='user_biases',
        trainable=train_biases)
    self.bi = tf.keras.layers.Embedding(
        input_dim=sizes[1],
        output_dim=1,
        embeddings_initializer='zeros',
        name='item_biases',
        trainable=train_biases)
    self.gamma = tf.Variable(
        initial_value=args.gamma * tf.keras.backend.ones(1), trainable=False)
    self.rhs_dep_lhs = False

  @abc.abstractmethod
  def get_queries(self, input_tensor):
    """Get query embeddings using user and item for an index tensor.

    Args:
      input_tensor: Tensor of size batch_size x 2 containing users and items'
        indices.

    Returns:
      Tensor of size batch_size x embedding_dimension representing users'
      embeddings.
    """
    pass

  @abc.abstractmethod
  def get_rhs(self, input_tensor):
    """Get right hand side (item) embeddings for an index tensor.

    Args:
      input_tensor: Tensor of size batch_size x 2 containing users and items'
        indices.

    Returns:
      Tensor of size batch_size x embedding_dimension representing item
      entities' embeddings.
    """
    pass

  @abc.abstractmethod
  def get_candidates(self, input_tensor=None):
    """Get all candidate item embeddings in a CF dataset.

    Args:
      input_tensor: Tensor of size batch_size x 2 containing users and items'
        indices, or None

    Returns:
      Tensor of size (batch_size x) n_items x embedding_dimension
      representing embeddings for all items in the CF
      if self.rhs_dep_lhs = False (True).
    """
    pass

  @abc.abstractmethod
  def similarity_score(self, lhs, rhs, eval_mode):
    """Computes a similarity score between user and item embeddings.

    Args:
      lhs: Tensor of size B1 x embedding_dimension containing users'
        embeddings.
      rhs: Tensor of size (B1 x) B2 x embedding_dimension containing items'
        embeddings if self.rhs_dep_lhs = False (True).
      eval_mode: boolean to indicate whether to compute all pairs of scores or
        not. If False, B2 must be equal to 1.

    Returns:
      Tensor representing similarity scores. If eval_mode is False, this tensor
      has size B1 x 1, otherwise it has size B1 x B2.
    """
    pass

  def call(self, input_tensor, eval_mode=False):
    """Forward pass of CF embedding models.

    Args:
      input_tensor: Tensor of size batch_size x 2 containing pairs' indices.
      eval_mode: boolean to indicate whether to compute scores against all
        possible item entities in the CF, or only individual pairs' scores.

    Returns:
      Tensor containing pairs scores. If eval_mode is False, this tensor
      has size batch_size x 1, otherwise it has size batch_size x n_item
      where n_item is the total number of items in the CF.
    """
    lhs = self.get_queries(input_tensor)
    lhs_biases = self.bu(input_tensor[:, 0])
    if eval_mode:
      rhs = self.get_candidates(input_tensor)
      rhs_biases = self.bi.embeddings
    else:
      rhs = self.get_rhs(input_tensor)
      rhs_biases = self.bi(input_tensor[:, 1])
    predictions = self.score(lhs, lhs_biases, rhs, rhs_biases, eval_mode)
    return predictions

  def score(self, lhs, lhs_biases, rhs, rhs_biases, eval_mode):
    """Compute pairs scores using embeddings and biases."""
    score = self.similarity_score(lhs, rhs, eval_mode)
    if self.bias == 'constant':
      return score + self.gamma
    elif self.bias == 'learn':
      if eval_mode:
        return score + tf.reshape(rhs_biases, [1, -1])
      else:
        return score + rhs_biases
    else:
      return score

  def get_scores_targets(self, input_tensor):
    """Computes pairs' scores as well as scores againts all possible entities.

    Args:
      input_tensor: Tensor of size batch_size x 2 containing pairs' indices.

    Returns:
      scores: Numpy array of size batch_size x n_items containing users'
              scores against all possible items in the CF.
      targets: Numpy array of size batch_size x 1 containing pairs' scores.
    """
    cand = self.get_candidates(input_tensor)
    cand_biases = self.bi.embeddings
    lhs = self.get_queries(input_tensor)
    lhs_biases = self.bu(input_tensor[:, 0])
    rhs = self.get_rhs(input_tensor)
    rhs_biases = self.bi(input_tensor[:, 1])
    scores = self.score(lhs, lhs_biases, cand, cand_biases, eval_mode=True)
    targets = self.score(lhs, lhs_biases, rhs, rhs_biases, eval_mode=False)
    return scores.numpy(), targets.numpy()

  def eval(self, examples, filters, batch_size=500):
    """Compute ranking-based evaluation metrics in full setting.

    Args:
      examples: Tensor of size n_examples x 2 containing pairs' indices.
      filters: Dict representing items to skip per user for evaluation in
        the filtered setting.
      batch_size: batch size to use to compute scores.

    Returns:
      Numpy array of shape (n_examples, ) containing the rank of each example.
    """
    total_examples = tf.data.experimental.cardinality(examples).numpy()
    batch_size = min(batch_size, total_examples)
    ranks = np.ones(total_examples)
    for counter, input_tensor in enumerate(examples.batch(batch_size)):
      if batch_size * counter >= total_examples:
        break
      scores, targets = self.get_scores_targets(input_tensor)
      for i, query in enumerate(input_tensor):
        query = query.numpy()
        filter_out = filters[query[0]]
        scores[i, filter_out] = -1e6
      ranks[counter * batch_size:(counter + 1) * batch_size] += np.sum(
          (scores >= targets), axis=1)
    return ranks

  def random_eval(self,
                  examples,
                  filters,
                  batch_size=500,
                  num_rand=100,
                  seed=1234):
    """Compute ranking-based evaluation metrics in both full and random settings.

    Args:
      examples: Tensor of size n_examples x 2 containing pairs' indices.
      filters: Dict representing items to skip per user for evaluation in
        the filtered setting.
      batch_size: batch size to use to compute scores.
      num_rand: number of negative samples to draw.
      seed: seed for random sampling.

    Returns:
    ranks: Numpy array of shape (n_examples, ) containing the rank of each
      example in full setting (ranking against the full item corpus).
    ranks_random: Numpy array of shape (n_examples, ) containing the rank of
      each example in random setting (ranking against randomly selected
      num_rand items).
    """
    total_examples = tf.data.experimental.cardinality(examples).numpy()
    batch_size = min(batch_size, total_examples)
    ranks = np.ones(total_examples)
    ranks_random = np.ones(total_examples)
    for counter, input_tensor in enumerate(examples.batch(batch_size)):
      if batch_size * counter >= total_examples:
        break
      scores, targets = self.get_scores_targets(input_tensor)
      scores_random = np.ones(shape=(scores.shape[0], num_rand))
      for i, query in enumerate(input_tensor):
        query = query.numpy()
        filter_out = filters[query[0]]
        scores[i, filter_out] = -1e6
        comp_filter_out = list(
            set(list(range(scores.shape[1]))) - set(filter_out))
        np.random.seed(seed)
        random_indices = np.random.choice(
            comp_filter_out, num_rand, replace=False)
        scores_random[i, :] = scores[i, random_indices]
      ranks[counter * batch_size:(counter + 1) * batch_size] += np.sum(
          (scores >= targets), axis=1)
      ranks_random[counter * batch_size:(counter + 1) * batch_size] += np.sum(
          (scores_random >= targets), axis=1)

    return ranks, ranks_random
