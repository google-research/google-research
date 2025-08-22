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

"""Regularizers defined in Unified GSL paper."""

import abc
from typing import Callable, Optional
from ml_collections import config_dict
import tensorflow as tf
import tensorflow_gnn as tfgnn


class BaseRegularizer(abc.ABC):
  """Base class for calculating regularization on model and label GraphTensors.

  Some regularizers only accept model GraphTensor (and ignore label).
  """

  @abc.abstractmethod
  def call(
      self,
      *,
      model_graph,
      label_graph = None,
      edge_set_name = tfgnn.EDGES,
      weights_feature_name = 'weights'
  ):
    pass

  def __call__(
      self,
      *,
      model_graph,
      label_graph = None,
      edge_set_name = tfgnn.EDGES,
      weights_feature_name = 'weights'
  ):
    return self.call(
        model_graph=model_graph,
        label_graph=label_graph,
        edge_set_name=edge_set_name,
        weights_feature_name=weights_feature_name,
    )


class ClosenessRegularizer(BaseRegularizer):
  """Call Returns ||A_model - A_label||_F^2."""

  def call(
      self,
      *,
      model_graph,
      label_graph = None,
      edge_set_name = tfgnn.EDGES,
      weights_feature_name = 'weights'
  ):
    assert label_graph is not None
    # If A and B where vectors (e.g., rasterized adjacency matrices):
    # ||A - B||_F^2 = ||A - B||^2_2 == (A-B)^T (A-B) = A^T A + B^T B - 2 A^T B
    # The first two terms of the RHS are easy to compute: sum-of-squares.
    # The last entry, however, require us to know the *common* edges in the two
    # graph tensors. For this, we sort the edges of one and use tf.searchsorted.
    # "EX:" stands for "Running Example".
    # EX: == [w6, w3, w24]
    model_weight = model_graph.edge_sets[edge_set_name][weights_feature_name]
    if weights_feature_name in label_graph.edge_sets[edge_set_name].features:
      label_weight = label_graph.edge_sets[edge_set_name][weights_feature_name]
    else:
      label_weight = tf.ones(
          label_graph.edge_sets[edge_set_name].sizes, dtype=tf.float32
      )

    assert (
        model_graph.edge_sets[edge_set_name].adjacency.source_name
        == label_graph.edge_sets[edge_set_name].adjacency.source_name
    )
    assert (
        model_graph.edge_sets[edge_set_name].adjacency.target_name
        == label_graph.edge_sets[edge_set_name].adjacency.target_name
    )

    tgt_name = model_graph.edge_sets[edge_set_name].adjacency.target_name
    src_name = model_graph.edge_sets[edge_set_name].adjacency.source_name

    # EX: == 5  (i.e., 5 nodes in each graph).
    size_target = tf.reduce_sum(model_graph.node_sets[tgt_name].sizes)
    # TODO(baharef): add an assert checking if the two graphs have the same
    # number of nodes.
    if tgt_name == src_name:
      size_source = size_target
    else:
      size_source = tf.reduce_sum(model_graph.node_sets[src_name].sizes)
      tf.assert_equal(
          size_source,
          tf.reduce_sum(label_graph.node_sets[src_name].sizes),
          'model_graph and label_graph have different number of source nodes.',
      )

    label_adj = label_graph.edge_sets[edge_set_name].adjacency

    # tf can sort vectors. We combine pairs of ints (source & target vectors) to
    # int vector by finding a suitable "base", multiplying the source by the
    # "base" and adding target.
    combined_label_indices = (  # EX:=[4, 0, 2, 0]*5+[4, 0, 1, 3]=[24, 0, 11, 3]
        # EX: source=[4, 0, 2, 0]       target=[4, 0, 1, 3]
        tf.cast(label_adj.source, tf.int64) * tf.cast(size_target, tf.int64)
        + tf.cast(label_adj.target, tf.int64)
    )
    model_adj = model_graph.edge_sets[edge_set_name].adjacency
    combined_model_indices = (  # EX: = [1, 0, 4]*5 + [1, 3, 4] = [6, 3, 24]
        # EX: source=[0, 1, 4]       target=[3, 1, 4].
        tf.cast(model_adj.source, tf.int64) * tf.cast(size_target, tf.int64)
        + tf.cast(model_adj.target, tf.int64)
    )

    # Add phantom node (to prevent gather on empty array). Excluded from "EX:".
    combined_label_indices = tf.concat(
        [
            combined_label_indices,
            tf.cast(tf.expand_dims(size_source * size_target, 0), tf.int64),
        ],
        axis=0,
    )
    label_weight = tf.concat(
        [label_weight, tf.zeros(1, dtype=label_weight.dtype)], 0
    )

    # EX: [1, 3, 2, 0]
    argsort = tf.argsort(combined_label_indices)
    # EX: [0, 3, 11, 24]
    sorted_combined_label_indices = tf.gather(combined_label_indices, argsort)
    # EX: [2, 1, 3]
    positions = tf.searchsorted(
        sorted_combined_label_indices, combined_model_indices
    )

    # Boolean array. Entry is set to True if edge in model `GraphTensor` is also
    # present in label `GraphTensor`.
    correct_positions = (  # EX: [False, True, True]
        # EX: [11, 3, 24]
        tf.gather(sorted_combined_label_indices, positions)
        # EX: [6, 3, 24]
        == combined_model_indices
    )

    # Order label weights, in an order matching edge order of model.
    label_weight_reordered = tf.gather(  # EX: [W11, W3, W24]
        tf.gather(  # EX: = [W0, W3, W11, W24]
            # EX: = [W24, W0, W11, W3]
            label_weight,
            argsort,
        ),
        positions,
    )
    if not model_weight.dtype.is_floating:
      model_weight = tf.cast(model_weight, tf.float32)
    if not label_weight_reordered.dtype.is_floating:
      label_weight_reordered = tf.cast(label_weight_reordered, tf.float32)
    a_times_b = (  # EX: 0*0 + w3*W3 + w24*W24
        # EX: [False, True, True] * [w6, w3, w24] == [0, w3, w24]
        tf.where(correct_positions, model_weight, tf.zeros_like(model_weight))
        * tf.where(  # EX: [False, True, True] * [W11, W3, W24] = [0, W3, W24]
            correct_positions,
            label_weight_reordered,
            tf.zeros_like(label_weight_reordered),
        )
    )

    regularizer = (
        tf.reduce_sum(model_weight**2)
        + tf.reduce_sum(label_weight**2)
        - 2 * tf.reduce_sum(a_times_b)
    )
    return regularizer


def euclidean_distance_squared(v1, v2):
  displacement = v1 - v2
  return tf.reduce_sum(displacement**2, axis=-1)


class SmoothnessRegularizer(BaseRegularizer):
  r"""Call Returns \sum_{ij} A_{ij} dist(v_i, v_j)."""

  def __init__(
      self,
      source_feature_name = tfgnn.HIDDEN_STATE,
      distance_fn = euclidean_distance_squared,
      target_feature_name = None,
      differentiable_wrt_features = False,
  ):
    self._distance_fn = distance_fn
    self._source_feature_name = source_feature_name
    self._target_feature_name = target_feature_name or source_feature_name
    self._differentiable_wrt_features = differentiable_wrt_features

  def call(
      self,
      *,
      model_graph,
      label_graph = None,
      edge_set_name = tfgnn.EDGES,
      weights_feature_name = 'weights'
  ):
    del label_graph
    edge_set = model_graph.edge_sets[edge_set_name]
    source_ns = edge_set.adjacency.source_name
    target_ns = edge_set.adjacency.target_name
    source_features = tf.gather(
        model_graph.node_sets[source_ns][self._source_feature_name],
        edge_set.adjacency.source,
    )
    target_features = tf.gather(
        model_graph.node_sets[target_ns][self._target_feature_name],
        edge_set.adjacency.target,
    )
    distance = self._distance_fn(source_features, target_features)
    if not self._differentiable_wrt_features:
      distance = tf.stop_gradient(distance)
    return tf.reduce_sum(edge_set[weights_feature_name] * distance)


class SparseConnectRegularizer(BaseRegularizer):
  """Call Returns ||A||_F^2."""

  def call(
      self,
      *,
      model_graph,
      label_graph = None,
      edge_set_name = tfgnn.EDGES,
      weights_feature_name = 'weights'
  ):
    del label_graph
    edge_set = model_graph.edge_sets[edge_set_name]
    return tf.reduce_sum(edge_set[weights_feature_name] ** 2)


class LogBarrier(BaseRegularizer):
  """Call returns -1^T . log (A . 1) == -log(A.sum(1)).sum(0)."""

  def call(
      self,
      *,
      model_graph,
      label_graph = None,
      edge_set_name = tfgnn.EDGES,
      weights_feature_name = 'weights'
  ):
    del label_graph
    weights = model_graph.edge_sets[edge_set_name][weights_feature_name]
    adj = model_graph.edge_sets[edge_set_name].adjacency
    src_name = model_graph.edge_sets[edge_set_name].adjacency.source_name
    num_src_nodes = tf.reduce_sum(model_graph.node_sets[src_name].sizes)
    column_sum = tf.math.unsorted_segment_sum(
        weights, adj.source, num_src_nodes
    )
    column_sum += 1e-5  # avoid infinity values.
    return -tf.reduce_sum(tf.math.log(column_sum))


class InformationRegularizer(BaseRegularizer):
  """Call returns A[i][j] * log (A[i][j]/r) + (1 - A[i][j]) * log ((1 - A[i][j])/(1 - r))."""

  def __init__(self, r, do_sigmoid):
    self._r = r
    self._do_sigmoid = do_sigmoid

  def call(
      self,
      *,
      model_graph,
      label_graph = None,
      edge_set_name = tfgnn.EDGES,
      weights_feature_name = 'weights',
  ):
    del label_graph
    weights = model_graph.edge_sets[edge_set_name][weights_feature_name]
    # If the weights are coming from a soft Bernoulli, a sigmoid has already
    # been applied on the weights.
    if self._do_sigmoid:
      weights = tf.sigmoid(weights)
    # Checking numerical stability
    close_to_0 = weights < 0.0000001
    close_to_1 = weights > 0.9999999
    pos_term = weights * tf.math.log(weights / self._r)
    neg_term = (1 - weights) * tf.math.log((1 - weights) / (1 - self._r))

    return tf.reduce_sum(
        tf.where(
            close_to_0,
            neg_term,
            tf.where(
                close_to_1,
                pos_term,
                pos_term + neg_term,
            ),
        )
    )


def add_loss_regularizers(
    model,
    model_graph,
    label_graph,
    cfg,
):
  """Adding corresponding regularizers to the model.

  Args:
    model: the keras model to add the regularizer for.
    model_graph: the graph generated at thi stage.
    label_graph: the input graph provided in the data.
    cfg: the regularizer config values.

  Returns:
    A keras model with the regularizers added in the loss.
  """
  if cfg.smoothness_enable:
    smoothness_regularizer = SmoothnessRegularizer()
    model.add_loss(
        cfg.smoothness_w
        * smoothness_regularizer(
            model_graph=model_graph,
            label_graph=None,
        )
    )
  if cfg.sparseconnect_enable:
    sparseconnect_regularizer = SparseConnectRegularizer()
    model.add_loss(
        cfg.sparseconnect_w
        * sparseconnect_regularizer(
            model_graph=model_graph,
            label_graph=None,
        )
    )
  if cfg.closeness_enable:
    closeness_regularizer = ClosenessRegularizer()
    model.add_loss(
        cfg.closeness_w
        * closeness_regularizer(
            model_graph=model_graph,
            label_graph=label_graph,
        )
    )
  if cfg.logbarrier_enable:
    log_barrier_regularizer = LogBarrier()
    model.add_loss(
        cfg.logbarrier_w
        * log_barrier_regularizer(
            model_graph=model_graph,
            label_graph=label_graph,
        )
    )
  if cfg.information_enable:
    information_regularizer = InformationRegularizer(
        cfg.information_r, cfg.information_do_sigmoid
    )
    model.add_loss(
        cfg.information_w
        * information_regularizer(
            model_graph=model_graph,
            label_graph=label_graph,
        )
    )
  return model
