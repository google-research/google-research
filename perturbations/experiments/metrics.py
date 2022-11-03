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

"""Metrics used in perturbations methods."""

import gin
import tensorflow.compat.v2 as tf

from perturbations.experiments import ops
from perturbations.experiments import shortest_path


class ShortestPathMetrics:
  """Wrap a tf.keras.metrics.Mean to be run smoothly in eager mode."""

  def __init__(self, name=''):
    self.name = name
    self._mean = tf.keras.metrics.Mean(name)

  def reset_states(self):
    self._mean.reset_states()

  def result(self):
    return self._mean.result()

  def cost(self, y, y_true):
    dijkstra = shortest_path.Dijkstra()
    path = tf.cast(dijkstra(y), dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)
    return tf.math.reduce_sum(
        tf.math.reduce_sum(y_true * path, axis=-1), axis=-1)

  def compute(self, y_true, y_pred):
    raise NotImplementedError()

  def update_state(self, y_true, y_pred, sample_weight=None):
    self._mean.update_state(
        self.compute(y_true, y_pred), sample_weight=sample_weight)


@gin.configurable
class ShortestPathBinaryAccuracy(ShortestPathMetrics):
  """Is the shortest path the optimal one ?"""

  def __init__(self):
    super().__init__(name='path_binary_acc')

  def compute(self, y_true, y_pred):
    true_cost = self.cost(y_true, y_true)
    pred_cost = self.cost(y_pred, y_true)
    return tf.math.reduce_mean(
        tf.cast(true_cost == pred_cost, dtype=y_pred.dtype))


@gin.configurable
class ShortestPathAccuracy(ShortestPathMetrics):
  """Is the shortest path close to the optimal one."""

  def __init__(self):
    super().__init__(name='path_acc')

  def compute(self, y_true, y_pred):
    true_cost = self.cost(y_true, y_true)
    pred_cost = self.cost(y_pred, y_true)
    return tf.math.reduce_mean(true_cost / pred_cost)


@gin.configurable
class PerfectRanksAccuracy(tf.keras.metrics.Mean):
  """This metric is 1.0 if all the ranks match, zero otherwise."""

  def __init__(self):
    super().__init__(name='perfect_ranks_acc')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.cast(ops.ranks_fn(y_pred, axis=-1), tf.int32)
    y_true = tf.cast(y_true, tf.int32)
    all_equals = tf.math.reduce_all(y_true == y_pred, axis=-1)
    result = tf.cast(all_equals, dtype=tf.float32)
    super().update_state(
        tf.reduce_mean(result, axis=-1), sample_weight=sample_weight)


@gin.configurable
class PartialRanksAccuracy(tf.keras.metrics.Mean):
  """This metric the proportion of matching ranks."""

  def __init__(self, name='partial_ranks_acc'):
    super().__init__(name=name)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.cast(ops.ranks_fn(y_pred, axis=-1), tf.int32)
    y_true = tf.cast(y_true, tf.int32)
    equals = tf.cast(y_true == y_pred, tf.float32)
    result = tf.math.reduce_mean(equals, axis=-1)
    super().update_state(
        tf.reduce_mean(result, axis=-1), sample_weight=sample_weight)


@gin.configurable
class ProjectedRanksAccuracy(tf.keras.metrics.Mean):
  """This metric is the normalized projection onto the permutahedron."""

  def __init__(self):
    super().__init__(name='projection_ranks_acc')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.cast(ops.ranks_fn(y_pred, axis=-1), tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    n = tf.cast(tf.shape(y_true)[-1], tf.float32)
    max_proj = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0
    result = tf.math.reduce_sum(y_true * y_pred, axis=-1) / max_proj
    super().update_state(
        tf.reduce_mean(result, axis=-1), sample_weight=sample_weight)
