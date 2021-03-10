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

"""Utility for training including contrastive helpers."""

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

EPS = 1e-9


@tf.function
def cosine_similarity(x, y):
  """Computes cosine similarity between all pairs of vectors in x and y."""
  x_expanded, y_expanded = x[:, tf.newaxis], y[tf.newaxis, :]
  similarity_matrix = tf.reduce_sum(x_expanded * y_expanded, axis=-1)
  similarity_matrix /= (
      tf.norm(x_expanded, axis=-1) * tf.norm(y_expanded, axis=-1) + EPS)
  return similarity_matrix


@tf.function
def sample_indices(dim_x, size=128, sort=False):
  dim_x = tf.cast(dim_x, tf.int32)
  indices = tf.range(0, dim_x, dtype=tf.int32)
  indices = tf.random.shuffle(indices)[:size]
  if sort:
    indices = tf.sort(indices)
  return indices


@tf.function
def representation_alignment_loss(nn_model,
                                  optimal_data_tuple,
                                  use_coupling_weights=False,
                                  coupling_temperature=0.1,
                                  return_representation=False,
                                  temperature=1.0):
  """PSE loss."""
  obs1, obs2, metric_vals = optimal_data_tuple
  if np.random.randint(2) == 1:
    obs2, obs1 = obs1, obs2
    metric_vals = tf.transpose(metric_vals)

  indices = sample_indices(tf.shape(metric_vals)[0], sort=return_representation)
  obs1 = tf.gather(obs1, indices, axis=0)
  metric_vals = tf.gather(metric_vals, indices, axis=0)

  representation_1 = nn_model.representation({'pixels': obs1})
  representation_2 = nn_model.representation({'pixels': obs2})
  similarity_matrix = cosine_similarity(representation_1, representation_2)
  alignment_loss = contrastive_loss(
      similarity_matrix,
      metric_vals,
      temperature,
      coupling_temperature=coupling_temperature,
      use_coupling_weights=use_coupling_weights)

  if return_representation:
    return alignment_loss, similarity_matrix
  else:
    return alignment_loss


@tf.function
def contrastive_loss(similarity_matrix,
                     metric_values,
                     temperature,
                     coupling_temperature=1.0,
                     use_coupling_weights=True):
  """Contrative Loss with soft coupling."""
  logging.info('Using alternative contrastive loss.')
  metric_shape = tf.shape(metric_values)
  similarity_matrix /= temperature
  neg_logits1 = similarity_matrix

  col_indices = tf.cast(tf.argmin(metric_values, axis=1), dtype=tf.int32)
  pos_indices1 = tf.stack(
      (tf.range(metric_shape[0], dtype=tf.int32), col_indices), axis=1)
  pos_logits1 = tf.gather_nd(similarity_matrix, pos_indices1)

  if use_coupling_weights:
    metric_values /= coupling_temperature
    coupling = tf.exp(-metric_values)
    pos_weights1 = -tf.gather_nd(metric_values, pos_indices1)
    pos_logits1 += pos_weights1
    negative_weights = tf.math.log((1.0 - coupling) + EPS)
    neg_logits1 += tf.tensor_scatter_nd_update(negative_weights, pos_indices1,
                                               pos_weights1)
  neg_logits1 = tf.math.reduce_logsumexp(neg_logits1, axis=1)
  return tf.reduce_mean(neg_logits1 - pos_logits1)


def _get_action(replay):
  if isinstance(replay, list):
    return np.array([x.action for x in replay])
  else:
    return replay.action


def _calculate_action_cost_matrix(ac1, ac2):
  diff = tf.expand_dims(ac1, axis=1) - tf.expand_dims(ac2, axis=0)
  return tf.cast(tf.reduce_mean(tf.abs(diff), axis=-1), dtype=tf.float32)


def metric_fixed_point_fast(cost_matrix, gamma=0.99, eps=1e-7):
  """Dynamic prograaming for calculating PSM."""
  d = np.zeros_like(cost_matrix)
  def operator(d_cur):
    d_new = 1 * cost_matrix
    discounted_d_cur = gamma * d_cur
    d_new[:-1, :-1] += discounted_d_cur[1:, 1:]
    d_new[:-1, -1] += discounted_d_cur[1:, -1]
    d_new[-1, :-1] += discounted_d_cur[-1, 1:]
    return d_new

  while True:
    d_new = operator(d)
    if np.sum(np.abs(d - d_new)) < eps:
      break
    else:
      d = d_new[:]
  return d


def compute_metric(replay1, replay2, gamma):
  actions1, actions2 = _get_action(replay1), _get_action(replay2)
  action_cost = _calculate_action_cost_matrix(actions1, actions2)
  return tf_metric_fixed_point(action_cost, gamma=gamma)


@tf.function
def tf_metric_fixed_point(action_cost_matrix, gamma):
  return tf.numpy_function(
      metric_fixed_point_fast, [action_cost_matrix, gamma], Tout=tf.float32)
