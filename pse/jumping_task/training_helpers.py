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

"""Helpers for training an agent using imitation learning."""

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

EPS = 1e-9
LARGE_NUM = 1e9


def metric_fixed_point(action_cost_matrix, gamma=1.0):
  """Computes the pseudo-metric satisfying F using fixed point iteration."""
  n, m = action_cost_matrix.shape
  d_metric = np.zeros_like(action_cost_matrix)
  def fixed_point_operator(d_metric):
    d_metric_new = np.empty_like(d_metric)
    for i in range(n):
      for j in range(m):
        d_metric_new[i, j] = action_cost_matrix[i, j] + \
            gamma * d_metric[min(i + 1, n - 1), min(j + 1, m - 1)]
    return d_metric_new

  while True:
    d_metric_new = fixed_point_operator(d_metric)
    if np.sum(np.abs(d_metric - d_metric_new)) < EPS:
      break
    else:
      d_metric = d_metric_new
  return d_metric


@tf.function
def tf_metric_fixed_point(action_cost_matrix, gamma):
  return tf.numpy_function(
      metric_fixed_point, [action_cost_matrix, gamma], Tout=tf.float32)


def calculate_action_cost_matrix(actions_1, actions_2):
  action_equality = tf.math.equal(
      tf.expand_dims(actions_1, axis=1), tf.expand_dims(actions_2, axis=0))
  return 1.0 - tf.cast(action_equality, dtype=tf.float32)


def calculate_reward_cost_matrix(rewards_1, rewards_2):
  diff = tf.expand_dims(rewards_1, axis=1) - tf.expand_dims(rewards_2, axis=0)
  return tf.cast(tf.abs(diff), dtype=tf.float32)


def ground_truth_coupling(actions_1, actions_2):
  """Calculates ground truth coupling using optimal actions on two envs."""
  diff = actions_2.index(1) - actions_1.index(1)
  assert diff >= 0, 'Please pass the actions_2 as actions_1 and vice versa!'
  n, m = len(actions_1), len(actions_2)
  cost_matrix = np.ones((n, m), dtype=np.float32)
  for i in range(n):
    j = i + diff
    if j < m:
      cost_matrix[i, j] = 0.0
    else:
      break
  return cost_matrix


@tf.function
def cosine_similarity(x, y):
  """Computes cosine similarity between all pairs of vectors in x and y."""
  x_expanded, y_expanded = x[:, tf.newaxis], y[tf.newaxis, :]
  similarity_matrix = tf.reduce_sum(x_expanded * y_expanded, axis=-1)
  similarity_matrix /= (
      tf.norm(x_expanded, axis=-1) * tf.norm(y_expanded, axis=-1) + EPS)
  return similarity_matrix


@tf.function
def l2_distance(x, y):
  """Computes cosine similarity between all pairs of vectors in x and y."""
  x_expanded, y_expanded = x[:, tf.newaxis], y[tf.newaxis, :]
  return tf.sqrt(tf.reduce_sum((x_expanded - y_expanded)**2, axis=-1))


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
  neg_logits1, neg_logits2 = similarity_matrix, similarity_matrix

  col_indices = tf.cast(tf.argmin(metric_values, axis=1), dtype=tf.int32)
  pos_indices1 = tf.stack(
      (tf.range(metric_shape[0], dtype=tf.int32), col_indices), axis=1)
  pos_logits1 = tf.gather_nd(similarity_matrix, pos_indices1)

  row_indices = tf.cast(tf.argmin(metric_values, axis=0), dtype=tf.int32)
  pos_indices2 = tf.stack(
      (row_indices, tf.range(metric_shape[1], dtype=tf.int32)), axis=1)
  pos_logits2 = tf.gather_nd(similarity_matrix, pos_indices2)

  if use_coupling_weights:
    metric_values /= coupling_temperature
    coupling = tf.exp(-metric_values)
    pos_weights1 = -tf.gather_nd(metric_values, pos_indices1)
    pos_weights2 = -tf.gather_nd(metric_values, pos_indices2)
    pos_logits1 += pos_weights1
    pos_logits2 += pos_weights2
    negative_weights = tf.math.log((1.0 - coupling) + EPS)
    neg_logits1 += tf.tensor_scatter_nd_update(
        negative_weights, pos_indices1, pos_weights1)
    neg_logits2 += tf.tensor_scatter_nd_update(
        negative_weights, pos_indices2, pos_weights2)

  neg_logits1 = tf.math.reduce_logsumexp(neg_logits1, axis=1)
  neg_logits2 = tf.math.reduce_logsumexp(neg_logits2, axis=0)

  loss1 = tf.reduce_mean(neg_logits1 - pos_logits1)
  loss2 = tf.reduce_mean(neg_logits2 - pos_logits2)
  return loss1 + loss2


def representation_alignment_loss(nn_model,
                                  optimal_data_tuple,
                                  use_bisim=False,
                                  gamma=0.99,
                                  use_l2_loss=False,
                                  use_coupling_weights=False,
                                  coupling_temperature=1.0,
                                  temperature=1.0,
                                  ground_truth=False):
  """Representation alignment loss."""
  obs_1, actions_1, rewards_1 = optimal_data_tuple[0]
  obs_2, actions_2, rewards_2 = optimal_data_tuple[1]

  representation_1 = nn_model.representation(obs_1)
  representation_2 = nn_model.representation(obs_2)

  if use_l2_loss:
    similarity_matrix = l2_distance(representation_1, representation_2)
  else:
    similarity_matrix = cosine_similarity(representation_1, representation_2)

  if ground_truth:
    metric_vals = tf.convert_to_tensor(
        ground_truth_coupling(actions_1, actions_2), dtype=tf.float32)
  else:
    if use_bisim:
      cost_matrix = calculate_reward_cost_matrix(rewards_1, rewards_2)
    else:
      cost_matrix = calculate_action_cost_matrix(actions_1, actions_2)
    metric_vals = tf_metric_fixed_point(cost_matrix, gamma)

  if use_l2_loss:
    # Directly match the l2 distance between representations to metric values
    alignment_loss = tf.reduce_mean((similarity_matrix - metric_vals)**2)
  else:
    alignment_loss = contrastive_loss(
        similarity_matrix,
        metric_vals,
        temperature,
        coupling_temperature=coupling_temperature,
        use_coupling_weights=use_coupling_weights)

  return alignment_loss, metric_vals, similarity_matrix


@tf.function
def cross_entropy(logits, targets):
  labels = tf.stack([1 - targets, targets], axis=1)
  loss_vals = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  return tf.reduce_mean(loss_vals)


def cross_entropy_loss(model, inputs, targets, training=False):
  predictions = model(inputs, training=training)
  return cross_entropy(predictions, targets)


@tf.function
def weight_decay(model):
  l2_losses = [tf.nn.l2_loss(x) for x in model.trainable_variables]
  return tf.add_n(l2_losses) / len(l2_losses)


def create_balanced_dataset(x_train, y_train, batch_size):
  """Creates a balanced training dataset by upsampling the rare class."""
  def partition_dataset(x_train, y_train):
    neg_mask = (y_train == 0)
    x_train_neg = x_train[neg_mask]
    y_train_neg = np.zeros(len(x_train_neg), dtype=np.float32)
    x_train_pos = x_train[~neg_mask]
    y_train_pos = np.ones(len(x_train_pos), dtype=np.float32)
    return (x_train_pos, y_train_pos), (x_train_neg, y_train_neg)

  pos, neg = partition_dataset(x_train, y_train)
  pos_dataset = tf.data.Dataset.from_tensor_slices(pos).apply(
      tf.data.experimental.shuffle_and_repeat(buffer_size=len(pos[0])))
  neg_dataset = tf.data.Dataset.from_tensor_slices(neg).apply(
      tf.data.experimental.shuffle_and_repeat(buffer_size=len(neg[0])))
  dataset = tf.data.experimental.sample_from_datasets(
      [pos_dataset, neg_dataset])
  ds_tensors = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
  return ds_tensors


def create_dataset(x_train, y_train, batch_size):
  """Creates a training dataset."""
  dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).apply(
      tf.data.experimental.shuffle_and_repeat(buffer_size=len(x_train[0])))
  ds_tensors = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
  return ds_tensors


def create_iterators(datasets, batch_size):
  """Create tf.Dataset iterators from a list of numpy datasets."""
  tf_datasets = [tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
                 for data in datasets]
  input_iterator = tf.data.Iterator.from_structure(
      tf_datasets[0].output_types, tf_datasets[0].output_shapes)
  init_ops = [input_iterator.make_initializer(data) for data in tf_datasets]
  x_batch = input_iterator.get_next()
  return x_batch, init_ops
