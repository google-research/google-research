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

"""MLP model for instance and bag-level training."""

import abc
from typing import Any, Dict, List, Optional

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from mir_uai24 import enum_utils
from mir_uai24 import losses


tfk = tf.keras
tfkl = tfk.layers


class MLPModel(tfk.Model, metaclass=abc.ABCMeta):
  """MLP model for instance and bag-level training."""

  def __init__(
      self, embedding_dim, num_hidden_layers, num_hidden_units,
      dataset_info):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.num_hidden_layers = num_hidden_layers
    self.num_hidden_units = num_hidden_units
    self.dataset_info = dataset_info
    if self.embedding_dim != 0:
      self.init_embedding_layers()
    self.init_hidden_layers()
    self.output_layer = tfkl.Dense(
        units=1, name='Output', activation=None)

  def init_embedding_layers(self):
    self.embedding_layers = {}
    for feature in self.dataset_info.features:
      logging.info('Initializing embedding layer for feature: %s', feature.key)
      if feature.type == enum_utils.FeatureType.CATEGORICAL:
        self.embedding_layers[feature.key] = tfkl.Embedding(
            input_dim=feature.num_categories,
            output_dim=self.embedding_dim,
            input_length=1,
            name=f'Embedding-{feature.key}',
        )
      else:
        self.embedding_layers[feature.key] = tfkl.Dense(
            units=self.embedding_dim,
            use_bias=False,
            name=f'Embedding-{feature.key}',
        )

  def init_hidden_layers(self):
    self.hidden_layers = []
    for layer_index in range(self.num_hidden_layers):
      self.hidden_layers.append(
          tfkl.Dense(
              units=self.num_hidden_units,
              activation='relu',
              name=f'Hidden-{layer_index+1}',
          )
      )

  def compute_expected_preds(
      self, posterior, preds, bag_ids
  ):
    normalized_posterior = posterior / tf.matmul(
        tf.cast(bag_ids[:, None] == bag_ids[None], dtype=tf.float32), posterior
    )
    _, bag_ids = tf.unique(bag_ids)
    aggregation_matrix = tf.transpose(
        tf.one_hot(bag_ids, tf.reduce_max(bag_ids) + 1)
    )
    expected_preds = tf.matmul(aggregation_matrix, normalized_posterior * preds)
    return expected_preds

  def compute_attribution_metrics(
      self, bag_batch, posterior
  ):
    attribution_true = (
        bag_batch[self.dataset_info.bag_id][:, None]
        == bag_batch[self.dataset_info.instance_id]
    )
    _, bag_ids = tf.unique(bag_batch[self.dataset_info.bag_id])
    normalized_posterior = posterior / tf.matmul(
        tf.cast(bag_ids[:, None] == bag_ids[None], dtype=tf.float32), posterior)
    attribution_xent = tf.reduce_mean(
        -tf.cast(attribution_true, dtype=tf.float32)
        * tf.math.log(normalized_posterior)
    )

    n_bags = tf.reduce_max(bag_ids) + 1
    mask = bag_ids[:, None] == tf.range(n_bags)[None]
    bag_sizes = tf.reduce_sum(tf.cast(mask, dtype=tf.int64), axis=0)
    attribution_pred = tf.argmax(
        tf.where(
            mask, posterior, -tf.ones_like(mask, dtype=tf.float32) * np.inf),
        axis=0
    )
    attribution_pred = attribution_pred - tf.cumsum(bag_sizes) + bag_sizes[0]

    attribution_true = (
        tf.where(attribution_true)[:, 0] - tf.cumsum(bag_sizes) + bag_sizes[0]
    )
    attribution_acc = tf.reduce_mean(
        tf.cast(attribution_pred == attribution_true, dtype=tf.float32))
    return attribution_xent, attribution_acc

  @abc.abstractmethod
  def get_prior(self, bag_batch):
    pass

  @abc.abstractmethod
  def compute_posterior(
      self, prior, preds, bag_batch
  ):
    pass

  def call(
      self,
      inputs,
      training = False,
      mask = None,
  ):
    del mask  # not used
    if self.embedding_dim != 0:
      embedded_inputs = {}
      for feature in self.embedding_layers:
        embedded_inputs[feature] = self.embedding_layers[feature](
            inputs[feature], training=training
        )
        if embedded_inputs[feature].ndim == 3:
          embedded_inputs[feature] = embedded_inputs[feature][:, 0, :]
      embedded_inputs = tf.nest.flatten(embedded_inputs)
      hidden_inputs = tfkl.concatenate(embedded_inputs)
    else:
      hidden_inputs = [
          tf.cast(inputs[feature.key], dtype=tf.float32)
          for feature in self.dataset_info.features
      ]
      hidden_inputs = tfkl.concatenate(hidden_inputs, axis=1)
    for layer_index in range(self.num_hidden_layers):
      hidden_inputs = self.hidden_layers[layer_index](
          hidden_inputs, training=training
      )
    return self.output_layer(hidden_inputs, training=training)

  def compile(
      self,
      optimizer = 'adam',
      loss = None,
      metrics = None,
      loss_weights = None,
      weighted_metrics = None,
      run_eagerly = False,
      steps_per_execution = None,
      jit_compile = None,
      pss_evaluation_shards = 0, **kwargs,):
    assert metrics is None and loss is None
    super().compile(
        optimizer=optimizer, loss_weights=loss_weights,
        run_eagerly=run_eagerly, steps_per_execution=steps_per_execution,
        jit_compile=jit_compile, pss_evaluation_shards=pss_evaluation_shards,
        **kwargs)

    self.loss_metrics = [
        tfk.metrics.Mean(name='loss'),
        tfk.metrics.Mean(name='bag_loss'),
        tfk.metrics.Mean(name='posterior_bce'),
        tfk.metrics.Mean(name='posterior_sum_1'),
        tfk.metrics.Mean(name='overlap_posterior_max_sum_1'),
    ]

    self.instance_regression_metrics = [
        tfk.metrics.MeanSquaredError(name='instance_mse'),
        tfk.metrics.Poisson(name='instance_poisson')
    ]

    self.bag_metrics = [
        tfk.metrics.MeanSquaredError(name='bag_mse'),
        tfk.metrics.Poisson(name='bag_poisson')
    ]

    self.bag_metrics_under_prior = [
        tfk.metrics.MeanSquaredError(name='bag_mse_under_prior'),
        tfk.metrics.Poisson(name='bag_poisson_under_prior')
    ]

    self.attribution_metrics = [
        tfk.metrics.Mean(name='attribution_cross_entropy'),
        tfk.metrics.Mean(name='attribution_accuracy')
    ]

  def update_metrics(
      self,
      labels,
      preds,
      bag_labels = None,
      bag_preds = None,
      bag_batch = None,
      prior = None,
      posterior = None,
      loss_list = None,
  ):

    if loss_list is not None:
      for loss_metric, loss_val in zip(self.loss_metrics, loss_list):
        loss_metric.update_state(loss_val)

    for instance_regression_metric in self.instance_regression_metrics:
      instance_regression_metric.update_state(labels, preds)

    for bag_metric in self.bag_metrics:
      bag_metric.update_state(bag_labels, bag_preds)

    if prior is not None:
      expected_preds_under_prior = self.compute_expected_preds(
          prior, preds, bag_batch[self.dataset_info.bag_id])
      for bag_metric_under_prior in self.bag_metrics_under_prior:
        bag_metric_under_prior.update_state(
            bag_labels, expected_preds_under_prior)

    if posterior is not None:
      attribution_metric_vals = self.compute_attribution_metrics(
          bag_batch, posterior)
      for attribution_metric, attribution_metric_val in zip(
          self.attribution_metrics, attribution_metric_vals):
        attribution_metric.update_state(attribution_metric_val)

  @property
  def metrics(self):
    return (
        self.loss_metrics
        + self.instance_regression_metrics
        + self.bag_metrics
        + self.bag_metrics_under_prior
        + self.attribution_metrics
    )


class InstanceMLPModel(MLPModel):
  """MLP model for instance-level training."""

  def get_prior(self, bag_batch):
    return np.ones(
        shape=(tf.shape(bag_batch[self.dataset_info.bag_id])[0], 1),
        dtype=np.float32,
    )

  def compute_posterior(
      self, prior, preds, bag_batch
  ):

    return prior, None

  def update_metrics(
      self,
      labels,
      preds,
      bag_labels = None,
      bag_preds = None,
      bag_batch = None,
      prior = None,
      posterior = None,
      loss_list = None,
  ):

    for instance_regression_metric in self.instance_regression_metrics:
      instance_regression_metric.update_state(labels, preds)

  @property
  def metrics(self):
    return self.instance_regression_metrics

  def train_step(self, bag_batch):
    with tf.GradientTape() as tape:
      preds = self(bag_batch, training=True)
      loss = losses.mse_loss(bag_batch[self.dataset_info.label], preds)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.update_metrics(
        bag_batch[self.dataset_info.label], preds)
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, bag_batch):
    preds = self(bag_batch, training=False)
    self.update_metrics(
        bag_batch[self.dataset_info.label], preds)
    return {m.name: m.result() for m in self.metrics}


class BagMLPModelWithERM(MLPModel):
  """MLP model for bag-level training with wtd-Assign."""

  class EmpiricalPosterior(tfkl.Layer):
    """Layer for tracking and learning the empirical posterior."""

    def __init__(self, prior):
      super().__init__()
      self.prior = prior

    def build(self, _):
      self.posterior = tf.Variable(
          initial_value=tf.math.log(self.prior/(1-self.prior)),
          trainable=True,
          dtype=tf.float32
      )

    def call(self, bag_id_x_instance_id):
      return tf.sigmoid(self.posterior.gather_nd(bag_id_x_instance_id))

  def __init__(
      self,
      embedding_dim,
      num_hidden_layers,
      num_hidden_units,
      dataset_info,
      prior,
  ):
    super().__init__(
        embedding_dim=embedding_dim,
        num_hidden_layers=num_hidden_layers,
        num_hidden_units=num_hidden_units,
        dataset_info=dataset_info,
    )
    self.prior = prior
    self.posterior = self.EmpiricalPosterior(prior)

  def compile(
      self,
      optimizer = 'adam',
      loss = None,
      metrics = None,
      loss_weights = None,
      weighted_metrics = None,
      run_eagerly = False,
      steps_per_execution = None,
      jit_compile = None,
      pss_evaluation_shards = 0,
      posterior_optimizer = 'adam',
      **kwargs,):

    super().compile(
        optimizer=optimizer, loss=loss, metrics=metrics,
        run_eagerly=run_eagerly, steps_per_execution=steps_per_execution,
        jit_compile=jit_compile, pss_evaluation_shards=pss_evaluation_shards,
        **kwargs)
    self.loss_weights = loss_weights
    self.posterior_optimizer = posterior_optimizer

  def get_prior(self, bag_batch):
    return tf.gather(
        self.prior,
        indices=bag_batch[self.dataset_info.bag_id_x_instance_id][:, 0]
    )

  def compute_posterior(
      self, prior, preds, bag_batch
  ):
    del prior, preds

    posterior = self.posterior(
        bag_batch[self.dataset_info.bag_id_x_instance_id]
    )
    batch_instance_ids = (
        bag_batch[self.dataset_info.instance_id].numpy().flatten().tolist()
    )
    bag_overlaps = list(
        map(
            lambda instance_id: self.dataset_info.memberships.instances[
                instance_id
            ],
            batch_instance_ids,
        )
    )
    overlap_posteriors = []
    n_overlapping_instances = []
    for bag_overlap in bag_overlaps:
      bag_id_x_instance_ids = tf.convert_to_tensor(
          list(
              map(lambda instance: instance.bag_id_x_instance_id, bag_overlap)
          ),
          dtype=tf.int64,
      )
      n_overlapping_instances.append(len(bag_overlap))
      overlap_posteriors.append(self.posterior(
          bag_id_x_instance_ids[:, None])[:, 0])
    overlap_posteriors = tf.concat(overlap_posteriors, axis=0)
    overlap_posteriors = tf.RaggedTensor.from_row_lengths(
        overlap_posteriors, row_lengths=n_overlapping_instances)

    return posterior, overlap_posteriors

  def train_step(self, bag_batch):
    instance_labels = tf.cast(bag_batch[self.dataset_info.label], tf.float32)
    bag_labels = tf.boolean_mask(
        instance_labels,
        (
            bag_batch[self.dataset_info.bag_id][:, None]
            == bag_batch[self.dataset_info.instance_id]
        ),
    )

    with tf.GradientTape() as tape:
      preds = self(bag_batch, training=True)
      prior = self.get_prior(bag_batch)
      posterior, overlap_posteriors = self.compute_posterior(
          prior, preds, bag_batch
      )
      expected_preds = self.compute_expected_preds(
          posterior, preds, bag_batch[self.dataset_info.bag_id]
      )
      bag_mse = losses.mse_loss(bag_labels, expected_preds)
      posterior_bce_loss = losses.bce_loss(posterior, posterior)
      posterior_sum_1 = losses.posterior_sum_1(
          bag_batch[self.dataset_info.bag_id], posterior
      )
      overlap_posterior_max_sum_1 = losses.overlap_posterior_max_sum_1(
          overlap_posteriors)
      loss = (
          self.loss_weights['bag_mse'] * bag_mse
          + self.loss_weights['posterior_bce_loss'] * posterior_bce_loss
          + self.loss_weights['posterior_sum_1'] * posterior_sum_1
          + self.loss_weights['overlap_posterior_max_sum_1']
          * overlap_posterior_max_sum_1
      )

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(
        zip(gradients[:-1], self.trainable_variables[:-1])
    )
    self.posterior_optimizer.apply_gradients(
        [(gradients[-1], self.trainable_variables[-1])]
    )
    self.update_metrics(
        instance_labels,
        preds,
        bag_labels,
        expected_preds,
        bag_batch,
        prior,
        posterior,
        [
            loss,
            bag_mse,
            posterior_bce_loss,
            posterior_sum_1,
            overlap_posterior_max_sum_1,
        ],
    )
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, bag_batch):
    instance_labels = tf.cast(bag_batch[self.dataset_info.label], tf.float32)
    bag_labels = tf.boolean_mask(
        instance_labels,
        (
            bag_batch[self.dataset_info.bag_id][:, None]
            == bag_batch[self.dataset_info.instance_id]
        )
    )
    preds = self(bag_batch, training=True)
    prior = self.get_prior(bag_batch)
    posterior = tf.ones_like(
        bag_batch[self.dataset_info.instance_id], dtype=tf.float32)
    expected_preds = self.compute_expected_preds(
        posterior, preds, bag_batch[self.dataset_info.bag_id])
    self.update_metrics(
        instance_labels, preds, bag_labels, expected_preds,
        bag_batch, prior, posterior)
    return {m.name: m.result() for m in self.metrics}


class BagMLPModelWithBP(MLPModel):
  """MLP model for bag-level training with Balanced Pruning MIR."""

  def __init__(
      self, pred_aggregation, embedding_dim,
      num_hidden_layers, num_hidden_units,
      dataset_info):

    def median(preds, bag_ids):
      preds = tf.reshape(preds, [-1])
      preds = tf.RaggedTensor.from_value_rowids(
          preds, value_rowids=bag_ids).to_tensor()
      return tfp.stats.percentile(preds, 50.0, interpolation='midpoint', axis=1)

    def mean(preds, bag_ids):
      preds = tf.reshape(preds, [-1])
      preds = tf.RaggedTensor.from_value_rowids(
          preds, value_rowids=bag_ids).to_tensor()
      return tf.reduce_mean(preds, axis=1)

    self.pred_aggregation_fn = {
        'median': median, 'mean': mean}[pred_aggregation]
    super().__init__(
        embedding_dim=embedding_dim,
        num_hidden_layers=num_hidden_layers,
        num_hidden_units=num_hidden_units,
        dataset_info=dataset_info,
    )

  def get_prior(self, bag_batch):
    return np.ones(
        shape=(tf.shape(bag_batch[self.dataset_info.bag_id])[0], 1),
        dtype=np.float32
    )

  def compute_posterior(
      self, prior, preds, bag_batch
  ):

    return prior, None

  def train_step(self, bag_batch):
    instance_labels = tf.cast(bag_batch[self.dataset_info.label], tf.float32)
    bag_ids = tf.unique(bag_batch['bag_id'])[1]

    with tf.GradientTape() as tape:
      preds = self(bag_batch, training=True)
      agg_preds = self.pred_aggregation_fn(preds, bag_ids)
      agg_labels = self.pred_aggregation_fn(instance_labels, bag_ids)
      loss = losses.mse_loss(agg_labels, agg_preds)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.update_metrics(
        instance_labels, preds, agg_labels, agg_preds)
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, bag_batch):
    instance_labels = tf.cast(bag_batch[self.dataset_info.label], tf.float32)
    bag_labels = tf.boolean_mask(
        instance_labels,
        bag_batch['bag_id'][:, None] == bag_batch['instance_id']
    )
    bag_ids = tf.unique(bag_batch['bag_id'])[1]
    preds = self(bag_batch, training=False)
    agg_preds = self.pred_aggregation_fn(preds, bag_ids)
    self.update_metrics(
        instance_labels, preds, bag_labels, agg_preds)
    return {m.name: m.result() for m in self.metrics}

  @property
  def metrics(self):
    return (self.loss_metrics
            + self.instance_regression_metrics
            + self.bag_metrics
            )
