# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Trainer class."""

# pylint: disable=logging-fstring-interpolation

from absl import logging
import tensorflow as tf


class PerFeatureMetric(tf.keras.metrics.Metric):
  """Wraps a Keras metric to calculate results per feature."""

  def __init__(
      self,
      num_features,
      metric_cls,
      name='per_feature_metric',
      base_name='f',
      **kwargs,
  ):
    super().__init__(name=name, **kwargs)
    self.num_features = num_features
    self._metrics = [
        metric_cls(name=f'{name}_{base_name}{i}') for i in range(num_features)
    ]

  def update_state(self, *args, sample_weight=None):
    """Updates inner metrics by slicing args along the feature dimension."""
    for i in range(self.num_features):
      sliced_args = [arg[:, i] for arg in args]
      self._metrics[i].update_state(*sliced_args, sample_weight=sample_weight)

  def result(self):
    return tf.stack([m.result() for m in self._metrics])

  def reset_states(self):
    for m in self._metrics:
      m.reset_states()


class Trainer(tf.Module):
  """Model trainer engine for coordinating epochs and metrics."""

  def __init__(
      self,
      model,
      optimizer,
      lr_reducer,
      loss_fn,
      outdir_path,
      target_var_scales,
      train_metrics_dict,
      val_metrics_dict,
      global_batch_size,
      budget_eq_weight=0.0,
      var_idx_in_budget_eq=None,
      var_scales_in_budget_eq=None,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.model = model
    self.optimizer = optimizer
    self.lr_reducer = lr_reducer
    self.loss_fn = loss_fn  # Expects Reduction.NONE
    self.outdir_path = outdir_path
    self.global_batch_size = global_batch_size
    self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
    self.best_val_metric = tf.Variable(
        float('inf'), dtype=tf.float32, trainable=False
    )
    self.best_val_metric_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
    self.model_save_path = f'{outdir_path}/best_model'
    self.target_var_scales = tf.constant(target_var_scales)

    self.model.lr_reducer = self.lr_reducer
    self.model.optimizer = self.optimizer
    self.lr_reducer.model = model

    self.train_metrics_dict = train_metrics_dict
    self.val_metrics_dict = val_metrics_dict

    self.budget_eq_weight = tf.constant(budget_eq_weight)
    self.var_idx_in_budget_eq = tf.convert_to_tensor(var_idx_in_budget_eq)
    self.var_scales_in_budget_eq = tf.convert_to_tensor(var_scales_in_budget_eq)

  def transform_target(self, labels):
    return labels / (self.target_var_scales + 1e-6)

  def inverse_transform_target(self, predictions):
    return predictions * self.target_var_scales

  def _reset_metrics(self, metrics_dict):
    for metric in metrics_dict.values():
      metric.reset_states()

  def _get_metrics_result(self, metrics_dict):
    return {k: v.result().numpy() for k, v in metrics_dict.items()}

  def _concat_batch_if_needed(self, *args):
    if isinstance(args[0], tuple):  # Balanced case: args = (b1, b2, ..., b6)
      features = tf.concat([a[0] for a in args], axis=0)
      labels = tf.concat([a[1] for a in args], axis=0)
      land_fractions = tf.concat([a[2] for a in args], axis=0)
      sample_weights = tf.concat([a[3] for a in args], axis=0)
      return features, labels, land_fractions, sample_weights
    else:  # Unbalanced case: args = (f, l, lf, sw)
      return args

  def compute_budget_equation_loss(self, features, predictions):
    b_eq_radiation_vars = tf.gather(
        features, self.var_idx_in_budget_eq, axis=-1
    )
    b_eq_radiation_vars = b_eq_radiation_vars * self.var_scales_in_budget_eq
    b_eq_energy_vars = predictions
    balance_residue = tf.reduce_sum(
        b_eq_radiation_vars, axis=-1
    ) + tf.reduce_sum(b_eq_energy_vars, axis=-1)

    balance_residue = tf.square(balance_residue)
    return balance_residue

  @tf.function
  def _forward_pass_and_loss(
      self, features, labels, land_fractions, sample_weights, training
  ):
    features = tf.reshape(features, [-1, tf.shape(features)[-1]])
    labels = tf.reshape(labels, [-1, tf.shape(labels)[-1]])
    land_fractions = tf.reshape(
        land_fractions, [-1, tf.shape(land_fractions)[-1]]
    )
    sample_weights = tf.reshape(
        sample_weights,
        [-1],  # last dim is 1
    )

    labels_scaled = self.transform_target(labels)

    model_output = self.model(
        (features, land_fractions), training=training
    )  # shape: (Batch, num_targets, num_experts) or (Batch, num_targets)

    is_soft_routing_moe = isinstance(model_output, tuple)
    if is_soft_routing_moe:
      predictions, land_fractions = model_output
    else:
      predictions = model_output

    if len(predictions.shape) == 2:
      predictions = tf.expand_dims(predictions, axis=-1)
      land_fractions = tf.ones_like(land_fractions[:, :1])

    weights_expanded = tf.expand_dims(land_fractions, axis=1)
    predictions_agg = tf.reduce_sum(predictions * weights_expanded, axis=-1)
    per_example_loss = self.loss_fn(labels_scaled, predictions_agg)

    predictions_agg_unscaled = self.inverse_transform_target(predictions_agg)
    per_example_budget_residue_loss = self.compute_budget_equation_loss(
        features, predictions_agg_unscaled
    )
    per_example_total_loss = (
        tf.reduce_sum(per_example_loss, axis=-1)
        + self.budget_eq_weight * per_example_budget_residue_loss
    )

    per_example_total_loss = per_example_total_loss * sample_weights

    metrics_dict = (
        self.train_metrics_dict if training else self.val_metrics_dict
    )
    metrics_dict['loss'].update_state(tf.stop_gradient(per_example_loss))
    metrics_dict['rmse'].update_state(
        tf.stop_gradient(labels), tf.stop_gradient(predictions_agg_unscaled)
    )
    metrics_dict['budget_loss'].update_state(
        tf.stop_gradient(per_example_budget_residue_loss)
    )
    metrics_dict['total_loss'].update_state(
        tf.stop_gradient(per_example_total_loss)
    )
    return per_example_total_loss

  @tf.function
  def train_step(self, *args):
    features, labels, land_fractions, sample_weights = (
        self._concat_batch_if_needed(*args)
    )

    all_vars = self.model.trainable_variables
    with tf.GradientTape() as tape:
      per_example_total_loss = self._forward_pass_and_loss(
          features, labels, land_fractions, sample_weights, training=True
      )
      per_replica_loss = tf.nn.compute_average_loss(
          per_example_total_loss, global_batch_size=self.global_batch_size
      )

    grads = tape.gradient(per_replica_loss, all_vars)
    self.optimizer.apply_gradients(zip(grads, all_vars))

  @tf.function
  def test_step(self, *args):
    features, labels, land_fractions, sample_weights = (
        self._concat_batch_if_needed(*args)
    )
    self._forward_pass_and_loss(
        features, labels, land_fractions, sample_weights, training=False
    )

  def train_and_validate_epoch(self, ds_train, ds_val, strategy):
    """Runs one full epoch of training and validation."""

    self._reset_metrics(self.train_metrics_dict)
    for batch_data in ds_train:
      strategy.run(
          self.train_step,
          args=batch_data,
      )
    results_dict = {
        f'train_{k}': v
        for k, v in self._get_metrics_result(self.train_metrics_dict).items()
    }

    self._reset_metrics(self.val_metrics_dict)
    for batch_data in ds_val:
      strategy.run(
          self.test_step,
          args=batch_data,
      )
    results_dict.update({
        f'val_{k}': v
        for k, v in self._get_metrics_result(self.val_metrics_dict).items()
    })

    self.lr_reducer.on_epoch_end(
        self.epoch.numpy(),
        logs={'val_total_loss': results_dict['val_total_loss']},
    )
    results_dict['lr'] = self.optimizer.learning_rate.numpy()

    curr_val_total_loss = results_dict['val_total_loss']
    if curr_val_total_loss < self.best_val_metric.numpy():
      self.save_model()
      self.best_val_metric.assign(curr_val_total_loss)
      self.best_val_metric_epoch.assign(self.epoch.numpy())

    self.increment_epoch()

    return results_dict

  def save_model(self):
    tf.saved_model.save(self.model, self.model_save_path)
    logging.info(
        f'Model saved to {self.model_save_path} at epoch {self.epoch.numpy()}'
    )

  def increment_epoch(self):
    self.epoch.assign_add(1)

  def model_stop_training(self):
    return self.model.stop_training
