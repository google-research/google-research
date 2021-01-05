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

"""Utilities to help implement switching non-linear dynamical systems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

layers = tf.keras.layers
FLOAT_TYPE = tf.float32


def build_birnn(rnn_type, rnn_hidden_dim):
  """Helper function for building bidirectional RNN."""
  rnn_type = rnn_type.lower()
  if rnn_type == "gru":
    rnn_unit = layers.GRU(units=rnn_hidden_dim,
                          return_sequences=True)
  elif rnn_type == "lstm":
    rnn_unit = layers.LSTM(units=rnn_hidden_dim,
                           return_sequences=True)
  return layers.Bidirectional(rnn_unit)


def build_dense_network(layer_sizes,
                        layer_activations,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="random_uniform"):
  """Helper function for building a multi-layer network."""
  nets = tf.keras.models.Sequential()
  for lsize, activation in zip(layer_sizes, layer_activations):
    nets.add(layers.Dense(
        lsize,
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer))
  return nets


def build_rnn_cell(rnn_type, rnn_hidden_dim):
  """Helper function for building RNN cells."""
  rnn_type = rnn_type.lower()
  if rnn_type == "gru":
    rnn_cell = layers.GRUCell(units=rnn_hidden_dim)
  elif rnn_type == "lstm":
    rnn_cell = layers.LSTMCell(units=rnn_hidden_dim)
  elif rnn_type == "simplernn":
    rnn_cell = layers.SimpleRNNCell(units=rnn_hidden_dim)
  return rnn_cell


def get_posterior_crossentropy(log_posterior, prior_probs):
  """Calculate cross entropy between prior and posterior distributions.

  Args:
    log_posterior: a `float` `Tensor` of shape [batch_size, num_steps,
      num_states].
    prior_probs: a `float` `Tensor` of shape [num_states].

  Returns:
    cross_entropy: a `float` `Tensor` of shape [batch_size].
  """
  log_posterior = tf.convert_to_tensor(log_posterior, dtype_hint=FLOAT_TYPE)
  prior_probs = tf.convert_to_tensor(prior_probs, dtype_hint=FLOAT_TYPE)
  entropy_mat = tf.einsum("ijk, k->ij", log_posterior, prior_probs)
  # when it is cross entropy, we want to minimize the cross entropy,
  # i.e. we want to maximize the sum(prior_prob * log_posterior)
  return tf.reduce_sum(entropy_mat, axis=1)


def normalize_logprob(logmat, axis=-1, temperature=1.):
  """Normalizing log probability with `reduce_logsumexp`."""
  logmat = tf.convert_to_tensor(logmat, dtype_hint=FLOAT_TYPE)
  logmat = logmat / temperature
  normalizer = tf.math.reduce_logsumexp(logmat, axis=axis, keepdims=True)
  return logmat - normalizer, normalizer


def tensor_for_ta(input_ta, swap_batch_time=True):
  """Creates a `Tensor` for the input `TensorArray`."""
  if swap_batch_time:
    res = input_ta.stack()
    return tf.transpose(
        res,
        np.concatenate([[1, 0], np.arange(2, res.shape.ndims)])
    )
  else:
    return input_ta.stack()


def write_updates_to_tas(tensor_arrays, t, tensor_updates):
  """Write updates to corresponding TensorArrays at time step t."""
  assert len(tensor_arrays) == len(tensor_updates)
  num_updates = len(tensor_updates)
  return [tensor_arrays[i].write(t, tensor_updates[i])
          for i in range(num_updates)]


def learning_rate_warmup(global_step,
                         warmup_end_lr,
                         warmup_start_lr,
                         warmup_steps):
  """Linear learning rate warm-up."""
  p = tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
  diff = warmup_end_lr - warmup_start_lr
  return warmup_start_lr +  diff * p


def learning_rate_schedule(global_step,
                           config):
  """Learning rate schedule with linear warm-up and cosine decay."""
  warmup_schedule = learning_rate_warmup(
      global_step=global_step,
      warmup_end_lr=config.learning_rate,
      warmup_start_lr=config.warmup_start_lr,
      warmup_steps=config.warmup_steps)
  decay_schedule = tf.keras.experimental.CosineDecay(
      initial_learning_rate=config.learning_rate,
      decay_steps=config.decay_steps - config.warmup_steps,
      alpha=config.decay_alpha,
      name=None)(tf.math.maximum(global_step - config.warmup_steps, 0))

  return tf.cond(global_step < config.warmup_steps,
                 lambda: warmup_schedule,
                 lambda: decay_schedule)


def inverse_annealing_learning_rate(global_step,
                                    target_lr,
                                    learning_rate_ramp=1e3,
                                    learning_rate_min=1e-10,
                                    decreasing_learning_rate_ramp=1e4):
  """Inverse annealing learning rate."""
  decreasing_gate = 1.0 * tf.pow(
      tf.constant(0.66, dtype=tf.float32),
      tf.to_float(global_step) / decreasing_learning_rate_ramp)
  increasing_gate = (1 - (1 - learning_rate_min) * tf.pow(
      tf.constant(0.66, dtype=tf.float32),
      tf.to_float(global_step) / learning_rate_ramp))
  lr = target_lr * increasing_gate * decreasing_gate + learning_rate_min
  return lr


def schedule_exponential_decay(global_step, config, min_val=1e-10,
                               dtype=tf.float32):
  """Flat and exponential decay schedule."""
  global_step = tf.cast(global_step, dtype)
  decay_steps = tf.cast(config.decay_steps, dtype)
  kickin_steps = tf.cast(config.kickin_steps, dtype)
  decay_schedule = (
      config.initial_temperature *
      config.decay_rate ** (
          tf.math.maximum(global_step - kickin_steps, 0.)
          / decay_steps))
  temp_schedule = tf.cond(global_step < config.kickin_steps,
                          lambda: config.initial_temperature,
                          lambda: tf.maximum(decay_schedule, min_val))
  return temp_schedule

