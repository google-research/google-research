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

# Lint as: python3
"""Random Network Distillation module."""

import numpy as np
import tensorflow as tf


class RunningStats:
  """Computes streaming statistics.

  Attributes:
    mean: running mean
    var: running variance
    count: number of samples that have been seen
  """

  def __init__(self, shape=()):
    self.mean = np.zeros(shape, dtype=np.float64)
    self.var = np.ones(shape, dtype=np.float64)
    self.count = 0

  def update(self, data):
    """Update the stats based on a batch of data."""
    batch_mean = np.mean(data, axis=0)
    batch_var = np.var(data, axis=0)
    batch_size = len(data)
    self.update_with_moments(batch_mean, batch_var, batch_size)

  def update_with_moments(self, batch_mean, batch_var, batch_size):
    """Distributed update of moments."""
    delta = batch_mean - self.mean
    new_count = self.count + batch_size

    if self.count == 0:
      new_mean = batch_mean
      new_var = batch_var
    else:
      new_mean = self.mean + delta * batch_size / new_count
      m_a = self.var * (self.count)
      m_b = batch_var * (batch_size)
      m2 = m_a + m_b + np.square(delta) * self.count * batch_size / (
          self.count + batch_size)
      new_var = m2 / (self.count + batch_size)

    self.mean = new_mean
    self.var = new_var
    self.count = new_count


class StateRND:
  """RND model from state space alone.

  Attributes:
    output_dim: dimension of the output
    predictor: prediction that maps an observation to a vector
    target: a random model that predictor tries to imitate
    opt: optimizer
    running_stats: object that keeps track of the running statistics of output
  """

  def __init__(self, input_dim=10, output_dim=5):
    """Initialize StateRND.

    Args:
      input_dim: dimension of the input
      output_dim: dimension of the output
    """
    self.output_dim = output_dim
    self.predictor = tf.keras.Sequential([
        tf.keras.layers.Dense(
            128, input_shape=(input_dim,), activation='leaky_relu'),
        tf.keras.layers.Dense(128, activation='leaky_relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_dim),
    ])
    self.target = tf.keras.Sequential([
        tf.keras.layers.Dense(
            128, input_shape=(input_dim,), activation='leaky_relu'),
        tf.keras.layers.Dense(128, activation='leaky_relu'),
        tf.keras.layers.Dense(output_dim),
    ])
    self.opt = tf.keras.optimizers.Adam(lr=1e-4)
    self.predictor.build()
    self.target.build()
    self.running_stats = RunningStats(shape=(input_dim,))

  def update_stats(self, states):
    """Update the running statistics of the states."""
    self.running_stats.update(np.array(states))

  def _whiten(self, states):
    """Whiten with running statistics."""
    centered = (states-self.running_stats.mean)/np.sqrt(self.running_stats.var)
    return centered.clip(-5, 5)

  def compute_intrinsic_reward(self, states):
    """Compute the intrinsic reward/novelty of a batch of states."""
    whitened_states = self._whiten(states)
    states = tf.convert_to_tensor(whitened_states)
    intrinsic_reward = self._diff_norm(states)
    return intrinsic_reward

  def train(self, states):
    """Train the predicotr network with a batch of states."""
    whitened_states = self._whiten(states)
    states = tf.convert_to_tensor(whitened_states)
    error = self._train(states)
    return {'prediction_loss': error}

  @tf.function
  def _train(self, states):
    pred_variables = self.predictor.variables
    with tf.GradientTape() as tape:
      tape.watch(pred_variables)
      error = self._diff_norm(states)
    grads = tape.gradient(error, pred_variables)
    self.opt.apply_gradients(zip(grads, pred_variables))
    return error

  @tf.function
  def _diff_norm(self, states):
    diff = self.predictor(states) - self.target(states)
    error = tf.reduce_mean(tf.reduce_sum(diff**2, axis=1))
    return error
