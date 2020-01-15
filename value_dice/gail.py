# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""An implementation of a GAIL discriminator (https://arxiv.org/abs/1606.03476).

In order to make training more stable, this implementation also uses
gradient penalty from WGAN-GP (https://arxiv.org/abs/1704.00028) or spectral
normalization (https://openreview.net/forum?id=B1QRgziT-).
"""
import tensorflow.compat.v2 as tf
import tensorflow_gan.python.losses.losses_impl as tfgan_losses


class SpectralNorm(tf.keras.layers.Wrapper):
  """Spectral Norm wrapper for tf.layers.Dense."""

  def build(self, input_shape):
    assert isinstance(self.layer,
                      tf.keras.layers.Dense), 'The class wraps only Dense layer'
    if not self.layer.built:
      self.layer.build(input_shape)

      self.kernel = self.layer.kernel

      shape = self.kernel.shape

      self.u = tf.random.truncated_normal(
          shape=[1, shape[-1]], dtype=tf.float32)

  def call(self, inputs, training=True):
    u = self.u
    u_wt = tf.matmul(u, self.kernel, transpose_b=True)
    u_wt_norm = tf.nn.l2_normalize(u_wt)
    u_wt_w_norm = tf.nn.l2_normalize(tf.matmul(u_wt_norm, self.kernel))
    sigma = tf.squeeze(
        tf.matmul(
            tf.matmul(u_wt_norm, self.kernel), u_wt_w_norm, transpose_b=True))
    self.layer.kernel = self.kernel / sigma

    if training:
      self.u = u_wt_w_norm
    return self.layer(inputs)


class RatioGANSN(object):
  """An implementation of GAIL discriminator with spectral normalization (https://openreview.net/forum?id=B1QRgziT-)."""

  def __init__(self, state_dim, action_dim, log_interval):
    """Creates an instance of the discriminator.

    Args:
      state_dim: State size.
      action_dim: Action size.
      log_interval: Log losses every N steps.
    """
    dense = tf.keras.layers.Dense
    self.discriminator = tf.keras.Sequential([
        SpectralNorm(
            dense(256, activation=tf.nn.tanh, kernel_initializer='orthogonal')),
        SpectralNorm(
            dense(256, activation=tf.nn.tanh, kernel_initializer='orthogonal')),
        SpectralNorm(dense(1, kernel_initializer='orthogonal'))
    ])

    self.discriminator.build(input_shape=(None, state_dim + action_dim))

    self.log_interval = log_interval

    self.avg_loss = tf.keras.metrics.Mean('gail loss', dtype=tf.float32)

    self.optimizer = tf.keras.optimizers.Adam()

  @tf.function
  def get_occupancy_ratio(self, states, actions):
    """Returns occupancy ratio between two policies.

    Args:
      states: A batch of states.
      actions: A batch of actions.

    Returns:
      A batch of occupancy ratios.
    """
    return tf.exp(self.get_log_occupancy_ratio(states, actions))

  @tf.function
  def get_log_occupancy_ratio(self, states, actions):
    """Returns occupancy ratio between two policies.

    Args:
      states: A batch of states.
      actions: A batch of actions.

    Returns:
      A batch of occupancy ratios.
    """
    inputs = tf.concat([states, actions], -1)
    return self.discriminator(inputs, training=False)

  @tf.function
  def update(self, expert_dataset_iter, replay_buffer_iter):
    """Performs a single training step for critic and actor.

    Args:
      expert_dataset_iter: An tensorflow graph iteratable over expert data.
      replay_buffer_iter: An tensorflow graph iteratable over replay buffer.
    """
    expert_states, expert_actions, _ = next(expert_dataset_iter)
    policy_states, policy_actions, _, _, _ = next(replay_buffer_iter)[0]

    policy_inputs = tf.concat([policy_states, policy_actions], -1)
    expert_inputs = tf.concat([expert_states, expert_actions], -1)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.discriminator.variables)
      inputs = tf.concat([policy_inputs, expert_inputs], 0)
      outputs = self.discriminator(inputs)

      policy_output, expert_output = tf.split(
          outputs, num_or_size_splits=2, axis=0)

      # Using the standard value for label smoothing instead of 0.25.
      classification_loss = tfgan_losses.modified_discriminator_loss(
          expert_output, policy_output, label_smoothing=0.0)

    grads = tape.gradient(classification_loss, self.discriminator.variables)

    self.optimizer.apply_gradients(zip(grads, self.discriminator.variables))

    self.avg_loss(classification_loss)

    if tf.equal(self.optimizer.iterations % self.log_interval, 0):
      tf.summary.scalar(
          'train gail/loss',
          self.avg_loss.result(),
          step=self.optimizer.iterations)
      self.avg_loss.reset_states()


class RatioGANGP(object):
  """An implementation of GAIL discriminator with gradient penalty  (https://arxiv.org/abs/1704.00028)."""

  def __init__(self, state_dim, action_dim, log_interval,
               grad_penalty_coeff=10):
    """Creates an instance of the discriminator.

    Args:
      state_dim: State size.
      action_dim: Action size.
      log_interval: Log losses every N steps.
      grad_penalty_coeff: A cofficient for gradient penalty.
    """
    self.discriminator = tf.keras.Sequential([
        tf.keras.layers.Dense(
            256, input_shape=(state_dim + action_dim,), activation=tf.nn.tanh),
        tf.keras.layers.Dense(256, activation=tf.nn.tanh),
        tf.keras.layers.Dense(1)
    ])

    self.log_interval = log_interval
    self.grad_penalty_coeff = grad_penalty_coeff

    self.avg_classification_loss = tf.keras.metrics.Mean(
        'classification loss', dtype=tf.float32)
    self.avg_gp_loss = tf.keras.metrics.Mean(
        'gradient penalty', dtype=tf.float32)
    self.avg_total_loss = tf.keras.metrics.Mean(
        'total gan loss', dtype=tf.float32)

    self.optimizer = tf.keras.optimizers.Adam()

  @tf.function
  def get_occupancy_ratio(self, states, actions):
    """Returns occupancy ratio between two policies.

    Args:
      states: A batch of states.
      actions: A batch of actions.

    Returns:
      A batch of occupancy ratios.
    """
    inputs = tf.concat([states, actions], -1)
    return tf.exp(self.discriminator(inputs))

  @tf.function
  def get_log_occupancy_ratio(self, states, actions):
    """Returns occupancy ratio between two policies.

    Args:
      states: A batch of states.
      actions: A batch of actions.

    Returns:
      A batch of occupancy ratios.
    """
    inputs = tf.concat([states, actions], -1)
    return self.discriminator(inputs)

  @tf.function
  def update(self, expert_dataset_iter, replay_buffer_iter):
    """Performs a single training step for critic and actor.

    Args:
      expert_dataset_iter: An tensorflow graph iteratable over expert data.
      replay_buffer_iter: An tensorflow graph iteratable over replay buffer.
    """
    expert_states, expert_actions, _ = next(expert_dataset_iter)
    policy_states, policy_actions, _, _, _ = next(replay_buffer_iter)[0]

    policy_inputs = tf.concat([policy_states, policy_actions], -1)
    expert_inputs = tf.concat([expert_states, expert_actions], -1)

    alpha = tf.random.uniform(shape=(policy_inputs.get_shape()[0], 1))
    inter = alpha * policy_inputs + (1 - alpha) * expert_inputs

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.discriminator.variables)
      policy_output = self.discriminator(policy_inputs)
      expert_output = self.discriminator(expert_inputs)

      # Using the standard value for label smoothing instead of 0.25.
      classification_loss = tfgan_losses.modified_discriminator_loss(
          expert_output, policy_output, label_smoothing=0.0)

      with tf.GradientTape(watch_accessed_variables=False) as tape2:
        tape2.watch(inter)
        output = self.discriminator(inter)

      grad = tape2.gradient(output, [inter])[0]
      grad_penalty = tf.reduce_mean(tf.pow(tf.norm(grad, axis=-1) - 1, 2))
      total_loss = classification_loss + self.grad_penalty_coeff * grad_penalty

    grads = tape.gradient(total_loss, self.discriminator.variables)

    self.optimizer.apply_gradients(zip(grads, self.discriminator.variables))

    self.avg_classification_loss(classification_loss)
    self.avg_gp_loss(grad_penalty)
    self.avg_total_loss(total_loss)

    if tf.equal(self.optimizer.iterations % self.log_interval, 0):
      tf.summary.scalar(
          'train gail/classification loss',
          self.avg_classification_loss.result(),
          step=self.optimizer.iterations)
      self.avg_classification_loss.reset_states()

      tf.summary.scalar(
          'train gail/gradient penalty',
          self.avg_gp_loss.result(),
          step=self.optimizer.iterations)
      self.avg_gp_loss.reset_states()

      tf.summary.scalar(
          'train gail/loss',
          self.avg_total_loss.result(),
          step=self.optimizer.iterations)
      self.avg_total_loss.reset_states()
