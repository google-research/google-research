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

"""Tabular contrastive learner."""

from typing import Optional, Union
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents import specs


class TabularBCEnergy(tf.keras.Model):
  """Tabular behavioral cloning with contrastive Fourier features."""

  def __init__(self,
               dataset_spec,
               gamma,
               embed_dim = 64,
               fourier_dim = None,
               embed_learning_rate = 0.01,
               learning_rate = 0.01,
               finetune = False):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      gamma: The discount factor to use.
      embed_dim: State embedder dimensioin.
      fourier_dim: Fourier feature dimensioin.
      embed_learning_rate: Representation learning rate.
      learning_rate: Policy learning rate.
      finetune: Whether to finetune fourier feature during policy learning.
    """
    super().__init__()
    self._gamma = gamma
    self._reward_fn = lambda env_step: env_step.reward

    # Get number of states/actions.
    observation_spec = dataset_spec.observation
    action_spec = dataset_spec.action
    self._num_states = observation_spec.maximum + 1
    self._num_actions = action_spec.maximum + 1

    self._data_transition = np.zeros(
        [self._num_states, self._num_states, self._num_actions],
        dtype=np.float32)
    self._state_prior = np.zeros([self._num_states], dtype=np.float32)
    self._embed_dim = min(embed_dim, self._num_states)
    self._fourier_dim = fourier_dim or self._embed_dim

    # Initialize embed parameters.
    self._state_embedder = tf.Variable(
        tf.random.truncated_normal([self._num_states, self._embed_dim]))
    self._state_action_embedder = tf.Variable(
        tf.random.truncated_normal(
            [self._num_states, self._num_actions, self._embed_dim]))

    # Initialize policy parameters.
    if self._fourier_dim:
      self._omega = tf.Variable(
          tf.random.normal([self._fourier_dim, self._embed_dim]),
          trainable=finetune)
      self._shift = tf.Variable(
          tf.random.uniform([self._fourier_dim], minval=0, maxval=2 * 3.14159),
          trainable=finetune)
      self.average_embed = tf.Variable(
          tf.zeros([self._embed_dim]), trainable=False)
      self.average_square = tf.Variable(
          tf.ones([self._embed_dim]), trainable=False)

    self._embed_policy = tf.Variable(
        tf.zeros([self._fourier_dim or self._embed_dim, self._num_actions]))

    self._embed_optimizer = tf.keras.optimizers.Adam(embed_learning_rate)
    self._optimizer = tf.keras.optimizers.Adam(learning_rate)

  def _get_index(self, observation, action):
    return observation * self._num_actions + action

  def _state_fourier(self, observation):
    embed = tf.gather(self._state_embedder, observation)
    if not self._fourier_dim:
      return embed

    average_embed = self.average_embed
    average_square = self.average_square
    stddev_embed = tf.sqrt(tf.maximum(1e-8, average_square - average_embed**2))
    normalized_omegas = self._omega / stddev_embed[None, :]
    projection = tf.matmul(
        embed - tf.stop_gradient(average_embed),
        normalized_omegas,
        transpose_b=True)
    projection /= self._embed_dim**0.5
    embed_linear = tf.math.cos(projection + self._shift)
    self.update_moving_averages(embed)
    return embed_linear

  def _state_action_fourier(self, observation, action):
    embed = tf.gather_nd(self._state_action_embedder,
                         tf.stack([observation, action], axis=-1))
    if self._fourier_dim:
      embed = tf.math.cos(
          tf.matmul(embed, self._omega, transpose_b=True) +
          self._shift[None, :])
    return embed

  def prepare_datasets(self, dataset, expert_dataset):
    episodes, valid_steps = dataset.get_all_episodes()
    for episode_num in range(tf.shape(valid_steps)[0]):
      for step_num in range(tf.shape(valid_steps)[1] - 1):
        this_step = tf.nest.map_structure(lambda t: t[episode_num, step_num],  # pylint: disable=cell-var-from-loop
                                          episodes)
        next_step = tf.nest.map_structure(
            lambda t: t[episode_num, step_num + 1], episodes)  # pylint: disable=cell-var-from-loop
        if this_step.is_last() or not valid_steps[episode_num, step_num]:  # pylint: disable=cell-var-from-loop
          continue
        self._state_prior[next_step.observation] += 1.
        self._data_transition[this_step.observation, next_step.observation,
                              this_step.action] += 1.
    self._data_transition = tf.math.divide_no_nan(
        self._data_transition,
        tf.reduce_sum(self._data_transition, axis=1, keepdims=True))
    self._state_prior = self._state_prior / np.sum(self._state_prior)

  def update_moving_averages(self, embeds):
    tt = 0.0005
    _ = self.average_embed.assign((1 - tt) * self.average_embed +
                                  tt * tf.reduce_mean(embeds, [0])),
    _ = self.average_square.assign((1 - tt) * self.average_square +
                                   tt * tf.reduce_mean(embeds**2, [0]))

  @tf.function
  def train_embed(self, transitions):
    this_step = tf.nest.map_structure(lambda t: t[:, 0, Ellipsis], transitions)
    next_step = tf.nest.map_structure(lambda t: t[:, 1, Ellipsis], transitions)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self._state_embedder)
      tape.watch(self._state_action_embedder)

      this_embed = tf.gather(self._state_embedder, this_step.observation)
      next_embed = tf.gather_nd(
          self._state_action_embedder,
          tf.stack([next_step.observation, this_step.action], axis=-1))
      all_next_embed = tf.gather(
          self._state_action_embedder, this_step.action, axis=1)
      pos_loss = tf.reduce_sum(-0.5 * (this_embed - next_embed)**2, axis=-1)
      neg_loss = tfp.math.reduce_weighted_logsumexp(
          tf.reduce_sum(
              -0.5 * (this_embed[None, Ellipsis] - all_next_embed)**2, axis=-1),
          w=self._state_prior[:, None],
          axis=0)
      prior = tf.math.log(
          tf.gather(self._state_prior, next_step.observation) + 1e-8)
      loss = tf.reduce_mean(prior - pos_loss + neg_loss)

    grads = tape.gradient(loss,
                          [self._state_embedder, self._state_action_embedder])
    self._embed_optimizer.apply_gradients(
        zip(grads, [self._state_embedder, self._state_action_embedder]))

    data_transition = tf.gather_nd(
        self._data_transition,
        tf.stack(
            [this_step.observation, next_step.observation, this_step.action],
            axis=-1))
    kld = tf.reduce_sum(
        data_transition * (tf.math.log(data_transition + 1e-8) -
                           (prior - pos_loss + neg_loss)),
        axis=-1)
    return {
        'loss': loss,
        'pos': tf.reduce_mean(pos_loss),
        'neg': tf.reduce_mean(neg_loss),
        'kld': tf.reduce_mean(kld),
    }

  @tf.function
  def train_step(self, transitions):
    this_step = tf.nest.map_structure(lambda t: t[:, 0, Ellipsis], transitions)
    next_step = tf.nest.map_structure(lambda t: t[:, 1, Ellipsis], transitions)
    variables = [self._embed_policy]
    if self._fourier_dim:
      variables += [self._omega, self._shift]
    with tf.GradientTape(watch_accessed_variables=False) as tape2:
      tape2.watch(variables)
      embed = self._state_fourier(this_step.observation)
      with tf.GradientTape(
          watch_accessed_variables=False, persistent=True) as tape1:
        tape1.watch(self._embed_policy)
        embed_policy = tf.nn.softmax(
            tf.matmul(embed, self._embed_policy), axis=-1)
        action = tf.reduce_sum(
            embed_policy * tf.one_hot(this_step.action, self._num_actions), -1)
        neg_ll = tf.reduce_mean(-tf.math.log(action + 1e-8))
      loss = tf.reduce_sum(tape1.gradient(neg_ll, self._embed_policy)**2)

    grads = tape2.gradient(loss, variables)
    self._optimizer.apply_gradients(zip(grads, variables))

    embed = self._state_fourier(this_step.observation)
    next_embed = self._state_action_fourier(next_step.observation,
                                            this_step.action)
    embed_transition = (
        tf.math.sqrt(2. / self._fourier_dim) *
        tf.reduce_sum(embed * next_embed, axis=-1))
    data_transition = tf.gather_nd(
        self._data_transition,
        tf.stack(
            [this_step.observation, next_step.observation, this_step.action],
            axis=-1))
    tvd = tf.reduce_mean(tf.abs(embed_transition - data_transition))
    return {'loss': loss, 'tvd': tvd}

  def get_policy(self):

    def policy_fn(observation, dtype=tf.int32):
      if tf.rank(observation) < 1:
        observation = [observation]

      embed = self._state_fourier(observation)
      distribution = tf.nn.softmax(
          tf.matmul(embed, self._embed_policy), axis=-1)

      policy_info = {'distribution': distribution}
      return (tfp.distributions.Categorical(probs=distribution,
                                            dtype=dtype), policy_info)

    policy_info_spec = {
        'log_probability':
            specs.TensorSpec([], tf.float32),
        'distribution':
            specs.BoundedTensorSpec([self._num_actions],
                                    tf.float32,
                                    minimum=0.0,
                                    maximum=1.0)
    }
    return policy_fn, policy_info_spec
