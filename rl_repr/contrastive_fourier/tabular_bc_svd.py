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

"""Tabular SVD learner."""

from typing import Union
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents import specs


class TabularBCSVD(tf.keras.Model):
  """Tabular behavioral cloning with representations learned via SVD."""

  def __init__(self,
               dataset_spec,
               gamma,
               embed_dim = 64,
               learning_rate = 0.01):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      gamma: The discount factor to use.
      embed_dim: State embedder dimensioin.
      learning_rate: Policy learning rate.
    """
    super().__init__()
    self._gamma = gamma
    self._reward_fn = lambda env_step: env_step.reward

    # Get number of states/actions.
    observation_spec = dataset_spec.observation
    action_spec = dataset_spec.action
    self._num_states = observation_spec.maximum + 1
    self._num_actions = action_spec.maximum + 1

    self._embedder = None

    # Initialize policy parameters.
    self._embed_dim = min(embed_dim, self._num_states)
    self._embed_policy = tf.Variable(
        tf.zeros([self._embed_dim, self._num_actions]))

    self._optimizer = tf.keras.optimizers.Adam(learning_rate)

  def _get_index(self, observation, action):
    return observation * self._num_actions + action

  def solve(self, dataset):
    data_transition = np.zeros(
        [self._num_states, self._num_states, self._num_actions],
        dtype=np.float32)
    weight = np.zeros([self._num_states, 1], dtype=np.float32)

    episodes, valid_steps = dataset.get_all_episodes()
    for episode_num in range(tf.shape(valid_steps)[0]):
      for step_num in range(tf.shape(valid_steps)[1] - 1):
        this_step = tf.nest.map_structure(lambda t: t[episode_num, step_num],  # pylint: disable=cell-var-from-loop
                                          episodes)
        next_step = tf.nest.map_structure(
            lambda t: t[episode_num, step_num + 1], episodes)  # pylint: disable=cell-var-from-loop
        if this_step.is_last() or not valid_steps[episode_num, step_num]:  # pylint: disable=cell-var-from-loop
          continue

        data_transition[this_step.observation, next_step.observation,
                        this_step.action] += 1.
        weight[this_step.observation] += 1.

    data_transition = tf.math.divide_no_nan(
        data_transition, tf.reduce_sum(data_transition, axis=1, keepdims=True))
    data_transition = tf.reshape(data_transition, [self._num_states, -1])
    print('rank', tf.linalg.matrix_rank(data_transition))

    s, u, v = tf.linalg.svd(
        data_transition * tf.math.sqrt(weight), full_matrices=True)
    u = tf.math.divide_no_nan(u, tf.math.sqrt(weight))
    self._embedder = u[:, :self._embed_dim] * s[None, :self._embed_dim]
    embed_transition = tf.matmul(
        self._embedder, v[:, :self._embed_dim], adjoint_b=True)
    embed_transition = tf.where(embed_transition < 0,
                                tf.zeros_like(embed_transition),
                                embed_transition)
    kl_loss = tf.reduce_sum(
        data_transition * (tf.math.log(data_transition + 1e-8) -
                           tf.math.log(embed_transition + 1e-8)),
        axis=-1)

    return {'transition_loss': tf.reduce_mean(kl_loss)}

  @tf.function
  def train_step(self, transitions):
    this_step = tf.nest.map_structure(lambda t: t[:, 0, Ellipsis], transitions)
    with tf.GradientTape(watch_accessed_variables=False) as tape2:
      tape2.watch(self._embed_policy)
      with tf.GradientTape(
          watch_accessed_variables=False, persistent=True) as tape1:
        tape1.watch(self._embed_policy)
        embed = tf.matmul(
            tf.one_hot(this_step.observation, self._num_states), self._embedder)
        embed_policy = tf.nn.softmax(
            tf.matmul(embed, self._embed_policy), axis=-1)
        action = tf.reduce_sum(
            embed_policy * tf.one_hot(this_step.action, self._num_actions), -1)
        neg_ll = tf.reduce_mean(-tf.math.log(action + 1e-8))
      loss = tf.reduce_sum(tape1.gradient(neg_ll, self._embed_policy)**2)

    grads = tape2.gradient(loss, [self._embed_policy])
    self._optimizer.apply_gradients(zip(grads, [self._embed_policy]))
    return {'loss': loss}

  def get_policy(self):

    def policy_fn(observation, dtype=tf.int32):
      if tf.rank(observation) < 1:
        observation = [observation]

      embed = tf.matmul(
          tf.one_hot(observation, self._num_states), self._embedder)
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
