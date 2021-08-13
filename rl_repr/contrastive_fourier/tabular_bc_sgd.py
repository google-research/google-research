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

"""Tabular SGD learner."""

from typing import Union
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents import specs


class TabularBCSGD(tf.keras.Model):
  """Tabular behavioral cloning with representations learned via SGD."""

  def __init__(self,
               dataset_spec,
               gamma,
               embed_dim = 64,
               embed_learning_rate = 0.001,
               learning_rate = 0.01,
               finetune = False,
               latent_policy = True):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      gamma: The discount factor to use.
      embed_dim: State embedder dimensioin.
      embed_learning_rate: Reward and transition model learning rate.
      learning_rate: Policy learning rate.
      finetune: Whether to finetune embedder during policy learning.
      latent_policy: Whether to learn a latent policy.
    """
    super().__init__()
    self._gamma = gamma
    self._reward_fn = lambda env_step: env_step.reward
    if not finetune:
      # Pretraining case must train a latent policy
      assert latent_policy
    self._finetune = finetune
    self._latent_policy = latent_policy

    # Get number of states/actions.
    observation_spec = dataset_spec.observation
    action_spec = dataset_spec.action
    self._num_states = observation_spec.maximum + 1
    self._num_actions = action_spec.maximum + 1

    self._data_transition = np.zeros(
        [self._num_states * self._num_actions, self._num_states],
        dtype=np.float32)
    self._data_reward = np.zeros([self._num_states * self._num_actions],
                                 dtype=np.float32)
    self._expert_policy = np.zeros([self._num_states, self._num_actions],
                                   dtype=np.float32)

    # Initialize embedding, reward, and transition parameters.
    inputs = tf.keras.Input(shape=(self._num_states,))
    self._embed_dim = min(embed_dim, self._num_states)
    embedder = tf.keras.layers.Dense(self._embed_dim)
    self._embedder = tf.keras.Model(inputs=inputs, outputs=embedder(inputs))
    self._embed_reward = tf.Variable(
        tf.zeros([self._embed_dim * self._num_actions]))
    self._embed_transition_logits = tf.Variable(
        tf.zeros([self._embed_dim * self._num_actions, self._num_states]))
    if self._latent_policy:
      self._embed_policy_logits = tf.Variable(
          tf.zeros([self._embed_dim, self._num_actions]))
    else:
      self._embed_policy_logits = tf.Variable(
          tf.zeros([self._num_states, self._num_actions]))

    self._embed_optimizer = tf.keras.optimizers.Adam(embed_learning_rate)
    self._optimizer = tf.keras.optimizers.Adam(learning_rate)

  def _get_index(self, observation, action):
    return observation * self._num_actions + action

  def _embed_state(self, observation):
    """Embed categorical observation into a k-dimensional embedding vector."""

    logits = self._embedder(tf.one_hot(observation, self._num_states))
    probs = tf.nn.softmax(logits, axis=-1)
    samples = tfp.distributions.Categorical(logits=logits).sample()
    onehot_samples = tf.one_hot(samples, self._embed_dim)

    # Straight-through gradients
    return onehot_samples + probs - tf.stop_gradient(probs)

  def prepare_datasets(self, dataset, expert_dataset):
    total_weight = np.zeros_like(self._data_reward)

    episodes, valid_steps = dataset.get_all_episodes()
    for episode_num in range(tf.shape(valid_steps)[0]):
      for step_num in range(tf.shape(valid_steps)[1] - 1):
        this_step = tf.nest.map_structure(lambda t: t[episode_num, step_num],  # pylint: disable=cell-var-from-loop
                                          episodes)
        next_step = tf.nest.map_structure(
            lambda t: t[episode_num, step_num + 1], episodes)  # pylint: disable=cell-var-from-loop
        if this_step.is_last() or not valid_steps[episode_num, step_num]:  # pylint: disable=cell-var-from-loop
          continue

        index = self._get_index(this_step.observation, this_step.action)
        self._data_transition[index, next_step.observation] += 1.
        self._data_reward[index] += self._reward_fn(this_step)
        total_weight[index] += 1.

    self._data_transition = tf.math.divide_no_nan(
        self._data_transition,
        tf.reduce_sum(self._data_transition, axis=-1, keepdims=True))
    self._data_reward = tf.math.divide_no_nan(self._data_reward, total_weight)

    expert_steps = expert_dataset.get_all_steps()
    np.add.at(self._expert_policy,
              [expert_steps.observation, expert_steps.action], 1.)
    self._expert_policy = tf.math.divide_no_nan(
        self._expert_policy,
        tf.reduce_sum(self._expert_policy, axis=-1, keepdims=True))

  @tf.function
  def train_embed(self, transitions):
    this_step = tf.nest.map_structure(lambda t: t[:, 0, Ellipsis], transitions)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self._embedder.trainable_variables)
      tape.watch(self._embed_reward)
      tape.watch(self._embed_transition_logits)

      reward_loss, transition_loss = 0., 0.
      for action in range(self._num_actions):
        state_action = tf.one_hot(
            self._get_index(this_step.observation, action),
            self._num_states * self._num_actions)
        data_reward = tf.matmul(state_action, self._data_reward[:, None])
        data_transition = tf.matmul(state_action, self._data_transition)

        embed = self._embed_state(this_step.observation)
        embed_action = tf.reshape(
            (tf.repeat(embed[Ellipsis, None], self._num_actions, axis=-1) *
             tf.one_hot(action, self._num_actions)), [tf.shape(embed)[0], -1])
        embed_reward = tf.matmul(embed_action, self._embed_reward[:, None])
        embed_transition = tf.matmul(
            embed_action, tf.nn.softmax(self._embed_transition_logits, axis=-1))

        reward_loss += tf.reduce_mean((data_reward - embed_reward)**2)
        kl_loss = tf.reduce_sum(
            data_transition * (tf.math.log(data_transition + 1e-8) -
                               tf.math.log(embed_transition + 1e-8)),
            axis=-1)
        transition_loss += tf.reduce_mean(kl_loss)

      loss = (1 / (1 - self._gamma) * reward_loss +
              2 * tf.reduce_max(self._data_reward) /
              (1 - self._gamma)**2 * transition_loss)
    variables = (
        self._embedder.trainable_variables +
        [self._embed_reward, self._embed_transition_logits])
    grads = tape.gradient(loss, variables)
    self._embed_optimizer.apply_gradients(zip(grads, variables))
    return {'reward_loss': reward_loss, 'transition_loss': transition_loss}

  @tf.function
  def train_step(self, transitions):
    this_step = tf.nest.map_structure(lambda t: t[:, 0, Ellipsis], transitions)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      if self._finetune:
        tape.watch(self._embedder.trainable_variables)
      tape.watch(self._embed_policy_logits)

      if self._latent_policy:
        embed = self._embed_state(this_step.observation)
      else:
        embed = tf.one_hot(this_step.observation, self._num_states)
      embed_policy = tf.matmul(
          embed, tf.nn.softmax(self._embed_policy_logits, axis=-1))
      expert_policy = tf.gather(self._expert_policy, this_step.observation)

      tf.debugging.check_numerics(embed, 'embed')
      tf.debugging.check_numerics(self._embed_policy_logits,
                                  'self._embed_policy_logits')
      tf.debugging.check_numerics(embed_policy, 'embed_policy')
      tf.debugging.check_numerics(expert_policy, 'expert_policy')

      loss = tf.reduce_mean(
          tf.reduce_sum(
              expert_policy * -tf.math.log(embed_policy + 1e-8), axis=-1))
      loss = (2 * tf.reduce_max(self._data_reward) * (2 - self._gamma) /
              (1 - self._gamma)**2 * loss)

    variables = [self._embed_policy_logits]
    if self._finetune:
      variables += self._embedder.trainable_variables
    grads = tape.gradient(loss, variables)
    self._optimizer.apply_gradients(zip(grads, variables))
    return {'loss': loss}

  def get_policy(self):

    def policy_fn(observation, dtype=tf.int32):
      if tf.rank(observation) < 1:
        observation = [observation]

      if self._latent_policy:
        embed = self._embed_state(observation)
      else:
        embed = tf.one_hot(observation, self._num_states)
      distribution = tf.matmul(
          embed, tf.nn.softmax(self._embed_policy_logits, axis=-1))

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
