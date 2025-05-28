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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents import specs

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union


class TabularSDT(tf.keras.Model):
  """Tabular decision transformer with discretized returns."""

  def __init__(self,
               dataset_spec,
               num_rtg = 2,
               min_rtg = None,
               max_rtg = None,
               energy_weight = 1.,
               learning_rate = 0.01):
    """Initializes the solver.

      Args:
        dataset_spec: The spec of the dataset that will be given.
        num_rtg: Number of returns-to-go.
        rtg_min: Minimum float returns-to-go for discretization.
        rtg_max: Maximum float returns-to-go for discretization.
        learning_rate: Policy learning rate.
    """
    super().__init__()

    observation_spec = dataset_spec.observation
    action_spec = dataset_spec.action
    self.num_states = observation_spec.maximum + 1
    self.num_actions = action_spec.maximum + 1
    self.min_rtg = min_rtg
    self.max_rtg = max_rtg
    self.num_rtg = num_rtg
    self.energy_weight = energy_weight
    self.num_latents = self.num_actions * self.num_rtg
    self.posterior_net = tf.Variable(
        tf.random.truncated_normal(
            [self.num_states, self.num_actions, self.num_rtg,
             self.num_latents]))
    self.value_net = tf.Variable(
        tf.random.truncated_normal(
            [self.num_states, self.num_latents, self.num_rtg]))
    self.policy = tf.Variable(
        tf.random.truncated_normal(
            [self.num_states, self.num_latents, self.num_actions]))
    self.energy_net = tf.Variable(
        tf.random.truncated_normal(
            [self.num_states, self.num_actions, self.num_rtg,
             self.num_latents]))

    self.rtg_prior = np.zeros([self.num_rtg], dtype=np.float32)
    self.optimizer = tf.keras.optimizers.Adam(learning_rate)

  def encode_rtg(self, rtg):
    if self.min_rtg and self.max_rtg:
      rtg = (rtg - self.min_rtg) / (self.max_rtg - self.min_rtg) * self.num_rtg
    return tf.one_hot(tf.cast(rtg, tf.int32), self.num_rtg)

  def prepare_dataset(self, dataset):
    episodes, valid_steps = dataset.get_all_episodes()
    for episode_num in range(tf.shape(valid_steps)[0]):
      for step_num in range(tf.shape(valid_steps)[1] - 1):
        this_step = tf.nest.map_structure(lambda t: t[episode_num, step_num],
                                          episodes)
        if this_step.is_last() or not valid_steps[episode_num, step_num]:
          continue
        self.rtg_prior[tf.cast(this_step.reward, this_step.action.dtype)] += 1.
    self.rtg_prior = self.rtg_prior / np.sum(self.rtg_prior)

  @tf.function
  def train_step(self, transitions):
    states, actions, rewards, _, mask = transitions
    states = tf.where(states < 0, 0, states)
    actions = tf.where(actions < 0, 0, actions)
    rewards = tf.where(rewards < 0., 0., rewards)
    states = tf.reshape(states, [-1])
    actions = tf.reshape(actions, [-1])
    rewards = tf.reshape(rewards, [-1])
    rewards = tf.cast(rewards, tf.int32)
    this_rtg = rewards

    with tf.GradientTape() as tape:
      this_rtg = tf.cast(this_rtg, actions.dtype)
      indices = tf.stack([states, actions, this_rtg], axis=-1)
      z_logits = tf.gather_nd(self.posterior_net, indices)
      z_probs = tf.nn.softmax(z_logits)
      z_preds = tf.squeeze(tf.random.categorical(z_logits, 1), -1)
      z_preds = tf.one_hot(z_preds, self.num_latents)
      # Straight through gradient
      z_preds = z_preds + z_probs - tf.stop_gradient(z_probs)

      value_logits = tf.reduce_sum(
          z_preds[Ellipsis, None] * tf.gather(self.value_net, states), axis=1)
      value_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          this_rtg, value_logits)
      action_logits = tf.reduce_sum(
          z_preds[Ellipsis, None] * tf.gather(self.policy, states), axis=1)
      action_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          actions, action_logits)

      pos_indices = tf.stack([states, actions, this_rtg], axis=-1)
      pos_embeds = tf.gather_nd(self.energy_net, pos_indices)
      pos_loss = tf.math.reduce_sum(pos_embeds * z_preds, axis=-1)
      neg_indices = tf.stack([states, actions], axis=-1)
      neg_embeds = tf.gather_nd(self.energy_net, neg_indices)
      neg_loss = tfp.math.reduce_weighted_logsumexp(
          tf.reduce_sum(neg_embeds * z_preds[:, None, :], axis=-1),
          w=self.rtg_prior,
          axis=-1)

      energy_loss = self.energy_weight * (-pos_loss + neg_loss)
      loss = (
          tf.reduce_mean(value_loss) + tf.reduce_mean(action_loss) +
          tf.reduce_mean(energy_loss))

    grads = tape.gradient(loss, self.variables)
    self.optimizer.apply_gradients(zip(grads, self.variables))
    return {
        'loss': loss,
        'value_loss': tf.reduce_mean(value_loss),
        'action_loss': tf.reduce_mean(action_loss),
        'pos_loss': tf.reduce_mean(pos_loss),
        'neg_loss': tf.reduce_mean(neg_loss),
        'energy_loss': tf.reduce_mean(energy_loss),
    }

  def get_policy(self):

    def policy_fn(observation, dtype=tf.int32):
      if tf.rank(observation) < 1:
        observation = [observation]

      # Take max rtg and most likely z that led to max rtg.
      value_logits = tf.gather(self.value_net, observation)  # [B, Z, R]
      value_probs = tf.nn.softmax(value_logits, axis=-1)
      z_preds = tf.argmax(value_probs[Ellipsis, -1], -1)

      action_logits = tf.gather_nd(self.policy,
                                   tf.stack([observation, z_preds], axis=-1))
      distribution = tf.nn.softmax(action_logits)
      policy_info = {'distribution': distribution}
      return (tfp.distributions.Categorical(probs=distribution,
                                            dtype=dtype), policy_info)

    policy_info_spec = {
        'log_probability':
            specs.TensorSpec([], tf.float32),
        'distribution':
            specs.BoundedTensorSpec([self.num_actions],
                                    tf.float32,
                                    minimum=0.0,
                                    maximum=1.0)
    }
    return policy_fn, policy_info_spec
