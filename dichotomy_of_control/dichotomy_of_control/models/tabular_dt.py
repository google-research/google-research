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


class TabularDT(tf.keras.Model):
  """Tabular decision transformer with discretized returns."""

  def __init__(self,
               dataset_spec,
               num_rtg = 2,
               min_rtg = None,
               max_rtg = None,
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
    self.policy = tf.Variable(
        tf.random.truncated_normal(
            [self.num_states, self.num_rtg, self.num_actions]))
    self.optimizer = tf.keras.optimizers.Adam(learning_rate)

  def discretize_rtg(self, rtg):
    if self.min_rtg and self.max_rtg:
      rtg = (rtg - self.min_rtg) / (self.max_rtg - self.min_rtg) * self.num_rtg
    return tf.cast(rtg, tf.int32)

  @tf.function
  def train_step(self, transitions):
    states, actions, rewards, _, mask = transitions
    # Replace padded -1s with 0s.
    states = tf.where(states < 0, 0, states)
    actions = tf.where(actions < 0, 0, actions)
    rewards = tf.where(rewards < 0., 0., rewards)
    states = tf.squeeze(states)
    actions = tf.squeeze(actions)
    rewards = tf.squeeze(tf.cast(rewards, tf.int32))
    rtgs = rewards
    if tf.rank(rtgs) > 1:
      rtgs = tf.math.cumsum(rewards, axis=-1, reverse=True)
    with tf.GradientTape() as tape:
      rtg = self.discretize_rtg(rtgs)
      preds = tf.gather_nd(self.policy, tf.stack([states, rtg], axis=-1))
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(actions, preds)
      loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, self.variables)
    self.optimizer.apply_gradients(zip(grads, self.variables))
    return {'loss': loss}

  def get_policy(self):

    def policy_fn(observation, dtype=tf.int32):
      if tf.rank(observation) < 1:
        observation = [observation]

      rtg = tf.ones_like(observation) * (self.num_rtg - 1)
      preds = tf.gather_nd(self.policy, tf.stack([observation, rtg], axis=-1))
      distribution = tf.nn.softmax(preds)
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
