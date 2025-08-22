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


class TabularBC(tf.keras.Model):
  """Tabular behavioral cloning."""

  def __init__(self,
               dataset_spec,
               learning_rate = 0.01):
    """Initializes the solver.

      Args:
        dataset_spec: The spec of the dataset that will be given.
        learning_rate: Policy learning rate.
    """
    super().__init__()

    observation_spec = dataset_spec.observation
    action_spec = dataset_spec.action
    self.num_states = observation_spec.maximum + 1
    self.num_actions = action_spec.maximum + 1
    self.policy = tf.Variable(
        tf.random.truncated_normal([self.num_states, self.num_actions]))
    self.optimizer = tf.keras.optimizers.Adam(learning_rate)

  @tf.function
  def train_step(self, transitions):
    states, actions, rewards, _, mask = transitions
    states = tf.where(states < 0, 0, states)
    actions = tf.where(actions < 0, 0, actions)
    rewards = tf.where(rewards < 0., 0., rewards)
    states = tf.reshape(states, [-1])
    actions = tf.reshape(actions, [-1])
    rewards = tf.reshape(rewards, [-1])

    with tf.GradientTape() as tape:
      action_logits = tf.gather(self.policy, states)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          actions, action_logits)
      action_preds = tf.argmax(
          action_logits, axis=-1, output_type=actions.dtype)
      action_acc = tf.reduce_mean(
          tf.cast(tf.equal(action_preds, actions), tf.float32))
      loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, self.variables)
    self.optimizer.apply_gradients(zip(grads, self.variables))
    return {'loss': loss, 'action_acc': action_acc}

  def get_policy(self):

    def policy_fn(observation, dtype=tf.int32):
      if tf.rank(observation) < 1:
        observation = [observation]
      distribution = tf.nn.softmax(tf.gather(self.policy, observation), axis=-1)
      batched = tf.rank(distribution) > 1
      if not batched:
        distributions = distribution[None, :]
      else:
        distributions = distribution

      batch_size = tf.shape(distributions)[0]

      actions = tf.random.categorical(
          tf.math.log(1e-8 + distributions), 1, dtype=dtype)
      actions = tf.squeeze(actions, -1)
      probs = tf.gather_nd(
          distributions,
          tf.stack([tf.range(batch_size, dtype=dtype), actions], -1))

      if not batched:
        action = actions[0]
        log_prob = tf.math.log(1e-8 + probs[0])
      else:
        action = actions
        log_prob = tf.math.log(1e-8 + probs)

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
