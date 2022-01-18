# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Wrapper for agent to use the critic function to compute reachability from critic functions."""

import tensorflow as tf
from tf_agents.utils import nest_utils

from reset_free_learning.agents import sac_agent
from reset_free_learning.agents import td3_agent


class SacAgent(sac_agent.SacAgent):

  def __init__(self, *sac_args, num_action_samples=1, **sac_kwargs):
    self._num_action_samples = num_action_samples
    super(SacAgent, self).__init__(*sac_args, **sac_kwargs)

  def compute_value(self, time_steps):
    nest_utils.assert_same_structure(time_steps, self.time_step_spec)
    # get number of actions from the policy
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    policy_state = self._train_policy.get_initial_state(batch_size)
    action_distribution = self._train_policy.distribution(
        time_steps, policy_state=policy_state).action
    actions = tf.nest.map_structure(
        lambda d: d.sample(self._num_action_samples), action_distribution)

    # repeat for multiple actions
    observations = tf.tile(time_steps.observation,
                           tf.constant([self._num_action_samples, 1]))
    actions = tf.reshape(actions, [-1, actions.shape[-1]])
    pred_input = (observations, actions)

    if self._critic_network_no_entropy_1 is None:
      critic_pred_1, _ = self._critic_network_1(
          pred_input, None, training=False)
      critic_pred_2, _ = self._critic_network_2(
          pred_input, None, training=False)
    else:
      critic_pred_1, _ = self._critic_network_no_entropy_1(
          pred_input, None, training=False)
      critic_pred_2, _ = self._critic_network_no_entropy_2(
          pred_input, None, training=False)

    # final value calculation
    critic_pred = tf.minimum(critic_pred_1, critic_pred_2)
    critic_pred = tf.reshape(critic_pred, [self._num_action_samples, -1])
    value = tf.reduce_mean(critic_pred, axis=0)
    return value


class Td3Agent(td3_agent.Td3Agent):

  def compute_value(self, time_steps):
    nest_utils.assert_same_structure(time_steps, self.time_step_spec)
    # get number of actions from the policy
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    policy_state = self._train_policy.get_initial_state(batch_size)
    action_distribution = self._train_policy.distribution(
        time_steps, policy_state=policy_state).action
    actions = tf.nest.map_structure(
        lambda d: d.sample(),
        action_distribution)  # this is a deterministic policy

    observations = time_steps.observation
    pred_input = (observations, actions)

    # TODO(architsh): check if the minimum should be taken before or after average
    critic_pred_1, _ = self._critic_network_1(pred_input, None, training=False)
    critic_pred_2, _ = self._critic_network_2(pred_input, None, training=False)

    # final value calculation
    value = tf.minimum(critic_pred_1, critic_pred_2)
    return value
