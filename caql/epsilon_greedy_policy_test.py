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

"""Tests for epsilon_greedy_policy."""

from unittest import mock

from absl import logging

import numpy as np
import tensorflow as tf
from tf_agents.specs import array_spec

from caql import agent_policy
from caql import epsilon_greedy_policy


class EpsilonGreedyPolicyTest(tf.test.TestCase):

  def setUp(self):
    super(EpsilonGreedyPolicyTest, self).setUp()
    seed = 199
    logging.info('Setting the numpy seed to %d', seed)
    np.random.seed(seed)

    self._mock_policy = mock.create_autospec(
        agent_policy.AgentPolicy, instance=True)
    self._mock_policy.continuous_action = True
    self._mock_policy.action_spec = array_spec.BoundedArraySpec(
        shape=(3,), dtype=np.float, minimum=[0, 0, 0], maximum=[1, 1, 1])
    self._mock_policy.action.return_value = np.array([.5, .5, .5])

  def testEpsilonGreedyAction(self):
    policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
        self._mock_policy, 0.9, 0.1, 0.01)
    action = policy.action(np.arange(2))
    self.assertAllClose([0.98203928, 0.3999047, 0.8441526], action)
    action = policy.action(np.arange(2))
    self.assertAllClose([0.54484307, 0.61947784, 0.33183679], action)
    action = policy.action(np.arange(2))
    self.assertAllClose([.5, .5, .5], action)

  def testEpsilonDecay(self):
    epsilon = 0.9
    decay_rate = 0.7
    epsilon_min = 0.01
    policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
        self._mock_policy, epsilon, decay_rate, epsilon_min)
    for _ in range(20):
      policy.update_params()
      epsilon = max(epsilon * decay_rate, epsilon_min)
      self.assertAlmostEqual(epsilon, policy.epsilon)


if __name__ == '__main__':
  tf.test.main()
