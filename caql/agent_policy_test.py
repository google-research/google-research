# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tests for agent_policy."""

from unittest import mock

import numpy as np
import tensorflow as tf
from tf_agents.specs import array_spec

from caql import agent_policy
from caql import caql_agent


class AgentPolicyTest(tf.test.TestCase):

  def setUp(self):
    super(AgentPolicyTest, self).setUp()
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(3,), dtype=np.float, minimum=[0, 0, 0], maximum=[1, 1, 1])

  def testBatchModeFalseWithOneDimensionalState(self):
    state = np.arange(2)
    mock_agent = mock.create_autospec(caql_agent.CaqlAgent, instance=True)
    mock_agent.best_action.return_value = (
        np.arange(3).reshape(1, 3), None, None, True)
    policy = agent_policy.AgentPolicy(self._action_spec, mock_agent)
    action = policy.action(state, batch_mode=False)
    self.assertAllEqual(np.arange(3), action)

  def testBatchModeFalseWithTwoDimensionalState(self):
    state = np.arange(2).reshape(1, 2)
    mock_agent = mock.create_autospec(caql_agent.CaqlAgent, instance=True)
    mock_agent.best_action.return_value = (
        np.arange(3).reshape(1, 3), None, None, True)
    policy = agent_policy.AgentPolicy(self._action_spec, mock_agent)
    action = policy.action(state, batch_mode=False)
    self.assertAllEqual(np.arange(3), action)

  def testBatchModeTrueWithOneDimensionalState(self):
    state = np.arange(2)
    mock_agent = mock.create_autospec(caql_agent.CaqlAgent, instance=True)
    mock_agent.best_action.return_value = (
        np.arange(3).reshape(1, 3), None, None, True)
    policy = agent_policy.AgentPolicy(self._action_spec, mock_agent)
    action = policy.action(state, batch_mode=True)
    self.assertAllEqual(np.arange(3).reshape(1, 3), action)

  def testBatchModeTrueWithTwoDimensionalState(self):
    state = np.arange(2).reshape(1, 2)
    mock_agent = mock.create_autospec(caql_agent.CaqlAgent, instance=True)
    mock_agent.best_action.return_value = (
        np.arange(3).reshape(1, 3), None, None, True)
    policy = agent_policy.AgentPolicy(self._action_spec, mock_agent)
    action = policy.action(state, batch_mode=True)
    self.assertAllEqual(np.arange(3).reshape(1, 3), action)


if __name__ == '__main__':
  tf.test.main()
