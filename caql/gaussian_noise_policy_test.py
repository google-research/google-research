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

"""Tests for gaussian_noise_policy."""

from unittest import mock

from absl import logging

import numpy as np
import tensorflow as tf
from tf_agents.specs import array_spec

from caql import agent_policy
from caql import gaussian_noise_policy


class GaussianNoisePolicyTest(tf.test.TestCase):

  def setUp(self):
    super(GaussianNoisePolicyTest, self).setUp()
    seed = 199
    logging.info('Setting the numpy seed to %d', seed)
    np.random.seed(seed)

    self._mock_policy = mock.create_autospec(
        agent_policy.AgentPolicy, instance=True)
    self._mock_policy.continuous_action = True
    self._mock_policy.action_spec = array_spec.BoundedArraySpec(
        shape=(3,), dtype=np.float, minimum=[0, 0, 0], maximum=[1, 1, 1])

  def testNoneAction(self):
    self._mock_policy.action.return_value = None
    policy = gaussian_noise_policy.GaussianNoisePolicy(
        self._mock_policy, 0.9, 0.7, 0.01)
    self.assertIsNone(policy.action(np.arange(2)))

  def testGaussianNoiseAction(self):
    self._mock_policy.action.return_value = np.array([0.2, 0.5, 0.8])
    policy = gaussian_noise_policy.GaussianNoisePolicy(
        self._mock_policy, 0.9, 0.1, 0.01)
    action = policy.action(np.arange(2))
    policy.update_params()
    self.assertAllClose([1.19726433, 0.20994996, 2.85986947], action)
    action = policy.action(np.arange(2))
    policy.update_params()
    self.assertAllClose([0.03390958, 0.36193276, 0.89809503], action)
    action = policy.action(np.arange(2))
    policy.update_params()
    self.assertAllClose([0.21209471, 0.49707365, 0.79036936], action)

  def testSigmaDecay(self):
    sigma = 0.9
    decay_rate = 0.7
    sigma_min = 0.01
    policy = gaussian_noise_policy.GaussianNoisePolicy(
        self._mock_policy, sigma, decay_rate, sigma_min)
    for _ in range(20):
      policy.update_params()
      sigma = max(sigma * decay_rate, sigma_min)
      self.assertAlmostEqual(sigma, policy.sigma)


if __name__ == '__main__':
  tf.test.main()
