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

"""Unit tests for Bisimulation module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from ksme.random_mdps import bisimulation
from ksme.random_mdps import metric
from ksme.random_mdps import random_mdp


mock = absltest.mock


class BisimulationTest(parameterized.TestCase):

  def test_object_is_a_metric(self):
    env = mock.Mock()
    self.assertIsInstance(bisimulation.Bisimulation(
        'bisimulation', 'bisim', env, '/tmp/foo'), metric.Metric)

  # We use a simple 4-state MDP to test π-bisimulation. x_2 and x_3 are
  # absorbing states with (possibly) varying rewards. x_0 transitions with equal
  # probability to x_2 and x_3, while x_1 transitions deterministically to x_2.
  @parameterized.parameters(
      {'r': [0.0, 0.0, 1.0, 1.0],
       'gamma': 0.9,
       'expected_metric': [[0.0, 0.0, 1.0, 1.0],
                           [0.0, 0.0, 1.0, 1.0],
                           [1.0, 1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0, 0.0]],
       },
      {'r': [0.0, 0.0, 0.0, 1.0],
       'gamma': 0.9,
       'expected_metric': [[0.0, 4.5, 4.5, 5.5],
                           [4.5, 0.0, 0.0, 10.0],
                           [4.5, 0.0, 0.0, 10.0],
                           [5.5, 10.0, 10.0, 0.0]],
       },
      {'r': [0.0, 0.0, 1.0, 1.0],
       'gamma': 0.0,
       'expected_metric': [[0.0, 0.0, 1.0, 1.0],
                           [0.0, 0.0, 1.0, 1.0],
                           [1.0, 1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0, 0.0]],
       },
  )
  def test_compute_metric(self, r, gamma, expected_metric):
    env = random_mdp.RandomMDP(4, 1)
    env.policy_transition_probs = np.array(
        [[0.0, 0.0, 0.5, 0.5],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]])
    env.policy_rewards = np.array(r)
    env.gamma = gamma
    pi_bisim = bisimulation.Bisimulation(
        'bisimulation', 'bisim', env, '/tmp/foo')
    pi_bisim._compute()
    self.assertTrue(np.allclose(pi_bisim.metric, expected_metric,
                                atol=1e-2))


if __name__ == '__main__':
  absltest.main()
