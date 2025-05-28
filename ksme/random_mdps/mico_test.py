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

"""Unit tests for MICo module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from ksme.random_mdps import metric
from ksme.random_mdps import mico
from ksme.random_mdps import random_mdp


class MICoTest(parameterized.TestCase):

  def _create_test_mdp(self):
    # We construct the simple 2-state MDP from Lemma 5.1 in the paper to test
    # computation of the metric.
    env = random_mdp.RandomMDP(2, 1)
    env.policy_transition_probs[0, :] = [0.5, 0.5]
    env.policy_transition_probs[1, :] = [0.0, 1.0]
    env.policy_rewards = [1.0, 0.0]
    return env

  def test_object_is_a_metric(self):
    env = self._create_test_mdp()
    self.assertIsInstance(mico.MICo('mico', 'MICo', env, '/tmp/foo'),
                          metric.Metric)

  @parameterized.parameters(
      {'reduced': False,
       'expected_metric': [[1.06, 1.82], [1.82, 0.0]],
       'gamma': 0.9},
      {'reduced': True,
       'expected_metric': [[0.0, 1.29], [1.29, 0.0]],
       'gamma': 0.9},
      {'reduced': False,
       'expected_metric': [[0.0, 1.0], [1.0, 0.0]],
       'gamma': 0.0},
      {'reduced': True,
       'expected_metric': [[0.0, 1.0], [1.0, 0.0]],
       'gamma': 0.0},
  )
  def test_compute_metric(self, reduced, expected_metric, gamma):
    env = self._create_test_mdp()
    env.gamma = gamma
    mico_metric = mico.MICo('mico', 'MICo', env, '/tmp/foo', reduced=reduced)
    mico_metric._compute()
    self.assertTrue(np.allclose(mico_metric.metric, expected_metric,
                                atol=1e-2))


if __name__ == '__main__':
  absltest.main()
