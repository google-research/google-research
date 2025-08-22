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

"""Unit tests for Utils module."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from ksme.random_mdps import utils


class UtilsTest(parameterized.TestCase):

  def test_make_auxiliary_mdp(self):
    # The original MDP has 3 states.
    p = np.array([[0.1, 0.2, 0.7],
                  [0.2, 0.4, 0.4],
                  [1.0, 0.0, 0.0]])
    r = np.array([1., 2., 3.])
    # The auxiliary MDP will have 3^2 = 9 states.
    # We compute it explicitly here for clarity.
    expected_aux_p = np.zeros((9, 9))
    expected_aux_r = np.zeros(9)
    for x1 in range(3):
      for y1 in range(3):
        aux_state_1 = x1 * 3 + y1
        expected_aux_r[aux_state_1] = abs(r[x1] - r[y1])
        for x2 in range(3):
          for y2 in range(3):
            aux_state_2 = x2 * 3 + y2
            expected_aux_p[aux_state_1, aux_state_2] = (
                p[x1, x2] * p[y1, y2])

    aux_p, aux_r = utils.build_auxiliary_mdp(p, r)
    self.assertTrue(np.array_equal(expected_aux_p, aux_p))
    self.assertTrue(np.array_equal(expected_aux_r, aux_r))

  @parameterized.parameters(
      {'p': [[0.5, 0.5], [0.8, 0.2]], 'r': [1.0, 0.0], 'gamma': 0.9,
       'expected_v': [6.46, 5.67]},
      {'p': [[0.5, 0.5], [0.8, 0.2]], 'r': [1.0, 0.0], 'gamma': 0.0,
       'expected_v': [1.0, 0.0]},
      {'p': [[1.0, 0.0], [0.0, 1.0]], 'r': [1.0, 0.0], 'gamma': 0.9,
       'expected_v': [10.0, 0.0]},
      {'p': [[0.0, 1.0], [0.0, 1.0]], 'r': [1.0, 0.0], 'gamma': 0.9,
       'expected_v': [1.0, 0.0]},
  )
  def test_compute_value(self, p, r, gamma, expected_v):
    # We test value function computation on a simple two-state MDP for which we
    # know the value.
    p = np.array(p)
    r = np.array(r)
    expected_v = np.array(expected_v)
    v = utils.compute_value(p, r, gamma)
    self.assertTrue(np.allclose(expected_v, v, atol=1e-2))


if __name__ == '__main__':
  absltest.main()
