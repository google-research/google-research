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

"""Tests for mcts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import mock
import numpy as np
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.models import mcts


class LeadingPowersTest(parameterized.TestCase):

  @parameterized.parameters(
      # Exact expression.
      ('1 / x + x * x', 0.),
      # Leading powers not match.
      ('1 / x / x + x + x', -2.),
      # leading_power_error is nan.
      ('1 / ( x - x )', -50.),
      )
  def test_evaluate(self, expression_string, expected_reward):
    reward = mcts.LeadingPowers(
        leading_at_0=-1.,
        leading_at_inf=2.,
        allow_nonterminal=True,
        default_value=-50.)
    mock_state = mock.MagicMock()
    mock_state.get_expression.return_value = expression_string
    self.assertAlmostEqual(reward._evaluate(mock_state), expected_reward)


class NumericalPointsAndLeadingPowersTest(parameterized.TestCase):

  @parameterized.parameters(
      # Exact expression.
      ('1 / x + x * x', None, True, 0.),
      ('1 / x + x * x', -100., True, 0.),
      ('1 / x + x * x', None, False, 0.),
      ('1 / x + x * x', -100., False, 0.),
      # Leading powers are correct. But wrong expression.
      # Soft penalty
      ('1 / x + x + x * x', None, True, -2.6457513110645907),
      # Hard penalty
      ('1 / x + x + x * x', -100., True, -2.6457513110645907),
      # Not include leading powers.
      ('1 / x + x + x * x', None, False, -2.6457513110645907),
      ('1 / x + x + x * x', -100., False, -2.6457513110645907),
      # Leading powers are wrong.
      # Soft penalty
      ('x', None, True, -10.2413741789801191),
      # Hard penalty
      ('x', -100., True, -100.),
      # Not include leading powers.
      ('x', None, False, -7.2413741789801191),
      ('x', -100., False, -7.2413741789801191),
      # leading_power_error is nan.
      ('1 / ( x - x )', None, True, -50.),
      ('1 / ( x - x )', -100., True, -100.),
      ('1 / ( x - x )', None, False, -9.8032732628784078),
      ('1 / ( x - x )', -100., False, -9.8032732628784078),
      )
  def test_evaluate(
      self,
      expression_string,
      hard_penalty_default_value,
      include_leading_powers,
      expected_reward):
    reward = mcts.NumericalPointsAndLeadingPowers(
        input_values=np.array([1., 2., 4.]),
        output_values=np.array([2., 4.5, 16.25]),
        leading_at_0=-1.,
        leading_at_inf=2.,
        hard_penalty_default_value=hard_penalty_default_value,
        include_leading_powers=include_leading_powers,
        allow_nonterminal=True,
        default_value=-50.)
    mock_state = mock.MagicMock()
    mock_state.get_expression.return_value = expression_string
    self.assertAlmostEqual(reward._evaluate(mock_state), expected_reward)


if __name__ == '__main__':
  tf.test.main()
