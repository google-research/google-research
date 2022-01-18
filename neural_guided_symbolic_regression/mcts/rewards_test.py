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

"""Tests for rewards."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.mcts import rewards
from neural_guided_symbolic_regression.mcts import states


class RewardBaseTest(tf.test.TestCase):

  def test_set_post_transformer_not_callable(self):
    with self.assertRaisesRegexp(TypeError,
                                 'post_transformer is not callable'):
      reward = rewards.RewardBase()
      reward.set_post_transformer(post_transformer=42)

  def test_set_default_value(self):
    reward = rewards.RewardBase()
    # Default None.
    self.assertIsNone(reward._default_value)
    # The default value can be changed.
    reward.set_default_value(42)
    self.assertAlmostEqual(reward._default_value, 42.)
    # The default value can be changed multiple times.
    reward.set_default_value(-1.5)
    self.assertAlmostEqual(reward._default_value, -1.5)

  def test_evaluate_not_implemented(self):
    state = states.ProductionRulesState(production_rules_sequence=[])
    reward = rewards.RewardBase()
    with self.assertRaisesRegexp(NotImplementedError,
                                 'Must be implemented by subclass'):
      reward.evaluate(state)

  def test_evaluate_not_terminal_without_default_value(self):
    not_terminal_state = states.ProductionRulesState(
        production_rules_sequence=[])
    not_terminal_state.is_terminal = mock.MagicMock(return_value=False)
    reward = rewards.RewardBase(allow_nonterminal=False, default_value=None)
    with self.assertRaisesRegexp(ValueError,
                                 'allow_nonterminal is False and '
                                 'default_value is None, but state is not '
                                 'terminal'):
      reward.evaluate(not_terminal_state)

    # ValueError will not be raised if default value is set.
    reward.set_default_value(42)
    self.assertAlmostEqual(reward.evaluate(not_terminal_state), 42.)

  def test_evaluate_not_terminal_with_default_value(self):
    not_terminal_state = states.ProductionRulesState(
        production_rules_sequence=[])
    not_terminal_state.is_terminal = mock.MagicMock(return_value=False)
    reward = rewards.RewardBase(allow_nonterminal=False, default_value=42)
    self.assertAlmostEqual(reward.evaluate(not_terminal_state), 42)


if __name__ == '__main__':
  tf.test.main()
