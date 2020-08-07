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

"""Tests for policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
import numpy as np
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.mcts import policies
from neural_guided_symbolic_regression.mcts import states


class _StateForTest(object):

  def __init__(self, value):
    self.value = value

  def __eq__(self, other):
    return np.isclose(self.value, other.value)


def _action_for_test(state, factor):
  return _StateForTest(factor * state.value)


class PolicyBaseTest(tf.test.TestCase):

  def test_get_new_states_probs(self):
    policy = policies.PolicyBase()
    with self.assertRaisesRegexp(NotImplementedError,
                                 'Must be implemented by subclass'):
      policy.get_new_states_probs(_StateForTest(0))


class ProductionRuleAppendPolicyTest(tf.test.TestCase):

  def setUp(self):
    super(ProductionRuleAppendPolicyTest, self).setUp()
    grammar_rules = [
        'S -> S "+" T',
        'S -> T',
        'T -> "(" S ")"',
        'T -> "x"',
    ]
    self.grammar = nltk.CFG.fromstring(grammar_rules)
    self.production_rules_dict = {
        k: v for k, v in zip(grammar_rules, self.grammar.productions())}

  def _strings_to_production_rules(self, production_rules_strings):
    return [
        self.production_rules_dict[production_rules_string]
        for production_rules_string in production_rules_strings]

  def test_get_new_states_probs_type_error(self):
    policy = policies.ProductionRuleAppendPolicy(grammar=self.grammar)
    with self.assertRaisesRegexp(TypeError,
                                 r'Input state shoud be an instance of '
                                 r'states\.ProductionRulesState'):
      policy.get_new_states_probs(states.StateBase())

  def test_get_new_states_probs(self):
    state = states.ProductionRulesState(self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> T',
    ]))
    # The above production rules sequence are parsed as
    #                 S
    #                 |
    #              S '+' T
    #              |
    #              T
    #
    # Since the order of the production rules sequence is the preorder traversal
    # of the parsing tree, the next symbol to parse is the 'T' on the left side
    # of the above parsing tree. Only production rule with left hand side symbol
    # T are valid production rule.
    # Thus, for grammar with production rules:
    # 'S -> S "+" T'
    # 'S -> T'
    # 'T -> "(" S ")"'
    # 'T -> "x"'
    # Appending the first two production rules will create invalid state, with
    # prior probabilities nan. The last two production rules can be appended
    # and will create new states, with equal prior probabilities.
    expected_new_states = [
        None,
        None,
        states.ProductionRulesState(self._strings_to_production_rules([
            'S -> S "+" T',
            'S -> T',
            'T -> "(" S ")"',
        ])),
        states.ProductionRulesState(self._strings_to_production_rules([
            'S -> S "+" T',
            'S -> T',
            'T -> "x"',
        ])),
    ]

    policy = policies.ProductionRuleAppendPolicy(grammar=self.grammar)
    new_states, action_probs = policy.get_new_states_probs(state)

    np.testing.assert_allclose(action_probs, [np.nan, np.nan, 0.5, 0.5])
    self.assertEqual(len(new_states), len(expected_new_states))
    for new_state, expected_new_state in zip(new_states, expected_new_states):
      self.assertEqual(new_state, expected_new_state)


if __name__ == '__main__':
  tf.test.main()
