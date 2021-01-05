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

"""Tests for tree."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import mock
import numpy as np
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.mcts import tree


class UtilitiesTest(parameterized.TestCase, tf.test.TestCase):

  def test_get_max_values_indices(self):
    array = [0., 0.3, 0.1, 0.3, 0.3]
    np.testing.assert_array_equal(tree._get_max_values_indices(array),
                                  [1, 3, 4])

  def test_random_argmax(self):
    # The maximum values has index [1, 3, 4].
    array = [0., 0.3, 0.1, 0.3, 0.3]
    random_state = np.random.RandomState(2)
    # Make sure every time the returned index are different.
    # Those indices are fixed for give random_state.
    self.assertEqual(tree.random_argmax(array, random_state), 1)
    self.assertEqual(tree.random_argmax(array, random_state), 3)
    self.assertEqual(tree.random_argmax(array, random_state), 1)
    self.assertEqual(tree.random_argmax(array, random_state), 4)
    self.assertEqual(tree.random_argmax(array, random_state), 4)

  @parameterized.parameters([
      # All states are terminal and there is one unique maximum.
      ([True, True, True], [1., 3., 2.], True, 3., 1),
      ([True, True, True], [1., 3., 2.], False, 3., 1),
      # There are non-terminal states and ignore_nonterminal is False.
      # In these cases, the expected max_state is always the one with largest
      # reward_value and no matter whether it is terminal.
      ([False, True, True], [1., 3., 2.], False, 3., 1),
      ([True, False, True], [1., 3., 2.], False, 3., 1),
      ([True, True, False], [1., 3., 2.], False, 3., 1),
      ([True, False, False], [1., 3., 2.], False, 3., 1),
      # There are non-terminal states and ignore_nonterminal is True.
      ([False, True, True], [1., 3., 2.], True, 3., 1),
      ([True, False, True], [1., 3., 2.], True, 2., 2),
      ([True, True, False], [1., 3., 2.], True, 3., 1),
      ([True, False, False], [1., 3., 2.], True, 1., 0),
  ])
  def test_max_reward_and_state_unique_maximum(self,
                                               states_terminal,
                                               reward_values,
                                               ignore_nonterminal,
                                               expected_max_reward_value,
                                               expected_max_state_index):
    mock_state0 = mock.MagicMock()
    mock_state0.is_terminal.return_value = states_terminal[0]
    mock_state1 = mock.MagicMock()
    mock_state1.is_terminal.return_value = states_terminal[1]
    mock_state2 = mock.MagicMock()
    mock_state2.is_terminal.return_value = states_terminal[2]
    mock_states_list = [mock_state0, mock_state1, mock_state2]

    max_reward_value, max_state = tree.max_reward_and_state(
        reward_values=reward_values,
        states_list=mock_states_list,
        ignore_nonterminal=ignore_nonterminal)

    self.assertAlmostEqual(max_reward_value, expected_max_reward_value)
    self.assertEqual(max_state, mock_states_list[expected_max_state_index])

  @parameterized.parameters([
      # All states are terminal and there are two state with maximum reward
      # value.
      ([True, True, True], [1., 3., 3.], True, 3., [1, 2, 2, 1, 1, 2]),
      ([True, True, True], [1., 3., 3.], False, 3., [1, 2, 2, 1, 1, 2]),
      # There are non-terminal states and ignore_nonterminal is False.
      # The returned results will not change.
      ([False, True, True], [1., 3., 3.], False, 3., [1, 2, 2, 1, 1, 2]),
      ([True, False, True], [1., 3., 3.], False, 3., [1, 2, 2, 1, 1, 2]),
      ([True, True, False], [1., 3., 3.], False, 3., [1, 2, 2, 1, 1, 2]),
      # There are non-terminal states and ignore_nonterminal is True.
      ([False, True, True], [1., 3., 3.], True, 3., [1, 2, 2, 1, 1, 2]),
      ([True, False, True], [1., 3., 3.], True, 3., [2, 2, 2, 2, 2, 2]),
      ([True, True, False], [1., 3., 3.], True, 3., [1, 1, 1, 1, 1, 1]),
  ])
  def test_max_reward_and_state_multiple_maximum(self,
                                                 states_terminal,
                                                 reward_values,
                                                 ignore_nonterminal,
                                                 expected_max_reward_value,
                                                 expected_max_state_indices):
    # In order to test the random selection, a fixed random seed is used
    # the expected_max_state_indices is a sequence of index of state
    # returned. This ensures that the states with maximum reward value
    # are selected randomly.
    random_state = np.random.RandomState(2)

    mock_state0 = mock.MagicMock()
    mock_state0.is_terminal.return_value = states_terminal[0]
    mock_state1 = mock.MagicMock()
    mock_state1.is_terminal.return_value = states_terminal[1]
    mock_state2 = mock.MagicMock()
    mock_state2.is_terminal.return_value = states_terminal[2]
    mock_states_list = [mock_state0, mock_state1, mock_state2]

    for expected_max_state_index in expected_max_state_indices:
      max_reward_value, max_state = tree.max_reward_and_state(
          reward_values=reward_values,
          states_list=mock_states_list,
          ignore_nonterminal=ignore_nonterminal,
          random_state=random_state)
      self.assertAlmostEqual(max_reward_value, expected_max_reward_value)
      self.assertEqual(max_state, mock_states_list[expected_max_state_index])

  def test_max_reward_and_state_length_not_match(self):
    with self.assertRaisesRegex(
        ValueError,
        r'The length of reward_values \(2\) does not match the length of '
        r'states_list \(1\)'):
      tree.max_reward_and_state(
          reward_values=[42., 9.], states_list=[mock.MagicMock()])

  def test_max_reward_and_state_allowed_states_list_empty(self):
    with self.assertRaisesRegex(
        ValueError, 'The number of allowed states to choose is 0'):
      tree.max_reward_and_state(
          reward_values=[], states_list=[], ignore_nonterminal=False)

    mock_state = mock.MagicMock()
    mock_state.is_terminal.return_value = False
    with self.assertRaisesRegex(
        ValueError, 'The number of allowed states to choose is 0'):
      tree.max_reward_and_state(
          reward_values=[42.],
          states_list=[mock_state],
          ignore_nonterminal=True)


class BackPropagationTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(BackPropagationTest, self).setUp()
    # Since back propagration will not affect state, the states of each node are
    # set as None for simplicity.
    #
    #                 root
    #                 /  \
    #            child1  child2
    #             /  \
    #  grandchild1   grandchild2
    root = tree.Node(None)
    child1 = tree.Node(None)
    root.add_child(child1)
    child2 = tree.Node(None)
    root.add_child(child2)
    grandchild1 = tree.Node(None)
    child1.add_child(grandchild1)
    grandchild2 = tree.Node(None)
    child1.add_child(grandchild2)
    self.root = root
    self.child1 = child1
    self.child2 = child2
    self.grandchild1 = grandchild1
    self.grandchild2 = grandchild2

  def test_back_propagation_add(self):
    # First back propapate the reward on grandchild1.
    tree.back_propagation(self.grandchild1, 1., update_method='add')
    # Only nodes on lineage:
    # grandchild1 -- child1 -- root will be updated.
    self.assertEqual(self.grandchild1.visits, 1)
    self.assertAlmostEqual(self.grandchild1.quality, 1.)
    self.assertEqual(self.child1.visits, 1)
    self.assertAlmostEqual(self.child1.quality, 1.)
    self.assertEqual(self.root.visits, 1)
    self.assertAlmostEqual(self.root.quality, 1.)
    # Other nodes will not be affected.
    self.assertEqual(self.grandchild2.visits, 0)
    self.assertAlmostEqual(self.grandchild2.quality, 0.)
    self.assertEqual(self.child2.visits, 0)
    self.assertAlmostEqual(self.child2.quality, 0.)

    # Then back propapate the reward on child2.
    tree.back_propagation(self.child2, 9., update_method='add')
    # Only nodes on lineage:
    # child2 -- root will be updated.
    self.assertEqual(self.child2.visits, 1)
    self.assertAlmostEqual(self.child2.quality, 9.)
    self.assertEqual(self.root.visits, 2)
    self.assertAlmostEqual(self.root.quality, 10.)
    # Other nodes will not be affected.
    self.assertEqual(self.grandchild1.visits, 1)
    self.assertAlmostEqual(self.grandchild1.quality, 1.)
    self.assertEqual(self.grandchild2.visits, 0)
    self.assertAlmostEqual(self.grandchild2.quality, 0.)
    self.assertEqual(self.child1.visits, 1)
    self.assertAlmostEqual(self.child1.quality, 1.)

  def test_back_propagation_max(self):
    # First back propapate the reward on grandchild1.
    tree.back_propagation(self.grandchild1, 1., update_method='max')
    # Only nodes on lineage:
    # grandchild1 -- child1 -- root will be updated.
    self.assertEqual(self.grandchild1.visits, 1)
    self.assertAlmostEqual(self.grandchild1.quality, 1.)
    self.assertEqual(self.child1.visits, 1)
    self.assertAlmostEqual(self.child1.quality, 1.)
    self.assertEqual(self.root.visits, 1)
    self.assertAlmostEqual(self.root.quality, 1.)
    # Other nodes will not be affected.
    self.assertEqual(self.grandchild2.visits, 0)
    self.assertAlmostEqual(self.grandchild2.quality, 0.)
    self.assertEqual(self.child2.visits, 0)
    self.assertAlmostEqual(self.child2.quality, 0.)

    # Then back propapate the reward on child2.
    tree.back_propagation(self.child2, 9., update_method='max')
    # Only nodes on lineage:
    # child2 -- root will be updated.
    self.assertEqual(self.child2.visits, 1)
    self.assertAlmostEqual(self.child2.quality, 9.)
    self.assertEqual(self.root.visits, 2)
    self.assertAlmostEqual(self.root.quality, 9.)
    # Other nodes will not be affected.
    self.assertEqual(self.grandchild1.visits, 1)
    self.assertAlmostEqual(self.grandchild1.quality, 1.)
    self.assertEqual(self.grandchild2.visits, 0)
    self.assertAlmostEqual(self.grandchild2.quality, 0.)
    self.assertEqual(self.child1.visits, 1)
    self.assertAlmostEqual(self.child1.quality, 1.)

  @parameterized.parameters([(np.nan, 'max'),
                             (np.inf, 'max'),
                             (-np.inf, 'max'),
                             (np.nan, 'add'),
                             (np.inf, 'add'),
                             (-np.inf, 'add')])
  def test_back_propagation_reward_value_not_finite(
      self, reward_value, update_method):
    # Back propapate the reward on grandchild1.
    tree.back_propagation(
        self.grandchild1, reward_value, update_method=update_method)
    # Nodes on lineage
    # grandchild1 -- child1 -- root
    # will not be affected since the back propagation step is skipped:
    self.assertEqual(self.grandchild1.visits, 0)
    self.assertAlmostEqual(self.grandchild1.quality, 0.)
    self.assertEqual(self.child1.visits, 0)
    self.assertAlmostEqual(self.child1.quality, 0.)
    self.assertEqual(self.root.visits, 0)
    self.assertAlmostEqual(self.root.quality, 0.)
    # Other nodes will not be affected.
    self.assertEqual(self.grandchild2.visits, 0)
    self.assertAlmostEqual(self.grandchild2.quality, 0.)
    self.assertEqual(self.child2.visits, 0)
    self.assertAlmostEqual(self.child2.quality, 0.)


class ProbsRemoveNaNTest(tf.test.TestCase):

  def test_probs_remove_nan_all_nan(self):
    with self.assertRaisesRegexp(ValueError,
                                 'All the elements in probs are nan.'):
      tree.probs_remove_nan(np.array([np.nan, np.nan]))

  def test_probs_remove_nan_no_nan(self):
    np.testing.assert_allclose(
        tree.probs_remove_nan(np.array([0.1, 0.1])), [0.5, 0.5])

  def test_probs_remove_nan(self):
    np.testing.assert_allclose(
        tree.probs_remove_nan(np.array([0.1, 0.1, np.nan])), [0.5, 0.5, 0.])


if __name__ == '__main__':
  tf.test.main()
