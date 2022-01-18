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

"""Monte Carlo Tree Search (MCTS) algorithm.

This module defines the node and operations for MCTS.

Introduction to Monte Carlo Tree Search:
https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

Each trial of Monte Carlo tree search consists of four steps:

* Selection: Start from root and select successive child nodes down to a leaf
    node.
* Expansion: Unless the selected leaf node is terminal state, expand the
    selected node.
* Simulation: Start from the current state in the node, recursively simulate to
    the next state until terminal state. Get the reward score from the terminal
    state. This step is sometimes also called rollout. It is worth to note that
    another choice is to use a heuristic function or a neural network to
    evaluate the reward score of a non-terminal state without rollout.
    For example,

    "Mastering the game of Go with deep neural networks and tree search",
    Nature 2016
    Used a mix of rollout and reward directly from neural network as the
    simulation result.

    "Mastering the game of Go without human knowledge", Nature 2017
    Used a neural network alone to get the simulation result without rollout.

* Backpropagation: Update the reward score from the leaf node started the
    simulation to the parent node recursively.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np


class Node(object):
  """Node of Monte Carlo tree.

  Attributes:
    state: mcts.states.StateBase object. Records all the information of
        expression generation.
    parent: Node object. The parent node of this node in Monte Carlo tree.
    children: List of Node objects. The children nodes of this node in Monte
        Carlo tree.
    visits: Integer, the number this node has been visited.
    quality: Float, the accumulated reward of this node.
    prior: Float, the prior probability of reaching this node from its parent
        node. Must be in the range [0., 1.] or None. If prior is np.nan, it is
        equivalent to add a mask on this node. This node will not be selected in
        the selection step.
  """

  def __init__(self, state, prior=1.):
    """Initializer.

    Args:
      state: mcts.states.StateBase object, recording the state of this node.
      prior: Float, the prior probability of reaching this node from its parent
          node. Must be in the range [0., 1.] or nan. If prior is np.nan, it is
          equivalent to add a mask on this node. This node will not be selected
          in the selection step.

    Raises:
      ValueError: If prior is not nan and out of the range [0, 1].
    """
    self.state = state
    # The parent node of this node in Monte Carlo tree.
    self.parent = None
    self.children = []
    self.visits = 0
    self.quality = 0.
    if not np.isnan(prior) and not 0 <= prior <= 1:
      raise ValueError(
          'prior must be nan or in the range [0, 1], but got %4.2f' % prior)
    self.prior = prior
    logging.info('Create %s', self)

  def get_ratio(self):
    """Gets the quality / visits ratio.

    Returns:
      Float.
    """
    if self.visits == 0:
      return 0.
    else:
      return self.quality / self.visits

  def update(self, reward_value, update_method='add'):
    """Updates the quality value and number of visits.

    Args:
      reward_value: Float add to quality value.
      update_method: String, how the quality is updated. {'add', 'max'}.
          'add': quality = quality + reward_value
          'max': quality = max(quality, reward_value)

    Raises:
      ValueError: If how is not in {'add', 'max'}.
    """
    self.visits += 1
    if update_method == 'add':
      self.quality += reward_value
    elif update_method == 'max':
      self.quality = max(self.quality, reward_value)
    else:
      raise ValueError(
          'update_method is expected to be in {\'add\',\'max\'}, '
          'but got %s' % update_method)

  def set_parent(self, parent_node):
    """Sets the parent node.

    Args:
      parent_node: Node object.

    Raises:
      ValueError: If the parent node already exists.
    """
    if self.parent is not None:
      raise ValueError('Try to set parent node but the current node %s already '
                       'has a parent node %s.'
                       % (str(self), str(self.parent)))
    self.parent = parent_node

  def reset_parent(self):
    """Resets the parent to None."""
    self.parent = None

  def add_child(self, child_node):
    """Adds child node.

    Args:
      child_node: Node object.
    """
    child_node.set_parent(self)
    self.children.append(child_node)

  def __repr__(self):
    """Defines behavior for when repr() is called on an instance of this class.

    Returns:
      String.
    """
    return ('Node [prior: %4.2f, quality / visits: %4.2f / %d, State: %s]'
            % (self.prior, self.quality, self.visits, str(self.state)))


def puct_alphago_score(node, c):
  """Scores node by a variant of PUCT algorithm used in Alpha Go.

  score = quality / visits
          + c * prior * sqrt(parent_total_visits) / (1 + visits)

  The score is used to compare the input node with its brother nodes from the
  same parent node, to decide which child nodes of this parent node will be
  selected. This search control strategy initially prefers actions with high
  prior probability and low visits, but asympotically prefers child nodes with
  high quality / visits ratio.

  This algorithm is used in AlphaGo Zero.
  https://www.nature.com/articles/nature24270

  Comparing to the classic UCT algorithm, the exploration term in this variant
  decays fast due since there is no sqrt on the denominator (1 + visits).

  Args:
    node: Node object.
    c: Float, a constant determining the level of exploration. Larger
        value will prefer more exploration. Range [0, inf).

  Returns:
    Float, the score from PUCT algorithm.
  """
  return node.get_ratio() + c * node.prior * np.sqrt(
      node.parent.visits) / (1 + node.visits)


def uct_score(node, c):
  """Scores node by UCT algorithm.

  score = quality / visits + c * sqrt(ln(parent_total_visits) / (1 + visits))

  See "Exploration and exploitation" section in
  https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

  Notice the denominator in the sqrt() is 1 + visits rather than visits in the
  wikipedia. This change is to smooth and avoid error of the UCT score for node
  with visits = 0.

  Args:
    node: Node object.
    c: Float, a constant determining the level of exploration. Larger
        value will prefer more exploration. Range [0, inf).

  Returns:
    Float, the score from UCT algorithm.
  """
  return node.get_ratio() + c * np.sqrt(
      np.log(node.parent.visits) / (1 + node.visits))


def _get_max_values_indices(array):
  """Gets the indices of the maximum values in 1d array.

  Args:
    array: Numpy array.

  Returns:
    Numpy array of the indices of maximum values.
  """
  return np.nonzero(array == np.amax(array))[0]


def random_argmax(array, random_state=None):
  """Returns the indices of the maximum values in 1d array.

  For numpy.argmax(), in case of multiple occurrences of the maximum values, the
  index corresponding to the first occurrence are returned. This can be biased
  since smaller indices are preferred by numpy.argmax().

  random_argmax() will randomly select an index of maximum value in case of
  multiple occurrences of the maximum values.

  Args:
    array: Numpy array.
    random_state: np.random.RandomState object.

  Returns:
    An index of maximum value.
  """
  if random_state is None:
    random_state = np.random.RandomState()
  return random_state.choice(_get_max_values_indices(array))


def max_reward_and_state(reward_values,
                         states_list,
                         ignore_nonterminal=False,
                         random_state=None):
  """Gets the maximum reward value and its corresponding state.

  If there are multiple states with maximum reward value, one of them will be
  returned.

  Args:
    reward_values: List of float numbers. The reward values for input states.
    states_list: List of mcts.states.SymbolsState objects.
    ignore_nonterminal: Boolean, whether to ignore nonterminal states while
        getting the maximum reward and its corresponding states.
    random_state: np.random.RandomState object.

  Returns:
    max_reward_value: Float, the maximum reward value.
    max_state: A mcts.states.SymbolsState object, the corresponding state of
        max_reward_value.

  Raises:
    ValueError: If the length of reward_values and states does not match, or the
        number of allowed states to choose is 0.
  """
  if len(reward_values) != len(states_list):
    raise ValueError('The length of reward_values (%d) does not match '
                     'the length of states_list (%d).'
                     % (len(reward_values), len(states_list)))

  allowed_reward_values = []
  allowed_states_list = []
  for reward_value, state in zip(reward_values, states_list):
    if ignore_nonterminal and not state.is_terminal():
      continue
    else:
      allowed_reward_values.append(reward_value)
      allowed_states_list.append(state)

  if not allowed_states_list:
    raise ValueError('The number of allowed states to choose is 0.')

  max_index = random_argmax(allowed_reward_values, random_state)
  return allowed_reward_values[max_index], allowed_states_list[max_index]


def selection(node, score_function, random_state=None):
  """Selection step in the Monte Carlo Tree Search trial.

  While the current node is not a leaf node, visits one of its child node with
  highest score from score_function. The score function balanced the opportunity
  of exploitation and exploration.

  See "Exploration and exploitation" section in
  https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

  Args:
    node: Node object.
    score_function: Callable, taking an Node object as single input argument
        and returning the score for selection.
    random_state: np.random.RandomState object.

  Returns:
    node: Node object. The selected node.

  Raises:
    ValueError: If the prior of input node is nan.
  """
  if np.isnan(node.prior):
    raise ValueError('The prior of the input node is nan.')
  # Leaf node will have no children.
  while node.children:
    # Returns the current node if the priors of all its children are nan.
    if all(np.isnan(child_node.prior) for child_node in node.children):
      return node
    # Only child nodes with prior not nan will be selected.
    allowed_child_scores = []
    allowed_children = []
    for child_node in node.children:
      if np.isfinite(child_node.prior):
        allowed_child_scores.append(score_function(child_node))
        allowed_children.append(child_node)
    node = allowed_children[random_argmax(allowed_child_scores, random_state)]

  logging.info('Select %s', node)
  return node


def expansion(node, policy):
  """Expansion step in the Monte Carlo Tree Search trial.

  If the current node contains a non-terminal state, expand the current node by
  creating all the child nodes, each of which contains a possible new state for
  next step.

  Args:
    node: Node object.
    policy: policies.PolicyBase object. The policy used for expansion.

  Raises:
    ValueError: If node already has children.
  """
  if node.children:
    raise ValueError('Input node is expected to have no child '
                     'but got %d children.' % len(node.children))
  current_state = node.state
  if not current_state.is_terminal():
    for new_state, prior in zip(*policy.get_new_states_probs(current_state)):
      node.add_child(Node(state=new_state, prior=prior))


def simulation(node, reward,
               policy=None, rollout_limit=None, random_state=None):
  """Simulation step in the Monte Carlo Tree Search trial.

  Note the simulation step will not create new node under the input node. The
  simulation starts at the state of the input node. New states will be created
  in the simulation until reach terminal state. The reward value of the terminal
  state will be used to update on the input node (quality and visits) as the
  result of this simulation, but the state of this input node will stay
  unchanged before and after the simulation.

  Args:
    node: Node object.
    reward: rewards.RewardBase object. Its evaluate() method is called to
        evaluate the reward of the state.
    policy: policies.PolicyBase object. The policy used for rollout.
        Rollout will repeatly evolve the state until it is terminal or the
        rollout limit is reached. Then reward object is used to get the reward
        value from the finished state. Default None for no rollout. In this
        case, the reward object must be able to evaluate the reward value from
        non-terminal state.
    rollout_limit: Integer or None. The maximum steps for rollout. Default None,
        continue rollout until terminal state.
    random_state: np.random.RandomState object.

  Returns:
    reward_value: Float, the reward of the finished state from the simulation in
        this trial.
    finished_state: mcts.states.SymbolsState object, the finished state in the
        simulation in this trial. Note the finished state may be nonterminal due
        to rollout limit.

  Raises:
    ValueError: If rollout_limit is negative.
  """
  if random_state is None:
    random_state = np.random.RandomState()

  if rollout_limit is not None and rollout_limit <= 0:
    raise ValueError('rollout_limit (%d) must be positive.' % rollout_limit)

  current_state = node.state
  logging.info('Simulation starts with %s', current_state)

  if policy is not None:
    num_steps = 0
    while not current_state.is_terminal():
      if rollout_limit is not None and num_steps == rollout_limit:
        logging.warning('rollout_limit (%d) is reached.', rollout_limit)
        break
      new_states, probs = policy.get_new_states_probs(current_state)
      # Stop rolling out if all new states are forbidden.
      if np.all(np.isnan(probs)):
        break
      # Convert nan to 0.
      current_state = random_state.choice(new_states, p=probs_remove_nan(probs))
      num_steps += 1
  # NOTE(leeley): The reward object will deal with the situation when
  # current_state is terminal or non-terminal. So whether current_state is
  # terminal is not checked here.
  reward_value = reward.evaluate(current_state)
  logging.info('Simulation finish at %s with %f', current_state, reward_value)

  return reward_value, current_state


def back_propagation(node, reward_value, update_method='add'):
  """Back propagation step in Monte Carlo Tree Search trial.

  Back propagation will update the visits and quality attributes of node object
  by update() method for all the ancestors of the input node. The states in
  nodes will not be modified in the back propagation step.

  Args:
    node: Node object.
    reward_value: Float, the reward value to propagate back to the ancestors of
        node.
    update_method: String, how the quality in each tree node is updated.
       {'add', 'max'}. This is passed to the update() method of node.
  """
  # NOTE(leeley): The reward object should take care of the evaluation of
  # reward value. There are two situations that the reward_value is not finite:
  #   * The simulation is invalid and we want to ignore it.
  #   * The simulation is valid. Although the evaluator in the reward object
  #         should ensure the output is a finite number. It takes care of
  #         special cases like dividing by zero. However, in some rares cases,
  #         the accumulation of numerical errors will cause problems. If this
  #         happens, I don't want to back propagate nan or inf to the parents.
  if np.isfinite(reward_value):
    while node is not None:
      node.update(reward_value, update_method)
      node = node.parent
  else:
    logging.warning('back propagation step on %s is skipped because '
                    'reward_value (%s) is not finite.',
                    str(node), str(reward_value))


def trial(node,
          score_function,
          expansion_policy,
          reward,
          rollout_policy=None,
          rollout_limit=None,
          update_method='add',
          random_state=None):
  """One trial of selection -> expansion -> simulation -> back propagation.

  This trial will create new nodes in the tree and update the visits and
  quality of the existing nodes.

  Args:
    node: Node object. This Monte Carlo Tree Search trial starts from this node.
    score_function: Callable, taking an Node object as single input argument
        and returning the score for selection.
    expansion_policy: policies.PolicyBase object. The policy used for
        expansion.
    reward: rewards.RewardBase object. Its evaluate() method is called to
        evaluate the reward of the state.
    rollout_policy: policies.PolicyBase object. The policy used for rollout.
        Rollout will repeatly evolve the state until it is terminal or the
        rollout limit is reached. Then reward object is used to get the reward
        value from the finished state. Default None for no rollout. In this
        case, the reward object must be able to evaluate the reward value from
        non-terminal state.
    rollout_limit: Integer or None. The maximum steps for rollout. Default
        None, continue rollout until terminal state.
    update_method: String, how the quality in each tree node is updated.
        {'add', 'max'}.
    random_state: np.random.RandomState object.

  Returns:
    reward_value: Float, the reward of the finished state from the simulation in
        this trial.
    finished_state: mcts.states.SymbolsState object, the finished state in the
        simulation in this trial. Note the finished state may be nonterminal due
        to rollout limit.
  """
  if random_state is None:
    random_state = np.random.RandomState()
  # Selection.
  selected_node = selection(node, score_function, random_state)
  # Expansion. Note the next simulation step starts at the state of
  # selected_node, not its children created in the expansion step.
  expansion(selected_node, expansion_policy)
  # Simulation.
  reward_value, finished_state = simulation(selected_node,
                                            reward,
                                            rollout_policy,
                                            rollout_limit,
                                            random_state)
  # Back propagation.
  back_propagation(selected_node, reward_value, update_method)
  return reward_value, finished_state


def repeat_trials(num_trials,
                  node,
                  score_function,
                  expansion_policy,
                  reward,
                  rollout_policy=None,
                  rollout_limit=None,
                  update_method='add',
                  random_state=None,
                  tuner=None,
                  report_measure_interval=None):
  """Repeats MCTS trials num_trials times while keeping the tree statistics.

  Args:
    num_trials: Integer, the number of trials before making a move.
    node: Node object. This Monte Carlo Tree Search trial starts from this node.
    score_function: Callable, taking an Node object as single input argument
        and returning the score for selection.
    expansion_policy: policies.PolicyBase object. The policy used for
        expansion.
    reward: rewards.RewardBase object. Its evaluate() method is called to
        evaluate the reward of the state.
    rollout_policy: policies.PolicyBase object. The policy used for rollout.
        Rollout will repeatly evolve the state until it is terminal or the
        rollout limit is reached. Then reward object is used to get the reward
        value from the finished state. Default None for no rollout. In this
        case, the reward object must be able to evaluate the reward value from
        non-terminal state.
    rollout_limit: Integer or None. The maximum steps for rollout. Default
        None, continue rollout until terminal state.
    update_method: String, how the quality in each tree node is updated.
        {'add', 'max'}.
    random_state: np.random.RandomState object.
    tuner: HPTuner. Used for Vizier study.
    report_measure_interval: Integer, after every report_measure_interval of
        trials, the current maximum reward value will be report to the tuner as
        a intermediate measure to vizier. Used when tuner is not None. Default
        report_measure_interval = int(num_trials / 10).

  Returns:
    reward_values: List of float numbers, the reward value of the finished state
        from the simulation in each trial.
    finished_states: List of mcts.states.SymbolsState objects, the finished
        state in the simulation in each trial. Note the finished state may be
        nonterminal due to rollout limit.
  """
  if tuner is not None and report_measure_interval is None:
    report_measure_interval = int(num_trials / 10)
  reward_values = []
  finished_states = []
  for i in range(num_trials):
    reward_value, finished_state = trial(
        node=node,
        score_function=score_function,
        expansion_policy=expansion_policy,
        reward=reward,
        rollout_policy=rollout_policy,
        rollout_limit=rollout_limit,
        update_method=update_method,
        random_state=random_state)
    reward_values.append(reward_value)
    finished_states.append(finished_state)
    if tuner and i % report_measure_interval == 0:
      # NOTE(leeley): global_step must be strictly positive so I set
      # global_step=i + 1.
      tuner.report_measure(np.amax(reward_values), global_step=i + 1)
  return reward_values, finished_states


def probs_remove_nan(probs):
  """Replaced nan in probs to zero and normalize the probabilities.

  This function replaces nan to zero and normalizes the probabilities.
  This step is essential if the probs is used as argument in np.random.choice().
  An error will be raised if probs are not summed to one.

  Args:
    probs: Numpy array. Probabilities.

  Returns:
    Numpy array.

  Raises:
    ValueError: If all the elements in probs are nan.
  """
  if np.all(np.isnan(probs)):
    raise ValueError('All the elements in probs are nan.')
  probs = np.nan_to_num(probs)
  return probs / np.sum(probs)
