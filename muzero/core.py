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

# Lint as: python3
# pylint: disable=missing-docstring
# pylint: disable=g-complex-comprehension
"""MuZero Core.

Based partially on https://arxiv.org/src/1911.08265v1/anc/pseudocode.py
"""

import collections
import logging
import math
from typing import List, Optional, Dict, Any, Tuple

from absl import flags
import attr
import gym
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS
MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', 'min max')

NetworkOutput = collections.namedtuple(
    'NetworkOutput',
    'value value_logits reward reward_logits policy_logits hidden_state')

Prediction = collections.namedtuple(
    'Prediction',
    'gradient_scale value value_logits reward reward_logits policy_logits')

Target = collections.namedtuple(
    'Target', 'value_mask reward_mask policy_mask value reward visits')

Range = collections.namedtuple('Range', 'low high')


class RLEnvironmentError(Exception):
  pass


class BadSupervisedEpisodeError(Exception):
  pass


class SkipEpisode(Exception):  # pylint: disable=g-bad-exception-name
  pass


class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value):
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      value = (value - self.minimum) / (self.maximum - self.minimum)
    value = max(min(1.0, value), 0.0)
    return value


class MuZeroConfig:
  """Config object for MuZero."""

  def __init__(self,
               action_space_size,
               max_moves,
               recurrent_inference_batch_size,
               initial_inference_batch_size,
               train_batch_size,
               discount = 0.99,
               dirichlet_alpha = 0.25,
               root_exploration_fraction = 0.25,
               num_simulations = 11,
               td_steps = 5,
               num_unroll_steps = 5,
               pb_c_base = 19652,
               pb_c_init = 1.25,
               visit_softmax_temperature_fn=None,
               known_bounds = None,
               use_softmax_for_action_selection = False,
               parent_base_visit_count=1,
               max_num_action_expansion = 0):

    ### Play
    self.action_space_size = action_space_size

    self.visit_softmax_temperature_fn = (visit_softmax_temperature_fn
                                         if visit_softmax_temperature_fn
                                         is not None else lambda a, b, c: 1.0)
    self.max_moves = max_moves
    self.num_simulations = num_simulations
    self.discount = discount

    # Root prior exploration noise.
    self.root_dirichlet_alpha = dirichlet_alpha
    self.root_exploration_fraction = root_exploration_fraction

    # UCB formula
    self.pb_c_base = pb_c_base
    self.pb_c_init = pb_c_init

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    self.known_bounds = known_bounds

    ### Training
    self.recurrent_inference_batch_size = recurrent_inference_batch_size
    self.initial_inference_batch_size = initial_inference_batch_size
    self.train_batch_size = train_batch_size
    self.num_unroll_steps = num_unroll_steps
    self.td_steps = td_steps

    self.use_softmax_for_action_selection = use_softmax_for_action_selection

    # This is 0 in the MuZero paper.
    self.parent_base_visit_count = parent_base_visit_count
    self.max_num_action_expansion = max_num_action_expansion

  def new_episode(self, environment, index=None):
    return Episode(
        environment, self.action_space_size, self.discount, index=index)


Action = np.int64  # pylint: disable=invalid-name


class TransitionModel:
  """Transition model providing additional information for MCTS transitions.

  An environment can provide a specialized version of a transition model via the
  info dict. This model then provides additional information, e.g. on the legal
  actions, between transitions in the MCTS.
  """

  def __init__(self, full_action_space_size):
    self.full_action_space_size = full_action_space_size

  def legal_actions_after_sequence(self,
                                   actions_sequence):  # pylint: disable=unused-argument
    """Returns the legal action space after a sequence of actions."""
    return range(self.full_action_space_size)

  def full_action_space(self):
    return range(self.full_action_space_size)

  def legal_actions_mask_after_sequence(self,
                                        actions_sequence):
    """Returns the legal action space after a sequence of actions as a mask."""
    mask = np.zeros(self.full_action_space_size, dtype=np.int64)
    for action in self.legal_actions_after_sequence(actions_sequence):
      mask[action] = 1
    return mask


class Node:
  """Node for MCTS."""

  def __init__(self, prior, config, is_root=False):
    self.visit_count = 0
    self.prior = prior
    self.is_root = is_root
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0
    self.discount = config.discount

  def expanded(self):
    return len(self.children) > 0  # pylint: disable=g-explicit-length-test

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

  def qvalue(self):
    return self.discount * self.value() + self.reward


class ActionHistory:
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history, action_space_size):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action):
    self.history.append(Action(action))

  def last_action(self):
    return self.history[-1]

  def action_space(self):
    return [Action(i) for i in range(self.action_space_size)]


class Episode:
  """A single episode of interaction with the environment."""

  def __init__(self,
               environment,
               action_space_size,
               discount,
               index=None):
    self.environment = environment
    self.history = []
    self.observations = []
    self.rewards = []
    self.child_visits = []
    self.root_values = []
    self.mcts_visualizations = []
    self.action_space_size = action_space_size
    self.discount = discount
    self.failed = False

    if index is None:
      self._observation, self._info = self.environment.reset()
    else:
      self._observation, self._info = self.environment.reset(index)
    self.observations.append(self._observation)
    self._reward = None
    self._done = False

  def terminal(self):
    return self._done

  def get_info(self, kword):
    if not self._info:
      return None
    return self._info.get(kword, None)

  def total_reward(self):
    return sum(self.rewards)

  def __len__(self):
    return len(self.history)

  def special_statistics(self):
    try:
      return self.environment.special_episode_statistics()
    except AttributeError:
      return {}

  def special_statistics_learner(self):
    try:
      return self.environment.special_episode_statistics_learner()
    except AttributeError:
      return ()

  def visualize_mcts(self, root):
    history = self.action_history().history
    try:
      treestr = self.environment.visualize_mcts(root, history)
    except AttributeError:
      treestr = ''
    if treestr:
      self.mcts_visualizations.append(treestr)

  def legal_actions(self,
                    actions_sequence = None
                   ):
    """Returns the legal actions after an actions sequence.

    Args:
      actions_sequence: Past sequence of actions.

    Returns:
      A list of full_action_space size. At each index a 1 corresponds to a legal
      action and a 0 to an illegal action.
    """
    transition_model = self.get_info('transition_model') or TransitionModel(
        self.action_space_size)
    actions_sequence = tuple(actions_sequence or [])
    return transition_model.legal_actions_mask_after_sequence(actions_sequence)

  def apply(self, action, training_steps = 0):
    (self._observation, self._reward, self._done,
     self._info) = self.environment.step(
         action, training_steps=training_steps)
    self.rewards.append(self._reward)
    self.history.append(action)
    self.observations.append(self._observation)

  def history_range(self, start, end):
    rng = []
    for i in range(start, end):
      if i < len(self.history):
        rng.append(self.history[i])
      else:
        rng.append(0)
    return np.array(rng, np.int64)

  def store_search_statistics(self, root, use_softmax=False):
    sum_visits = sum(child.visit_count for child in root.children.values())
    sum_visits = max(sum_visits, 1)
    action_space = (Action(index) for index in range(self.action_space_size))
    if use_softmax:
      child_visits, mask = zip(*[(root.children[a].visit_count,
                                  1) if a in root.children else (0, 0)
                                 for a in action_space])
      child_visits_distribution = masked_softmax(child_visits, mask)
    else:
      child_visits_distribution = [
          root.children[a].visit_count / sum_visits if a in root.children else 0
          for a in action_space
      ]

    self.child_visits.append(child_visits_distribution)
    self.root_values.append(root.value())

  def make_image(self, state_index):
    if state_index == -1 or state_index < len(self.observations):
      return self.observations[state_index]
    return self._observation

  @staticmethod
  def make_target(state_index,
                  num_unroll_steps,
                  td_steps,
                  rewards,
                  policy_distributions,
                  discount,
                  value_approximations = None):
    num_steps = len(rewards)
    if td_steps == -1:
      td_steps = num_steps  # for sure go to the end of the game

    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < num_steps and value_approximations is not None:
        value = value_approximations[bootstrap_index] * discount**td_steps
      else:
        value = 0

      for i, reward in enumerate(rewards[current_index:bootstrap_index]):
        value += reward * discount**i  # pytype: disable=unsupported-operands

      reward_mask = 1.0 if current_index > state_index else 0.0
      if current_index < num_steps:
        targets.append(
            (1.0, reward_mask, 1.0, value, rewards[current_index - 1],
             policy_distributions[current_index]))
      elif current_index == num_steps:
        targets.append((1.0, reward_mask, 0.0, 0.0, rewards[current_index - 1],
                        policy_distributions[0]))
      else:
        # States past the end of games are treated as absorbing states.
        targets.append((1.0, 0.0, 0.0, 0.0, 0.0, policy_distributions[0]))
    target = Target(*zip(*targets))
    return target

  def action_history(self):
    return ActionHistory(self.history, self.action_space_size)


def prepare_root_node(config, legal_actions,
                      initial_inference_output):
  root = Node(0, config, is_root=True)
  expand_node(root, legal_actions, initial_inference_output, config)
  add_exploration_noise(config, root)
  return root


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config,
             root,
             action_history,
             legal_actions_fn,
             recurrent_inference_fn,
             visualization_fn=None):
  min_max_stats = MinMaxStats(config.known_bounds)

  for _ in range(config.num_simulations):
    history = action_history.clone()
    node = root
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node, min_max_stats)
      history.add_action(action)
      search_path.append(node)

    # Inside the search tree we use the dynamics function to obtain the next
    # hidden state given an action and the previous hidden state.
    parent = search_path[-2]
    network_output = recurrent_inference_fn(parent.hidden_state,
                                            history.last_action())
    legal_actions = legal_actions_fn(
        history.history[len(action_history.history):])
    expand_node(node, legal_actions, network_output, config)

    backpropagate(search_path, network_output.value, config.discount,
                  min_max_stats)

  if visualization_fn:
    visualization_fn(root)


def masked_distribution(x,
                        use_exp,
                        mask = None):
  if mask is None:
    mask = [1] * len(x)
  assert sum(mask) > 0, 'Not all values can be masked.'
  assert len(mask) == len(x), (
      'The dimensions of the mask and x need to be the same.')
  x = np.exp(x) if use_exp else np.array(x, dtype=np.float64)
  mask = np.array(mask, dtype=np.float64)
  x *= mask
  if sum(x) == 0:
    # No unmasked value has any weight. Use uniform distribution over unmasked
    # tokens.
    x = mask
  return x / np.sum(x, keepdims=True)


def masked_softmax(x, mask=None):
  x = np.array(x) - np.max(x, axis=-1)  # to avoid overflow
  return masked_distribution(x, use_exp=True, mask=mask)


def masked_count_distribution(x, mask=None):
  return masked_distribution(x, use_exp=False, mask=mask)


def histogram_sample(distribution,
                     temperature,
                     use_softmax=False,
                     mask=None):
  actions = [d[1] for d in distribution]
  visit_counts = np.array([d[0] for d in distribution], dtype=np.float64)
  if temperature == 0.:
    probs = masked_count_distribution(visit_counts, mask=mask)
    return actions[np.argmax(probs)]
  if use_softmax:
    logits = visit_counts / temperature
    probs = masked_softmax(logits, mask)
  else:
    logits = visit_counts**(1. / temperature)
    probs = masked_count_distribution(logits, mask)
  return np.random.choice(actions, p=probs)


def select_action(config,
                  num_moves,
                  node,
                  train_step,
                  use_softmax=False,
                  is_training=True):
  visit_counts = [
      (child.visit_count, action) for action, child in node.children.items()
  ]
  t = config.visit_softmax_temperature_fn(
      num_moves=num_moves, training_steps=train_step, is_training=is_training)
  action = histogram_sample(visit_counts, t, use_softmax=use_softmax)
  return action


# Select the child with the highest UCB score.
def select_child(config, node, min_max_stats):
  ucb_scores = [(ucb_score(config, node, child, min_max_stats), action, child)
                for action, child in node.children.items()]
  _, action, child = max(ucb_scores)
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config, parent, child,
              min_max_stats):
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count + config.parent_base_visit_count) / (
      child.visit_count + 1)

  prior_score = pb_c * child.prior
  if child.visit_count > 0:
    value_score = min_max_stats.normalize(child.qvalue())
  else:
    value_score = 0.
  return prior_score + value_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node, legal_actions,
                network_output, config):
  node.hidden_state = network_output.hidden_state
  node.reward = network_output.reward
  policy_probs = masked_softmax(
      network_output.policy_logits, mask=legal_actions.astype(np.float64))
  actions = np.where(legal_actions == 1)[0]

  if (config.max_num_action_expansion > 0 and
      len(actions) > config.max_num_action_expansion):
    # get indices of the max_num_action_expansion largest probabilities
    actions = np.argpartition(
        policy_probs,
        -config.max_num_action_expansion)[-config.max_num_action_expansion:]

  policy = {a: policy_probs[a] for a in actions}
  for action, p in policy.items():
    node.children[action] = Node(p, config)


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path, value, discount,
                  min_max_stats):
  for node in search_path[::-1]:
    node.value_sum += value
    node.visit_count += 1
    min_max_stats.update(node.qvalue())
    value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config, node):
  actions = list(node.children.keys())
  noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


class ValueEncoder:
  """Encoder for reward and value targets from Appendix of MuZero Paper."""

  def __init__(self,
               min_value,
               max_value,
               num_steps,
               use_contractive_mapping=True):
    if not max_value > min_value:
      raise ValueError('max_value must be > min_value')
    min_value = float(min_value)
    max_value = float(max_value)
    if use_contractive_mapping:
      max_value = contractive_mapping(max_value)
      min_value = contractive_mapping(min_value)
    if num_steps <= 0:
      num_steps = int(math.ceil(max_value) + 1 - math.floor(min_value))
    logging.info('Initializing ValueEncoder with range (%d, %d) and %d steps',
                 min_value, max_value, num_steps)
    self.min_value = min_value
    self.max_value = max_value
    self.value_range = max_value - min_value
    self.num_steps = num_steps
    self.step_size = self.value_range / (num_steps - 1)
    self.step_range_int = tf.range(self.num_steps, dtype=tf.int32)
    self.step_range_float = tf.cast(self.step_range_int, tf.float32)
    self.use_contractive_mapping = use_contractive_mapping

  def encode(self, value):
    if len(value.shape) != 1:
      raise ValueError(
          'Expected value to be 1D Tensor [batch_size], but got {}.'.format(
              value.shape))
    if self.use_contractive_mapping:
      value = contractive_mapping(value)
    value = tf.expand_dims(value, -1)
    clipped_value = tf.clip_by_value(value, self.min_value, self.max_value)
    above_min = clipped_value - self.min_value
    num_steps = above_min / self.step_size
    lower_step = tf.math.floor(num_steps)
    upper_mod = num_steps - lower_step
    lower_step = tf.cast(lower_step, tf.int32)
    upper_step = lower_step + 1
    lower_mod = 1.0 - upper_mod
    lower_encoding, upper_encoding = (
        tf.cast(tf.math.equal(step, self.step_range_int), tf.float32) * mod
        for step, mod in (
            (lower_step, lower_mod),
            (upper_step, upper_mod),
        ))
    return lower_encoding + upper_encoding

  def decode(self, logits):
    if len(logits.shape) != 2:
      raise ValueError(
          'Expected logits to be 2D Tensor [batch_size, steps], but got {}.'
          .format(logits.shape))
    num_steps = tf.reduce_sum(logits * self.step_range_float, -1)
    above_min = num_steps * self.step_size
    value = above_min + self.min_value
    if self.use_contractive_mapping:
      value = inverse_contractive_mapping(value)
    return value


# From the MuZero paper.
def contractive_mapping(x, eps=0.001):
  return tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1.) - 1.) + eps * x


# From the MuZero paper.
def inverse_contractive_mapping(x, eps=0.001):
  return tf.math.sign(x) * (
      tf.math.square(
          (tf.sqrt(4 * eps *
                   (tf.math.abs(x) + 1. + eps) + 1.) - 1.) / (2. * eps)) - 1.)


@attr.s(auto_attribs=True)
class EnvironmentDescriptor:
  """Descriptor for Environments."""
  observation_space: gym.spaces.Space
  action_space: gym.spaces.Space
  reward_range: Range
  value_range: Range
  pretraining_space: gym.spaces.Space = None
  extras: Dict[str, Any] = None
