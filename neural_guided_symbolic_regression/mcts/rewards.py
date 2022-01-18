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

"""Classes to compute the rewards of state in Monte Carlo Tree Search.

The classes in this module are reward functions used to evaluate the reward
value of a state in the node of Monte Carlo tree.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np


class RewardBase(object):
  """Base class to evaluate the reward of state.

  Subclasses should define the following method:
    * _evaluate
  """

  def __init__(self,
               post_transformer=None,
               allow_nonterminal=False,
               default_value=None):
    """Initializer.

    Args:
      post_transformer: Callable. This function takes one float number and
          output a float number as the transformed value of input. It is used
          to post-transformation the reward evaluated on a state. Default None,
          no post-transformation will be applied.
      allow_nonterminal: Boolean, if False, ValueError will be raised when
          list of symbols to evaluate contains non-terminal symbol and
          default_value is None. Default False.
      default_value: Float, if allow_nonterminal is False and non-terminal
          symbol exists, instead of raising a ValueError, return default_value
          as the reward value.
    """
    self._allow_nonterminal = allow_nonterminal
    self.set_default_value(default_value)
    self.set_post_transformer(post_transformer)

  def set_default_value(self, default_value):
    """Sets default value if nonterminal is allowed for evaluation.

    Args:
      default_value: Float, if allow_nonterminal is False and non-terminal
          symbol exists, instead of raising a ValueError, return default_value
          as the reward value.
    """
    if default_value is not None:
      default_value = float(default_value)
    self._default_value = default_value

  def set_post_transformer(self, post_transformer):
    """Sets post transformer.

    Args:
      post_transformer: Callable. This function takes one float number and
          output a float number as the transformed value of input. It is used
          to post-transformation the reward evaluated on a state. Default None,
          no post-transformation will be applied.

    Raises:
      TypeError: If post_transformer is not callable.
    """
    if post_transformer is not None and not callable(post_transformer):
      raise TypeError('post_transformer is not callable.')
    self._post_transformer = post_transformer

  def _evaluate(self, state):
    """Evaluates the reward from input state.

    Args:
      state: mcts.states.StateBase object. Records all the information of
          a state.

    Returns:
      Float, the reward of the current state.
    """
    raise NotImplementedError('Must be implemented by subclass.')

  def evaluate(self, state):
    """Evaluates the reward from input state.

    Args:
      state: mcts.states.StateBase object. Records all the information of
          a state.

    Returns:
      Float, the reward of the current state.

    Raises:
      ValueError: If allow_nonterminal is False and default_value is None, but
          state is not terminal.
    """
    # Check whether nonterminal state is allowed.
    if not self._allow_nonterminal and not state.is_terminal():
      if self._default_value is not None:
        logging.info(
            '%s is not terminal, use default_value (%5.3f) as reward_value.',
            state, self._default_value)
        reward_value = self._default_value
      else:
        raise ValueError('allow_nonterminal is False and default_value is '
                         'None, but state is not terminal: %s' % state)
    else:
      reward_value = self._evaluate(state)

    if not np.isfinite(reward_value):
      logging.warning('reward_value (%s) for input state %s is not finite.',
                      str(reward_value), str(state))

    # Add post transformer.
    if self._post_transformer is not None:
      reward_value = self._post_transformer(reward_value)

    return reward_value
