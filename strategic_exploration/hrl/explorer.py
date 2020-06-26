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

import abc
import numpy as np
import random
from strategic_exploration.hrl.action import DefaultAction
from strategic_exploration.hrl.justification import Justification
from collections import Counter


class Explorer(object):
  """Defines an exploration scheme to find new abstract states."""
  __metaclass__ = abc.ABCMeta

  @classmethod
  def from_config(cls, config, num_actions):
    """Creates an Explorer from a Config."""
    if config.type == "uniform":
      factory = UniformRandomExplorer
    elif config.type == "repeat":
      factory = RepeatedActionsExplorer
    elif config.type == "mixture":
      factory = MixtureExplorer
    else:
      raise ValueError("{} is not a valid Explorer type".format(config.type))

    return factory.from_config(config, num_actions)

  @abc.abstractmethod
  def act(self, state):
    """Returns an Action and a string justification."""
    raise NotImplementedError()

  @abc.abstractmethod
  def activate(self, node):
    """Starts the exploration episode.

    Can be called again after
        exploration episode terminates to begin a new exploration episode.

        Args:
            node (AbstractNode): node at which exploration episode is invoked
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def active(self):
    """Returns True while the exploration episode is ongoing.

        Returns False before being activated or after exploration episode is
        done.
        """
    raise NotImplementedError()


class UniformRandomExplorer(Explorer):
  """Uniformly at random samples an action at each timestep.

  Starts
    exploration episode off by taking a random number of no-ops between 0 and
    value specified in config (inclusive).
    """

  @classmethod
  def from_config(cls, config, num_actions):
    # Make top inclusive
    no_ops = np.random.randint(0, config.no_ops + 1)
    return cls(config.exploration_horizon, num_actions, no_ops)

  def __init__(self, exploration_horizon, num_actions, no_ops):
    """Constructs.

        Args:
            exploration_horizon (int): number of steps in exploration episode
            num_actions (int): size of the action space
            no_ops (int): number of no-ops to start exploration episode with
    """
    self._exploration_horizon = exploration_horizon
    self._steps_left = 0  # Not yet active
    self._num_actions = num_actions
    self._no_ops = no_ops

  def act(self, state):
    if not self.active():
      raise ValueError("Exploration not active")

    self._steps_left -= 1
    if self._no_ops_left > 0:
      action = DefaultAction(0)
      s = "{} no-ops at {}: {} / {} visits, steps left {}".format(
          self._no_ops, self._node.uid, self._node.visit_count,
          self._node.min_visit_count, self._steps_left)
      self._no_ops_left -= 1
      return action, s

    action = DefaultAction(random.randint(0, self._num_actions - 1))
    s = "uniform random from {}: {} / {} visits, steps left {}".format(
        self._node.uid, self._node.visit_count, self._node.min_visit_count,
        self._steps_left)
    return action, s

  def activate(self, node):
    if self.active():
      raise ValueError("Exploration already active")

    self._node = node
    self._steps_left = self._exploration_horizon
    self._no_ops_left = self._no_ops

  def active(self):
    return self._steps_left > 0


class RepeatedActionsExplorer(Explorer):
  """Samples an action and number of timesteps to repeat the action for.

    Sampling is specified either as uniform or log uniform in config.
    """

  @classmethod
  def from_config(cls, config, num_actions):
    if config.log_uniform:
      sampler = lambda: np.exp(np.random.uniform(config.low, config.high))
    else:
      sampler = lambda: np.random.randint(config.low, config.high)
    discrete_sampler = lambda: int(sampler())
    return cls(config.exploration_horizon, num_actions, discrete_sampler)

  def __init__(self, exploration_horizon, num_actions, repeat_sampler):
    """
        Args:
            exploration_horizon (int): number of steps in exploration episode
            num_actions (int): size of the action space
            repeat_sampler (Callable): returns an int for the number of actions
              to repeat for
    """
    self._exploration_horizon = exploration_horizon
    self._steps_left = 0  # Not yet active
    self._num_actions = num_actions
    self._repeat_sampler = repeat_sampler

  def act(self, state):
    if not self.active():
      raise ValueError("RepeatedActionsExplorer not active")

    self._steps_left -= 1
    if self._repeat == 0:
      self._repeat = self._repeat_sampler()
      self._repeated_action = DefaultAction(
          random.randint(0, self._num_actions - 1))

    self._repeat -= 1
    s = "repeat {} random from {}: {} / {} visits, steps left {}".format(
        self._repeat, self._node.uid, self._node.visit_count,
        self._node.min_visit_count, self._steps_left)
    return self._repeated_action, s

  def activate(self, node):
    if self.active():
      raise ValueError("Exploration already active")

    self._node = node
    self._steps_left = self._exploration_horizon
    self._repeat = 0

  def active(self):
    return self._steps_left > 0


class MixtureExplorer(Explorer):
  """On each activation, uniformly at random selects one of its explorers and

    follows it for the entire exploration episode.
    """

  @classmethod
  def from_config(cls, config, num_actions):
    explorers = [
        super(MixtureExplorer, cls).from_config(subconfig, num_actions)
        for subconfig in config.mixture
    ]
    return cls(explorers)

  def __init__(self, explorers):
    """Constructs.

        Args:
            explorers (Explorer): different explorers to select from
    """
    self._explorers = explorers
    self._active_explorer = None

  def act(self, state):
    if not self.active():
      raise ValueError("MixtureExplorer not active")

    return self._active_explorer.act(state)

  def activate(self, node):
    if self.active():
      raise ValueError("MixtureExplorer already active")

    self._active_explorer = np.random.choice(self._explorers)
    self._active_explorer.activate(node)

  def active(self):
    return self._active_explorer is not None and \
            self._active_explorer.active()
