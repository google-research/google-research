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
import torch.nn as nn


# If you implement a subclass, you should register it in the from_config
# method and override the from_config method.
class Policy(nn.Module):
  """Defines a mapping from state to action"""
  __metaclass__ = abc.ABCMeta

  @classmethod
  def from_config(cls, config, num_actions):
    """Constructs a Policy from a config.

        Args:
            config (Config): config.type specifies the Policy subclass
            num_actions (int): number of possible actions at each state

        Returns:
            Policy
        """
    if config.type == "dqn":
      # TODO: Properly break circular dependencies
      from strategic_exploration.hrl.dqn import DQNPolicy
      return DQNPolicy.from_config(config, num_actions)
    elif config.type == "systematic_exploration":
      from strategic_exploration.hrl.abstract_exploration import AbstractExploration
      return AbstractExploration.from_config(config, num_actions)
    else:
      raise ValueError("{} not a supported policy type.".format(config.type))

  def act(self, state):
    """Takes a state and returns an action to take.

        Args: state (State)

        Returns:
            action (int)
        """
    raise NotImplementedError()

  @abc.abstractmethod
  def stats(self):
    """Returns a dictionary whose (key, val) pairs correspond to relevant

        statistics for the policy. Primary use is for printing these statistics
        periodically

        Returns:
            dict
        """
    raise NotImplementedError()
