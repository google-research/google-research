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

r"""Library for creating different reinforcement learning environments.

Responsible for creating different reinforcement learning environments, in
particular wrappers for Brain Robotics versions of OpenAIGym envs.
"""
import abc
import gym

from gym.envs.classic_control import continuous_mountain_car
from gym.envs.classic_control import pendulum
import numpy as np


class BlackboxRLEnvironment(abc.ABC, gym.Env):
  """Base class for all environments used for the RL functions below."""

  @abc.abstractmethod
  def deterministic_start(self):
    """Starts/resets the environment to a deterministic state for reproducibility."""
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def state_dimensionality(self):
    """Returns the 1-D state dimension (for input to policies)."""
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def action_dimensionality(self):
    """Returns the 1-D action dimension (for policy outputs)."""
    raise NotImplementedError('Abstract method')


class ContMountainCar(continuous_mountain_car.Continuous_MountainCarEnv,
                      BlackboxRLEnvironment):
  """Class representing ContinuousMountainCar OpenAIGym env for RL.

  OpenAIGym style environment for testing reinforcement learning algorithms.
  This is a wrapper on the Brain Robotics version of the ContinuousMountainCar
  OpenAIGym env with extra functionalities (such as deterministically setting up
  the
  initial configuration).
  """

  def deterministic_start(self):
    self._reset()
    value = -0.55
    self.state = np.array([value, 0.0])
    return self.state

  def state_dimensionality(self):
    return 2

  def action_dimensionality(self):
    return 1


class Pendulum(pendulum.PendulumEnv, BlackboxRLEnvironment):
  """Class representing Pendulum OpenAIGym env for RL.

  OpenAIGym style environment for testing reinforcement learning algorithms.
  This is a wrapper on the Brain Robotics version of the Pendulum
  OpenAIGym env with extra functionalities (such as deterministically setting up
  the
  initial configuration).
  """

  def deterministic_start(self):
    self.reset()
    self.state = np.array([-0.55, 0])
    theta, thetadot = self.state
    return np.array([np.cos(theta), np.sin(theta), thetadot])

  def state_dimensionality(self):
    return 3

  def action_dimensionality(self):
    return 1
