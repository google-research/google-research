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

"""Cartpole environment with sensor bias and continuous action space."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from gym import spaces
from gym.envs.classic_control import cartpole
import numpy as np


class CartpoleSensorBiasEnv(cartpole.CartPoleEnv):
  """Cartpole environment with sensor bias and continuous action space.

  Makes two changes to the standard Cartpole environment.
  1) Add a random bias to the angle observation.
  2) The action space is continuous, where the 1-dimensional action is the
    signed force applied to the cart.

  Args:
    angle_observation_bias: The bias to be added to angle observation.
  """

  def __init__(self, angle_observation_bias=0.):
    super(CartpoleSensorBiasEnv, self).__init__()
    self.action_space = spaces.Box(
        low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)
    self._angle_observation_bias = angle_observation_bias

  def reset(self):
    """Resets the environment and returns the initial obervation."""
    observation = np.array(super(CartpoleSensorBiasEnv, self).reset())
    observation[2] += self._angle_observation_bias
    return observation

  def step(self, action):
    """Steps the environment forward.

    Args:
      action: The signed force to be applied to the cart.

    Returns:
      observation: A tuple of 4 real numbers representing (x, xdot, theta,
        thetadot), where x, theta are the position of the cart and pole.
      reward: Reward for current step, which is 1 for every step in which the
        episode has not finished.
      done: Whether the episode has finished.
      info: A dictionary that stores diagnostic information (default empty).
    """
    self.total_mass = (self.masspole + self.masscart)
    state = self.state
    x, x_dot, theta, theta_dot = state
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    force = action[0]
    temp = (force + self.polemass_length * theta_dot * theta_dot *
            sintheta) / self.total_mass
    thetaacc = (self.gravity * sintheta - costheta * temp) / (
        self.length *
        (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
    xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
    x = x + self.tau * x_dot
    x_dot = x_dot + self.tau * xacc
    theta = theta + self.tau * theta_dot
    theta_dot = theta_dot + self.tau * thetaacc

    self.state = (x, x_dot, theta, theta_dot)
    done = True
    if abs(x) < self.x_threshold and abs(theta) < self.theta_threshold_radians:
      done = False

    reward = float(not done)
    observation = np.array(self.state)
    observation[2] += self._angle_observation_bias
    # Get rid of velocity components at index 1, and 3.
    return observation, reward, done, {}
