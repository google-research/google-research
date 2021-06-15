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

"""MinAtar environment made compatible for Dopamine."""

from dopamine.discrete_domains import atari_lib
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import minatar


gin.constant('minatar_env.ASTERIX_SHAPE', (10, 10, 4))
gin.constant('minatar_env.BREAKOUT_SHAPE', (10, 10, 4))
gin.constant('minatar_env.FREEWAY_SHAPE', (10, 10, 7))
gin.constant('minatar_env.SEAQUEST_SHAPE', (10, 10, 10))
gin.constant('minatar_env.SPACE_INVADERS_SHAPE', (10, 10, 6))
gin.constant('minatar_env.DTYPE', jnp.float64)


class MinAtarEnv(object):
  """Wrapper class for MinAtar environments."""

  def __init__(self, game_name):
    self.env = minatar.Environment(env_name=game_name)
    self.env.n = self.env.num_actions()
    self.game_over = False

  @property
  def observation_space(self):
    return self.env.state_shape()

  @property
  def action_space(self):
    return self.env  # Only used for the `n` parameter.

  @property
  def reward_range(self):
    pass  # Unused

  @property
  def metadata(self):
    pass  # Unused

  def reset(self):
    self.game_over = False
    self.env.reset()
    return self.env.state()

  def step(self, action):
    r, terminal = self.env.act(action)
    self.game_over = terminal
    return self.env.state(), r, terminal, None


@gin.configurable
def create_minatar_env(game_name):
  return MinAtarEnv(game_name)


@gin.configurable
class MinatarDQNNetwork(nn.Module):
  """JAX DQN Network for Minatar environments."""
  num_actions: int

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32)
    x = nn.Conv(features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = x.reshape(-1)  # flatten
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class MinatarRainbowNetwork(nn.Module):
  """Jax Rainbow network for Minatar."""
  num_actions: int
  num_atoms: int

  @nn.compact
  def __call__(self, x, support):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32)
    x = nn.Conv(features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = x.reshape(-1)  # flatten
    x = nn.Dense(features=self.num_actions * self.num_atoms,
                 kernel_init=initializer)(x)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.sum(support * probabilities, axis=1)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


@gin.configurable
class MinatarQuantileNetwork(nn.Module):
  """Convolutional network used to compute the agent's return quantiles."""
  num_actions: int
  num_atoms: int

  @nn.compact
  def __call__(self, x):
    initializer = jax.nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    x = x.astype(jnp.float32)
    x = nn.Conv(features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = x.reshape(-1)  # flatten
    x = nn.Dense(features=self.num_actions * self.num_atoms,
                 kernel_init=initializer)(x)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.mean(logits, axis=1)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)
