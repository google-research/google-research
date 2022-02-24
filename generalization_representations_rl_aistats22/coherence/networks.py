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

"""Networks with support for feature outputs."""

import time

from dopamine.jax import networks
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp


### DQN Networks ###
@gin.configurable
class NatureDQNNetworkWithFeatures(networks.NatureDQNNetwork):
  """The convolutional network used to compute feature representations."""
  num_actions: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = networks.preprocess_atari_inputs(x)
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    return x


### Rainbow Networks ###
@gin.configurable
class RainbowNetworkWithFeatures(networks.RainbowNetwork):
  """Convolutional network used to compute the feature representations."""
  num_actions: int
  num_atoms: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x, support=None):
    initializer = nn.initializers.variance_scaling(
        scale=1.0 / jax.numpy.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    if not self.inputs_preprocessed:
      x = networks.preprocess_atari_inputs(x)
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    return x


### Implicit Quantile Networks ###
class ImplicitQuantileNetworkWithFeatures(networks.ImplicitQuantileNetwork):
  """IQN convolutional network used to compute feature representations."""
  num_actions: int
  quantile_embedding_dim: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x, num_quantiles=32, rng=None):
    initializer = nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    if not self.inputs_preprocessed:
      x = networks.preprocess_atari_inputs(x)
    if rng is None:
      seed = int(time.time() * 1e6)
      rng = jax.random.PRNGKey(seed)
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    state_vector_length = x.shape[-1]
    state_net_tiled = jnp.tile(x, [num_quantiles, 1])
    quantiles_shape = [num_quantiles, 1]
    quantiles = jax.random.uniform(rng, shape=quantiles_shape)
    quantile_net = jnp.tile(quantiles, [1, self.quantile_embedding_dim])
    quantile_net = (
        jnp.arange(1, self.quantile_embedding_dim + 1, 1).astype(jnp.float32)
        * onp.pi
        * quantile_net)
    quantile_net = jnp.cos(quantile_net)
    quantile_net = nn.Dense(features=state_vector_length,
                            kernel_init=initializer)(quantile_net)
    quantile_net = nn.relu(quantile_net)
    x = state_net_tiled * quantile_net
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = jnp.mean(x, axis=0)
    return x
