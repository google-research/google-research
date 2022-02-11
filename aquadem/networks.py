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

"""Networks for Random Expert Distillation."""

import dataclasses
from typing import Any, Sequence, Tuple

from acme import specs
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from flax import linen
import haiku as hk
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class AquademNetworks:
  """Container of Aquadem networks factories."""
  encoder: networks_lib.FeedForwardNetwork
  discrete_rl_networks: Any = None


class Encoder(linen.Module):
  """Encoder for the multi BC."""
  action_dim: int
  num_actions: int
  torso_layer_sizes: Sequence[int]
  head_layer_sizes: Sequence[int]
  input_dropout_rate: float
  hidden_dropout_rate: float

  @linen.compact
  def __call__(self, observation, is_training=False):
    initializer = linen.initializers.xavier_uniform()
    deterministic = not is_training

    x = linen.Dropout(
        rate=self.input_dropout_rate, deterministic=deterministic)(
            observation)
    for layer_size in self.torso_layer_sizes:
      x = linen.Dense(features=layer_size, kernel_init=initializer)(x)
      x = linen.relu(x)
      x = linen.Dropout(
          rate=self.hidden_dropout_rate, deterministic=deterministic)(
              x)
    actions = []
    for _ in range(self.num_actions):
      z = x
      for layer_size in self.head_layer_sizes:
        z = linen.Dense(features=layer_size, kernel_init=initializer)(z)
        z = linen.relu(z)
        z = linen.Dropout(
            rate=self.hidden_dropout_rate, deterministic=deterministic)(
                z)
      actions.append(
          linen.Dense(features=self.action_dim, kernel_init=initializer)(z))
    return jnp.stack(actions, axis=-1)


def make_action_candidates_network(
    spec,
    num_actions,
    discrete_rl_networks,
    torso_layer_sizes = (256,),
    head_layer_sizes = (256,),
    input_dropout_rate = 0.1,
    hidden_dropout_rate = 0.1):
  """Creates networks used by the agent and wraps it into Flax Model.

  Args:
    spec: Environment spec.
    num_actions: the number of actions proposed by the multi-modal model.
    discrete_rl_networks: Direct RL algorithm networks.
    torso_layer_sizes: Layer sizes of the torso.
    head_layer_sizes: Layer sizes of the heads.
    input_dropout_rate: Dropout rate input.
    hidden_dropout_rate: Dropout rate hidden.
  Returns:
    The Flax model.
  """
  dummy_obs, _ = get_dummy_batched_obs_and_actions(spec)
  encoder_module = Encoder(
      action_dim=np.prod(spec.actions.shape, dtype=int),
      num_actions=num_actions,
      torso_layer_sizes=torso_layer_sizes,
      head_layer_sizes=head_layer_sizes,
      input_dropout_rate=input_dropout_rate,
      hidden_dropout_rate=hidden_dropout_rate,)

  encoder = networks_lib.FeedForwardNetwork(
      lambda key: encoder_module.init(key, dummy_obs, is_training=False),
      encoder_module.apply)

  return AquademNetworks(
      encoder=encoder,
      discrete_rl_networks=discrete_rl_networks)


def make_q_network(spec,
                   hidden_layer_sizes=(512, 512, 256),
                   architecture='LayerNorm'):
  """DQN network for Aquadem algo."""

  def _q_fn(obs):
    if architecture == 'MLP':  # AQuaOff architecture
      network_fn = hk.nets.MLP
    elif architecture == 'LayerNorm':  # Original AQuaDem architecture
      network_fn = networks_lib.LayerNormMLP
    else:
      return ValueError('Architecture not recognized')

    network = network_fn(list(hidden_layer_sizes) + [spec.actions.num_values])
    value = network(obs)
    return value

  critic = hk.without_apply_rng(hk.transform(_q_fn))
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  critic_network = networks_lib.FeedForwardNetwork(
      lambda key: critic.init(key, dummy_obs), critic.apply)
  return critic_network


def get_dummy_batched_obs_and_actions(
    environment_spec):
  """Generates dummy batched (batch_size=1) obs and actions."""
  dummy_observation = utils.tile_nested(
      utils.zeros_like(environment_spec.observations), 1)
  dummy_action = utils.tile_nested(
      utils.zeros_like(environment_spec.actions), 1)
  return dummy_observation, dummy_action
