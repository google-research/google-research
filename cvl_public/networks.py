# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Contrastive RL networks definition."""
import dataclasses
from typing import Callable, Optional, Tuple, Sequence

from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.networks import resnet
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class DeepAtariTorso(hk.Module):
  """Deep torso for Atari, from the IMPALA paper."""

  def __init__(
      self,
      channels_per_group = (16, 32, 32),
      blocks_per_group = (2, 2, 2),
      downsampling_strategies = (
          resnet.DownsamplingStrategy.CONV_MAX,) * 3,
      hidden_sizes = (256,),
      use_layer_norm = False,
      name = 'deep_atari_torso'):
    super().__init__(name=name)
    self._use_layer_norm = use_layer_norm
    self.resnet = resnet.ResNetTorso(
        channels_per_group=channels_per_group,
        blocks_per_group=blocks_per_group,
        downsampling_strategies=downsampling_strategies,
        use_layer_norm=use_layer_norm)
    # Make sure to activate the last layer as this torso is expected to feed
    # into the rest of a bigger network.
    self.mlp_head = hk.nets.MLP(output_sizes=hidden_sizes, activate_final=True)

  def __call__(self, x):
    output = self.resnet(x)
    output = jax.nn.relu(output)
    output = hk.Flatten(preserve_dims=-3)(output)
    output = self.mlp_head(output)
    return output


@dataclasses.dataclass
class ContrastiveNetworks:
  """Network and pure functions for the Contrastive RL agent."""
  policy_network: networks_lib.FeedForwardNetwork
  q_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  repr_fn: Callable[Ellipsis, networks_lib.NetworkOutput]
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None


def apply_policy_and_sample(
    networks,
    eval_mode = False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return apply_and_sample


def make_networks(
    spec,
    repr_dim = 64,
    repr_norm = False,
    repr_norm_temp = True,
    hidden_layer_sizes = (256, 256),
    actor_min_std = 1e-6,
    twin_q = False,
    use_image_obs = False):
  """Creates networks used by the agent."""

  num_dimensions = np.prod(spec.actions.shape, dtype=int)
  TORSO = DeepAtariTorso  # pylint: disable=invalid-name

  def _repr_fn(obs, action, goal=None, hidden=None):
    # The optional input hidden is the image representations. We include this
    # as an input for the second Q value when twin_q = True, so that the two Q
    # values use the same underlying image representation.
    if hidden is None:
      if use_image_obs:
        img_encoder = TORSO()
        if obs is not None:
          state = img_encoder(obs)
        if goal is not None:
          goal = img_encoder(goal)
      else:
        state = obs
    else:
      state = hidden

    if obs is not None and action is not None:
      sa_encoder = hk.nets.MLP(
          list(hidden_layer_sizes) + [repr_dim],
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
          activation=jax.nn.relu,
          name='sa_encoder')
      sa_repr = sa_encoder(jnp.concatenate([state, action], axis=-1))

    if goal is not None:
      g_encoder = hk.nets.MLP(
          list(hidden_layer_sizes) + [repr_dim],
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
          activation=jax.nn.relu,
          name='g_encoder')
      g_repr = g_encoder(goal)

    if repr_norm:
      if obs is not None and action is not None:
        sa_repr = sa_repr / jnp.linalg.norm(sa_repr, axis=1, keepdims=True)
      if goal is not None:
        g_repr = g_repr / jnp.linalg.norm(g_repr, axis=1, keepdims=True)

      if repr_norm_temp:
        if obs is not None and action is not None:
          log_scale = hk.get_parameter(
              'repr_log_scale', [], dtype=sa_repr.dtype, init=jnp.zeros)
          sa_repr = sa_repr / jnp.exp(log_scale)
    if obs is None or action is None:
      sa_repr = None
    if goal is None:
      g_repr = None
    return sa_repr, g_repr, (state, goal)

  def _combine_repr(sa_repr, g_repr):
    return jax.numpy.einsum('ik,jk->ij', sa_repr, g_repr)

  def _critic_fn(obs, action, goal, sa_repr=None, g_repr=None):
    sa_repr_new, g_repr_new, hidden = _repr_fn(obs, action, goal)
    # If we re-computed phi(s,a), overwrite
    if sa_repr is None:
      sa_repr = sa_repr_new
    # If we re-computed psi(s_g), overwrite
    if g_repr is None:
      g_repr = g_repr_new
    outer = _combine_repr(sa_repr, g_repr)
    if twin_q:
      sa_repr2, g_repr2, _ = _repr_fn(obs, action, goal=goal, hidden=hidden)
      outer2 = _combine_repr(sa_repr2, g_repr2)
      # outer.shape = [batch_size, batch_size, 2]
      outer = jnp.stack([outer, outer2], axis=-1)
    return outer

  def _actor_fn(obs):
    if use_image_obs:
      obs = TORSO()(obs)
    network = hk.Sequential([
        hk.nets.MLP(
            list(hidden_layer_sizes),
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            activation=jax.nn.relu,
            activate_final=True),
        networks_lib.NormalTanhDistribution(num_dimensions,
                                            min_scale=actor_min_std),
    ])
    return network(obs)

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))
  repr_fn = hk.without_apply_rng(hk.transform(_repr_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  return ContrastiveNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_obs), policy.apply),
      q_network=networks_lib.FeedForwardNetwork(
          lambda key: critic.init(key, dummy_obs, dummy_action, dummy_obs),
          critic.apply),
      repr_fn=repr_fn.apply,
      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode(),
      )
