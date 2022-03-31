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

# python3
"""BC networks definition."""

import dataclasses
from typing import Optional, Tuple

from acme import specs
from acme.agents.jax import actor_core, actors
from acme.jax import networks as networks_lib
from acme.jax import utils
import gin
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jrl.utils.networks import procgen_networks

distributional = networks_lib.distributional
atari = networks_lib.atari


@dataclasses.dataclass
class BCNetworks:
  """Network and pure functions for the BC agent."""
  policy_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None
  img_encoder: Optional[networks_lib.FeedForwardNetwork] = None


def apply_policy_and_sample(
    networks, eval_mode = False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return actor_core.batched_feed_forward_to_actor_core(apply_and_sample)


def apply_policy_and_sample_with_img_encoder(
    networks, eval_mode = False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    img = obs['state_image']
    img_embedding = networks.img_encoder.apply(params[1], img)
    x = dict(state_image=img_embedding, state_dense=obs['state_dense'])
    return sample_fn(networks.policy_network.apply(params[0], x), key)
  return actor_core.batched_feed_forward_to_actor_core(apply_and_sample)

w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
b_init = jnp.zeros
dist_w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
dist_b_init = jnp.zeros

@gin.register
def build_standard_actor_fn(
    num_dimensions,
    actor_hidden_layer_sizes = (256, 256, 256),):
  def _actor_fn(obs):
    # # for matching Ilya's codebase
    # relu_orthogonal = hk.initializers.Orthogonal(scale=2.0**0.5)
    # near_zero_orthogonal = hk.initializers.Orthogonal(1e-2)
    # x = obs
    # for hid_dim in actor_hidden_layer_sizes:
    #   x = hk.Linear(hid_dim, w_init=relu_orthogonal, b_init=jnp.zeros)(x)
    #   x = jax.nn.relu(x)
    # dist = networks_lib.NormalTanhDistribution(
    #     num_dimensions,
    #     w_init=near_zero_orthogonal,
    #     b_init=jnp.zeros)(x)
    # return dist

    network = hk.Sequential([
        hk.nets.MLP(
            list(actor_hidden_layer_sizes),
            # w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            # w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
            w_init=w_init,
            b_init=b_init,
            activation=jax.nn.relu,
            activate_final=True),
        # networks_lib.NormalTanhDistribution(num_dimensions),
        networks_lib.NormalTanhDistribution(
            num_dimensions,
            w_init=dist_w_init,
            b_init=dist_b_init,
            min_scale=1e-2,
        ),
    ])
    return network(obs)
  return _actor_fn


def make_networks(
    spec,
    build_actor_fn=build_standard_actor_fn,
    img_encoder_fn=None,
    ):
  """Creates networks used by the agent."""
  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  if isinstance(spec.actions, specs.DiscreteArray):
    num_dimensions = spec.actions.num_values
    # _actor_fn = procgen_networks.build_procgen_actor_fn(num_dimensions)
  else:
    num_dimensions = np.prod(spec.actions.shape, dtype=int)

  _actor_fn = build_actor_fn(num_dimensions)

  if img_encoder_fn is not None:
    img_encoder = hk.without_apply_rng(
        hk.transform(img_encoder_fn, apply_rng=True))
    key = jax.random.PRNGKey(seed=42)
    temp_encoder_params = img_encoder.init(key, dummy_obs['state_image'])
    dummy_hidden = img_encoder.apply(temp_encoder_params, dummy_obs['state_image'])
    img_encoder_network = networks_lib.FeedForwardNetwork(
        lambda key: img_encoder.init(key, dummy_hidden), img_encoder.apply)
    dummy_policy_input = dict(
        state_image=dummy_hidden,
        state_dense=dummy_obs['state_dense'],)
  else:
    img_encoder_fn = None
    dummy_policy_input = dummy_obs
    img_encoder_network = None

  policy = hk.without_apply_rng(hk.transform(_actor_fn, apply_rng=True))

  return BCNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_policy_input), policy.apply),
      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode(),
      img_encoder=img_encoder_network,)
