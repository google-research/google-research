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

"""SNR networks definition."""

from typing import Optional, Tuple, Callable

from acme import specs
from acme.agents.jax import actors, actor_core
from acme.jax import networks as networks_lib
from acme.jax import utils
import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp
import neural_tangents as nt
from neural_tangents import stax
from neural_tangents._src.utils import kernel
import numpy as np


@dataclasses.dataclass
class SNRNetworks:
  """Network and pure functions for the SNR agent."""
  policy_network: networks_lib.FeedForwardNetwork
  q_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None
  q_kernel_fn: Optional[Callable] = None
  nt_critic_apply_fn: Optional[Callable] = None


def apply_policy_and_sample(
    networks, eval_mode = False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return actor_core.batched_feed_forward_to_actor_core(apply_and_sample)


def make_networks(
    spec,
    actor_hidden_layer_sizes = (256, 256),
    critic_hidden_layer_sizes = (256, 256),
    num_critics = 1,):
  """Creates networks used by the agent."""
  assert num_critics == 1, 'Otherwise need to fix critic parametrization'

  num_dimensions = np.prod(spec.actions.shape, dtype=int)

  def _actor_fn(obs):
    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
    b_init = jnp.zeros
    dist_w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
    dist_b_init = jnp.zeros

    network = hk.Sequential([
        hk.nets.MLP(
            list(actor_hidden_layer_sizes),
            w_init=w_init,
            b_init=b_init,
            activation=jax.nn.relu,
            activate_final=True),
        networks_lib.NormalTanhDistribution(
            num_dimensions,
            w_init=dist_w_init,
            b_init=dist_b_init),
    ])
    return network(obs)

  def build_critic_network_components():
    layers = []
    for hid_dim in critic_hidden_layer_sizes:
      W_std = 1.0
      b_std = 0.05
      layers += [
          stax.Dense(hid_dim, W_std=W_std, b_std=b_std),
          stax.Relu()
      ]
    layers += [stax.Dense(1, W_std=W_std, b_std=0.05)]
    nt_init_fn, nt_apply_fn, nt_kernel_fn = stax.serial(*layers)
    # kernel_fn = jax.jit(nt_kernel_fn, static_argnums=(2,))

    return nt_init_fn, nt_apply_fn, nt_kernel_fn

  nt_critic_init_fn, nt_critic_apply_fn, nt_critic_kernel_fn = build_critic_network_components()
  nt_critic_kernel_fn = jax.jit(nt_critic_kernel_fn, static_argnums=(2,))

  def critic_init_fn(key, obs, act):
    all_params = []
    for _ in range(num_critics):
      key, sub_key = jax.random.split(key)
      x = jnp.concatenate([obs, act], axis=-1)
      x_shape = list(x.shape)
      x_shape[0] = -1
      _, params = nt_critic_init_fn(sub_key, tuple(x_shape))
      all_params.append(params)
    # return all_params
    return all_params[0]

  def critic_apply_fn(params, obs, act):
    preds = []
    for i in range(num_critics):
      x = jnp.concatenate([obs, act], axis=-1)
      # preds.append(nt_critic_apply_fn(params[i], x))
      preds.append(nt_critic_apply_fn(params, x))
    # preds = jnp.concatenate(preds, axis=-1)
    # return preds
    return preds[0]

  policy = hk.without_apply_rng(hk.transform(_actor_fn, apply_rng=True))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  return SNRNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_obs), policy.apply),
      q_network=networks_lib.FeedForwardNetwork(
          lambda key: critic_init_fn(key, dummy_obs, dummy_action),
          critic_apply_fn),
      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode(),
      q_kernel_fn=nt_critic_kernel_fn,
      nt_critic_apply_fn=nt_critic_apply_fn,)
