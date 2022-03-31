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
"""Batch Ensemble MSG networks definition."""

import dataclasses
from typing import Callable, Optional, Tuple
# import gin

from acme import specs
from acme.agents.jax import actor_core, actors
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from jrl.utils import ensemble_utils


@dataclasses.dataclass
class BatchEnsembleMSGNetworks:
  """Network and pure functions for the Efficient MSG agent.."""
  policy_network: networks_lib.FeedForwardNetwork
  q_ensemble_init: Callable
  q_ensemble_member_apply: Callable
  q_ensemble_same_batch_apply: Callable
  q_ensemble_different_batch_apply: Callable
  log_prob: networks_lib.LogProbFn
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None


def apply_policy_and_sample(
    networks, eval_mode = False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return actor_core.batched_feed_forward_to_actor_core(apply_and_sample)


def build_mlp_actor_fn(
    action_dim,
    hidden_layer_sizes):
  w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
  b_init = jnp.zeros
  dist_w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
  dist_b_init = jnp.zeros

  def _actor_fn(obs):
    network = hk.Sequential([
        hk.nets.MLP(
            list(hidden_layer_sizes),
            # w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            w_init=w_init,
            b_init=b_init,
            activation=jax.nn.relu,
            activate_final=True),
        networks_lib.NormalTanhDistribution(
            action_dim,
            w_init=dist_w_init,
            b_init=dist_b_init),
    ])
    return network(obs)

  return _actor_fn


def build_mlp_critic_fn(
    hidden_layer_sizes,
    use_double_q,
    ):
  w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
  b_init = jnp.zeros
  def _critic_fn(obs, action):
    # bug in implementation, needs non-empty shared params
    fake_shared_params = hk.get_parameter(
        'fake_shared_params',
        shape=(128,),
        dtype=float,
        init=hk.initializers.Constant(0.))
    network1 = hk.Sequential([
        hk.nets.MLP(
            list(hidden_layer_sizes) + [1],
            # w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            w_init=w_init,
            b_init=b_init,
            activation=jax.nn.relu,
            name=ensemble_utils.ENSEMBLE_PARAM_KEYWORD + '_network1'),
    ])
    input_ = jnp.concatenate([obs, action], axis=-1)
    value1 = network1(input_)
    if use_double_q:
      network2 = hk.Sequential([
          hk.nets.MLP(
              list(hidden_layer_sizes) + [1],
              # w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
              w_init=w_init,
              b_init=b_init,
              activation=jax.nn.relu,
              name=ensemble_utils.ENSEMBLE_PARAM_KEYWORD + '_network2'),
      ])
      value2 = network2(input_)
      return jnp.concatenate([value1, value2], axis=-1)
    else:
      return value1

  return _critic_fn


def build_hk_batch_ensemble_mlp_critic_fn(
    hidden_layer_sizes,
    use_doule_q,):
  # def _single_critic(x):
  #   layers = []
  #   for hid_dim in hidden_layer_sizes:
  #     layers.append(
  #         ensemble_utils.DenseBatchEnsembleLayer(hid_dim, jax.nn.relu, 0.5))
  #   layers.append(
  #       ensemble_utils.DenseBatchEnsembleLayer(1, None, 0.5))
  #   seq = hk.Sequential(layers)
  #   return seq(x)

  def _hk_batch_ensemble_mlp_critic_fn(obs, act):
    input_ = jnp.concatenate([obs, act], axis=-1)

    layers = []
    for hid_dim in hidden_layer_sizes:
      layers.append(
          ensemble_utils.DenseBatchEnsembleLayer(hid_dim, jax.nn.relu, 0.5))
    layers.append(
        ensemble_utils.DenseBatchEnsembleLayer(1, None, 0.5))
    seq = hk.Sequential(layers)
    v1 = seq(input_)

    if use_doule_q:
      layers = []
      for hid_dim in hidden_layer_sizes:
        layers.append(
            ensemble_utils.DenseBatchEnsembleLayer(hid_dim, jax.nn.relu, 0.5))
      layers.append(
          ensemble_utils.DenseBatchEnsembleLayer(1, None, 0.5))
      seq = hk.Sequential(layers)
      v2 = seq(input_)
      return jnp.concatenate([v1, v2], axis=-1)
    else:
      return v1

  return _hk_batch_ensemble_mlp_critic_fn


def make_networks(
    spec,
    actor_fn_build_fn = build_mlp_actor_fn,
    actor_hidden_layer_sizes = (256, 256),
    critic_fn_build_fn = build_hk_batch_ensemble_mlp_critic_fn,
    # critic_fn_build_fn: Callable = build_mlp_critic_fn,
    critic_hidden_layer_sizes = (256, 256),
    use_double_q = False,
    ):
  """Creates networks used by the agent."""

  num_dimensions = np.prod(spec.actions.shape, dtype=int)

  _actor_fn = actor_fn_build_fn(num_dimensions, actor_hidden_layer_sizes)
  policy = hk.without_apply_rng(hk.transform(_actor_fn, apply_rng=True))

  _critic_fn = critic_fn_build_fn(critic_hidden_layer_sizes, use_double_q)
  critic = hk.without_apply_rng(hk.transform(_critic_fn, apply_rng=True))
  critic_ensemble_init = ensemble_utils.transform_init_for_ensemble(critic.init, init_same=False)
  critic_ensemble_member_apply = ensemble_utils.transform_apply_for_ensemble_member(critic.apply)
  critic_same_batch_ensemble_apply = ensemble_utils.build_same_batch_ensemble_apply_fn(critic_ensemble_member_apply, 2)
  critic_diff_batch_ensemble_apply = ensemble_utils.build_different_batch_ensemble_apply_fn(critic_ensemble_member_apply, 2)

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  return BatchEnsembleMSGNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_obs), policy.apply),
      q_ensemble_init=lambda ensemble_size, key: critic_ensemble_init(ensemble_size, key, dummy_obs, dummy_action),
      q_ensemble_member_apply=critic_ensemble_member_apply,
      q_ensemble_same_batch_apply=critic_same_batch_ensemble_apply,
      q_ensemble_different_batch_apply=critic_diff_batch_ensemble_apply,
      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode())
