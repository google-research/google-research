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
"""CQL networks definition."""

from typing import Optional, Tuple, Callable

from acme import specs
from acme.agents.jax import actors, actor_core
from acme.jax import networks as networks_lib
from acme.jax import utils
import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class CQLNetworks:
  """Network and pure functions for the SAC agent.."""
  policy_network: networks_lib.FeedForwardNetwork
  q_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None
  compute_kernel_features: Optional[Callable] = None


def apply_policy_and_sample(
    networks, eval_mode = False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return actor_core.batched_feed_forward_to_actor_core(apply_and_sample)


def build_q_filtered_actor(
    networks,
    num_samples,
    with_uniform = True,):
  def select_action(
      params,
      key,
      obs,
      ):
    key, sub_key = jax.random.split(key)
    policy_params = params[0]
    q_params = params[1]

    dist = networks.policy_network.apply(policy_params, obs)
    acts = dist._sample_n(num_samples, sub_key)
    acts = acts[:,0,:] # N x act_dim

    if with_uniform:
      key, sub_key = jax.random.split(sub_key)
      unif_acts = jax.random.uniform(sub_key, acts.shape, dtype=acts.dtype, minval=-1., maxval=1.)
      acts = jnp.concatenate([acts, unif_acts], axis=0)

    def obs_tile_fn(t):
      # t = jnp.expand_dims(t, axis=0)
      tile_shape = [1] * t.ndim
      # tile_shape[0] = num_samples
      tile_shape[0] = acts.shape[0]
      return jnp.tile(t, tile_shape)
    tiled_obs = jax.tree_map(obs_tile_fn, obs)

    # batch_size x num_critics
    all_q = networks.q_network.apply(q_params, tiled_obs, acts)
    # num_devices x num_per_device x batch_size
    q_score = jnp.min(all_q, axis=-1)
    best_idx = jnp.argmax(q_score)
    # return acts[best_idx], key
    return acts[best_idx][None,:]

  # return actor_core.ActorCore(
  #     init=lambda key: key,
  #     select_action=select_action,
  #     get_extras=lambda x: ())
  return actor_core.batched_feed_forward_to_actor_core(select_action)


def make_networks(
    spec,
    actor_hidden_layer_sizes = (256, 256),
    critic_hidden_layer_sizes = (256, 256, 256),
    num_critics = 2):
  """Creates networks used by the agent."""

  num_dimensions = np.prod(spec.actions.shape, dtype=int)

  # w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')
  # w_init = hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal') # to match CQLSACAgent
  # b_init = jnp.zeros
  # dist_w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform') # to match CQLSACAgent
  # dist_b_init = jnp.zeros
  # def _actor_fn(obs):
  #   network = hk.Sequential([
  #       hk.nets.MLP(
  #           list(actor_hidden_layer_sizes),
  #           # w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
  #           w_init=w_init,
  #           b_init=b_init,
  #           activation=jax.nn.relu,
  #           activate_final=True),
  #       networks_lib.NormalTanhDistribution(
  #           num_dimensions,
  #           w_init=dist_w_init,
  #           b_init=dist_b_init),
  #   ])
  #   return network(obs)

  w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
  b_init = jnp.zeros
  dist_w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
  dist_b_init = jnp.zeros

  def _actor_fn(obs):
    # # for matching Ilya's codebase
    # relu_orthogonal = hk.initializers.Orthogonal(scale=2.0**0.5)
    # near_zero_orthogonal = hk.initializers.Orthogonal(1e-2)
    x = obs
    for hid_dim in actor_hidden_layer_sizes:
      x = hk.Linear(hid_dim, w_init=w_init, b_init=b_init)(x)
      x = jax.nn.relu(x)
    dist = networks_lib.NormalTanhDistribution(
        num_dimensions,
        w_init=dist_w_init,
        b_init=dist_b_init)(x)
    return dist

  # w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')
  # w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform') # to match CQLSACAgent
  # b_init = jnp.zeros
  # for matching Ilya's codebase
  # relu_orthogonal = hk.initializers.Orthogonal(scale=2.0**0.5)
  # near_zero_orthogonal = hk.initializers.Orthogonal(1e-2)

  w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
  b_init = jnp.zeros
  dist_w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
  dist_b_init = jnp.zeros

  def _all_critic_stuff(obs, action):
    input_ = jnp.concatenate([obs, action], axis=-1)
    critic_preds = []
    critic_hiddens = []

    for _ in range(num_critics):
      # for matching Ilya's codebase
      # w_init = relu_orthogonal
      # b_init = jnp.zeros

      cnet = hk.Sequential([
          hk.nets.MLP(
              list(critic_hidden_layer_sizes),
              # w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
              w_init=w_init,
              b_init=b_init,
              activation=jax.nn.relu,
              activate_final=True),
      ])
      hidden = cnet(input_)

      # for matching Ilya's codebase
      # w_init = near_zero_orthogonal
      # b_init = jnp.zeros

      pred = hk.Linear(1, w_init=w_init, b_init=b_init)(hidden)
      critic_hiddens.append(hidden)
      critic_preds.append(pred)

    return (jnp.concatenate(critic_preds, axis=-1), jnp.concatenate(critic_hiddens, axis=-1))

  policy = hk.without_apply_rng(hk.transform(_actor_fn, apply_rng=True))
  # critic = hk.without_apply_rng(hk.transform(_critic_fn, apply_rng=True))
  all_critic_stuff = hk.without_apply_rng(hk.transform(_all_critic_stuff, apply_rng=True))
  critic_apply_fn = lambda p, obs, act: all_critic_stuff.apply(p, obs, act)[0]

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  def critic_feats(p, X):
    obs_dim = dummy_obs.shape[-1]
    obs = X[:, :obs_dim]
    act = X[:, obs_dim:]
    return all_critic_stuff.apply(p, obs, act)[1] * ((1. / float(critic_hidden_layer_sizes[-1]))**0.5)

  return CQLNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_obs), policy.apply),
      # q_network=networks_lib.FeedForwardNetwork(
      #     lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
      q_network=networks_lib.FeedForwardNetwork(
          lambda key: all_critic_stuff.init(key, dummy_obs, dummy_action),
          critic_apply_fn),
      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode(),
      compute_kernel_features=critic_feats,)
