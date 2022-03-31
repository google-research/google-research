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
"""MSG networks definition."""

import dataclasses
from typing import Callable, Optional, Tuple

from acme import specs
from acme.agents.jax import actor_core
from acme.agents.jax import actor_core, actors
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import neural_tangents as nt
from neural_tangents import stax
from neural_tangents._src.utils import kernel
import numpy as np

from jrl.utils import network_utils
from jrl.utils.networks import bimanual_sweep

import tensorflow_probability
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions



@dataclasses.dataclass
class MSGNetworks:
  """Network and pure functions for the SAC agent.."""
  policy_network: networks_lib.FeedForwardNetwork
  q_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None
  img_encoder: Optional[networks_lib.FeedForwardNetwork] = None
  kernel_fn: Optional[kernel.Kernel] = None
  get_particular_critic_init: Optional[Callable] = None
  get_critic_repr: Optional[Callable] = None
  simclr_encoder: Optional[networks_lib.FeedForwardNetwork] = None


_min_stddev = 1e-2
def build_gaussian_mixture_log_prob(num_dimensions, num_components):

  def relaxed_gaussian_mixture_log_prob(params, actions):
    logits = params[:, :num_components]
    log_pis = jax.nn.log_softmax(logits, axis=-1) # B x num_components

    means = jnp.reshape(
        params[:, num_components:num_components + num_components*num_dimensions],
        [actions.shape[0], num_components, num_dimensions])
    log_stddevs = jnp.reshape(
        params[:, -(num_components*num_dimensions):],
        [actions.shape[0], num_components, num_dimensions])
    stddevs = jax.nn.softplus(log_stddevs) + _min_stddev

    actions = jnp.expand_dims(actions, axis=1)
    actions = jnp.tile(actions, [1, num_components, 1])
    log_prob_given_pi = tfd.MultivariateNormalDiag(
        loc=means, scale_diag=stddevs).log_prob(actions) # B x num_components

    log_probs = jax.nn.logsumexp(log_pis + log_prob_given_pi, axis=-1)
    return log_probs

  return relaxed_gaussian_mixture_log_prob


def build_gaussian_mixture_sample(num_dimensions, num_components, eval_mode=False):
  def relaxed_gaussian_mixture_sample(params, key):
    logits = params[:, :num_components]
    pis = jax.nn.softmax(logits, axis=-1)
    pis = jnp.expand_dims(pis, axis=2)
    key, sub_key = jax.random.split(key)
    # cat_samples = tfd.OneHotCategorical(
    #     logits=logits, dtype=jnp.float32).sample(seed=sub_key)
    cat_samples = tfd.RelaxedOneHotCategorical(
        temperature=0.1, logits=logits).sample(seed=sub_key)
    cat_samples = jnp.expand_dims(cat_samples, axis=2)

    means = jnp.reshape(
        params[:, num_components:num_components + num_components*num_dimensions],
        [params.shape[0], num_components, num_dimensions])
    log_stddevs = jnp.reshape(
        params[:, -(num_components*num_dimensions):],
        [params.shape[0], num_components, num_dimensions])
    stddevs = jax.nn.softplus(log_stddevs) + _min_stddev
    if eval_mode:
      stddevs = jnp.full_like(stddevs, 1e-4)

    key, sub_key = jax.random.split(key)
    gaussian_samples = tfd.MultivariateNormalDiag(
        loc=means, scale_diag=stddevs).sample(seed=sub_key) # B x num_components x num_dims

    if eval_mode:
      pis = jnp.argmax(pis, axis=1)
      pis = jax.nn.one_hot(pis, num_classes=num_components, dtype=jnp.float32, axis=1)
    else:
      # pass
      pis = cat_samples

    samples = jnp.sum(gaussian_samples * pis, axis=1)

    return samples

  return relaxed_gaussian_mixture_sample


def build_q_filtered_actor(
    networks,
    beta,
    num_samples,
    use_img_encoder = False,
    with_uniform = True,
    ensemble_method = 'deep_ensembles',
    ensemble_size = None, # not used for deep ensembles
    mimo_using_obs_tile = False,
    mimo_using_act_tile = False,
  ):
  if ensemble_method not in ['deep_ensembles', 'mimo',]:
    raise NotImplementedError()

  def select_action(
      params,
      key,
      obs,
      ):
    key, sub_key = jax.random.split(key)
    policy_params = params[0]
    all_q_params = params[1]
    if use_img_encoder:
      img_encoder_params = params[2]
      obs = {
          'state_image': networks.img_encoder.apply(
              img_encoder_params, obs['state_image']),
          'state_dense': obs['state_dense']
      }

    dist = networks.policy_network.apply(policy_params, obs)
    acts = dist._sample_n(num_samples, sub_key)
    acts = acts[:,0,:] # N x act_dim

    if with_uniform:
      key, sub_key = jax.random.split(sub_key)
      unif_acts = jax.random.uniform(sub_key, acts.shape, dtype=acts.dtype, minval=-1., maxval=1.)
      acts = jnp.concatenate([acts, unif_acts], axis=0)

    if ensemble_method == 'deep_ensembles':
      get_all_q_values = jax.pmap(
          jax.vmap(networks.q_network.apply, in_axes=(0, None, None), out_axes=0),
          in_axes=(0, None, None),
          out_axes=0)
    elif ensemble_method == 'mimo':
      get_all_q_values = jax.pmap(
          jax.vmap(networks.q_network.apply, in_axes=(0, None, None), out_axes=0),
          in_axes=(0, None, None),
          out_axes=0)
    else:
      raise NotImplementedError()

    def obs_tile_fn(t):
      # t = jnp.expand_dims(t, axis=0)
      tile_shape = [1] * t.ndim
      # tile_shape[0] = num_samples
      tile_shape[0] = acts.shape[0]
      return jnp.tile(t, tile_shape)
    tiled_obs = jax.tree_map(obs_tile_fn, obs)

    if ensemble_method == 'deep_ensembles':
      # num_devices x num_per_device x batch_size x 2(because of double-Q)
      all_q = get_all_q_values(all_q_params, tiled_obs, acts)
      # num_devices x num_per_device x batch_size
      all_q = jnp.min(all_q, axis=-1)

      q_mean = jnp.mean(all_q, axis=(0, 1))
      q_std = jnp.std(all_q, axis=(0, 1))
      q_score = q_mean + beta * q_std # batch_size
      best_idx = jnp.argmax(q_score)
    elif ensemble_method == 'mimo':
      if mimo_using_obs_tile:
        # if using the version where we also tile the obs
        tile_shape = [1] * tiled_obs.ndim
        tile_shape[-1] = ensemble_size
        tiled_obs = jnp.tile(tiled_obs, tile_shape)

      if mimo_using_act_tile:
        # if using the version where we are tiling the acts
        tile_shape = [1] * acts.ndim
        tile_shape[-1] = ensemble_size
        tiled_acts = jnp.tile(acts, tile_shape)
      else:
        # otherwise
        tiled_acts = acts

      all_q = get_all_q_values(all_q_params, tiled_obs, tiled_acts) # 1 x 1 x batch_size x ensemble_size x (num_qs_per_member)
      all_q = jnp.min(all_q, axis=-1) # 1 x 1 x batch_size x ensemble_size

      q_mean = jnp.mean(all_q, axis=(0, 1, 3))
      q_std = jnp.std(all_q, axis=(0, 1, 3))
      q_score = q_mean + beta * q_std # batch_size
      best_idx = jnp.argmax(q_score)
    else:
      raise NotImplementedError()

    # return acts[best_idx], key
    return acts[best_idx][None,:]

  # return actor_core.ActorCore(
  #     init=lambda key: key,
  #     select_action=select_action,
  #     get_extras=lambda x: ())
  # return select_action
  return actor_core.batched_feed_forward_to_actor_core(select_action)


def apply_policy_and_sample(
    networks,
    eval_mode = False,
    use_img_encoder = False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    if use_img_encoder:
      params, encoder_params = params[0], params[1]
      obs = {
          'state_image': networks.img_encoder.apply(
              encoder_params, obs['state_image']),
          'state_dense': obs['state_dense']
      }
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



def make_networks(
    spec,
    actor_hidden_layer_sizes = (256, 256),
    critic_hidden_layer_sizes = (256, 256),
    init_type = 'glorot_except_dist',
    critic_init_scale = 1.0,
    use_double_q = True,
    img_encoder_fn=None,
    build_kernel_fn = False,
    ensemble_method = 'deep_ensembles',
    ensemble_size = None, # this is not used for deep ensembles
    mimo_using_obs_tile = False,
    mimo_using_act_tile = False,):
  """Creates networks used by the agent."""
  assert not (build_kernel_fn and (img_encoder_fn is not None))
  if ensemble_method not in [
      'deep_ensembles', 'mimo', 'tree_deep_ensembles',
      'efficient_tree_deep_ensembles']:
    raise NotImplementedError()

  num_dimensions = np.prod(spec.actions.shape, dtype=int)

  if init_type == 'glorot_except_dist':
    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
    b_init = jnp.zeros
    dist_w_init = hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform')
    dist_b_init = jnp.zeros
  elif init_type == 'glorot_also_dist':
    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
    b_init = jnp.zeros
    dist_w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
    dist_b_init = jnp.zeros
  elif init_type =='he_normal':
    w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")
    b_init = jnp.zeros
    dist_w_init = w_init
    dist_b_init = b_init
  elif init_type == 'Ilya':
    assert False, 'This is not correct'
    relu_orthogonal = hk.initializers.Orthogonal(scale=2.0**0.5)
    near_zero_orthogonal = hk.initializers.Orthogonal(1e-2)
    w_init = relu_orthogonal
    b_init = jnp.zeros
    dist_w_init = near_zero_orthogonal
    dist_b_init = jnp.zeros
  else:
    raise NotImplementedError

  NUM_MIXTURE_COMPONENTS = 5 # if using gaussian mixtures
  rlu_uniform_initializer = hk.initializers.VarianceScaling(
      distribution='uniform', mode='fan_out', scale=0.333)
  # rlu_uniform_initializer = hk.initializers.VarianceScaling(scale=1e-4)
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

    # w_init = hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
    # b_init = jnp.zeros

    # PAPER VERSION
    network = hk.Sequential([
        hk.nets.MLP(
            list(actor_hidden_layer_sizes),
            # w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            # w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
            w_init=w_init,
            b_init=b_init,
            activation=jax.nn.relu,
            # activation=jax.nn.tanh,
            activate_final=True),
        networks_lib.NormalTanhDistribution(
            num_dimensions,
            w_init=dist_w_init,
            b_init=dist_b_init,
            min_scale=1e-2,
            ),
        # networks_lib.MultivariateNormalDiagHead(
        #     num_dimensions,
        #     w_init=w_init,
        #     b_init=b_init),
        # networks_lib.GaussianMixture(
        #     num_dimensions,
        #     num_components=5,
        #     multivariate=True),
        # hk.Linear(
        #     NUM_MIXTURE_COMPONENTS + 2 * NUM_MIXTURE_COMPONENTS * num_dimensions,
        #     with_bias=True,
        #     w_init=dist_w_init,
        #     b_init=dist_b_init,),
    ])
    return network(obs)

#   def _actor_fn(obs):
#     # inspired by the ones used in RL Unplugged
#     x = obs
#     x = hk.Sequential([
#         hk.Linear(300, w_init=rlu_uniform_initializer),
#         hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
#         jax.lax.tanh,])(x)
#     x = hk.Linear(1024, w_init=rlu_uniform_initializer)(x)
#     for i in range(4):
#       x = network_utils.ResidualLayerNormBlock(
#           [1024, 1024],
#           activation=jax.nn.relu,
#           w_init=rlu_uniform_initializer,)(x)

#     # a = hk.Linear(
#     #     NUM_MIXTURE_COMPONENTS + 2 * NUM_MIXTURE_COMPONENTS * num_dimensions,
#     #     with_bias=True,
#     #     w_init=hk.initializers.VarianceScaling(scale=1e-5, mode='fan_in'),)(x)
#     a = networks_lib.NormalTanhDistribution(
#         num_dimensions,
#         w_init=dist_w_init,
#         b_init=dist_b_init,
#         min_scale=1e-2,)(x)
#     # a = networks_lib.MultivariateNormalDiagHead(
#     #     num_dimensions,
#     #     min_scale=1e-2,
#     #     w_init=dist_w_init,
#     #     b_init=dist_b_init,)(x)
#     return a

  critic_output_dim = 1
  if ensemble_method in ['mimo', 'tree_deep_ensembles',
                         'efficient_tree_deep_ensembles']:
    critic_output_dim = ensemble_size

  def small_critic(x):
    # i.e. what people typically use for d4rl benchmark
    _mlp = hk.nets.MLP(
        list(critic_hidden_layer_sizes),
        # w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
        # w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
        # w_init=hk.initializers.VarianceScaling(critic_init_scale, "fan_avg", "truncated_normal"),
        w_init=w_init,
        b_init=b_init,
        activation=jax.nn.relu,
        # activation=jax.nn.tanh,
        activate_final=True)
    h = _mlp(x)
    _linear = hk.Linear(critic_output_dim, w_init=w_init, b_init=b_init)
    v = _linear(h)
    return v, h

  # def small_critic(x):
  #   # this one is for exploring maximal parameterization
  #   width = 256
  #   x = hk.Linear(
  #       width,
  #       w_init=hk.initializers.VarianceScaling(scale=1.0, mode='fan_out', distribution='truncated_normal'),
  #       b_init=hk.initializers.VarianceScaling(scale=0.05, mode='fan_out', distribution='truncated_normal'))(x)
  #   x = x * (float(width) ** 0.5)
  #   x = jax.nn.relu(x)
  #   x = hk.Linear(
  #       width,
  #       w_init=hk.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal'),
  #       b_init=hk.initializers.VarianceScaling(scale=0.05, mode='fan_in', distribution='truncated_normal'),)(x)
  #   x = jax.nn.relu(x)
  #   x = hk.Linear(
  #       width,
  #       w_init=hk.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal'),
  #       b_init=hk.initializers.VarianceScaling(scale=0.05, mode='fan_in', distribution='truncated_normal'),)(x)
  #   x = jax.nn.relu(x)
  #   h = x
  #   x = hk.Linear(
  #       1,
  #       w_init=hk.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal'),
  #       b_init=hk.initializers.VarianceScaling(scale=0.05, mode='fan_in', distribution='truncated_normal'),)(x)
  #   x = x / (float(width) ** 0.5)
  #   return x, h

  # def large_critic(x):
  #   # inspired by the ones used in RL Unplugged, but smaller hidden layer sizes
  #   hid_dim = 256
  #   _encoder = hk.Linear(hid_dim, w_init=w_init, b_init=b_init)
  #   x = _encoder(x)
  #   for i in range(4):
  #     x = network_utils.ResidualLayerNormBlock(
  #         [hid_dim, hid_dim],
  #         activation=jax.nn.relu,
  #         w_init=w_init,
  #         b_init=b_init,)(x)
  #   h = hk.Linear(hid_dim, w_init=w_init, b_init=b_init)(x)
  #   v = hk.Linear(critic_output_dim, w_init=w_init, b_init=b_init)(h)
  #   return v, h
  def large_critic(x):
    # inspired by the ones used in RL Unplugged
    x = hk.Sequential([
        hk.Linear(400, w_init=rlu_uniform_initializer),
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        jax.lax.tanh,])(x)
    x = hk.Linear(1024, w_init=rlu_uniform_initializer)(x)
    for i in range(4):
      x = network_utils.ResidualLayerNormBlock(
          [1024, 1024],
          activation=jax.nn.relu,
          w_init=rlu_uniform_initializer,)(x)
    h = x
    # v = hk.Linear(1, w_init=rlu_uniform_initializer)(h)
    # v = hk.Linear(critic_output_dim)(h)
    all_vs = []
    for _ in range(critic_output_dim):
      head_v = hk.Linear(256, w_init=rlu_uniform_initializer)(h)
      head_v = jax.nn.relu(head_v)
      head_v = hk.Linear(1, w_init=rlu_uniform_initializer)(head_v)
      all_vs.append(head_v)
    v = jnp.concatenate(all_vs, axis=-1)
    return v, h

  # def _critic_fn(obs, action):
  def _all_critic_stuff(obs, action):
    # for matching Ilya's codebase
    # relu_orthogonal = hk.initializers.Orthogonal(scale=2.0**0.5)
    # near_zero_orthogonal = hk.initializers.Orthogonal(1e-2)
    # def _cn(x):
    #   for hid_dim in critic_hidden_layer_sizes:
    #     x = hk.Linear(hid_dim, w_init=relu_orthogonal, b_init=jnp.zeros)(x)
    #     x = jax.nn.relu(x)
    #   x = hk.Linear(1, w_init=near_zero_orthogonal, b_init=jnp.zeros)(x)
    #   return x
    # input_ = jnp.concatenate([obs, action], axis=-1)
    # if use_double_q:
    #   value1 = _cn(input_)
    #   value2 = _cn(input_)
    #   return jnp.concatenate([value1, value2], axis=-1)
    # else:
    #   return _cn(input_)

    # w_init = hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
    # b_init = jnp.zeros

    #####################################
    input_ = jnp.concatenate([obs, action], axis=-1)

    if ensemble_method == 'tree_deep_ensembles':
      critic_network_builder = network_utils.build_tree_deep_ensemble_critic(
          w_init, b_init, use_double_q)
    elif ensemble_method == 'efficient_tree_deep_ensembles':
      critic_network_builder = network_utils.build_efficient_tree_deep_ensemble_critic(
          w_init, b_init, use_double_q)
    else:
      # for standard d4rl architecture
      critic_network_builder = small_critic
      # for larger architecture inspired by rl unplugged
      # critic_network_builder = large_critic

    value1, h1 = critic_network_builder(input_)
    if ensemble_method in ['mimo', 'tree_deep_ensembles',
                           'efficient_tree_deep_ensembles']:
      value1 = jnp.reshape(value1, [-1, ensemble_size, 1])

    if use_double_q:
      value2, h2 = critic_network_builder(input_)
      if ensemble_method in ['mimo', 'tree_deep_ensembles',
                             'efficient_tree_deep_ensembles']:
        value2 = jnp.reshape(value2, [-1, ensemble_size, 1])
      return jnp.concatenate([value1, value2], axis=-1), jnp.concatenate([h1, h2], axis=-1)
    else:
      return value1, h1

  def get_particular_critic_init(w_init, b_init, key, obs, act):
    def _critic_with_particular_init(obs, action):
      raise NotImplementedError('Not implemented for MIMO, Not implemented for new version that also returns h1, h2')
      network1 = hk.Sequential([
          hk.nets.MLP(
              list(critic_hidden_layer_sizes) + [1],
              w_init=w_init,
              b_init=b_init,
              activation=jax.nn.relu,
              activate_final=False),
      ])
      input_ = jnp.concatenate([obs, action], axis=-1)
      value1 = network1(input_)
      if use_double_q:
        network2 = hk.Sequential([
            hk.nets.MLP(
                list(critic_hidden_layer_sizes) + [1],
                w_init=w_init,
                b_init=b_init,
                activation=jax.nn.relu,
                activate_final=False),
        ])
        value2 = network2(input_)
        return jnp.concatenate([value1, value2], axis=-1)
      else:
        return value1

    init_fn = hk.without_apply_rng(
        hk.transform(_critic_with_particular_init, apply_rng=True)).init
    return init_fn(key, obs, act)

  kernel_fn = None
  if build_kernel_fn:
    layers = []
    for hid_dim in critic_hidden_layer_sizes:
      # W_std = 1.5
      W_std = 2.0
      layers += [
          stax.Dense(hid_dim, W_std=W_std, b_std=0.05),
          stax.Relu()
      ]
    layers += [stax.Dense(1, W_std=W_std, b_std=0.05)]
    nt_init_fn, nt_apply_fn, nt_kernel_fn = stax.serial(*layers)
    kernel_fn = jax.jit(nt_kernel_fn, static_argnums=(2,))

  if img_encoder_fn is not None:
    # _actor_fn = bimanual_sweep.policy_on_encoder_v0(num_dimensions)
    # _critic_fn = bimanual_sweep.critic_on_encoder_v0(use_double_q=use_double_q)
    _actor_fn = bimanual_sweep.policy_on_encoder_v1(num_dimensions)
    raise NotImplementedError('Need to handle the returning of h1, h2 with new version of all_critic_stuff')
    _critic_fn = bimanual_sweep.critic_on_encoder_v1(use_double_q=use_double_q)

  def _simclr_encoder(h):
    # return hk.nets.MLP(
    #     [256, 128],
    #     # [256, 256, 256],
    #     w_init=w_init,
    #     # b_init=b_init, # b_init should not be set when not using bias
    #     with_bias=False,
    #     activation=jax.nn.relu,
    #     activate_final=False)(h)

    # IF YOU CHANGE THIS AND USE SASS, YOU NEED TO FIX THE SASS ENCODER OPTIM STEP
    return h  # i.e. no encoder (sometimes referred to as "projection")

  policy = hk.without_apply_rng(hk.transform(_actor_fn, apply_rng=True))
  # critic = hk.without_apply_rng(hk.transform(_critic_fn, apply_rng=True))
  all_critic_stuff = hk.without_apply_rng(hk.transform(_all_critic_stuff, apply_rng=True))
  critic_init = all_critic_stuff.init
  critic_apply = lambda p, obs, act: all_critic_stuff.apply(p, obs, act)[0]
  critic_repr = lambda p, obs, act: all_critic_stuff.apply(p, obs, act)[1]

  simclr_encoder = hk.without_apply_rng(hk.transform(_simclr_encoder, apply_rng=True))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)
  tile_shape = [1 for _ in range(dummy_action.ndim)]
  tile_shape[0] = 256
  dummy_action = jnp.tile(dummy_action, tile_shape)
  tile_shape = [1 for _ in range(dummy_obs.ndim)]
  tile_shape[0] = 256
  dummy_obs = jnp.tile(dummy_obs, tile_shape)

  if img_encoder_fn is not None:
    img_encoder = hk.without_apply_rng(
        hk.transform(img_encoder_fn, apply_rng=True))
    key = jax.random.PRNGKey(seed=42)
    temp_encoder_params = img_encoder.init(key, dummy_obs['state_image'])
    dummy_hidden = img_encoder.apply(temp_encoder_params, dummy_obs['state_image'])
    img_encoder_network = networks_lib.FeedForwardNetwork(
        lambda key: img_encoder.init(key, dummy_hidden), img_encoder.apply)
    dummy_encoded_input = dict(
        state_image=dummy_hidden,
        state_dense=dummy_obs['state_dense'],)
  else:
    img_encoder_fn = None
    dummy_encoded_input = dummy_obs
    img_encoder_network = None

  critic_dummy_encoded_input = dummy_encoded_input
  critic_dummy_action = dummy_action
  if ensemble_method == 'mimo':
    if mimo_using_obs_tile:
      # if using the version where we are also tiling the obs
      tile_array = [1]*len(critic_dummy_encoded_input.shape) # type: ignore
      tile_array[-1] = ensemble_size
      critic_dummy_encoded_input = jnp.tile(critic_dummy_encoded_input, tile_array)

    if mimo_using_act_tile:
      # if using the version where we are also tiling the acts
      tile_array = [1]*len(critic_dummy_action.shape)
      tile_array[-1] = ensemble_size
      critic_dummy_action = jnp.tile(critic_dummy_action, tile_array)

  temp_critic_params = critic_init(
      jax.random.PRNGKey(42), critic_dummy_encoded_input, critic_dummy_action)
  dummy_critic_repr = critic_repr(
      temp_critic_params, critic_dummy_encoded_input, critic_dummy_action)

  # mixture_sample = build_gaussian_mixture_sample(num_dimensions, NUM_MIXTURE_COMPONENTS, eval_mode=False)
  # mixture_sample_eval = build_gaussian_mixture_sample(num_dimensions, NUM_MIXTURE_COMPONENTS, eval_mode=True)
  # mixture_log_prob = build_gaussian_mixture_log_prob(num_dimensions, NUM_MIXTURE_COMPONENTS)


  return MSGNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_encoded_input), policy.apply),
      q_network=networks_lib.FeedForwardNetwork(
          lambda key: critic_init(key, critic_dummy_encoded_input, critic_dummy_action),
          critic_apply),

      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      # sample_eval=lambda params, key: params.mode(),
      sample_eval=lambda params, key: params.sample(seed=key),

      # log_prob=mixture_log_prob,
      # sample=mixture_sample,
      # # sample_eval=lambda params, key: params.mode(),
      # sample_eval=mixture_sample_eval,

      img_encoder=img_encoder_network,
      kernel_fn=kernel_fn,
      get_particular_critic_init=lambda w_init, b_init, key: get_particular_critic_init(
          w_init, b_init, key, dummy_encoded_input, dummy_action),
      get_critic_repr=critic_repr,
      simclr_encoder=networks_lib.FeedForwardNetwork(
          lambda key: simclr_encoder.init(key, dummy_critic_repr),
          simclr_encoder.apply),
      )
