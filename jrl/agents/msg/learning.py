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
"""MSG learner implementation."""

import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
from collections import OrderedDict
from copy import deepcopy

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
import tensorflow_probability
from jrl.agents.msg import networks as msg_networks
from jrl.utils.sass import sass_utils

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
distributional = networks_lib.distributional



class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  all_q_optimizer_states: optax.OptState
  policy_params: networks_lib.Params
  all_q_params: networks_lib.Params
  all_target_q_params: networks_lib.Params
  key: networks_lib.PRNGKey
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[networks_lib.Params] = None
  img_encoder_params: Optional[networks_lib.Params] = {}
  simclr_encoder_params: Optional[networks_lib.Params] = {}
  simclr_encoder_optim_state: Optional[optax.OptState] = {}


class MSGLearner(acme.Learner):
  """MSG learner."""

  _state: TrainingState

  def __init__(
      self,
      ensemble_size,
      beta,
      networks,
      rng,
      iterator,
      ensemble_method = 'deep_ensembles',
      perform_sarsa_q_eval = False,
      num_q_repr_pretrain_iters = 0,
      pretrain_temp = 1.,
      use_sass = False,
      num_bc_iters = 50_000,
      use_random_weighting_in_critic_loss = True,
      use_ema_target_critic_params = True,
      tau = 0.005,
      reward_scale = 1.0,
      discount = 0.99,
      entropy_coefficient = None,
      target_entropy = 0,
      use_entropy_regularization = True,
      behavior_regularization_type = 'none',
      behavior_regularization_alpha = 1.0,
      num_cql_actions = 1,
      td_target_method = 'independent', # MSG is 'independent'
      critic_random_init = False,
      use_img_encoder = False,
      img_encoder_params_ckpt_path = '',
      rem_mode = False, # REM is a baseline with close implementation
      # some params for MIMO or multi-head MSG
      mimo_using_obs_tile = False,
      mimo_using_act_tile = False,
      mimo_using_adamw = False,
      policy_lr = 1e-4,
      q_lr = 3e-4,
      counter = None,
      logger = None,
      num_sgd_steps_per_step = 1):
    """Initialize the MSG learner.

    Args:
      ensemble_size: Number of elements in the ensemble.
      beta: the beta value for MSG.
      networks: MSG networks
      rng: a key for random number generation.
      iterator: an iterator over training data.
      perform_sarsa_q_eval: whether to do sarsa-style q-evaluation using
        actions from the offline dataset, no policy training.
      num_q_repr_pretrain_iters: number of iterations where we pretrain the Q-fn
        representations e.g. using SIMCLR
      pretrain_temp: temperature for the pretraining objective
      use_sass: whether to use the pretrain objective during training as well
      num_bc_iters: number of initial iterations for bc as the actor update
      use_random_weighting_in_critic_loss: whether to use the random weighting
      use_ema_target_critic_params: whether to use EMA for the target critic
        params, otherwise copy the critic params after every
        utils.process_multiple_batches which puts a jax.lax.scan on top of
        the full update step.
      tau: target smoothing coefficient.
      reward_scale: reward scale.
      discount: discount to use for TD updates.
      entropy_coefficient: coefficient applied to the entropy bonus. If None, an
        adaptative coefficient will be used.
      target_entropy: Used to normalize entropy. Only used when
        entropy_coefficient is None.
      use_entropy_regularization: whether to do entropy regularization for the
        policy.
      behavior_regularization_type: which kind of behavior regularization to
        use.
      behavior_regularization_alpha: the weight for behavior regularization
      num_cql_actions: if using cql for regularization, how many actions to use
        for importance sampling.
      use_img_encoder: Whether to first process the image with a pretrained
        encoder
      img_encoder_params_ckpt_path: Path to the checkpoint for the image encoder
      rem_mode: setting this mode runs the REM baseline instead
      policy_lr: learning rate for the policy
      q_lr: learning rate for the q functions
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      num_sgd_steps_per_step: number of sgd steps to perform per learner 'step'.
    """
    num_devices = len(jax.devices())
    assert ensemble_size % num_devices == 0
    if ensemble_method == 'deep_ensembles':
      num_models_per_device = ensemble_size // num_devices
    else:
      num_models_per_device = 1

    if ensemble_method not in ['deep_ensembles', 'mimo', 'tree_deep_ensembles',
                               'efficient_tree_deep_ensembles']:
      raise NotImplementedError(f'{ensemble_method} not implemented!')
    if ensemble_method in ['mimo', 'tree_deep_ensembles',
                           'efficient_tree_deep_ensembles']:
      if use_sass:
        raise NotImplementedError('Sass not implemented for this ensemble method!')
    if any([mimo_using_obs_tile, mimo_using_act_tile, mimo_using_adamw]):
      assert ensemble_method == 'mimo', 'These params are for MIMO or multi-head only!'
    if num_devices > 1:
      assert ensemble_method == 'deep_ensembles', \
          'Only deep ensembles is implemented for multi-devices (e.g. TPU)!'

    if num_q_repr_pretrain_iters > 0:
      supported_pretrain = ['deep_ensembles', 'mimo', 'tree_deep_ensembles',
                            'efficient_tree_deep_ensembles']
      if ensemble_method not in supported_pretrain:
        raise NotImplementedError(f'Pretraining not implemented for'
                                  'ensemble method {ensemble_method}!')
      if mimo_using_obs_tile or mimo_using_act_tile:
        raise NotImplementedError('So far only multi-head implemented!')

    self._ensemble_method = ensemble_method
    self._perform_sarsa_q_eval = perform_sarsa_q_eval
    self._num_q_repr_pretrain_iters = num_q_repr_pretrain_iters
    self._num_bc_iters = num_bc_iters
    self._num_sgd_steps_per_step = num_sgd_steps_per_step
    self._use_img_encoder = use_img_encoder
    self._use_ema_target_critic_params = use_ema_target_critic_params

    policy_optimizer = optax.adam(learning_rate=policy_lr)

    if mimo_using_adamw:
      q_optimizer = optax.adamw(learning_rate=q_lr, weight_decay=1e-3)
    else:
      q_optimizer = optax.adam(learning_rate=q_lr)

    ################
    pretrain_optimizer = optax.adam(learning_rate=q_lr)
    ################
    # lamb_kwargs=dict(
    #     b1=0.9,
    #     b2=0.999,
    #     eps=1e-6,)
    # optax_chain = []
    # optax_chain.extend([
    #     optax.scale_by_adam(**lamb_kwargs),
    #     # optimizers.add_weight_decay(1e-6),
    #     optax.add_decayed_weights(1e-6, True),
    #     optax.scale_by_trust_ratio(),
    #   ])
    # # warmup_steps = 50_000
    # warmup_steps = 100_000
    # # Batch scale the other lr values as well:
    # init_value = 0.
    # end_value = 0.
    # # base_lr = (2.5e-4 * 32.) / 256.
    # base_lr = 0.3
    # total_steps = 1_000_000
    # schedule_fn = optax.warmup_cosine_decay_schedule(
    #     init_value=init_value,
    #     peak_value=base_lr,
    #     warmup_steps=warmup_steps,
    #     decay_steps=total_steps,
    #     end_value=end_value)
    # optax_chain.extend([
    #     optax.scale_by_schedule(schedule_fn),
    #     optax.scale(-1),
    # ])
    # pretrain_optimizer = optax.chain(*optax_chain)
    ################

    adaptive_entropy_coefficient = entropy_coefficient is None
    if adaptive_entropy_coefficient:
      # alpha is the temperature parameter that determines the relative
      # importance of the entropy term versus the reward.
      log_alpha = jnp.asarray(0., dtype=jnp.float32)
      alpha_optimizer = optax.adam(learning_rate=3e-4)
      alpha_optimizer_state = alpha_optimizer.init(log_alpha)
    else:
      if target_entropy:
        raise ValueError('target_entropy should not be set when '
                         'entropy_coefficient is provided')

    def per_critic_loss(
        critic_params,
        target_critic_params,
        transitions,
        cur_act_dist_params,
        next_act_dist_params,
        key,
        encoder_params,):
      '''Implements per ensemble member critic loss for deep ensembles.'''

      obs = transitions.observation
      acts = transitions.action
      next_obs = transitions.next_observation

      # cur_act_dist = cur_act_dist_params
      cur_act_dist = tfd.Normal(
          loc=cur_act_dist_params[0], scale=cur_act_dist_params[1])
      cur_act_dist = tfd.Independent(
          distributional.TanhTransformedDistribution(cur_act_dist),
          reinterpreted_batch_ndims=1)
      # cur_act_dist = tfd.MultivariateNormalDiag(
      #     loc=cur_act_dist_params[0], scale_diag=cur_act_dist_params[1])

      # next_act_dist = next_act_dist_params
      next_act_dist = tfd.Normal(
          loc=next_act_dist_params[0], scale=next_act_dist_params[1])
      next_act_dist = tfd.Independent(
          distributional.TanhTransformedDistribution(next_act_dist),
          reinterpreted_batch_ndims=1)
      # next_act_dist = tfd.MultivariateNormalDiag(
      #     loc=next_act_dist_params[0], scale_diag=next_act_dist_params[1])

      if self._perform_sarsa_q_eval:
        next_acts = transitions.extras
      else:
        key, sub_key = jax.random.split(key)
        next_acts = networks.sample(next_act_dist, sub_key)

      q_cur = networks.q_network.apply(critic_params, obs, acts)
      target_q_next = networks.q_network.apply(
          target_critic_params, next_obs, next_acts)
      target_q_next = jnp.min(target_q_next, axis=-1)
      y = jax.lax.stop_gradient(transitions.reward * reward_scale +
                                transitions.discount * discount * target_q_next)
      if td_target_method == 'independent':
        pass
      elif td_target_method == 'shared_mean':
        y = jax.lax.pmean(y, axis_name='within_device')
        y = jax.lax.pmean(y, axis_name='across_devices')
      elif td_target_method == 'shared_min':
        y = jax.lax.pmin(y, axis_name='within_device')
        y = jax.lax.pmin(y, axis_name='across_devices')
      elif td_target_method == 'shared_lcb':
        y_mean = jax.lax.pmean(y, axis_name='within_device')
        y_mean = jax.lax.pmean(y_mean, axis_name='across_devices')
        y_squared_mean = jax.lax.pmean(y**2, axis_name='within_device')
        y_squared_mean = jax.lax.pmean(y_squared_mean, axis_name='across_devices')
        y_var = jax.lax.max(0., y_squared_mean - y_mean**2)
        y_std = y_var ** 0.5
        y = y_mean + beta * y_std
      else:
        raise NotImplementedError()

      if rem_mode:
        key, sub_key = jax.random.split(key)
        rem_alpha_shape = list(q_cur.shape)
        rem_alpha_shape[-1] = 1
        rem_alpha_shape = tuple(rem_alpha_shape)
        rem_alpha = jax.random.uniform(sub_key, rem_alpha_shape, q_cur.dtype)
        rem_alpha_sum = jax.lax.psum(
            jax.lax.psum(rem_alpha, axis_name='within_device'),
            axis_name='across_devices')
        rem_alpha = jax.lax.stop_gradient(rem_alpha / rem_alpha_sum)

        rem_q_cur = rem_alpha * q_cur
        rem_q_cur = jax.lax.psum(
            jax.lax.psum(rem_q_cur, axis_name='within_device'),
            axis_name='across_devices')

        rem_y = rem_alpha * y
        rem_y = jax.lax.psum(
            jax.lax.psum(rem_y, axis_name='within_device'),
            axis_name='across_devices')
        rem_y = jax.lax.stop_gradient(rem_y)

        q_cur = rem_q_cur
        y = rem_y

      q_error = jnp.square(q_cur - jnp.expand_dims(y, axis=-1))
      # In our experience this doesn't matter, we just leave it on by default
      if use_random_weighting_in_critic_loss:
        q_error *= jax.random.uniform(
            key, q_error.shape, q_error.dtype, minval=0., maxval=2.)
      q_loss = 0.5 * jnp.mean(q_error)

      q_preds = jnp.min(q_cur, axis=-1)
      behavior_reg_loss = 0.
      fraction_active = 0.

      if behavior_regularization_type == 'v1':
        key, sub_key = jax.random.split(key)
        policy_acts_for_obs = networks.sample(cur_act_dist, sub_key)
        q_for_policy_acts = networks.q_network.apply(
            critic_params, obs, policy_acts_for_obs)
        q_for_policy_acts = jnp.min(q_for_policy_acts, axis=-1)
        behavior_reg_loss = jnp.mean(q_for_policy_acts - q_preds)
        q_loss = q_loss + behavior_regularization_alpha * behavior_reg_loss
        fraction_active = 1.

      elif behavior_regularization_type == 'v1_prime':
        key, sub_key = jax.random.split(key)
        policy_acts_for_obs = networks.sample(cur_act_dist, sub_key)
        q_for_policy_acts = networks.q_network.apply(
            critic_params, obs, policy_acts_for_obs)
        behavior_reg_loss = jnp.mean(
            jnp.sum(q_for_policy_acts - q_cur, axis=-1))
        q_loss = q_loss + behavior_regularization_alpha * behavior_reg_loss
        fraction_active = 1.

      elif behavior_regularization_type == 'v2':
        key, sub_key = jax.random.split(key)
        policy_acts_for_obs = networks.sample(cur_act_dist, sub_key)
        q_for_policy_acts = networks.q_network.apply(
            critic_params, obs, policy_acts_for_obs)
        q_for_policy_acts = jnp.min(q_for_policy_acts, axis=-1)
        q_sign = (jnp.sign(q_for_policy_acts - q_preds) + 1.)/2.
        q_sign = jax.lax.stop_gradient(q_sign)
        behavior_reg_loss = jnp.mean(q_sign * q_for_policy_acts)
        q_loss = q_loss + behavior_regularization_alpha * behavior_reg_loss
        fraction_active = jnp.mean(q_sign)

      elif behavior_regularization_type == 'v3':
        # same as v1 but clipped
        key, sub_key = jax.random.split(key)
        policy_acts_for_obs = networks.sample(cur_act_dist, sub_key)
        q_for_policy_acts = networks.q_network.apply(
            critic_params, obs, policy_acts_for_obs)
        q_for_policy_acts = jnp.min(q_for_policy_acts, axis=-1)
        behavior_reg_loss = jnp.mean(
            jax.nn.relu(q_for_policy_acts - q_preds))
        q_loss = q_loss + behavior_regularization_alpha * behavior_reg_loss
        fraction_active = 1.

      elif behavior_regularization_type == 'cql':
        def tile_fn(t):
          tile_shape = [1] * t.ndim
          tile_shape[0] = num_cql_actions
          return jnp.tile(t, tile_shape)

        obs = transitions.observation
        # policy actions for cql
        key, sub_key = jax.random.split(key)
        policy_dist = cur_act_dist
        policy_acts = policy_dist._sample_n(num_cql_actions, sub_key) # NxBx1
        policy_acts_log_probs = networks.log_prob(policy_dist, policy_acts) # NxB

        # uniform actions for cql
        key, sub_key = jax.random.split(key)
        unif_acts = jax.random.uniform(
            sub_key,
            shape=policy_acts.shape,
            dtype=policy_acts.dtype,
            minval=-1.,
            maxval=1.,)
        unif_acts_log_prob = jnp.ones(
            policy_acts_log_probs.shape, policy_acts_log_probs.dtype) * (
                jnp.log(1./2.) * transitions.action.shape[-1])

        # compute the regularization loss
        # get the q-values
        def obs_tile_fn(t):
          tile_shape = [1] * t.ndim
          tile_shape[0] = 2*num_cql_actions
          return jnp.tile(t, tile_shape)
        tiled_obs = jax.tree_map(obs_tile_fn, obs)
        concat_acts = jnp.concatenate([policy_acts, unif_acts], axis=0)
        concat_acts = jnp.reshape(
            concat_acts,
            [2*num_cql_actions*acts.shape[0], concat_acts.shape[-1]])
        # q_for_cql_acts: 2(NxB) x (2 or 1)
        q_for_cql_acts = networks.q_network.apply(
            critic_params, tiled_obs, concat_acts)
        q_for_cql_acts = jnp.reshape(
            q_for_cql_acts,
            [2*num_cql_actions, acts.shape[0], -1]) # 2N x B x (2 or 1)
        all_log_probs = jnp.concatenate(
            [policy_acts_log_probs, unif_acts_log_prob], axis=0) # 2N x B
        all_log_probs = jnp.expand_dims(all_log_probs, axis=-1)
        # all_log_probs = jnp.reshape(
        #     all_log_probs, [2*num_cql_actions, obs.shape[0], 1]) # 2N x B x 1
        combined = q_for_cql_acts - all_log_probs # 2N x B x (2 or 1)
        logsumexp_term = jax_logsumexp(combined, axis=0) # B x (2 or 1)

        cql_term = jnp.mean(jnp.sum(logsumexp_term - q_cur, axis=-1))
        q_loss = q_loss + behavior_regularization_alpha * cql_term

        behavior_reg_loss = cql_term
        fraction_active = 1.
      else:
        assert behavior_regularization_type in ['none', 'spec_norm']

      # if using the pretrain objective during training as well
      if use_sass:
        key, sub_key = jax.random.split(key)
        sass_loss, sass_acc = per_q_repr_pretrain_loss(
            q_params=critic_params,
            target_q_params=target_critic_params,
            encoder_params=encoder_params,
            transitions=transitions,
            key=sub_key)
      else:
        sass_loss, sass_acc = 0., 0.

      # q_loss = q_loss + 0.1 * sass_loss
      q_loss = q_loss + 1.0 * sass_loss

      return q_loss, (q_preds, behavior_reg_loss, fraction_active, sass_loss, sass_acc)


    per_critic_loss_and_grad = jax.value_and_grad(per_critic_loss, has_aux=True, argnums=(0, 6))


    def per_critic_update_step(
        critic_params,
        target_critic_params,
        transitions,
        optim_state,
        cur_act_dist_params,
        next_act_dist_params,
        key,
        encoder_params,
        encoder_optim_state,):
      key, sub_key = jax.random.split(key)
      (critic_loss, (q_preds, behavior_reg_loss, fraction_active, sass_loss, sass_acc)), (critic_grad, encoder_grad) = per_critic_loss_and_grad(
          critic_params,
          target_critic_params,
          transitions,
          cur_act_dist_params,
          next_act_dist_params,
          sub_key,
          encoder_params,)

      critic_update, optim_state = q_optimizer.update(critic_grad, optim_state)
      critic_params = optax.apply_updates(critic_params, critic_update)

      if self._use_ema_target_critic_params:
        target_critic_params = jax.tree_multimap(
            lambda x, y: x * (1 - tau) + y * tau,
            target_critic_params, critic_params)

      # if use_sass:
      #   encoder_update, optim_state = pretrain_optimizer.update(encoder_grad, encoder_optim_state)
      #   encoder_params = optax.apply_updates(encoder_params, encoder_update)

      return critic_params, target_critic_params, optim_state, encoder_params, encoder_optim_state, critic_loss, q_preds, behavior_reg_loss, fraction_active, sass_loss, sass_acc


    def _mimo_batch(key, x):
      batch_size = x[0].shape[0]
      main_shuffle = jnp.arange(batch_size)
      to_shuffle = batch_size // 2

      key, sub_key = jax.random.split(key)
      inds = jnp.concatenate(
          [
              jax.random.permutation(sub_key, main_shuffle[:to_shuffle]),
              main_shuffle[to_shuffle:],

              # jax.random.permutation(sub_key, main_shuffle),
              # main_shuffle,

              # main_shuffle,
          ], axis=0)
      # inds = main_shuffle
      x = jax.tree_map(lambda t: jnp.take(t, inds, axis=0), x)
      return x

    mimo_batch = jax.vmap(_mimo_batch, in_axes=(0, None), out_axes=0)

    def mimo_critic_loss(
        critic_params,
        target_critic_params,
        transitions,
        cur_act_dist_params,
        next_act_dist_params,
        key):
      '''Critics loss when ensmeble method is MIMO.'''
      # cur_act_dist = cur_act_dist_params
      cur_act_dist = tfd.Normal(
          loc=cur_act_dist_params[0], scale=cur_act_dist_params[1])
      cur_act_dist = tfd.Independent(
          distributional.TanhTransformedDistribution(cur_act_dist),
          reinterpreted_batch_ndims=1)
      # cur_act_dist = tfd.MultivariateNormalDiag(
      #     loc=cur_act_dist_params[0], scale_diag=cur_act_dist_params[1])

      # next_act_dist = next_act_dist_params
      next_act_dist = tfd.Normal(
          loc=next_act_dist_params[0], scale=next_act_dist_params[1])
      next_act_dist = tfd.Independent(
          distributional.TanhTransformedDistribution(next_act_dist),
          reinterpreted_batch_ndims=1)
      # next_act_dist = tfd.MultivariateNormalDiag(
      #     loc=next_act_dist_params[0], scale_diag=next_act_dist_params[1])

      obs = transitions.observation
      acts = transitions.action
      next_obs = transitions.next_observation
      key, sub_key = jax.random.split(key)

      ####### DEBUGGED MIMO ########
      next_acts = networks.sample(next_act_dist, sub_key)
      key, sub_key = jax.random.split(key)
      policy_acts_for_obs = networks.sample(cur_act_dist, sub_key)
      sarsa_tuple = (obs, acts, transitions.reward, transitions.discount, next_obs, next_acts, policy_acts_for_obs)

      if mimo_using_obs_tile or mimo_using_act_tile:
        # if doing MIMO, otherwise it means we are just doing multi-head
        all_keys = jax.random.split(key, ensemble_size + 1)
        key = all_keys[0]
        batch_keys = all_keys[1:]
        sarsa_tuple = mimo_batch(batch_keys, sarsa_tuple) # ensemble_size x batch x ...
        sarsa_tuple = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), sarsa_tuple) # batch x ensemble_size x ...
        sarsa_tuple = jax.tree_map(lambda x: jnp.reshape(x, [x.shape[0], -1]), sarsa_tuple)

      obs, acts, rews, discs, next_obs, next_acts, policy_acts_for_obs = sarsa_tuple

      if not (mimo_using_obs_tile or mimo_using_act_tile):
        # multi-head
        rews = jnp.expand_dims(rews, axis=-1)
        discs = jnp.expand_dims(discs, axis=-1)
      ###############


#       # next_acts = msg_networks.apply_policy_and_sample(
#       #     networks, eval_mode=False)(policy_params, sub_key, next_obs)
#       # sarsa_tuple = (obs, acts, transitions.reward, transitions.discount, next_obs, next_acts)

#       if mimo_using_obs_tile and mimo_using_act_tile:
#         # if using the version where we also tile the obs
#         obs, acts, rews, discs, next_obs = jax.tree_map(
#             lambda x: jnp.tile(x, [1]*(x.ndim-1) + [ensemble_size]),
#             (obs, acts, transitions.reward[:, None], transitions.discount[:, None], next_obs))
#       elif (not mimo_using_obs_tile) and mimo_using_act_tile:
#         # otherwise not tiling obs
#         acts, rews, discs = jax.tree_map(
#             lambda x: jnp.tile(x, [1]*(x.ndim-1) + [ensemble_size]),
#             (acts, transitions.reward[:, None], transitions.discount[:, None]))
#       elif (not mimo_using_obs_tile) and (not mimo_using_act_tile):
#         # otherwise not tile obs or acts
#         rews, discs = jax.tree_map(
#             lambda x: jnp.tile(x, [1]*(x.ndim-1) + [ensemble_size]),
#             (transitions.reward[:, None], transitions.discount[:, None]))
#       else:
#         raise NotImplementedError()

#       key, sub_key = jax.random.split(key)
#       if mimo_using_act_tile:
#         # if using act tiling
#         next_acts = next_act_dist._sample_n(ensemble_size, sub_key) # ens_size x batch x dim
#         next_acts = jnp.swapaxes(next_acts, 0, 1) # batch x ens_size x dim
#       else:
#         # otherwise
#         next_acts = networks.sample(next_act_dist, sub_key) # batch x dim

#       # if you want to repeat the acts some percentage of the time
#       # to_keep = next_acts.shape[0] // 2
#       # if always repeating
#       # to_keep = 0
#       # diff_next_acts = next_acts[:to_keep]
#       # repeat_next_acts = next_acts[to_keep:, 0:1]
#       # repeat_next_acts = jnp.tile(repeat_next_acts, [1, ensemble_size, 1])
#       # next_acts = jnp.concatenate([diff_next_acts, repeat_next_acts], axis=0)

#       if mimo_using_act_tile:
#         # if using act tiling
#         next_acts = jnp.reshape(next_acts, [obs.shape[0], -1]) # batch x (ensemble_size x dim)

#       # much older
#       # batch_size = transitions.reward.shape[0] // ensemble_size
#       # sarsa_tuple = jax.tree_map(
#       #     lambda x: jnp.reshape(x, (batch_size, -1)),
#       #     sarsa_tuple)
#       # obs, acts, rews, discs, next_obs, next_acts = sarsa_tuple

#       # all_keys = jax.random.split(key, ensemble_size + 1)
#       # key = all_keys[0]
#       # batch_keys = all_keys[1:]
#       # sarsa_tuple = mimo_batch(batch_keys, sarsa_tuple) # ensemble_size x normal batch size x dim
#       # sarsa_tuple = jax.tree_map(lambda t: jnp.swapaxes(t, 0, 1), sarsa_tuple)
#       # pre_flatten_obs = sarsa_tuple[0] # normal batch size x ensemble size x dim
#       # batch_size = transitions.reward.shape[0]
#       # sarsa_tuple = jax.tree_map(
#       #     lambda x: jnp.reshape(x, (batch_size, -1)),
#       #     sarsa_tuple)
#       # obs, acts, rews, discs, next_obs, next_acts = sarsa_tuple

      q_cur = networks.q_network.apply(critic_params, obs, acts) # B x N x num_qs_per_member
      target_q_next = networks.q_network.apply(
          target_critic_params, next_obs, next_acts) # B x N x num_qs_per_member
      target_q_next = jnp.min(target_q_next, axis=-1) # B x N
      y = jax.lax.stop_gradient(rews * reward_scale +
                                discs * discount * target_q_next) # B x N

      if td_target_method == 'independent':
        pass
      # The other ones can't be done in current MIMO implementation and would be costly
      # This is because different members are trained with difference batches of data
      # for MIMO.
      # For the version of MIMO where they are all the same we can do it.
      elif td_target_method == 'shared_mean':
        y = jnp.mean(y, axis=-1, keepdims=True)
      # elif td_target_method == 'shared_min':
      #   y = jnp.min(y, axis=-1, keepdims=True)
      # elif td_target_method == 'shared_lcb':
      #   y_mean = jnp.mean(y, axis=-1, keepdims=True)
      #   y_std = jnp.std(y, axis=-1, keepdims=True)
      #   y = y_mean + beta * y_std
      else:
        raise NotImplementedError()

      q_error = jnp.square(q_cur - jnp.expand_dims(y, axis=-1)) # B x N x num_qs_per_member
      # In our experience this doesn't matter, we just leave it on by default
      if use_random_weighting_in_critic_loss:
        q_error *= jax.random.uniform(
            key, q_error.shape, q_error.dtype, minval=0., maxval=2.)
      # q_loss = 0.5 * jnp.mean(jnp.sum(q_error, axis=1))
      q_loss = 0.5 * jnp.mean(q_error)

      # Time for some pessimism

      q_preds = jnp.min(q_cur, axis=-1) # B x N

      # if you want to add regularization on q_pred magnitudes
      # q_loss = q_loss + 0.5 * jnp.mean(q_preds ** 2)
      # q_loss = q_loss + jnp.mean(jnp.abs(q_cur))

      if behavior_regularization_type == 'none':
        behavior_reg_loss = 0.
        fraction_active = 0.
      elif behavior_regularization_type == 'v1':
        key, sub_key = jax.random.split(key)

        # much older
        # policy_acts_for_obs = msg_networks.apply_policy_and_sample(
        #     networks, eval_mode=False)(policy_params, sub_key, transitions.observation)
        # policy_acts_for_obs = jnp.reshape(policy_acts_for_obs, (batch_size, -1))


        # policy_acts_for_obs = msg_networks.apply_policy_and_sample(
        #     networks, eval_mode=False)(
        #         policy_params,
        #         sub_key,
        #         jnp.reshape(pre_flatten_obs, [-1, transitions.observation.shape[-1]])
        #     )
        # policy_acts_for_obs = jnp.reshape(policy_acts_for_obs, (batch_size, -1))

        #         key, sub_key = jax.random.split(key)

        #         if mimo_using_act_tile:
        #           # if using act tiling
        #           # policy_acts_for_obs = cur_act_dist._sample_n(ensemble_size, sub_key) # ens_size x batch x dim
        #           # policy_acts_for_obs = jnp.swapaxes(policy_acts_for_obs, 0, 1) # batch x ens_size x dim
        #           # policy_acts_for_obs = jnp.reshape(policy_acts_for_obs, [obs.shape[0], -1]) # batch x (ensemble_size x dim)
        #           # if always repeating
        #           policy_acts_for_obs = networks.sample(cur_act_dist, sub_key)
        #           policy_acts_for_obs = jnp.tile(policy_acts_for_obs, [1, ensemble_size])
        #         else:
        #           # otherwise
        #           policy_acts_for_obs = networks.sample(cur_act_dist, sub_key)


        q_for_policy_acts = networks.q_network.apply(
            critic_params, obs, policy_acts_for_obs) # B x N x num_qs_per_member
        q_for_policy_acts = jnp.min(q_for_policy_acts, axis=-1) # B x N

        # behavior_reg_loss = jnp.mean(jnp.sum(q_for_policy_acts - q_preds, axis=1))
        behavior_reg_loss = jnp.mean(q_for_policy_acts - q_preds)

        q_loss = q_loss + behavior_regularization_alpha * behavior_reg_loss
        fraction_active = 1.
      else:
        raise NotImplementedError()

      return q_loss, (q_preds, behavior_reg_loss, fraction_active)

    mimo_critic_loss_and_grad = jax.value_and_grad(mimo_critic_loss, has_aux=True)

    def mimo_critic_update_step(
        critic_params,
        target_critic_params,
        transitions,
        optim_state,
        cur_act_dist_params,
        next_act_dist_params,
        key,
        encoder_params,
        encoder_optim_state):
      key, sub_key = jax.random.split(key)
      (critic_loss, (q_preds, behavior_reg_loss, fraction_active)), critic_grad = mimo_critic_loss_and_grad(
          critic_params,
          target_critic_params,
          transitions,
          cur_act_dist_params,
          next_act_dist_params,
          sub_key)

      if mimo_using_adamw:
        critic_update, optim_state = q_optimizer.update(critic_grad, optim_state, critic_params) # for adamw
      else:
        critic_update, optim_state = q_optimizer.update(critic_grad, optim_state)
      critic_params = optax.apply_updates(critic_params, critic_update)

      if self._use_ema_target_critic_params:
        target_critic_params = jax.tree_multimap(
            lambda x, y: x * (1 - tau) + y * tau,
            target_critic_params, critic_params)

      sass_loss, sass_acc = 0., 0.
      return critic_params, target_critic_params, optim_state, encoder_params, encoder_optim_state, critic_loss, q_preds, behavior_reg_loss, fraction_active, sass_loss, sass_acc

    def get_all_mimo_q_values(critic_params, obs, act):
      if mimo_using_obs_tile:
        # if using the version where we also tile the obs
        obs = jnp.tile(obs, [1, ensemble_size])
      if mimo_using_act_tile:
        # if using the version where we also tile the acts
        act = jnp.tile(act, [1, ensemble_size])
      q_preds = networks.q_network.apply(critic_params, obs, act)
      return q_preds

    if ensemble_method == 'deep_ensembles':
      all_critics_update_step = jax.pmap(
          jax.vmap(
              per_critic_update_step,
              in_axes=(0, 0, None, 0, None, None, 0, 0, 0),
              # in_axes=(0, 0, None, 0, None, None, None, 0, 0),
              out_axes=0,
              axis_name='within_device'),
          in_axes=(0, 0, None, 0, None, None, 0, 0, 0),
          # in_axes=(0, 0, None, 0, None, None, None, 0, 0),
          out_axes=0,
          axis_name='across_devices')

      get_all_q_values = jax.pmap(
          jax.vmap(networks.q_network.apply, in_axes=(0, None, None), out_axes=0),
          in_axes=(0, None, None),
          out_axes=0)
    elif ensemble_method in ['mimo', 'tree_deep_ensembles',
                             'efficient_tree_deep_ensembles']:
      '''for MIMO there is no need to pmap and vmap, we just do this to stay
      consistent with the rest of the implementation and not have to deal with
      shaping bugs.'''
      all_critics_update_step = jax.pmap(
          jax.vmap(
              mimo_critic_update_step,
              in_axes=(0, 0, None, 0, None, None, 0, 0, 0),
              # in_axes=(0, 0, None, 0, None, None, None, 0, 0),
              out_axes=0,
              axis_name='within_device'),
          in_axes=(0, 0, None, 0, None, None, 0, 0, 0),
          # in_axes=(0, 0, None, 0, None, None, None, 0, 0),
          out_axes=0,
          axis_name='across_devices')
      get_all_q_values = jax.pmap(
          jax.vmap(get_all_mimo_q_values, in_axes=(0, None, None), out_axes=0),
          in_axes=(0, None, None),
          out_axes=0)
    else:
      raise NotImplementedError()


    def msg_loss(
        all_critic_params,
        obs,
        acts,
        key,
    ):
      if rem_mode:
        raise NotImplementedError()
        rem_alphas = jax.random.uniform(
            key, all_q_vals.shape, dtype=all_q_vals.dtype)
        rem_alphas = rem_alphas / jnp.sum(rem_alphas, axis=(0, 1), keepdims=True)
        rem_all_q_vals = rem_alphas * all_q_vals
        REM = jnp.sum(rem_all_q_vals, axis=(0, 1))
        return -1. * jnp.mean(REM), all_q_vals
      else:
        if ensemble_method == 'deep_ensembles':
          # num_devices x num_per_device x batch_size x num_Q_per_critic
          all_q_vals = get_all_q_values(all_critic_params, obs, acts)
          all_q_vals = jnp.min(all_q_vals, axis=-1)

          mean_q_vals = jnp.mean(all_q_vals, axis=(0, 1))
          std_q_vals = jnp.std(all_q_vals, axis=(0, 1))
          LCB = mean_q_vals + beta * std_q_vals
        elif ensemble_method in ['mimo', 'tree_deep_ensembles',
                                 'efficient_tree_deep_ensembles']:
          # 1 x 1 x batch_size x num_ensembles x num_qs_per_member
          all_q_vals = get_all_q_values(all_critic_params, obs, acts)
          all_q_vals = jnp.min(all_q_vals, axis=-1)

          mean_q_vals = jnp.mean(all_q_vals, axis=(0, 1, 3))
          # std_q_vals = jnp.std(all_q_vals, axis=(0, 1, 3))
          std_q_vals = jnp.std(all_q_vals, axis=(0, 1, 3), ddof=1)
          LCB = mean_q_vals + beta * std_q_vals
        else:
          raise NotImplementedError()

        # when not independent, the Q vals are already estimating a lowerbound
        if td_target_method == 'independent':
          msg_loss = -1. * jnp.mean(LCB)
        elif td_target_method == 'shared_mean':
          # msg_loss = -1. * jnp.mean(LCB)
          msg_loss = -1. * jnp.mean(mean_q_vals)
        elif td_target_method == 'shared_lcb':
          msg_loss = -1. * jnp.mean(LCB)
          # msg_loss = -1. * jnp.mean(mean_q_vals)
        elif td_target_method == 'shared_min':
          # msg_loss = -1. * jnp.mean(LCB)
          # msg_loss = -1. * jnp.mean(mean_q_vals)
          msg_loss = -1. * jnp.min(all_q_vals)
        else:
          raise NotImplementedError()

        return msg_loss, all_q_vals


    def compute_spec_norm_loss(X, X_prime, key):
      """This is not relevant for MSG,
      it was being used for testing out an orthogonal idea"""
      # mode = 'nngp'
      mode = 'ntk'
      kernel_fn = networks.kernel_fn
      K_X_X = kernel_fn(X, X, mode)
      K_X_X = K_X_X + (1e-4) * jnp.eye(K_X_X.shape[0])
      K_inv = jnp.linalg.inv(K_X_X)
      K_Xp_X = kernel_fn(X_prime, X, mode)

      C = K_Xp_X @ K_inv
      C = C.T @ C
      key, sub_key = jax.random.split(key)
      v = jax.random.normal(
          key=sub_key,
          shape=[C.shape[0], 1],
          dtype=C.dtype)

      for i in range(20):
        v = C @ v
        v = v / jnp.linalg.norm(v)

      v = jax.lax.stop_gradient(v)
      sn = jnp.sqrt(v.T @ C @ v)[0, 0]
      # dont regularize beyond spec norm 1
      sn = jax.nn.relu(sn - 1.) + 1.

      return sn


    def total_actor_loss(
        actor_params,
        all_critic_params,
        transitions,
        alpha,
        key,
    ):
      """This is the loss that is used after the initial num_bc_iters."""
      obs = transitions.observation
      act_dists = networks.policy_network.apply(actor_params, obs)
      key, sub_key = jax.random.split(key)
      acts = networks.sample(act_dists, sub_key)
      log_probs = networks.log_prob(act_dists, acts)

      key, sub_key = jax.random.split(key)
      msg_loss_term, all_q_vals = msg_loss(
          all_critic_params,
          obs,
          acts,
          sub_key)
      ent_loss = jnp.mean(log_probs)

      if use_entropy_regularization:
        total_loss = msg_loss_term + alpha * ent_loss
      else:
        total_loss = msg_loss_term

      spec_norm_loss = 0.
      if behavior_regularization_type == 'spec_norm':
        next_act_dists = networks.policy_network.apply(
            actor_params, transitions.next_observation)
        key, sub_key = jax.random.split(key)
        pi_next_acts = networks.sample(next_act_dists, sub_key)

        X = jnp.concatenate(
            [transitions.observation, transitions.action], axis=-1)
        X_prime = jnp.concatenate(
            [transitions.next_observation, pi_next_acts], axis=-1)

        key, sub_key = jax.random.split(key)
        spec_norm_loss = compute_spec_norm_loss(X, X_prime, sub_key)
        total_loss = total_loss + behavior_regularization_alpha * spec_norm_loss

      return total_loss, (ent_loss, all_q_vals, spec_norm_loss)

    total_actor_loss_val_and_grad = jax.value_and_grad(
        total_actor_loss, has_aux=True)

    def actor_bc_loss(
        actor_params,
        obs,
        acts):
      act_dists = networks.policy_network.apply(actor_params, obs)
      log_probs = networks.log_prob(act_dists, acts)
      avg_log_probs = jnp.mean(log_probs)
      return -avg_log_probs, avg_log_probs

    bc_actor_loss_val_and_grad = jax.value_and_grad(
        actor_bc_loss, has_aux=True)

    def _actor_update_step(
        actor_params,
        all_critics_params,
        transitions,
        optim_state,
        alpha,
        key,
        in_initial_bc_iters):
      """The unjitted version of the bc update step.

      in_initial_bc_iters is used to indicate whether we are in the initial
      phase of training using a BC loss. In the jit call, this variable will be
      marked as a static arg name, so when it changes value from true to false
      it will be re-jitted."""
      if not in_initial_bc_iters:
        (total_loss, (avg_log_prob, all_q_vals, spec_norm_loss)), total_loss_grads = total_actor_loss_val_and_grad(
            actor_params,
            all_critics_params,
            transitions,
            alpha,
            key)
      else:
        spec_norm_loss = 0.
        (total_loss, avg_log_prob), total_loss_grads = bc_actor_loss_val_and_grad(
            actor_params,
            transitions.observation,
            transitions.action)
        all_q_vals = jnp.zeros(
            [num_devices, num_models_per_device, transitions.action.shape[0]],
            total_loss.dtype)
      actor_updates, optim_state = policy_optimizer.update(
          total_loss_grads, optim_state)
      actor_params = optax.apply_updates(actor_params, actor_updates)

      return actor_params, optim_state, avg_log_prob, total_loss, all_q_vals, spec_norm_loss

    actor_update_step = jax.jit(
        _actor_update_step, static_argnames='in_initial_bc_iters')

    def alpha_loss_fn(alpha_params, avg_log_prob):
      alpha = jnp.exp(alpha_params)
      return alpha * jax.lax.stop_gradient(-avg_log_prob - target_entropy)

    alpha_loss_val_and_grad = jax.jit(jax.value_and_grad(alpha_loss_fn))


    if use_img_encoder:
      pmapped_img_encoder = jax.pmap(
          networks.img_encoder.apply,
          in_axes=0,
          out_axes=0)


    def _full_update_step(
        state,
        transitions,
        in_initial_bc_iters,
    ):
      """The unjitted version of the full update step.

      in_initial_bc_iters is used to indicate whether we are in the initial
      phase of training using a BC loss. In the jit call, this variable will be
      marked as a static arg name, so when it changes value from true to false
      it will be re-jitted."""

      # preprocess the obs if we need to pass the images through the encoder
      if self._use_img_encoder:
        obs = transitions.observation['state_image']
        next_obs = transitions.next_observation['state_image']
        B = obs.shape[0]

        def reshape_for_devices(t):
          rest_t_shape = list(t.shape[1:])
          new_shape = [num_devices, B//num_devices,] + rest_t_shape
          return jnp.reshape(t, new_shape)
        def reshape_back(t):
          rest_t_shape = list(t.shape[2:])
          new_shape = [B,] + rest_t_shape
          return jnp.reshape(t, new_shape)

        obs_for_devices = jax.tree_map(reshape_for_devices, obs)
        next_obs_for_devices = jax.tree_map(reshape_for_devices, next_obs)

        encoded_obs = pmapped_img_encoder(
            state.img_encoder_params,
            obs_for_devices,)
        encoded_next_obs = pmapped_img_encoder(
            state.img_encoder_params,
            next_obs_for_devices,)

        encoded_obs = jax.tree_map(reshape_back, encoded_obs)
        encoded_next_obs = jax.tree_map(reshape_back, encoded_next_obs)

        obs = dict(
            state_image=encoded_obs,
            state_dense=transitions.observation['state_dense'])
        next_obs = dict(
            state_image=encoded_next_obs,
            state_dense=transitions.next_observation['state_dense'])

        transitions = types.Transition(
            observation=obs,
            action=transitions.action,
            reward=transitions.reward,
            discount=transitions.discount,
            next_observation=next_obs,
            extras=transitions.extras)

      metrics = OrderedDict()

      key = state.key
      if adaptive_entropy_coefficient:
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = entropy_coefficient

      # critic update step
      if ensemble_method == 'deep_ensembles':
        all_keys = jax.random.split(key, ensemble_size + 1)
      elif ensemble_method in ['mimo', 'tree_deep_ensembles',
                               'efficient_tree_deep_ensembles']:
        all_keys = jax.random.split(key, 1 + 1)
      key = all_keys[0]
      critic_update_keys = jnp.reshape(
          all_keys[1:], (num_devices, num_models_per_device, 2))

      cur_act_dist_params = networks.policy_network.apply(
          state.policy_params, transitions.observation)
      cur_act_dist_params = (
          cur_act_dist_params._distribution._distribution.loc,
          cur_act_dist_params._distribution._distribution.scale,)
      # cur_act_dist_params = (
      #     cur_act_dist_params.parameters['loc'],
      #     cur_act_dist_params.parameters['scale_diag'],)
      next_act_dist_params = networks.policy_network.apply(
          state.policy_params, transitions.next_observation)
      next_act_dist_params = (
          next_act_dist_params._distribution._distribution.loc,
          next_act_dist_params._distribution._distribution.scale,)
      # next_act_dist_params = (
      #     next_act_dist_params.parameters['loc'],
      #     next_act_dist_params.parameters['scale_diag'],)

      new_all_q_params, new_all_target_q_params, new_all_q_optimizer_states, new_simclr_encoder_params, new_simclr_encoder_optim_state, all_q_losses, q_preds, behavior_reg_loss, fraction_active, sass_loss, sass_acc = \
          all_critics_update_step(
              state.all_q_params,
              state.all_target_q_params,
              transitions,
              state.all_q_optimizer_states,
              cur_act_dist_params,
              next_act_dist_params,
              critic_update_keys,
              state.simclr_encoder_params,
              state.simclr_encoder_optim_state,)
      # all_keys[1])
      metrics['avg_critic_loss'] = jnp.mean(all_q_losses)
      metrics['std_critic_loss'] = jnp.std(all_q_losses)
      behavior_reg_loss = jnp.reshape(behavior_reg_loss, [-1])
      metrics['behavior_reg_loss/avg'] = jnp.mean(behavior_reg_loss)
      metrics['behavior_reg_loss/std'] = jnp.std(behavior_reg_loss)
      metrics['behavior_reg_loss/max'] = jnp.max(behavior_reg_loss)
      metrics['behavior_reg_loss/min'] = jnp.min(behavior_reg_loss)
      fraction_active = jnp.reshape(fraction_active, [-1])
      metrics['fraction_behavior_reg_active/avg'] = jnp.mean(fraction_active)
      metrics['fraction_behavior_reg_active/std'] = jnp.std(fraction_active)
      metrics['fraction_behavior_reg_active/max'] = jnp.max(fraction_active)
      metrics['fraction_behavior_reg_active/min'] = jnp.min(fraction_active)

      metrics['sass_loss'] = jnp.mean(sass_loss)
      metrics['sass_acc'] = jnp.mean(sass_acc)

      if self._perform_sarsa_q_eval:
        q_preds = jnp.transpose(q_preds, axes=(2, 0, 1))
        q_preds = jnp.reshape(q_preds, [q_preds.shape[0], -1])
        q_means = jnp.mean(q_preds, axis=-1)
        q_stds = jnp.std(q_preds, axis=-1)

        metrics['q_mean/avg'] = jnp.mean(q_means)
        metrics['q_mean/std'] = jnp.std(q_means)
        metrics['q_mean/max'] = jnp.max(q_means)
        metrics['q_mean/min'] = jnp.mean(q_means)

        metrics['q_std/avg'] = jnp.mean(q_stds)
        metrics['q_std/std'] = jnp.std(q_stds)
        metrics['q_std/max'] = jnp.max(q_stds)
        metrics['q_std/min'] = jnp.mean(q_stds)

      # actor update step
      if not self._perform_sarsa_q_eval:
        key, sub_key = jax.random.split(key)
        # if ensemble_method == 'mimo':
        #   transitions = jax.tree_map(lambda x: x[:256], transitions)
        new_policy_params, new_policy_optimizer_state, avg_log_prob, act_loss, all_q_vals, spec_norm_loss = actor_update_step(
            state.policy_params,
            state.all_q_params,
            transitions,
            state.policy_optimizer_state,
            alpha,
            sub_key,
            in_initial_bc_iters)
        metrics['actor_loss'] = act_loss
        metrics['avg_log_prob'] = jnp.mean(avg_log_prob)
        if ensemble_method == 'deep_ensembles':
          q_preds = jnp.transpose(all_q_vals, axes=(2, 0, 1))
          q_preds = jnp.reshape(q_preds, [q_preds.shape[0], -1])
        elif ensemble_method in ['mimo', 'tree_deep_ensembles',
                                 'efficient_tree_deep_ensembles']:
          q_preds = all_q_vals[0, 0]
        else:
          raise NotImplementedError()
        ensemble_q_avgs = jnp.mean(q_preds, axis=0)
        q_means = jnp.mean(q_preds, axis=-1)
        q_stds = jnp.std(q_preds, axis=-1)

        metrics['q_mean/avg'] = jnp.mean(q_means)
        metrics['q_mean/std'] = jnp.std(q_means)
        metrics['q_mean/max'] = jnp.max(q_means)
        metrics['q_mean/min'] = jnp.mean(q_means)

        metrics['q_std/avg'] = jnp.mean(q_stds)
        metrics['q_std/std'] = jnp.std(q_stds)
        metrics['q_std/max'] = jnp.max(q_stds)
        metrics['q_std/min'] = jnp.mean(q_stds)

        metrics['argmax_q'] = jnp.argmax(ensemble_q_avgs)
        metrics['argmin_q'] = jnp.argmin(ensemble_q_avgs)

        metrics['spec_norm'] = jnp.mean(spec_norm_loss)
      else:
        new_policy_params = state.policy_params
        new_policy_optimizer_state = state.policy_optimizer_state

      # create new state
      new_state = TrainingState(
          new_policy_optimizer_state,
          new_all_q_optimizer_states,
          new_policy_params,
          new_all_q_params,
          new_all_target_q_params,
          key,
          img_encoder_params=state.img_encoder_params,
          simclr_encoder_params=new_simclr_encoder_params,
          simclr_encoder_optim_state=new_simclr_encoder_optim_state,)

      # alpha update step
      if use_entropy_regularization and (not self._perform_sarsa_q_eval) and (not in_initial_bc_iters) and adaptive_entropy_coefficient:
        alpha_loss, alpha_grads = alpha_loss_val_and_grad(
            state.alpha_params, avg_log_prob)
        alpha_update, alpha_optimizer_state = alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state)
        alpha_params = optax.apply_updates(state.alpha_params, alpha_update)

        # # not used in generating paper results, just for playing around
        # alpha_params = jnp.clip(alpha_params, a_max=0.) # alpha params is log of alpha

        # metrics.update({
        #     'alpha_loss': alpha_loss,
        #     'alpha': jnp.exp(alpha_params),
        # })
        new_state = new_state._replace(
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params)
        metrics['alpha'] = jnp.exp(alpha_params)
        metrics['alpha_loss'] = alpha_loss
      else:
        new_state = new_state._replace(
            alpha_optimizer_state=state.alpha_optimizer_state,
            alpha_params=state.alpha_params)
        metrics['alpha'] = alpha
        metrics['alpha_loss'] = 0.

      return new_state, metrics


    simclr_loss_fn = sass_utils.build_simclr_loss(networks.simclr_encoder.apply)

    def per_q_repr_pretrain_loss(
        q_params,
        target_q_params,
        encoder_params,
        transitions,
        key):
      def mask_input(x):
        obs, act, rng = x
        rng, sub_rng = jax.random.split(rng)
        mask = jax.random.bernoulli(
            sub_rng, p=0.5, shape=obs.shape)
        obs = obs * mask
        rng, sub_rng = jax.random.split(rng)
        mask = jax.random.bernoulli(
            sub_rng, p=0.5, shape=act.shape)
        act = act * mask
        return (obs, act)

      # key, sub_key = jax.random.split(key)
      # dropout_cur_sa = jax.random.bernoulli(sub_key, 0.5)
      # key, sub_key = jax.random.split(key)
      # cur_obs, cur_act = jax.lax.cond(
      #     dropout_cur_sa,
      #     mask_input,
      #     lambda x: (x[0], x[1]),
      #     (transitions.observation, transitions.action, sub_key))

      # IF NO INPUT DROPOUT
      cur_obs = transitions.observation
      cur_act = transitions.action

      h1 = networks.get_critic_repr(
          q_params, cur_obs, cur_act)

      # # So far the best results
      # h2 = networks.get_critic_repr(
      #     q_params, transitions.next_observation, transitions.extras)
      ###########
      key, sub_key = jax.random.split(key)
      half_batch_size = transitions.observation.shape[0] // 2
      data_next_acts = transitions.extras[:half_batch_size] # we put the next acts in the extras field
      unif_next_acts = jax.random.uniform(
          sub_key,
          shape=(half_batch_size, *transitions.action.shape[1:]),
          dtype=transitions.action.dtype,
          minval=-1.,
          maxval=1.
      )
      next_acts_to_use = jnp.concatenate(
          [data_next_acts, unif_next_acts], axis=0)
      ###########
      # key, sub_key = jax.random.split(key)
      # unif_next_acts = jax.random.uniform(
      #     sub_key,
      #     shape=transitions.action.shape,
      #     dtype=transitions.action.dtype,
      #     minval=-1.,
      #     maxval=1.
      # )
      # next_acts_to_use = unif_next_acts
      ###########
      # next_acts_to_use = transitions.extras

      # key, sub_key = jax.random.split(key)
      # dropout_next_sa = jax.random.bernoulli(sub_key, 0.5)
      # key, sub_key = jax.random.split(key)
      # next_obs, next_act = jax.lax.cond(
      #     dropout_next_sa,
      #     mask_input,
      #     lambda x: (x[0], x[1]),
      #     (transitions.next_observation, next_acts_to_use, sub_key))

      # IF NO INPUT DROPOUT
      next_obs = transitions.next_observation
      next_act = next_acts_to_use

      # h2 = networks.get_critic_repr(
      #     q_params,
      #     next_obs,
      #     next_act,
      # )
      stop_grad_target_q_params = jax.tree_map(
          jax.lax.stop_gradient, target_q_params)
      h2 = networks.get_critic_repr(
          stop_grad_target_q_params,
          next_obs,
          next_act,
      )
      loss, acc = simclr_loss_fn(encoder_params, h1, h2, temp=pretrain_temp)
      # loss, acc = simclr_loss_fn(encoder_params, h1, h2, temp=128**0.5)
      # loss, acc = simclr_loss_fn(encoder_params, h1, h2, temp=0.1)
      # loss, acc = simclr_loss_fn(encoder_params, h1, h2, temp=0.05)

      # if also matching (s,a) to itself
      loss_self, acc_self = simclr_loss_fn(encoder_params, h1, h1, temp=pretrain_temp)
      loss = 0.5 * (loss + loss_self)
      acc = 0.5 * (acc + acc_self)

      return loss, acc

    per_q_repr_pretrain_loss_and_grad = jax.value_and_grad(per_q_repr_pretrain_loss, argnums=(0, 2), has_aux=True)

    def per_q_repr_pretrain_step(
        q_params,
        target_q_params,
        encoder_params,
        transitions,
        q_optim_state,
        encoder_optim_state,
        key):
      key, sub_key = jax.random.split(key)
      (q_loss, acc), (q_grad, encoder_grad) = per_q_repr_pretrain_loss_and_grad(
          q_params,
          target_q_params,
          encoder_params,
          transitions,
          sub_key)

      # q_update, q_optim_state = q_optimizer.update(q_grad, q_optim_state)
      q_update, q_optim_state = pretrain_optimizer.update(
          # q_grad, q_optim_state, q_params)
          q_grad, q_optim_state)
      q_params = optax.apply_updates(q_params, q_update)

      enc_update, encoder_optim_state = pretrain_optimizer.update(
          encoder_grad, encoder_optim_state)
      # encoder_grad, encoder_optim_state, encoder_params)
      encoder_params = optax.apply_updates(encoder_params, enc_update)

      return q_loss, acc, q_params, q_optim_state, encoder_params, encoder_optim_state

    if ensemble_method == 'deep_ensembles':
      all_q_repr_pretrain_step = jax.pmap(
          jax.vmap(
              per_q_repr_pretrain_step,
              in_axes=(0, 0, 0, None, 0, 0, 0),
              out_axes=0),
          in_axes=(0, 0, 0, None, 0, 0, 0),
          out_axes=0)
    else:
      all_q_repr_pretrain_step = jax.pmap(
          jax.vmap(
              per_q_repr_pretrain_step,
              in_axes=(0, 0, 0, None, 0, 0, 0),
              out_axes=0),
          in_axes=(0, 0, 0, None, 0, 0, 0),
          out_axes=0)

    def _full_pretraining_step(
        state,
        transitions,
    ):
      key, sub_key = jax.random.split(state.key)
      if ensemble_method == 'deep_ensembles':
        pretrain_keys = jax.random.split(sub_key, num=ensemble_size)
        pretrain_keys = jnp.reshape(
            pretrain_keys, [num_devices, num_models_per_device, 2])
      elif ensemble_method in ['mimo', 'tree_deep_ensembles',
                               'efficient_tree_deep_ensembles']:
        pretrain_keys = jnp.reshape(sub_key, [1, 1, 2])

      pretrain_loss, pretrain_acc, new_q_params, new_q_optim_state, new_enc_params, new_enc_optim_state = all_q_repr_pretrain_step(
          state.all_q_params,
          state.all_target_q_params,
          state.simclr_encoder_params,
          transitions,
          state.all_q_optimizer_states,
          state.simclr_encoder_optim_state,
          pretrain_keys)

      metrics = OrderedDict()
      metrics['pretraining/pretrain_loss'] = jnp.mean(pretrain_loss)
      metrics['pretraining/pretrain_acc'] = jnp.mean(pretrain_acc)

      new_state = TrainingState(
          state.policy_optimizer_state,
          new_q_optim_state,
          state.policy_params,
          new_q_params,
          new_q_params,
          key,)
      new_state = new_state._replace(
          alpha_optimizer_state=state.alpha_optimizer_state,
          alpha_params=state.alpha_params,
          img_encoder_params=state.img_encoder_params,
          simclr_encoder_params=new_enc_params,
          simclr_encoder_optim_state=new_enc_optim_state)

      return new_state, metrics


    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray)

    # Iterator on demonstration transitions.
    self._iterator = iterator

    # if num_sgd_steps_per_step != 1: raise NotImplementedError()
    # full_update_step = utils.process_multiple_batches(_full_update_step,
    #                                                   num_sgd_steps_per_step)
    # self._update_step = jax.jit(
    #     full_update_step, static_argnames='in_initial_bc_iters')
    # self._update_step = jax.jit(
    #     _full_update_step, static_argnames='in_initial_bc_iters')
    self._update_step_in_initial_bc_iters = utils.process_multiple_batches(
        lambda x, y: _full_update_step(x, y, True),
        num_sgd_steps_per_step)
    self._update_step_rest = utils.process_multiple_batches(
        lambda x, y: _full_update_step(x, y, False),
        num_sgd_steps_per_step)

    self._update_step_in_pretrain_iters = utils.process_multiple_batches(
        _full_pretraining_step,
        num_sgd_steps_per_step)

    def make_initial_state(key):
      """"""
      # critic stuff
      all_q_params = []
      all_q_optimizer_states = []
      all_pretrain_encoder_params = []
      all_pretrain_encoder_optimizer_states = []
      if ensemble_method == 'deep_ensembles':
        number_of_params = ensemble_size
      elif ensemble_method in ['mimo', 'tree_deep_ensembles',
                               'efficient_tree_deep_ensembles']:
        number_of_params = 1
      else:
        raise NotImplementedError()

      for _ in range(number_of_params):
        key, sub_key = jax.random.split(key)

        if critic_random_init:
          key, sub_key = jax.random.split(key)
          init_scale = float(jax.random.uniform(sub_key, shape=(), minval=0.5, maxval=2.0))
          key, sub_key = jax.random.split(key)
          init_type_options = []
          for fan in ['fan_in', 'fan_avg']:
            for dist in ['truncated_normal', 'uniform']:
              init_type_options.append(
                  hk.initializers.VarianceScaling(
                      init_scale, mode=fan, distribution=dist))
          idx = int(
              jax.random.randint(
                  sub_key, shape=(), minval=0, maxval=len(init_type_options)-1))
          key, sub_key = jax.random.split(key)
          q_params = networks.get_particular_critic_init(
              init_type_options[idx],
              jnp.zeros,
              sub_key,)
          # print(init_scale, init_type_options[idx])
        else:
          q_params = networks.q_network.init(sub_key)

        all_q_params.append(q_params)
        # q_optim_state = q_optimizer.init(q_params)
        q_optim_state = pretrain_optimizer.init(q_params)
        all_q_optimizer_states.append(q_optim_state)

        key, sub_key = jax.random.split(key)
        enc_params = networks.simclr_encoder.init(sub_key)
        all_pretrain_encoder_params.append(enc_params)
        enc_optim_state = pretrain_optimizer.init(enc_params)
        all_pretrain_encoder_optimizer_states.append(enc_optim_state)

      def stack_params(*args):
        return jnp.stack(list(args), axis=0)
      def reshape_params(p):
        p_shape = list(p.shape)
        new_shape = [num_devices, num_models_per_device] + p_shape[1:]
        return jnp.reshape(p, new_shape)

      all_q_params = jax.tree_multimap(stack_params, *all_q_params)
      all_q_params = jax.tree_map(reshape_params, all_q_params)
      # all_q_params = [
      #     [all_q_params[i] for i in range(num_devices*j, num_devices*j + num_models_per_device)]
      #     for j in range(num_devices)
      # ]

      all_q_optimizer_states = jax.tree_multimap(stack_params, *all_q_optimizer_states)
      all_q_optimizer_states = jax.tree_map(reshape_params, all_q_optimizer_states)
      # all_q_optimizer_states = [
      #     [all_q_optimizer_states[i] for i in range(num_devices*j, num_devices*j + num_models_per_device)]
      #     for j in range(num_devices)
      # ]

      all_pretrain_encoder_params = jax.tree_multimap(stack_params, *all_pretrain_encoder_params)
      all_pretrain_encoder_params = jax.tree_map(reshape_params, all_pretrain_encoder_params)

      all_pretrain_encoder_optimizer_states = jax.tree_multimap(stack_params, *all_pretrain_encoder_optimizer_states)
      all_pretrain_encoder_optimizer_states = jax.tree_map(reshape_params, all_pretrain_encoder_optimizer_states)

      # policy stuff
      key, sub_key = jax.random.split(key)
      policy_params = networks.policy_network.init(sub_key)
      policy_optimizer_state = policy_optimizer.init(policy_params)

      # img encoder
      devices = jax.local_devices()
      if use_img_encoder:
        """
        This is not the same thing as simclr encoder.
        Not sure if using img encoders still works, haven't used it in a while.
        Load pretrained img_encoder_params and do:
        replicated_img_encoder_params = jax.device_put_replicated(
            img_encoder_params, devices)
        """
        class EncoderTrainingState(NamedTuple):
          encoder_params: hk.Params
        img_encoder_params = {}
        replicated_img_encoder_params = img_encoder_params
        raise NotImplementedError('Need to load a checkpoint.')
      else:
        img_encoder_params = {}
        replicated_img_encoder_params = img_encoder_params

      state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          all_q_optimizer_states=all_q_optimizer_states,
          policy_params=policy_params,
          all_q_params=all_q_params,
          all_target_q_params=all_q_params,
          key=key,
          img_encoder_params=replicated_img_encoder_params,
          simclr_encoder_params=all_pretrain_encoder_params,
          simclr_encoder_optim_state=all_pretrain_encoder_optimizer_states,)

      # entropy stuff
      if adaptive_entropy_coefficient:
        state = state._replace(alpha_optimizer_state=alpha_optimizer_state,
                               alpha_params=log_alpha)

      # import pdb; pdb.set_trace()

      return state

    # Create initial state.
    self._state = make_initial_state(rng)
    self._duplicated_initial_all_q_optimizer_states = deepcopy(
        self._state.all_q_optimizer_states)
    self._switched_to_bc = False

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None


  def step(self):
    sample = next(self._iterator)
    transitions = types.Transition(*sample.data)

    counts = self._counter.get_counts()
    if 'learner_steps' not in counts:
      cur_step = 0
    else:
      cur_step = counts['learner_steps']

    in_pretrain_iters = cur_step < self._num_q_repr_pretrain_iters
    in_initial_bc_iters = self._num_q_repr_pretrain_iters <= cur_step < self._num_bc_iters + self._num_q_repr_pretrain_iters
    # self._state, metrics = self._update_step(
    #     self._state, transitions, in_initial_bc_iters)
    using_pretraining = self._num_q_repr_pretrain_iters > 0
    if in_pretrain_iters:
      # print('\nPRETRAINING')
      self._state, metrics = self._update_step_in_pretrain_iters(
          self._state, transitions)
    elif in_initial_bc_iters:
      if (not self._switched_to_bc) and using_pretraining:
        # reset the q optimizer state
        self._state = self._state._replace(
            all_q_optimizer_states=self._duplicated_initial_all_q_optimizer_states)
        self._switched_to_bc = True
      # print('\nBC TRAINING')
      self._state, metrics = self._update_step_in_initial_bc_iters(
          self._state, transitions)
    else:
      # print('\nMSG TRAINING')
      self._state, metrics = self._update_step_rest(
          self._state, transitions)

    if not self._use_ema_target_critic_params:
      # if we are not use EMA target params, then just copy them over after
      # many update steps.
      self._state = self._state._replace(
          all_target_q_params=self._state.all_q_params)

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(
        steps=self._num_sgd_steps_per_step, walltime=elapsed_time)

    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names):
    variables = {
        'policy': self._state.policy_params,
        'all_q': self._state.all_q_params,
    }
    if self._use_img_encoder:
      img_encoder_params = jax.tree_map(
          lambda x: x[0], self._state.img_encoder_params)
      variables['img_encoder'] = img_encoder_params
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state
