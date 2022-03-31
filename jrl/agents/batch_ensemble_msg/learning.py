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
"""BatchEnsembleMSG learner implementation.

This is the version of MSG that trains N independent Q-fns."""

from collections import OrderedDict
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
import tensorflow_probability
from jrl.agents.batch_ensemble_msg import networks as batch_ensemble_msg_networks
from jrl.utils import ensemble_utils
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
distributional = networks_lib.distributional


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  replicated_policy_optimizer_state: optax.OptState
  replicated_shared_q_optim_state: optax.OptState
  ensemble_q_optim_state: optax.OptState
  replicated_policy_params: networks_lib.Params
  replicated_shared_q_params: networks_lib.Params
  ensemble_q_params: networks_lib.Params
  target_replicated_shared_q_params: networks_lib.Params
  target_ensemble_q_params: networks_lib.Params
  key: networks_lib.PRNGKey
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[networks_lib.Params] = None


class BatchEnsembleMSGLearner(acme.Learner):
  """Efficient MSG learner."""

  _state: TrainingState

  def __init__(
      self,
      ensemble_size,
      beta,
      networks,
      rng,
      iterator,
      num_bc_iters = 50_000,
      use_random_weighting_in_critic_loss = True,
      tau = 0.005,
      reward_scale = 1.0,
      discount = 0.99,
      entropy_coefficient = None,
      target_entropy = 0,
      behavior_regularization_type = 'none',
      behavior_regularization_alpha = 1.0,
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
      num_bc_iters: number of initial iterations for bc as the actor update
      use_random_weighting_in_critic_loss: whether to use the random weighting
      tau: target smoothing coefficient.
      reward_scale: reward scale.
      discount: discount to use for TD updates.
      entropy_coefficient: coefficient applied to the entropy bonus. If None, an
        adaptative coefficient will be used.
      target_entropy: Used to normalize entropy. Only used when
        entropy_coefficient is None.
      policy_lr: learning rate for the policy
      q_lr: learning rate for the q functions
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      num_sgd_steps_per_step: number of sgd steps to perform per learner 'step'.
    """
    num_devices = len(jax.devices())
    assert ensemble_size % num_devices == 0
    num_models_per_device = ensemble_size // num_devices

    self._num_bc_iters = num_bc_iters
    self._num_sgd_steps_per_step = num_sgd_steps_per_step

    policy_optimizer = optax.adam(learning_rate=policy_lr)
    q_optimizer = optax.adam(learning_rate=q_lr)

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

    def member_q_loss(sp, ep, tsp, tep, transitions, act_dist_for_next_obs, act_dist_for_obs, key):
      obs = transitions.observation
      acts = transitions.action
      next_obs = transitions.next_observation

      # get actions for current obs
      cur_dist = tfd.Normal(
          loc=act_dist_for_obs[0], scale=act_dist_for_obs[1])
      cur_dist = tfd.Independent(
          distributional.TanhTransformedDistribution(cur_dist),
          reinterpreted_batch_ndims=1)
      key, sub_key = jax.random.split(key)
      policy_acts_for_obs = networks.sample(cur_dist, sub_key)

      # get actions for next obs
      next_dist = tfd.Normal(
          loc=act_dist_for_next_obs[0], scale=act_dist_for_next_obs[1])
      next_dist = tfd.Independent(
          distributional.TanhTransformedDistribution(next_dist),
          reinterpreted_batch_ndims=1)
      key, sub_key = jax.random.split(key)
      next_acts = networks.sample(next_dist, sub_key)

      q_cur = networks.q_ensemble_member_apply(sp, ep, obs, acts)
      target_q_next = networks.q_ensemble_member_apply(
          tsp, tep, next_obs, next_acts)
      target_q_next = jnp.min(target_q_next, axis=-1)
      y = jax.lax.stop_gradient(transitions.reward * reward_scale +
                                transitions.discount * discount * target_q_next)
      q_error = jnp.square(q_cur - jnp.expand_dims(y, axis=-1))
      if use_random_weighting_in_critic_loss:
        key, sub_key = jax.random.split(key)
        q_error *= jax.random.uniform(
            sub_key, q_error.shape, q_error.dtype, minval=0., maxval=2.)
      q_loss = 0.5 * jnp.mean(q_error)

      q_preds = jnp.min(q_cur, axis=-1)
      if behavior_regularization_type == 'none':
        behavior_reg_loss = 0.
        fraction_active = 0.
      elif behavior_regularization_type == 'v1':
        key, sub_key = jax.random.split(key)
        q_for_policy_acts = networks.q_ensemble_member_apply(sp, ep, obs, policy_acts_for_obs)
        q_for_policy_acts = jnp.min(q_for_policy_acts, axis=-1)
        behavior_reg_loss = jnp.mean(q_for_policy_acts - q_preds)
        q_loss = q_loss + behavior_regularization_alpha * behavior_reg_loss
        fraction_active = 1.
      else:
        raise NotImplementedError()

      # return q_loss
      return q_loss / float(ensemble_size)

    member_q_loss_val_and_grad = jax.value_and_grad(member_q_loss, argnums=(0,1))

    def member_q_update_fn(sp, ep, epos, tsp, tep, transitions, act_dist_for_next_obs, act_dist_for_obs, key):
      q_loss_val, q_loss_grad = member_q_loss_val_and_grad(
          sp,
          ep,
          tsp,
          tep,
          transitions,
          act_dist_for_next_obs,
          act_dist_for_obs,
          key)
      sp_grad = q_loss_grad[0]
      sp_grad = jax.lax.psum(sp_grad, axis_name='on_device')

      ep_grad = q_loss_grad[1]
      ep_update, new_epos = q_optimizer.update(ep_grad, epos)
      new_ep = optax.apply_updates(ep, ep_update)
      # update ensemble member specific target params
      new_tep = jax.tree_multimap(
          lambda x, y: x * (1 - tau) + y * tau,
          tep, new_ep)

      return sp_grad, new_ep, new_epos, new_tep, q_loss_val

    device_q_update_fn = jax.vmap(
        member_q_update_fn,
        # in_axes=(None, 0, 0, None, 0, None, None, None, 0),
        # in_axes=(None, 0, 0, None, 0, 0, None, None, 0),
        in_axes=(None, 0, 0, None, 0, 0, 0, 0, 0),
        out_axes=(None, 0, 0, 0, 0),
        axis_name='on_device')

    def _full_q_update_fn(sp, ep, spos, epos, tsp, tep, transitions, pp, act_key, q_key, trans_next_obs, trans_obs):
      act_key_cur, act_key_next = jax.random.split(act_key)

      next_dist = networks.policy_network.apply(pp, trans_next_obs)
      act_dist_for_next_obs = (
          next_dist._distribution._distribution.loc,
          next_dist._distribution._distribution.scale,)
      del trans_next_obs

      cur_dist = networks.policy_network.apply(pp, trans_obs)
      act_dist_for_obs = (
          cur_dist._distribution._distribution.loc,
          cur_dist._distribution._distribution.scale,)
      del trans_obs

      act_dist_for_next_obs, act_dist_for_obs = jax.tree_map(
          lambda t: jnp.reshape(
              t,
              (
                  transitions.observation.shape[0],
                  transitions.observation.shape[1],
                  *t.shape[1:])
              ),
          (act_dist_for_next_obs, act_dist_for_obs)
      )

      sp_grad, new_ep, new_epos, new_tep, q_loss_val = device_q_update_fn(
          sp,
          ep,
          epos,
          tsp,
          tep,
          transitions,
          act_dist_for_next_obs,
          act_dist_for_obs,
          q_key)
      sp_grad = jax.lax.psum(sp_grad, axis_name='across_devices')

      sp_update, new_spos = q_optimizer.update(sp_grad, spos)
      new_sp = optax.apply_updates(sp, sp_update)
      # update ensemble target params
      new_tsp = jax.tree_multimap(
          lambda x, y: x * (1 - tau) + y * tau,
          tsp, new_sp)

      return new_sp, new_spos, new_ep, new_epos, new_tsp, new_tep, q_loss_val

    full_q_update_fn = jax.pmap(
        _full_q_update_fn,
        # in_axes=(0, 0, 0, 0, 0, 0, None, 0, None, 0),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, 0, 0, 0),
        out_axes=0,
        axis_name='across_devices')

    def msg_loss(
        sp,
        ep,
        obs,
        acts,
    ):
      # batch_size x 2(because of double-Q)
      q_vals = networks.q_ensemble_member_apply(sp, ep, obs, acts)
      q_vals = jnp.min(q_vals, axis=-1)
      mean_q_vals = jax.lax.pmean(
          jax.lax.pmean(q_vals, axis_name='on_device'),
          axis_name='across_devices')
      q_vals_squared = jnp.square(q_vals)
      mean_q_vals_squared = jax.lax.pmean(
          jax.lax.pmean(q_vals_squared, axis_name='on_device'),
          axis_name='across_devices')
      std_q_vals = (mean_q_vals_squared - jnp.square(mean_q_vals))**0.5

      LCB = mean_q_vals + beta * std_q_vals
      return -1. * jnp.mean(LCB), q_vals

    vmapped_msg_loss = jax.vmap(
        msg_loss,
        in_axes=(None, 0, None, None),
        out_axes=0,
        axis_name='on_device')

    def rl_actor_loss(
        actor_params,
        sp,
        ep,
        obs,
        alpha,
        key,
    ):
      act_dists = networks.policy_network.apply(actor_params, obs)
      key, sub_key = jax.random.split(key)
      acts = networks.sample(act_dists, sub_key)
      log_probs = networks.log_prob(act_dists, acts)

      msg_loss_term, q_vals = vmapped_msg_loss(
          sp,
          ep,
          obs,
          acts)
      msg_loss_term = jnp.sum(msg_loss_term)

      ent_loss = jnp.mean(log_probs)

      # because we are going to sum grads over the devices, we need to
      # divide the ent_loss by num_devices
      total_loss = msg_loss_term + alpha * ent_loss / (float(num_devices))
      return total_loss, (ent_loss, q_vals)

    rl_actor_loss_val_and_grad = jax.value_and_grad(rl_actor_loss, has_aux=True)

    def _rl_actor_update_step(
        actor_params,
        actor_optim_state,
        sp,
        ep,
        transitions,
        alpha,
        key,):
      (total_loss, (avg_log_prob, q_vals)), total_loss_grads = rl_actor_loss_val_and_grad(
          actor_params,
          sp,
          ep,
          transitions.observation,
          alpha,
          key)
      total_loss_grads = jax.lax.psum(total_loss_grads, axis_name='across_devices')
      actor_updates, new_optim_state = policy_optimizer.update(
          total_loss_grads, actor_optim_state)
      new_actor_params = optax.apply_updates(actor_params, actor_updates)

      return new_actor_params, new_optim_state, avg_log_prob, total_loss, q_vals

    pmapped_rl_actor_update_step = jax.pmap(
        _rl_actor_update_step,
        in_axes=(0, 0, 0, 0, None, None, None),
        axis_name='across_devices')


    #     def rl_actor_loss(
    #         actor_params: networks_lib.Params,
    #         sp,
    #         ep,
    #         obs: types.NestedArray,
    #         alpha: jnp.ndarray,
    #         key: networks_lib.PRNGKey,
    #     ):
    #       """This is the loss that is used after the initial num_bc_iters."""
    #       act_dists = networks.policy_network.apply(actor_params, obs)
    #       key, sub_key = jax.random.split(key)
    #       acts = networks.sample(act_dists, sub_key)
    #       log_probs = networks.log_prob(act_dists, acts)

    #       msg_loss_term, all_q_vals = msg_loss(
    #           actor_params,
    #           sp,
    #           ep,
    #           obs,
    #           acts)
    #       ent_loss = jnp.mean(log_probs)

    #       total_loss = msg_loss_term + alpha * ent_loss
    #       return total_loss, (ent_loss, all_q_vals)

    #     rl_actor_loss_val_and_grad = jax.value_and_grad(
    #         rl_actor_loss, has_aux=True)

    #     def actor_bc_loss(
    #         actor_params: networks_lib.Params,
    #         obs: types.NestedArray,
    #         acts: types.NestedArray):
    #       act_dists = networks.policy_network.apply(actor_params, obs)
    #       log_probs = networks.log_prob(act_dists, acts)
    #       avg_log_probs = jnp.mean(log_probs)
    #       return -avg_log_probs, avg_log_probs

    #     bc_actor_loss_val_and_grad = jax.value_and_grad(
    #         actor_bc_loss, has_aux=True)

    # bc_actor_loss_val_and_grad = jax.value_and_grad(
    #     actor_bc_loss, has_aux=True)

    def bc_loss(
        actor_params,
        obs,
        acts):
      act_dists = networks.policy_network.apply(actor_params, obs)
      log_probs = networks.log_prob(act_dists, acts)
      avg_log_probs = jnp.mean(log_probs)
      return -avg_log_probs, avg_log_probs

    bc_loss_val_and_grad = jax.value_and_grad(bc_loss, has_aux=True)

    def _bc_actor_update_step(
        actor_params,
        actor_optim_state,
        transitions,):
      (total_loss, avg_log_prob), total_loss_grads = bc_loss_val_and_grad(
          actor_params,
          transitions.observation,
          transitions.action)
      q_vals = jnp.zeros([num_models_per_device], total_loss.dtype) # just placeholder filler
      actor_updates, new_optim_state = policy_optimizer.update(
          total_loss_grads, actor_optim_state)
      new_actor_params = optax.apply_updates(actor_params, actor_updates)

      return new_actor_params, new_optim_state, avg_log_prob, total_loss, q_vals

    pmapped_bc_actor_update_step = jax.pmap(
        _bc_actor_update_step,
        in_axes=(0, 0, None),
        axis_name='across_devices')

    #     def _actor_update_step(
    #         actor_params: networks_lib.Params,
    #         sp,
    #         ep,
    #         transitions: types.Transition,
    #         optim_state: optax.OptState,
    #         alpha: jnp.ndarray,
    #         key: networks_lib.PRNGKey,
    #         in_initial_bc_iters: bool):
    #       """The unjitted version of the bc update step.

    #       in_initial_bc_iters is used to indicate whether we are in the initial
    #       phase of training using a BC loss. In the jit call, this variable will be
    #       marked as a static arg name, so when it changes value from true to false
    #       it will be re-jitted."""
    #       if not in_initial_bc_iters:
    #         (total_loss, (avg_log_prob, all_q_vals)), total_loss_grads = total_actor_loss_val_and_grad(
    #             actor_params,
    #             sp,
    #             ep,
    #             transitions.observation,
    #             alpha,
    #             key)
    #         print(total_loss)
    #       else:
    #         (total_loss, avg_log_prob), total_loss_grads = bc_actor_loss_val_and_grad(
    #             actor_params,
    #             transitions.observation,
    #             transitions.action)
    #         all_q_vals = jnp.zeros(
    #             [num_devices, num_models_per_device], total_loss.dtype)
    #       actor_updates, optim_state = policy_optimizer.update(
    #           total_loss_grads, optim_state)
    #       actor_params = optax.apply_updates(actor_params, actor_updates)

    #       return actor_params, optim_state, avg_log_prob, total_loss, all_q_vals

    #     actor_update_step = jax.jit(
    #         _actor_update_step, static_argnames='in_initial_bc_iters')

    def alpha_loss_fn(alpha_params, avg_log_prob):
      alpha = jnp.exp(alpha_params)
      return alpha * jax.lax.stop_gradient(-avg_log_prob - target_entropy)

    alpha_loss_val_and_grad = jax.jit(jax.value_and_grad(alpha_loss_fn))

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

      metrics = OrderedDict()

      key = state.key
      if adaptive_entropy_coefficient:
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = entropy_coefficient

      # critic update step
      all_keys = jax.random.split(key, ensemble_size + 1)
      key = all_keys[0]
      critic_update_keys = jnp.reshape(
          all_keys[1:], (num_devices, num_models_per_device, 2))
      key, act_key = jax.random.split(key)
      # next_acts = batch_ensemble_msg_networks.apply_policy_and_sample(
      #     networks, eval_mode=False)(
      #         state.policy_params,
      #         sub_key,
      #         transitions.observation)

      reshaped_transitions = jax.tree_map(
          lambda t: jnp.reshape(t, (num_devices, num_models_per_device, int(t.shape[0] / ensemble_size), *t.shape[1:])),
          transitions)
      trans_next_obs, trans_obs = jax.tree_map(
          lambda t: jnp.reshape(t, (num_devices, int(t.shape[0] / num_devices), *t.shape[1:])),
          (transitions.next_observation, transitions.observation))

      new_sp, new_spos, new_ep, new_epos, new_tsp, new_tep, all_q_losses = \
          full_q_update_fn(
              state.replicated_shared_q_params,
              state.ensemble_q_params,
              state.replicated_shared_q_optim_state,
              state.ensemble_q_optim_state,
              state.target_replicated_shared_q_params,
              state.target_ensemble_q_params,
              # transitions,
              reshaped_transitions,
              state.replicated_policy_params,
              act_key,
              critic_update_keys,
              trans_next_obs,
              trans_obs,)
      metrics['avg_critic_loss'] = jnp.mean(all_q_losses)
      metrics['std_critic_loss'] = jnp.std(all_q_losses)
      # new_sp = state.replicated_shared_q_params
      # new_ep = state.ensemble_q_params
      # new_spos = state.replicated_shared_q_optim_state
      # new_epos = state.ensemble_q_optim_state
      # new_tsp = state.target_replicated_shared_q_params
      # new_tep = state.target_ensemble_q_params

      # actor update step
      key, sub_key = jax.random.split(key)
      # new_policy_params, new_policy_optimizer_state, avg_log_prob, act_loss, all_q_vals = actor_update_step(
      #     state.policy_params,
      #     state.replicated_shared_q_params,
      #     state.ensemble_q_params,
      #     transitions,
      #     state.policy_optimizer_state,
      #     alpha,
      #     sub_key,
      #     in_initial_bc_iters)
      if in_initial_bc_iters:
        new_policy_params, new_policy_optimizer_state, avg_log_prob, all_act_loss, all_q_vals = pmapped_bc_actor_update_step(
            state.replicated_policy_params,
            state.replicated_policy_optimizer_state,
            transitions)
      else:
        new_policy_params, new_policy_optimizer_state, avg_log_prob, all_act_loss, all_q_vals = pmapped_rl_actor_update_step(
            state.replicated_policy_params,
            state.replicated_policy_optimizer_state,
            state.replicated_shared_q_params,
            state.ensemble_q_params,
            transitions,
            alpha,
            sub_key)
      avg_log_prob = avg_log_prob[0]
      act_loss = jnp.mean(all_act_loss)
      metrics['actor_loss'] = act_loss
      metrics['actor_entropy'] = -avg_log_prob
      all_q_vals = jnp.reshape(all_q_vals, [-1])
      metrics['avg_q'] = jnp.mean(all_q_vals)
      metrics['std_q'] = jnp.std(all_q_vals)
      metrics['max_q'] = jnp.max(all_q_vals)
      metrics['min_q'] = jnp.min(all_q_vals)
      metrics['argmax_q'] = jnp.argmax(all_q_vals)
      metrics['argmin_q'] = jnp.argmin(all_q_vals)

      # create new state
      new_state = TrainingState(
          replicated_policy_optimizer_state=new_policy_optimizer_state,
          replicated_shared_q_optim_state=new_spos,
          ensemble_q_optim_state=new_epos,
          replicated_policy_params=new_policy_params,
          replicated_shared_q_params=new_sp,
          ensemble_q_params=new_ep,
          target_replicated_shared_q_params=new_tsp,
          target_ensemble_q_params=new_tep,
          key=key,)

      # alpha update step
      if (not in_initial_bc_iters) and adaptive_entropy_coefficient:
        alpha_loss, alpha_grads = alpha_loss_val_and_grad(
            state.alpha_params, avg_log_prob)
        alpha_update, alpha_optimizer_state = alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state)
        alpha_params = optax.apply_updates(state.alpha_params, alpha_update)
        alpha_params = jax.nn.relu(alpha_params + 20.) - 20.
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

    # self._update_step = _full_update_step

    self._update_step_in_initial_bc_iters = utils.process_multiple_batches(
        lambda x, y: _full_update_step(x, y, True),
        num_sgd_steps_per_step)
    self._update_step_rest = utils.process_multiple_batches(
        lambda x, y: _full_update_step(x, y, False),
        num_sgd_steps_per_step)

    def make_initial_state(key):
      """"""
      num_devices = jax.device_count()
      # critic stuff
      # model params
      key, sub_key = jax.random.split(key)
      shared_params, ensemble_params = networks.q_ensemble_init(ensemble_size, sub_key)
      # replicated_shared_params = jax.tree_map(
      #     lambda x: jnp.array([x] * num_devices), shared_params)
      replicated_shared_params = jax.device_put_replicated(
          shared_params, jax.local_devices())

      # optim params
      _, shared_params_optim_state, ensemble_params_optim_state = ensemble_utils.build_ensemble_optimizer(
          ensemble_size,
          shared_params,
          ensemble_params,
          optax.adam,
          {'learning_rate': q_lr})
      # replicated_shared_params_optim_state = jax.tree_map(
      #     lambda x: jnp.array([x] * num_devices), shared_params_optim_state)
      replicated_shared_params_optim_state = jax.device_put_replicated(
          shared_params_optim_state, jax.local_devices())


      # policy stuff
      key, sub_key = jax.random.split(key)
      policy_params = networks.policy_network.init(sub_key)
      policy_optimizer_state = policy_optimizer.init(policy_params)

      # replicated_policy_params = jax.tree_map(
      #     lambda x: jnp.array([x] * num_devices), policy_params)
      # replicated_policy_optimizer_state = jax.tree_map(
      #     lambda x: jnp.array([x] * num_devices), policy_optimizer_state)
      replicated_policy_params = jax.device_put_replicated(
          policy_params, jax.local_devices())
      replicated_policy_optimizer_state = jax.device_put_replicated(
          policy_optimizer_state, jax.local_devices())

      state = TrainingState(
          replicated_policy_optimizer_state=replicated_policy_optimizer_state,
          replicated_shared_q_optim_state=replicated_shared_params_optim_state,
          ensemble_q_optim_state=ensemble_params_optim_state,
          replicated_policy_params=replicated_policy_params,
          replicated_shared_q_params=replicated_shared_params,
          ensemble_q_params=ensemble_params,
          target_replicated_shared_q_params=replicated_shared_params,
          target_ensemble_q_params=ensemble_params,
          key=key,)

      # entropy stuff
      if adaptive_entropy_coefficient:
        state = state._replace(alpha_optimizer_state=alpha_optimizer_state,
                               alpha_params=log_alpha)

      # jax.tree_map(lambda t: print(t.shape), replicated_shared_params_optim_state)

      return state

    # Create initial state.
    self._state = make_initial_state(rng)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None


  def step(self):
    # print('DOING A STEP')
    sample = next(self._iterator)
    transitions = types.Transition(*sample.data)
    # print('GOT DATA')

    counts = self._counter.get_counts()
    if 'learner_steps' not in counts:
      cur_step = 0
    else:
      cur_step = counts['learner_steps']
    in_initial_bc_iters = cur_step < self._num_bc_iters
    if in_initial_bc_iters:
      self._state, metrics = self._update_step_in_initial_bc_iters(
          self._state, transitions)
    else:
      self._state, metrics = self._update_step_rest(
          self._state, transitions)

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
        # 'policy': self._state.policy_params,
        'policy': jax.tree_map(lambda x: x[0], self._state.replicated_policy_params),
    }
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state
