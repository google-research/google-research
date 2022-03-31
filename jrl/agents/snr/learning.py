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
"""Spectral Norm Regularization (SNR) learner implementation.

Almost like a vanilla actor-critic method with SNR ideas added on top.
"""

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
from jax.experimental import host_callback
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp
import numpy as np
import optax
import tensorflow_probability
from jrl.agents.snr import kmeans
from jrl.agents.snr import networks
from jrl.agents.snr import snr_utils
from jrl.agents.snr.config import SNRKwargs
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
distributional = networks_lib.distributional

SNRState = snr_utils.SNRState
jax_eig = jax.jit(snr_utils.jax_eig)


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  q_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  q_params: networks_lib.Params
  target_q_params: networks_lib.Params
  key: networks_lib.PRNGKey
  snr_state: SNRState
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[networks_lib.Params] = None


class SNRLearner(acme.Learner):
  """SNR learner."""

  _state: TrainingState

  def __init__(
      self,
      networks,
      rng,
      iterator,
      tau = 0.005,
      num_bc_iters = 50_000,
      reward_scale = 1.0,
      discount = 0.99,
      entropy_coefficient = None,
      target_entropy = 0,
      use_snr_in_bc_iters = False,
      snr_applied_to = 'policy',
      snr_alpha = 1.0,
      snr_kwargs = SNRKwargs(),
      policy_lr = 1e-4,
      q_lr = 3e-4,
      counter = None,
      logger = None,
      num_sgd_steps_per_step = 1):
    """Initialize the SNR learner.

    Args:
      networks: SNR networks
      rng: a key for random number generation.
      iterator: an iterator over training data.
      tau: target smoothing coefficient.
      reward_scale: reward scale.
      discount: discount to use for TD updates.
      entropy_coefficient: coefficient applied to the entropy bonus. If None, an
        adaptative coefficient will be used.
      target_entropy: Used to normalize entropy. Only used when
        entropy_coefficient is None.
      snr_alpha: weight for the SNR regularizer
      num_importance_acts: number of actions to take in the importance sampling
      policy_lr: learning rate for the policy
      q_lr: learning rate for the q functions
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      num_sgd_steps_per_step: number of sgd steps to perform per learner 'step'.
    """
    policy_optimizer = optax.adam(learning_rate=policy_lr)
    q_optimizer = optax.adam(learning_rate=q_lr)

    self._num_bc_iters = num_bc_iters
    self._num_sgd_steps_per_step = num_sgd_steps_per_step

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

    def alpha_loss_fn(alpha_params, avg_log_prob):
      alpha = jnp.exp(alpha_params)
      return alpha * jax.lax.stop_gradient(-avg_log_prob - target_entropy)


    def critic_loss(q_params,
                    policy_params,
                    target_q_params,
                    alpha,
                    transitions,
                    key,
                    snr_state,
                    in_initial_bc_iters,):
      q_old_action = networks.q_network.apply(
          q_params, transitions.observation, transitions.action)
      next_dist_params = networks.policy_network.apply(
          policy_params, transitions.next_observation)
      key, sub_key = jax.random.split(key)
      next_action = networks.sample(next_dist_params, sub_key)
      next_q = networks.q_network.apply(
          target_q_params, transitions.next_observation, next_action)
      # next_log_prob = networks.log_prob(next_dist_params, next_action)
      # next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
      next_v = jnp.min(next_q, axis=-1)
      target_q = jax.lax.stop_gradient(transitions.reward * reward_scale +
                                       transitions.discount * discount * next_v)
      q_error = q_old_action - jnp.expand_dims(target_q, -1)
      q_loss = 0.5 * jnp.mean(jnp.square(q_error))

      if snr_applied_to == 'critic':
        key, sub_key = jax.random.split(key)
        snr_next_dist_params = [
            next_dist_params._distribution._distribution.loc,
            next_dist_params._distribution._distribution.scale,
        ]
        snr_loss, (masked_s, c_matrix, snr_state) = snr_loss_fn(
            snr_next_dist_params,
            transitions.observation,
            transitions.action,
            transitions.next_observation,
            transitions.discount,
            sub_key,
            snr_state,
            q_params,
            target_q_params,)
        snr_loss_weight = 1.0

        total_q_loss = q_loss
        if (not in_initial_bc_iters) or use_snr_in_bc_iters:
          total_q_loss = total_q_loss + snr_alpha * snr_loss_weight * snr_loss
      else:
        snr_loss = 0.
        masked_s = 0.
        c_matrix = 0.
        snr_loss_weight = 0.
        total_q_loss = q_loss

      return total_q_loss, (q_loss, snr_loss, masked_s, c_matrix, snr_state, snr_loss_weight)

    def critic_apply_separate_last_layer(
        last_layer_params,
        rest_params,
        x):
      return networks.nt_critic_apply_fn(
          rest_params + [last_layer_params],
          x)
    dQ_dX = jax.jacfwd(critic_apply_separate_last_layer)

    def compute_kernel_features(params, X0):
      rest_params, last_layer_params = params[:-1], params[-1]
      dQ_dX0 = dQ_dX(last_layer_params, rest_params, X0)
      dQ_dX0 = jax.tree_map(
          lambda x: jnp.reshape(x, [x.shape[0], -1]),
          dQ_dX0)
      dQ_dX0 = jnp.concatenate(dQ_dX0, axis=-1)
      return dQ_dX0

    snr_loss_fn = snr_utils.build_snr_loss_fn(
        snr_kwargs,
        discount,
        networks,
        compute_kernel_features,)

    snr_loss_fn_val_and_grad = jax.value_and_grad(snr_loss_fn, has_aux=True)

    def actor_loss_fn(
        dist_params,
        q_params,
        alpha,
        transitions,
        key,
        in_initial_bc_iters,):
      distribution = tfd.Normal(loc=dist_params[0], scale=dist_params[1])
      dist_params = tfd.Independent(
          distributional.TanhTransformedDistribution(distribution),
          reinterpreted_batch_ndims=1)

      if not in_initial_bc_iters:
        key, sub_key = jax.random.split(key)
        action = networks.sample(dist_params, sub_key)
        log_prob = networks.log_prob(dist_params, action)
        q_action = networks.q_network.apply(
            q_params, transitions.observation, action)
        min_q = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - min_q
      else:
        log_prob = networks.log_prob(dist_params, transitions.action)
        actor_loss = -1. * log_prob
        min_q = 0.
      actor_loss = jnp.mean(actor_loss)

      return actor_loss, (min_q, log_prob)

    actor_loss_fn_val_and_grad = jax.value_and_grad(actor_loss_fn, has_aux=True)

    def compute_full_norm(p):
      """Compute the global norm across a nested structure of tensors."""
      return jnp.sqrt(
          sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(p)]))

    def total_actor_loss_fn(policy_params,
                            q_params,
                            target_q_params,
                            alpha,
                            transitions,
                            key,
                            snr_state,
                            in_initial_bc_iters,):
      dist_params = networks.policy_network.apply(
          policy_params, transitions.observation)
      dist_params = [
          dist_params._distribution._distribution.loc,
          dist_params._distribution._distribution.scale,
      ]
      next_dist_params = networks.policy_network.apply(
          policy_params, transitions.next_observation)
      next_dist_params = [
          next_dist_params._distribution._distribution.loc,
          next_dist_params._distribution._distribution.scale,
      ]

      key, sub_key = jax.random.split(key)
      (actor_loss, (min_q, log_prob)), dist_params_grad = actor_loss_fn_val_and_grad(
          dist_params,
          q_params,
          alpha,
          transitions,
          sub_key,
          in_initial_bc_iters)
      # dist_params_grad = jax.lax.stop_gradient(dist_params_grad)
      # dist_params_grad_norm = compute_full_norm(dist_params_grad)

      if snr_applied_to == 'policy':
        key, sub_key = jax.random.split(key)
        # (snr_loss, (masked_s, c_matrix, snr_state)), next_dist_params_grad = snr_loss_fn_val_and_grad(
        #     next_dist_params,
        #     transitions.observation,
        #     transitions.action,
        #     transitions.next_observation,
        #     sub_key,
        #     snr_state,
        #     q_params,)
        # next_dist_params_grad = jax.lax.stop_gradient(next_dist_params_grad)
        # next_dist_params_grad_norm = compute_full_norm(next_dist_params_grad)

        # snr_loss_weight = dist_params_grad_norm / (1e-6 + next_dist_params_grad_norm)
        # snr_loss_weight = jax.lax.stop_gradient(snr_loss_weight) # just in case

        snr_loss, (masked_s, c_matrix, snr_state) = snr_loss_fn(
            next_dist_params,
            transitions.observation,
            transitions.action,
            transitions.next_observation,
            transitions.discount,
            sub_key,
            snr_state,
            q_params,
            target_q_params,)
        snr_loss_weight = 1.0

        total_actor_loss = actor_loss
        if (not in_initial_bc_iters) or use_snr_in_bc_iters:
          total_actor_loss = total_actor_loss + snr_alpha * snr_loss_weight * snr_loss
      else:
        total_actor_loss = actor_loss
        snr_loss = 0.
        masked_s = 0.
        c_matrix = 0.
        snr_loss_weight = 0.

      return total_actor_loss, (min_q, log_prob, snr_loss, masked_s, c_matrix, snr_state, snr_loss_weight)

    alpha_val_and_grad = jax.jit(jax.value_and_grad(alpha_loss_fn))
    critic_val_and_grad = jax.value_and_grad(critic_loss, has_aux=True)
    actor_val_and_grad = jax.value_and_grad(total_actor_loss_fn, has_aux=True)

    def _full_update_step(
        state,
        transitions,
        in_initial_bc_iters,
    ):

      key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)
      if adaptive_entropy_coefficient:
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = entropy_coefficient

      if snr_applied_to == 'critic':
        (critic_loss_value, (q_loss, snr_term, masked_s, c_matrix, snr_state, snr_loss_weight)), critic_grads = critic_val_and_grad(state.q_params,
                                                              state.policy_params,
                                                              state.target_q_params,
                                                              alpha,
                                                              transitions,
                                                              key_critic,
                                                              state.snr_state,
                                                              in_initial_bc_iters)
      else:
        (critic_loss_value, (q_loss, _, _, _, _, _)), critic_grads = critic_val_and_grad(state.q_params,
                                                            state.policy_params,
                                                            state.target_q_params,
                                                            alpha,
                                                            transitions,
                                                            key_critic,
                                                            state.snr_state,
                                                            in_initial_bc_iters)

      if snr_applied_to == 'policy':
        (actor_loss_value, (min_q, log_prob, snr_term, masked_s, c_matrix, snr_state, snr_loss_weight)), actor_grads = actor_val_and_grad(
            state.policy_params,
            state.q_params,
            state.target_q_params,
            alpha,
            transitions,
            key_actor,
            state.snr_state,
            in_initial_bc_iters,)
      else:
        (actor_loss_value, (min_q, log_prob, _, _, _, _, _)), actor_grads = actor_val_and_grad(
          state.policy_params,
          state.q_params,
          state.target_q_params,
          alpha,
          transitions,
          key_actor,
          state.snr_state,
          in_initial_bc_iters,)

      # Apply policy gradients
      actor_update, policy_optimizer_state = policy_optimizer.update(
          actor_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, actor_update)

      # Apply critic gradients
      critic_update, q_optimizer_state = q_optimizer.update(
          critic_grads, state.q_optimizer_state)
      q_params = optax.apply_updates(state.q_params, critic_update)

      # new_target_q_params = jax.tree_multimap(
      #     lambda x, y: x * (1 - tau) + y * tau, state.target_q_params, q_params)
      new_target_q_params = q_params

      metrics = OrderedDict()
      metrics['critic_loss'] = q_loss
      metrics['actor_loss'] = actor_loss_value
      metrics['actor_log_probs'] = jnp.mean(log_prob)
      metrics['q/avg'] = jnp.mean(min_q)
      metrics['q/std'] = jnp.std(min_q)
      metrics['q/max'] = jnp.max(min_q)
      metrics['q/min'] = jnp.min(min_q)
      metrics['snr/loss'] = snr_term
      metrics['snr/loss_weight'] = snr_loss_weight
      num_gt_zero = jnp.sum(masked_s > 0.)
      metrics['snr/num_gt_zero'] = num_gt_zero
      min_s = jnp.take(masked_s, [num_gt_zero - 1], axis=0)[0]
      num_gt_zero = num_gt_zero + 1e-6
      mean_s = jnp.sum(masked_s) / num_gt_zero
      std_s = jnp.sqrt((jnp.sum(masked_s**2) / num_gt_zero) - mean_s**2)
      metrics['snr/avg'] = mean_s
      metrics['snr/std'] = std_s
      metrics['snr/max'] = jnp.max(masked_s)
      metrics['snr/min'] = min_s

      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          target_q_params=new_target_q_params,
          key=key,
          snr_state=snr_state,
      )

      # alpha update step
      if (not in_initial_bc_iters) and adaptive_entropy_coefficient:
        alpha_loss, alpha_grads = alpha_val_and_grad(
            state.alpha_params, jnp.mean(log_prob))
        alpha_update, alpha_optimizer_state = alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state)
        alpha_params = optax.apply_updates(state.alpha_params, alpha_update)
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

    # update_step = utils.process_multiple_batches(update_step,
    #                                              num_sgd_steps_per_step)
    # # Use the JIT compiler.
    # self._update_step = jax.jit(update_step)

    self._update_step_in_initial_bc_iters = jax.jit(utils.process_multiple_batches(
        lambda x, y: _full_update_step(x, y, True),
        num_sgd_steps_per_step))
    self._update_step_rest = jax.jit(utils.process_multiple_batches(
        lambda x, y: _full_update_step(x, y, False),
        num_sgd_steps_per_step))

    def make_initial_state(key):
      """Initialises the training state (parameters and optimiser state)."""
      key_policy, key_q, key = jax.random.split(key, 3)

      policy_params = networks.policy_network.init(key_policy)
      policy_optimizer_state = policy_optimizer.init(policy_params)

      q_params = networks.q_network.init(key_q)
      q_optimizer_state = q_optimizer.init(q_params)

      # Must be a nicer way to do this...
      # c_dim = q_params[0][-1][0].shape[0] + q_params[0][-1][1].shape[0]
      c_dim = q_params[-1][0].shape[0] + q_params[-1][1].shape[0]
      key, sub_key = jax.random.split(key)
      snr_state = snr_utils.snr_state_init(c_dim, sub_key, snr_kwargs)

      state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          target_q_params=q_params,
          key=key,
          snr_state=snr_state)

      if adaptive_entropy_coefficient:
        state = state._replace(alpha_optimizer_state=alpha_optimizer_state,
                               alpha_params=log_alpha)
      return state

    def compute_periodic_snr_debug_info(state):
      debug_info = OrderedDict()
      snr_state = state.snr_state

      if snr_state.snr_matrix is not None:
        w, v = jax_eig(snr_state.snr_matrix)
        D = jnp.linalg.inv(v) @ snr_state.snr_matrix @ v
        D = jnp.diag(D).real
        debug_info['debug_info/max_real_eig'] = jnp.max(D)
        debug_info['debug_info/min_real_eig'] = jnp.min(D)
        debug_info['debug_info/avg_real_eig'] = jnp.mean(D)
        debug_info['debug_info/std_real_eig'] = jnp.std(D)
        debug_info['debug_info/num_real_eig_gt_zero'] = jnp.sum(D > 0.)
        debug_info['debug_info/ratio_real_eig_gt_zero'] = jnp.mean(D > 0.)

      return debug_info

    self._compute_periodic_snr_debug_info = jax.jit(compute_periodic_snr_debug_info)

    # Create initial state.
    self._state = make_initial_state(rng)

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
    in_initial_bc_iters = cur_step < self._num_bc_iters

    if in_initial_bc_iters:
      self._state, metrics = self._update_step_in_initial_bc_iters(
          self._state, transitions)
    else:
      self._state, metrics = self._update_step_rest(
          self._state, transitions)

    periodic_debug_info = self._compute_periodic_snr_debug_info(self._state)
    metrics.update(periodic_debug_info)

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
    }
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state
