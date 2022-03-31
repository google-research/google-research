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
"""CQL learner implementation."""

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

from jrl.agents.cql import networks
from jrl.agents.snr import snr_utils
from jrl.agents.snr.config import SNRKwargs
from jrl.agents.snr.learning import SNRState


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


class CQLLearner(acme.Learner):
  """CQL learner."""

  _state: TrainingState

  def __init__(
      self,
      networks,
      rng,
      iterator,
      num_bc_iters = 50_000,
      tau = 0.005,
      reward_scale = 1.0,
      discount = 0.99,
      entropy_coefficient = None,
      target_entropy = 0,
      cql_alpha = 5.0,
      num_importance_acts = 16,
      snr_alpha = 0.0,
      snr_kwargs = SNRKwargs(),
      policy_lr = 1e-4,
      q_lr = 3e-4,
      counter = None,
      logger = None,
      num_sgd_steps_per_step = 1):
    """Initialize the SAC learner.

    Args:
      networks: CQL networks
      rng: a key for random number generation.
      iterator: an iterator over training data.
      tau: target smoothing coefficient.
      reward_scale: reward scale.
      discount: discount to use for TD updates.
      entropy_coefficient: coefficient applied to the entropy bonus. If None, an
        adaptative coefficient will be used.
      target_entropy: Used to normalize entropy. Only used when
        entropy_coefficient is None.
      cql_alpha: weight for the CQL regularizer
      num_importance_acts: number of actions to take in the importance sampling
      policy_lr: learning rate for the policy
      q_lr: learning rate for the q functions
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      num_sgd_steps_per_step: number of sgd steps to perform per learner 'step'.
    """
    self._use_snr = snr_alpha > 0.
    self._num_sgd_steps_per_step = num_sgd_steps_per_step
    self._num_bc_iters = num_bc_iters

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

    def alpha_loss(alpha_params,
                   avg_log_prob):
      """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
      alpha = jnp.exp(alpha_params)
      alpha_loss = alpha * jax.lax.stop_gradient(-avg_log_prob - target_entropy)
      return alpha_loss

    def critic_loss(q_params,
                    policy_params,
                    target_q_params,
                    alpha,
                    transitions,
                    key):
      q_old_action = networks.q_network.apply(
          q_params, transitions.observation, transitions.action)
      next_dist_params = networks.policy_network.apply(
          policy_params, transitions.next_observation)
      next_action = networks.sample(next_dist_params, key)
      next_log_prob = networks.log_prob(next_dist_params, next_action)
      next_q = networks.q_network.apply(
          target_q_params, transitions.next_observation, next_action)
      # next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
      next_v = jnp.min(next_q, axis=-1)
      target_q = jax.lax.stop_gradient(transitions.reward * reward_scale +
                                       transitions.discount * discount * next_v)
      q_error = q_old_action - jnp.expand_dims(target_q, -1)
      # q_loss = 0.5 * jnp.mean(jnp.square(q_error))
      q_loss = jnp.mean(jnp.square(q_error))
      q_loss = q_loss * q_error.shape[-1]
      return q_loss

    def cql_loss(q_params,
                 policy_params,
                 transitions,
                 key):
      obs = transitions.observation
      tiled_obs = jnp.tile(obs, [num_importance_acts, 1])

      # policy samples
      dist_params = networks.policy_network.apply(
          policy_params, obs)
      key, sub_key = jax.random.split(key)
      pi_acts = dist_params.sample(
          sample_shape=(num_importance_acts,), seed=sub_key)
      pi_log_probs = dist_params.log_prob(pi_acts)
      pi_log_probs = jnp.expand_dims(pi_log_probs, axis=2)
      acts_shape_prior = pi_acts.shape
      pi_acts = jnp.reshape(
          pi_acts,
          [acts_shape_prior[0]*acts_shape_prior[1], acts_shape_prior[2]])
      q_for_pi_acts = networks.q_network.apply(
          q_params, tiled_obs, pi_acts)
      q_for_pi_acts = jnp.reshape(q_for_pi_acts, list(acts_shape_prior[:2])+[q_for_pi_acts.shape[-1]])
      # q_for_pi_acts = jnp.min(q_for_pi_acts, axis=-1)

      # uniform samples
      key, sub_key = jax.random.split(key)
      unif_acts = jax.random.uniform(
          sub_key, pi_acts.shape, dtype=pi_acts.dtype, minval=-1., maxval=1.)
      unif_log_probs = -jnp.log(2.) * pi_acts.shape[-1]
      unif_acts = jnp.reshape(
          unif_acts,
          [acts_shape_prior[0]*acts_shape_prior[1], acts_shape_prior[2]])
      q_for_unif = networks.q_network.apply(
          q_params, tiled_obs, unif_acts)
      q_for_unif = jnp.reshape(q_for_unif, list(acts_shape_prior[:2])+[q_for_unif.shape[-1]])
      # q_for_unif = jnp.min(q_for_pi_acts, axis=-1)

      # compute the cql regularizer
      # 2N x batch_size x 2
      # 2 because two critics
      combined = jnp.concatenate(
          [q_for_pi_acts - pi_log_probs, q_for_unif - unif_log_probs], axis=0)
      logsumexp = jax_logsumexp(combined, axis=0) # batch_size x 2

      # get the data action q-values
      data_q = networks.q_network.apply(q_params, obs, transitions.action)

      cql_term = jnp.mean(logsumexp - data_q)

      return cql_term


    def total_critic_loss(q_params,
                          policy_params,
                          target_q_params,
                          alpha,
                          cql_alpha,
                          transitions,
                          key):
      critic_loss_term = critic_loss(
          q_params, policy_params, target_q_params, alpha, transitions, key)
      cql_term = cql_loss(q_params, policy_params, transitions, key)
      total = critic_loss_term + cql_alpha * cql_term
      return total, {'critic_loss': critic_loss_term, 'cql_loss': cql_term}

    critic_grad = jax.value_and_grad(total_critic_loss, has_aux=True)

    def critic_update_step(
        q_params,
        target_q_params,
        optim_state,
        policy_params,
        alpha,
        cql_alpha,
        transitions,
        key):
      total_critic_loss_and_aux, critic_grads = critic_grad(
          q_params,
          policy_params,
          target_q_params,
          alpha,
          cql_alpha,
          transitions,
          key)
      critic_grads = jax.lax.pmean(critic_grads, 'devices')
      # Apply critic gradients
      critic_update, optim_state = q_optimizer.update(
          critic_grads, optim_state)
      q_params = optax.apply_updates(q_params, critic_update)

      target_q_params = jax.tree_multimap(
          lambda x, y: x * (1 - tau) + y * tau, target_q_params, q_params)

      return total_critic_loss_and_aux, q_params, target_q_params, optim_state

    pmapped_critic_update = jax.pmap(
        critic_update_step,
        axis_name='devices',
        in_axes=(0, 0, 0, 0, None, None, 0, 0),
        out_axes=0)

    snr_loss_fn = snr_utils.build_snr_loss_fn(
        snr_kwargs,
        discount,
        networks,
        networks.compute_kernel_features,)

    def actor_loss(policy_params,
                   q_params,
                   target_q_params,
                   alpha,
                   transitions,
                   snr_state,
                   key,
                   in_initial_bc_iters):
      dist_params = networks.policy_network.apply(
          policy_params, transitions.observation)
      if in_initial_bc_iters:
        log_prob = networks.log_prob(dist_params, transitions.action)
        min_q = 0.
        actor_loss = -log_prob

        # No SNR in bc iters
        sn = 0.
        new_snr_state = snr_state
      else:
        key, sub_key = jax.random.split(key)
        action = networks.sample(dist_params, sub_key)
        log_prob = networks.log_prob(dist_params, action)
        q_action = networks.q_network.apply(
            q_params, transitions.observation, action)
        min_q = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - min_q

        # SNR only applied after initial BC iters
        if self._use_snr:
          next_dist_params = networks.policy_network.apply(
              policy_params, transitions.next_observation)
          next_dist_params = [
              next_dist_params._distribution._distribution.loc,
              next_dist_params._distribution._distribution.scale,
          ]
          key, sub_key = jax.random.split(key)
          sn, (masked_s, C, new_snr_state) = snr_loss_fn(
              next_dist_params,
              transitions.observation,
              transitions.action,
              transitions.next_observation,
              transitions.discount,
              sub_key,
              snr_state,
              q_params,
              target_q_params)
          actor_loss = actor_loss + snr_alpha * sn
        else:
          sn = 0.
          new_snr_state = snr_state

      return jnp.mean(actor_loss), (min_q, jnp.mean(log_prob), sn, new_snr_state)

    actor_grad = jax.value_and_grad(actor_loss, has_aux=True)

    def actor_update_step(
        in_initial_bc_iters,
        policy_params,
        optim_state,
        q_params,
        target_q_params,
        alpha,
        transitions,
        snr_state,
        key,):
      (actor_loss, (min_q, avg_log_prob, sn, new_snr_state)), actor_grads = actor_grad(
          policy_params,
          q_params,
          target_q_params,
          alpha,
          transitions,
          snr_state,
          key,
          in_initial_bc_iters,)

      actor_grads = jax.lax.pmean(actor_grads, 'devices')

      actor_update, optim_state = policy_optimizer.update(
          actor_grads, optim_state)
      policy_params = optax.apply_updates(policy_params, actor_update)

      return policy_params, optim_state, actor_loss, min_q, avg_log_prob, sn, new_snr_state

    # pmapped_actor_update_in_bc_iters = jax.pmap(
    #     lambda *args: actor_update_step(True, *args),
    #     axis_name='devices',
    #     in_axes=(0, 0, 0, 0, None, 0, 0, 0))
    # pmapped_actor_update_after_bc_iters = jax.pmap(
    #     lambda *args: actor_update_step(False, *args),
    #     axis_name='devices',
    #     in_axes=(0, 0, 0, 0, None, 0, 0, 0))

    pmapped_actor_update = jax.pmap(
        actor_update_step,
        axis_name='devices',
        in_axes=(None, 0, 0, 0, 0, None, 0, 0, 0),
        static_broadcasted_argnums=0)

    alpha_grad = jax.value_and_grad(alpha_loss)
    num_devices = jax.local_device_count()

    def update_step(
        state,
        transitions,
        in_initial_bc_iters,
    ):

      def reshape_for_devices(t):
        rest_t_shape = list(t.shape[1:])
        new_shape = [num_devices, t.shape[0]//num_devices,] + rest_t_shape
        return jnp.reshape(t, new_shape)
      transitions = jax.tree_map(reshape_for_devices, transitions)

      key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)
      if adaptive_entropy_coefficient:
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = entropy_coefficient

      key_critic = jax.random.split(key_critic, jax.local_device_count())
      # print(jax.tree_map(lambda t: t.shape, state.q_params))
      total_critic_loss_and_aux, q_params, new_target_q_params, q_optimizer_state = pmapped_critic_update(
          state.q_params,
          state.target_q_params,
          state.q_optimizer_state,
          state.policy_params,
          alpha,
          cql_alpha,
          transitions,
          key_critic,)
      # print(jax.tree_map(lambda t: t.shape, q_params))
      total_critic_loss_and_aux = jax.tree_map(jnp.mean, total_critic_loss_and_aux)

      key_actor = jax.random.split(key_actor, jax.local_device_count())
      # if in_initial_bc_iters:
      #   pmapped_actor_update = pmapped_actor_update_in_bc_iters
      # else:
      #   pmapped_actor_update = pmapped_actor_update_after_bc_iters
      policy_params, policy_optimizer_state, actor_loss, min_q, avg_log_prob, sn, new_snr_state = pmapped_actor_update(
          in_initial_bc_iters,
          state.policy_params,
          state.policy_optimizer_state,
          state.q_params,
          state.target_q_params,
          alpha,
          transitions,
          state.snr_state,
          key_actor,)
      avg_log_prob = jnp.mean(avg_log_prob)

      critic_loss_aux = total_critic_loss_and_aux[1]
      # metrics = {
      #     'critic_loss': critic_loss_aux['critic_loss'],
      #     'cql_loss': critic_loss_aux['cql_loss'],
      #     'actor_loss': actor_loss,
      # }
      metrics = OrderedDict()
      metrics['actor_loss'] = jnp.mean(actor_loss)
      metrics['avg_log_prob'] = avg_log_prob
      metrics['total_critic_loss'] = total_critic_loss_and_aux[0]
      metrics['critic_loss'] = critic_loss_aux['critic_loss']
      metrics['cql_loss'] = critic_loss_aux['cql_loss']
      metrics['q/avg'] = jnp.mean(min_q)
      metrics['q/std'] = jnp.std(min_q)
      metrics['q/max'] = jnp.max(min_q)
      metrics['q/min'] = jnp.min(min_q)
      metrics['SNR/loss'] = jnp.mean(sn)

      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          target_q_params=new_target_q_params,
          key=key,
          snr_state=new_snr_state,
      )
      if adaptive_entropy_coefficient and (not in_initial_bc_iters):
        # Apply alpha gradients
        alpha_loss, alpha_grads = alpha_grad(state.alpha_params,
                                             avg_log_prob)
        alpha_update, alpha_optimizer_state = alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state)
        alpha_params = optax.apply_updates(state.alpha_params, alpha_update)
        metrics['alpha_loss'] = alpha_loss
        metrics['alpha'] = jnp.exp(alpha_params)
        new_state = new_state._replace(
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params)
      else:
        metrics['alpha_loss'] = 0.
        metrics['alpha'] = jnp.exp(state.alpha_params)
        new_state = new_state._replace(
            alpha_optimizer_state=state.alpha_optimizer_state,
            alpha_params=state.alpha_params)

      # metrics['observations_mean'] = jnp.mean(
      #     utils.batch_concat(
      #         jax.tree_map(lambda x: jnp.abs(jnp.mean(x, axis=0)),
      #                      transitions.observation)))
      # metrics['observations_std'] = jnp.mean(
      #     utils.batch_concat(
      #         jax.tree_map(lambda x: jnp.std(x, axis=0),
      #                      transitions.observation)))
      # metrics['next_observations_mean'] = jnp.mean(
      #     utils.batch_concat(
      #         jax.tree_map(lambda x: jnp.abs(jnp.mean(x, axis=0)),
      #                      transitions.next_observation)))
      # metrics['next_observations_std'] = jnp.mean(
      #     utils.batch_concat(
      #         jax.tree_map(lambda x: jnp.std(x, axis=0),
      #                      transitions.next_observation)))

      return new_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray)

    # Iterator on demonstration transitions.
    self._iterator = iterator

    # update_step = utils.process_multiple_batches(update_step,
    #                                              num_sgd_steps_per_step)
    self._update_step_in_initial_bc_iters = utils.process_multiple_batches(
        lambda x, y: update_step(x, y, True),
        num_sgd_steps_per_step)
    self._update_step_rest = utils.process_multiple_batches(
        lambda x, y: update_step(x, y, False),
        num_sgd_steps_per_step)

    # Use the JIT compiler.
    self._update_step = jax.jit(update_step)

    def make_initial_state(key):
      """Initialises the training state (parameters and optimiser state)."""
      key_policy, key_q, key = jax.random.split(key, 3)
      devices = jax.local_devices()

      policy_params = networks.policy_network.init(key_policy)
      policy_optimizer_state = policy_optimizer.init(policy_params)
      policy_params = jax.device_put_replicated(policy_params, devices)
      policy_optimizer_state = jax.device_put_replicated(
          policy_optimizer_state, devices)

      q_params = networks.q_network.init(key_q)
      q_optimizer_state = q_optimizer.init(q_params)
      q_params = jax.device_put_replicated(q_params, devices)
      q_optimizer_state = jax.device_put_replicated(
          q_optimizer_state, devices)

      key, sub_key = jax.random.split(key)
      c_dim = 42 # TODO(kamyar): implement this
      snr_state = snr_utils.snr_state_init(
          c_dim,
          sub_key,
          snr_kwargs,)
      snr_state = jax.device_put_replicated(snr_state, devices)

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

    # self._state, metrics = self._update_step(self._state, transitions)

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
        'policy': jax.tree_map(lambda x: x[0], self._state.policy_params),
        'q': jax.tree_map(lambda x: x[0], self._state.q_params),
    }
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state
