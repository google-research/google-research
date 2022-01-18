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

# Copyright 2021 The Google Research Authors.
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
"""Implementation of Hamiltonian Monte Carlo."""

import jax
import jax.numpy as jnp

from bnn_hmc.utils import tree_utils


def make_leapfrog(log_prob_and_grad):
  """Leapfrog method."""

  # Return likelihood and prior separately in log_prob_and_grad to compute the
  # prior densities ratio more accurately in float32 in the accept-reject step.

  def leapfrog(dataset, init_params, init_net_state, init_momentum, init_grad,
               step_size, n_leapfrog):

    def _leapfrog_body(_, carry):
      params, net_state, momentum, grad, _, _ = carry
      momentum = jax.tree_multimap(lambda m, g: m + 0.5 * step_size * g,
                                   momentum, grad)
      params = jax.tree_multimap(lambda s, m: s + m * step_size, params,
                                 momentum)
      log_prob, grad, log_likelihood, net_state = log_prob_and_grad(
          dataset, params, net_state)
      momentum = jax.tree_multimap(lambda m, g: m + 0.5 * step_size * g,
                                   momentum, grad)
      return params, net_state, momentum, grad, log_prob, log_likelihood

    init_vals = (init_params, init_net_state, init_momentum, init_grad, 0., 0.)
    (new_params, new_net_state, new_momentum, new_grad, new_log_prob,
     new_log_likelihood) = jax.lax.fori_loop(
         jnp.zeros_like(n_leapfrog), n_leapfrog, _leapfrog_body, init_vals)
    return (new_params, new_net_state, new_momentum, new_grad,
            new_log_likelihood)

  return leapfrog


def _nan_to_inf(x):
  return jnp.where(jnp.isnan(x), jnp.inf + jnp.zeros_like(x), x)


def _first(xy):
  return xy[0]


def _second(xy):
  return xy[1]


def get_kinetic_energy_diff(momentum1, momentum2):
  return sum([
      0.5 * jnp.sum(m1**2 - m2**2)
      for m1, m2 in zip(jax.tree_leaves(momentum1), jax.tree_leaves(momentum2))
  ])


def make_accept_prob(log_prior_diff_fn):

  def get_accept_prob(log_likelihood1, params1, momentum1, log_likelihood2,
                      params2, momentum2):
    energy_diff = get_kinetic_energy_diff(momentum1, momentum2)
    energy_diff -= log_likelihood1 - log_likelihood2
    energy_diff -= log_prior_diff_fn(params1, params2)
    accept_prob = jnp.minimum(1., jnp.exp(energy_diff))
    return accept_prob

  return get_accept_prob


def adapt_step_size(step_size, target_accept_rate, accept_prob,
                    step_size_adaptation_speed):
  log_factor = jnp.where(
      jnp.logical_or(target_accept_rate <= 0, step_size_adaptation_speed <= 0),
      0., step_size_adaptation_speed * (accept_prob - target_accept_rate))
  return step_size * jnp.exp(log_factor)


def make_adaptive_hmc_update(log_prob_and_grad_fn, log_prior_diff_fn):
  """Returns an adaptive HMC update function."""
  leapfrog = make_leapfrog(log_prob_and_grad_fn)
  get_accept_prob = make_accept_prob(log_prior_diff_fn)

  def adaptive_hmc_update(dataset,
                          params,
                          net_state,
                          log_likelihood,
                          state_grad,
                          key,
                          step_size,
                          n_leapfrog_steps,
                          target_accept_rate,
                          step_size_adaptation_speed,
                          do_mh_correction=True):

    normal_key, uniform_key, jitter_key = jax.random.split(key, 3)
    momentum, _ = tree_utils.normal_like_tree(params, normal_key)

    new_params, net_state, new_momentum, new_grad, new_log_likelihood = (
        leapfrog(dataset, params, net_state, momentum, state_grad, step_size,
                 n_leapfrog_steps))
    accept_prob = get_accept_prob(log_likelihood, params, momentum,
                                  new_log_likelihood, new_params, new_momentum)
    accepted = jax.random.uniform(uniform_key) < accept_prob

    step_size = adapt_step_size(step_size, target_accept_rate, accept_prob,
                                step_size_adaptation_speed)

    if do_mh_correction:
      params = jax.lax.cond(accepted, _first, _second, (new_params, params))
      log_likelihood = jnp.where(accepted, new_log_likelihood, log_likelihood)
      state_grad = jax.lax.cond(accepted, _first, _second,
                                (new_grad, state_grad))
    else:
      params, log_likelihood, state_grad = (new_params, new_log_likelihood,
                                            new_grad)
    return (params, net_state, log_likelihood, state_grad, step_size,
            accept_prob, accepted)

  return adaptive_hmc_update
