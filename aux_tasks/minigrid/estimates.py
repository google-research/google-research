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

"""Different matrix estimates."""

import functools

import jax
import jax.numpy as jnp
import optax


def matrix_estimator(Phi, num_rows, key):  # pylint: disable=invalid-name
  r"""Computes an unbiased estimate of an input matrix.

  $\nu(s_i)^{-1}e_{s_i} \phi_{s_i}^\T$

  Args:
    Phi: S times d array
    num_rows: int: number of rows used in estimators
    key: prng key
  Returns:
    S times d array
  """
  S, _ = Phi.shape  # pylint: disable=invalid-name
  states = jax.random.randint(key, (num_rows,), 0, S)
  # states = jax.random.permutation(key, jnp.arange(S))[:num_rows]
  mask = jnp.zeros_like(Phi)
  mask = mask.at[states].set(1)
  return Phi * mask


def j_truncated_serie(Phi, j, coeff_alpha):  # pylint: disable=invalid-name
  """Computes j truncated serie."""
  _, d = Phi.shape  # pylint: disable=invalid-name
  norm = jnp.linalg.norm(Phi.T @ Phi, ord=2)
  alpha = coeff_alpha * 1 / norm
  return jnp.linalg.solve(Phi.T @ Phi, jnp.eye(d)) @ (
      jnp.eye(d) -
      jnp.linalg.matrix_power(jnp.eye(d) - alpha * Phi.T @ Phi, j + 1))


def compute_bias_cov_lissa(Phi, j, subkey, coeff_alpha):  # pylint: disable=invalid-name
  """Computes bias between lissa estimator and the inverse covariance matrix."""
  S, d = Phi.shape  # pylint: disable=invalid-name
  states = jax.random.randint(subkey, (j, 1), 0, S)
  norm = 2 * jnp.max(jnp.sum(jnp.square(Phi[states.reshape(j,)]), axis=1))
  alpha = coeff_alpha * 1 / norm
  return (-1) * jnp.linalg.solve(Phi.T @ Phi,
                                 jnp.eye(d)) @ jnp.linalg.matrix_power(
                                     jnp.eye(d) - alpha * Phi.T @ Phi, j + 1)


@functools.partial(jax.jit, static_argnums=(1, 2, 4, 5, 6))
def lissa(Phi,  # pylint: disable=invalid-name
          j,
          num_rows,
          subkey,
          coeff_alpha,
          use_penalty=False,
          reg_coeff=0.0):
  """Computes the lissa estimator.

  Args:
    Phi: S times d array
    j: int, index of the lissa estimator
    num_rows: int: number of rows used in estimators
    subkey: prng key
    coeff_alpha: float
    use_penalty: bool: whether to add "lambda * Id" term to features
    reg_coeff: float: coeff for regs

  Returns:
    d times d array
  """
  num_rows = 1  # use one sample per iteration
  S, d = Phi.shape  # pylint: disable=invalid-name
  I = jnp.eye(d)  # pylint: disable=invalid-name

  states = jax.random.randint(subkey, (j, 1), 0, S)
  # states = jnp.repeat(jnp.arange(S).reshape(1, S), j, axis=0)
  norm = 2 * jnp.max(jnp.sum(jnp.square(Phi[states.reshape(j,)]), axis=1))
  # norm = jnp.linalg.norm(Phi.T @ Phi, ord=2)
  alpha = coeff_alpha * 1 / norm
  lissa_init = alpha * I

  def _neumann_series(carry, state):
    A_j = alpha * I  # pylint: disable=invalid-name
    # pylint: disable=invalid-name
    if use_penalty:
      A_j += (I - alpha * (1 / num_rows) *
              (Phi[state, :].T @ Phi[state, :] + reg_coeff * I)) @ carry
    else:
      A_j += (I - alpha *
              (1 / num_rows) * Phi[state, :].T @ Phi[state, :]) @ carry
      # A_j += (I - alpha *
      #         (1 / num_rows) * Phi.T @ Phi) @ carry
    return A_j, A_j

  lisa_j, result = jax.lax.scan(_neumann_series, lissa_init, states)
  return lisa_j, result, lissa_init


@jax.jit
def _russian_roulette(Phi, states, coefficients, alpha):  # pylint: disable=invalid-name
  """Computes Russian roulette given fixed number of iterations."""
  S, d = Phi.shape  # pylint: disable=invalid-name
  I = jnp.eye(d)  # pylint: disable=invalid-name

  def _lissa_body(carry, state):
    lissa_j = alpha * I
    lissa_j += (
        I - alpha * S * jnp.einsum('i,j->ij', Phi[state], Phi[state])) @ carry
    return lissa_j, lissa_j

  lissa_init = alpha * I
  _, lissa_estimates = jax.lax.scan(_lissa_body, lissa_init, states)

  deltas = I - S * jnp.einsum('ni,nj,mjk->mik', Phi[states], Phi[states],
                              lissa_estimates)
  deltas *= coefficients.reshape(-1, 1, 1)

  return jnp.sum(deltas, axis=0)


def russian_roulette(Phi, p, key, coeff_alpha):  # pylint: disable=invalid-name
  """Computes the Russian roulette estimator from a LISSA sequence.

  Args:
    Phi: S times d array
    p: paramter of the bernoulli distribution
    key: prng key
    coeff_alpha: float
  Returns:
    array of shape d times d
  """
  S, _ = Phi.shape  # pylint: disable=invalid-name
  norm = jnp.linalg.norm(Phi.T @ Phi, ord=2)
  alpha = coeff_alpha * 1 / norm

  # Sample from geometric R.V. to get number of iterations
  key, subkey = jax.random.split(key)
  iterations = int(
      jnp.ceil(jnp.log(jax.random.uniform(subkey)) / jnp.log1p(-p)))

  # Sample states
  key, subkey = jax.random.split(key)
  states = jax.random.randint(subkey, (iterations,), 0, S)

  # Get delta coefficients
  coefficients = alpha / ((1 - p) ** jnp.arange(1, iterations + 1))

  return _russian_roulette(Phi, states, coefficients, alpha)


@functools.partial(jax.jit, static_argnums=(2, 3, 5, 6, 7, 8))
def least_square_estimator(  # pylint: disable=invalid-name
    Phi,
    Psi,
    num_rows,
    j,
    key,
    estimator='lissa',
    alpha=0.9,
    use_penalty=False,
    reg_coeff=0.0):  # pylint: disable=invalid-name
  r"""Computes an unbiased least squares estimate.

  $W^*_\Phi = (\Phi^T \Phi)^{-1} \Phi^T \Psi$

  Args:
    Phi: S times d array
    Psi: S times T array
    num_rows: int: number of rows used in estimators
    j: int: num of samples for lissa
    key: prng key
    estimator: str: russian_roulette, lissa, hat_w
    alpha: float: renormalize covariance term
    use_penalty: bool: whether to add "lambda * Id" term to features
    reg_coeff: float: coeff for reg

  Returns:
    array d times T
  """
  S, d = Phi.shape  # pylint: disable=invalid-name
  key, subkey = jax.random.split(key)
  states = jax.random.randint(subkey, (num_rows,), 0, S)
  _, subkey = jax.random.split(key)
  if estimator == 'lissa':
    cov_estim, _, _ = lissa(Phi, j, num_rows, subkey, alpha, use_penalty,
                            reg_coeff)
  # cov_estim = jnp.linalg.solve(Phi.T @ Phi, jnp.eye(d))
  elif estimator == 'naive':
    states_cov = jax.random.randint(subkey, (j,), 0, S)
    cov_estim = jnp.linalg.solve(Phi[states_cov].T @ Phi[states_cov],
                                 jnp.eye(d))
  elif estimator == 'rr':
    pass
  return cov_estim @ Phi[states, :].T @ Psi[
      states, :] / num_rows  # we use the same samples here


@functools.partial(jax.jit, static_argnums=(3, 5, 6, 7, 8, 9, 10, 11, 12))
def nabla_phi_analytical(  # pylint: disable=invalid-name
    Phi,
    Psi,
    key,
    optim,
    opt_state,
    estimator,
    alpha,
    use_l2_reg,
    reg_coeff,
    use_penalty,
    j,
    num_rows=1):

  r"""Computes unbiased estimate of 2 * (\Phi W^*_\Phi - \Psi)(W^*_\Phi)^T.

  Args:
    Phi: S times d array
    Psi: S times T array
    key: prng key
    optim: optax optimizer
    opt_state: optimizer initialization
    estimator: str: russian_roulette, lissa, hat_w
    alpha: float: used to nornalize covariance term
    use_l2_reg: bool: whether to use l2 reg
    reg_coeff: float: coeff for reg
    use_penalty: bool: whether to add "lambda * Id" term to features
    j: int: num of samples for lissa
    num_rows: int: number of rows used in estimators

  Returns:
    array S times d
  """
  key, subkey = jax.random.split(key)
  _, T = Psi.shape  # pylint: disable=invalid-name
  task = jax.random.randint(subkey, (1,), 0, T)
  key, subkey = jax.random.split(key)
  Phi_estim = matrix_estimator(Phi, num_rows, subkey)  # pylint: disable=invalid-name
  Psi_estim = matrix_estimator(Psi[:, task], num_rows, subkey)  # pylint: disable=invalid-name
  key, subkey = jax.random.split(key)

  least_square_estim_1 = least_square_estimator(Phi, Psi[:, task], num_rows, j,
                                                subkey, estimator, alpha,
                                                use_penalty, reg_coeff)

  key, subkey = jax.random.split(key)
  least_square_estim_2 = least_square_estimator(Phi, Psi[:, task], num_rows, j,
                                                subkey, estimator, alpha,
                                                use_penalty, reg_coeff)
  grads = (Phi_estim @ least_square_estim_1 -
           Psi_estim) @ least_square_estim_2.T
  if use_l2_reg:
    grads += Phi_estim * reg_coeff
  updates, opt_state = optim.update(grads, opt_state, Phi)
  beta = 1

  return optax.apply_updates(Phi, beta * updates), opt_state, grads
