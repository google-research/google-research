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

"""Differentiable top-k using sinkhorn and custom backward.

Implements the paper https://arxiv.org/abs/2002.06504.
"""

import chex
import jax
import jax.numpy as jnp


def smooth_min(x, epsilon, axis):
  return -epsilon * jax.scipy.special.logsumexp(
      -x / epsilon, axis=axis, keepdims=True)


def sinkhorn(costs, mu, nu, epsilon, num_iterations):
  """Sinkhorn algorithm."""
  n, m = costs.shape
  chex.assert_shape(mu, (n,))
  chex.assert_shape(nu, (m,))

  epsilon_log_mu = epsilon * jnp.log(mu)
  epsilon_log_nu = epsilon * jnp.log(nu)

  # Initial value of the for loop iteration. f is not used and is a placeholder
  # that need to be of the right shape and dtype.
  f = jnp.zeros((n, 1))
  g = jnp.zeros((1, m))
  f_and_g = (f, g)

  def for_step(i, f_and_g):
    del i
    f, g = f_and_g
    f = smooth_min(costs - g, epsilon, axis=1) + epsilon_log_mu[:, jnp.newaxis]
    g = smooth_min(costs - f, epsilon, axis=0) + epsilon_log_nu[jnp.newaxis, :]
    return (f, g)

  f, g = jax.lax.fori_loop(0, num_iterations, for_step, f_and_g)

  transport_plan = jnp.exp((-costs + f + g) / epsilon)
  return transport_plan


def smooth_top_k(x, k, epsilon, num_iterations):
  """Smooth Top K."""
  n = x.shape[0]
  y = jnp.array([0., 1.])
  mu = jnp.ones(n) / n
  nu = jnp.array([k / n, (n - k) / n])

  # shape: n, 2
  costs = (x[:, jnp.newaxis] - y[jnp.newaxis, :])**2

  transport_plan = sinkhorn(costs, mu, nu, epsilon, num_iterations)
  assignement = n * transport_plan[:, 1]
  return assignement


def smooth_sorted_top_k(x, k, epsilon, num_iterations):
  """Smooth Sorted Top K."""
  n = x.shape[0]
  y = jnp.arange(k + 1)
  mu = jnp.ones(n) / n
  nu = jnp.array([(n - k) / n] + [1. / n] * k)

  # shape: n, k + 1
  costs = (x[:, jnp.newaxis] - y[jnp.newaxis, :])**2

  transport_plan = sinkhorn(costs, mu, nu, epsilon, num_iterations)
  assignement = n * transport_plan[:, 1:]
  return assignement


def sinkhorn_forward(costs, mu, nu, epsilon, num_iterations):
  """Forward for Sinkhorn."""
  gamma = sinkhorn(costs, mu, nu, epsilon, num_iterations)
  saved_for_bwd = (mu, nu, epsilon, gamma)
  return gamma, saved_for_bwd


def sinkhorn_backward(saved_for_bwd, result_cotangent):
  """Backwords for Sinkhorn."""
  mu, nu, epsilon, gamma = saved_for_bwd
  grad_output_gamma = result_cotangent
  del saved_for_bwd, result_cotangent

  n, k_ = gamma.shape
  k = k_ - 1
  nu_ = nu[1:]
  gamma_ = gamma[:, 1:]

  inv_diag_mu = jnp.diag(1. / mu)
  diag_nu_ = jnp.diag(nu_)

  kappa = diag_nu_ - gamma_.transpose() @ inv_diag_mu @ gamma_
  inv_kappa = jax.scipy.linalg.inv(kappa)
  del kappa

  h1 = inv_diag_mu \
      + inv_diag_mu @ gamma_ @ inv_kappa @ gamma_.transpose() @ inv_diag_mu
  h2 = -inv_diag_mu @ gamma_ @ inv_kappa
  h3 = h2.transpose()
  h4 = inv_kappa
  del gamma_, inv_kappa, inv_diag_mu, diag_nu_

  padded_h2 = jnp.concatenate([jnp.zeros((n, 1)), h2], axis=1)
  padded_h4 = jnp.concatenate([jnp.zeros((k, 1)), h4], axis=1)
  del h2, h4

  dxidc = jnp.einsum("hi,ij->hij", h1, gamma) + \
      jnp.einsum("hj,ij->hij", padded_h2, gamma)
  dbdc = jnp.einsum("li,ij->lij", h3, gamma) + \
      jnp.einsum("lj,ij->lij", padded_h4, gamma)
  del h1, padded_h2, h3, padded_h4

  padded_dbdc = jnp.concatenate([jnp.zeros((1, n, k + 1)), dbdc], axis=0)
  del dbdc

  dldc = 1 / epsilon * (
      -jnp.einsum("ij,ij->ij", grad_output_gamma, gamma) +
      jnp.einsum("hl,hl,hij->ij", grad_output_gamma, gamma, dxidc) +
      jnp.einsum("hl,hl,lij->ij", grad_output_gamma, gamma, padded_dbdc))

  return (dldc, None, None, None, None)


# create differentiable Monte Carlo estimate with custom backward
sinkhord_differentiable = jax.custom_vjp(sinkhorn)
sinkhord_differentiable.defvjp(sinkhorn_forward, sinkhorn_backward)


def differentiable_smooth_sorted_top_k(x, k, epsilon, num_iterations):
  """Differentiable smooth sorted top k."""
  n = x.shape[0]
  y = jnp.arange(k + 1)
  mu = jnp.ones(n) / n
  nu = jnp.array([(n - k) / n] + [1. / n] * k)

  # shape: n, k + 1
  costs = (x[:, jnp.newaxis] - y[jnp.newaxis, :]) ** 2

  transport_plan = sinkhord_differentiable(costs, mu, nu, epsilon,
                                           num_iterations)
  assignement = n * transport_plan[:, 1:]
  return assignement
