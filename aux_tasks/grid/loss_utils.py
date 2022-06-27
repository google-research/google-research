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

"""Loss Utilities."""
import functools
import itertools
from typing import Optional, Sequence

import chex
import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=('normalization'))
def cov_inv_estimate(
    phis,
    *,
    alpha,
    normalization = 'max_feature_norm'):
  """Covariance inverse estimate."""
  # pylint: disable=invalid-name
  _, embedding_dim = phis.shape
  I = jnp.eye(embedding_dim)

  if normalization == 'top_singular_value':
    norm = jnp.linalg.norm(phis.T @ phis, ord=2)
    alpha = alpha * 1.0 / norm
  elif normalization == 'max_feature_norm':
    norm = 2 * jnp.amax(jnp.linalg.norm(phis, axis=1, ord=2))
    alpha = alpha * 1.0 / norm

  def _neumann_series(carry, phi):
    A_j = alpha * I
    A_j += (I - alpha * jnp.einsum('i,j->ij', phi, phi)) @ carry
    return A_j, None

  # pylint: enable=invalid-name
  cov_inv, _ = jax.lax.scan(_neumann_series, alpha * I, phis)
  return cov_inv


def weight_estimate(phis_for_cov, phis,
                    psis, *, alpha):
  cov_inv = cov_inv_estimate(phis_for_cov, alpha=alpha)
  return cov_inv @ phis.T @ psis / phis.shape[0]


@functools.partial(jax.custom_vjp, nondiff_argnums=(8,))
def implicit_least_squares(phis, phis_for_wi1,
                           phis_for_wi2, phis_for_cov1,
                           phis_for_cov2, psis,
                           psis_for_wi1, psis_for_wi2,
                           alpha):
  """Implicit least squares objective."""
  # Make sure all the shapes agree
  chex.assert_equal_shape([phis_for_cov1, phis_for_cov2])
  chex.assert_equal_shape([phis_for_wi1, phis_for_wi2])
  chex.assert_equal_shape([psis_for_wi1, psis_for_wi2])
  chex.assert_equal_shape_prefix([phis, psis], 1)
  chex.assert_scalar(alpha)
  chex.assert_rank([
      phis,
      phis_for_cov1,
      phis_for_cov2,
      phis_for_wi1,
      phis_for_wi2,
      psis,
      psis_for_wi1,
      psis_for_wi2,
  ], 2)

  # Get w_1 estimate on the forward pass when computing the loss
  w = weight_estimate(phis_for_cov1, phis_for_wi1, psis_for_wi1, alpha=alpha)
  # w = weight_estimate(phis_for_cov1, phis, psis, alpha=alpha)
  # Predict using w_1
  predictions = phis @ w
  # Least-squares cost
  cost = predictions - psis
  # MSE Loss
  mse = 0.5 * jnp.mean(cost**2)

  return mse


def implicit_least_squares_fwd(
    phis, phis_for_wi1, phis_for_wi2,
    phis_for_cov1, phis_for_cov2, psis,
    psis_for_wi1, psis_for_wi2,
    alpha):
  """Forward pass for implicit least squares objective."""
  chex.assert_equal_shape([phis_for_cov1, phis_for_cov2])
  chex.assert_equal_shape([phis_for_wi1, phis_for_wi2])
  chex.assert_equal_shape([psis_for_wi1, psis_for_wi2])
  chex.assert_equal_shape_prefix([phis, psis], 1)
  chex.assert_scalar(alpha)
  chex.assert_rank([
      phis,
      phis_for_cov1,
      phis_for_cov2,
      phis_for_wi1,
      phis_for_wi2,
      psis,
      psis_for_wi1,
      psis_for_wi2,
  ], 2)

  # Get w_1 estimate on the forward pass when computing the loss
  w = weight_estimate(phis_for_cov1, phis_for_wi1, psis_for_wi1, alpha=alpha)
  # w = weight_estimate(phis_for_cov1, phis, psis, alpha=alpha)

  # Predict using w_1
  predictions = phis @ w
  # Least-squares cost
  cost = predictions - psis
  # MSE Loss
  mse = implicit_least_squares(
      phis,
      phis_for_wi1,
      phis_for_wi2,
      phis_for_cov1,
      phis_for_cov2,
      psis,
      psis_for_wi1,
      psis_for_wi2,
      alpha=alpha)

  # Return appropriate residuals so we can compute w_2 on backward pass
  return mse, (cost, phis_for_cov2, phis_for_wi2, psis_for_wi2)


def implicit_least_squares_bwd(alpha, residuals,
                               g):
  """Backward pass for implicit least squares objective."""
  # Get residuals
  cost, phis_for_cov2, phis_for_wi2, psis_for_wi2 = residuals
  # Compute w_2
  w_prime = weight_estimate(
      phis_for_cov2, phis_for_wi2, psis_for_wi2, alpha=alpha)
  # w_prime = weight_estimate(phis_for_cov2, phis, psis, alpha=alpha)
  # Grad is cost @ w_2.T
  phi_grads = g * cost @ w_prime.T

  # There's no grads associated with any other inputs except for Phi(s)
  return phi_grads, None, None, None, None, None, None, None

implicit_least_squares.defvjp(implicit_least_squares_fwd,
                              implicit_least_squares_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def naive_implicit_least_squares(phis, psis, alpha):
  """Naive implicit least squares objective."""
  # Get w_1 estimate on the forward pass when computing the loss
  w = weight_estimate(phis, phis, psis, alpha=alpha)
  # Predict using w_1
  predictions = phis @ w
  # Least-squares cost
  cost = predictions - psis
  # MSE Loss
  mse = 0.5 * jnp.mean(cost**2)

  return mse


def naive_implicit_least_squares_fwd(
    phis, psis,
    alpha):
  """Forward pass for naive implicit least squares objective."""
  # Get w_1 estimate on the forward pass when computing the loss
  w = weight_estimate(phis, phis, psis, alpha=alpha)
  # Predict using w_1
  predictions = phis @ w
  # Least-squares cost
  cost = predictions - psis
  # MSE Loss
  mse = naive_implicit_least_squares(phis, psis, alpha=alpha)

  # Return appropriate residuals so we can compute w_2 on backward pass
  return mse, (cost, w)


def naive_implicit_least_squares_bwd(_, residuals,
                                     g):
  """Backward pass for naive implicit least squares objective."""
  cost, w = residuals
  grad = g * cost @ w.T
  return grad, None

naive_implicit_least_squares.defvjp(naive_implicit_least_squares_fwd,
                                    naive_implicit_least_squares_bwd)


# Helper function to split in chunks, e.g., split(x, [5, 2])
# would split x into two arrays of size 5 and 2.
def split_in_chunks(x,
                    chunks,
                    *,
                    axis=0):
  split_points = list(itertools.accumulate(chunks))
  splits = jnp.split(x, split_points, axis=axis)
  return splits[:len(chunks)]


def top_d_singular_vectors(x, d):
  """Get top-d singular vectors."""
  u, _, _ = jnp.linalg.svd(x, full_matrices=False)
  return u[:, :d]


def grassman_distance(y1, y2):
  """Grassman distance between subspaces spanned by Y1 and Y2."""
  q1, _ = jnp.linalg.qr(y1)
  q2, _ = jnp.linalg.qr(y2)

  _, sigma, _ = jnp.linalg.svd(q1.T @ q2)
  sigma = jnp.round(sigma, decimals=6)
  return jnp.linalg.norm(jnp.arccos(sigma))
