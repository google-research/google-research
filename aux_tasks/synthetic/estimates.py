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

from typing import Optional

import jax.numpy as jnp
import numpy as np

from aux_tasks.synthetic import utils


def naive_inverse_covariance_matrix(
    Phi,  # pylint: disable=invalid-name
    key,
    covariance_batch_size,
    *,
    sample_with_replacement = True):
  """Estimates the covariance matrix naively.

  We want to return a covariance matrix whose norm is "equivalent" to a single
  data point, so we multiply by the covariance_batch_size.

  Args:
    Phi: feature matrix.
    key: jax rng key.
    covariance_batch_size: how many states to sample to estimate.
    sample_with_replacement: whether to sample with replacement.

  Returns:
    A tuple of (naive inverse covariance matrix, rng key).
  """
  num_states, d = Phi.shape

  states, key = utils.draw_states(
      num_states,
      covariance_batch_size,
      key,
      replacement=sample_with_replacement)
  matrix_estimate = jnp.linalg.solve(Phi[states, :].T @ Phi[states, :],
                                     jnp.eye(d))

  return matrix_estimate * covariance_batch_size, key


def lissa_inverse_covariance_matrix(
    Phi,  # pylint: disable=invalid-name
    key,
    lissa_iterations,
    lissa_kappa,
    feature_norm = None,
    *,
    sample_with_replacement = True):
  """Estimates the covariance matrix by LISSA.

  By default this method returns a covariance matrix whose norm is equivalent
  to a single data point, no need to multiply.

  Args:
    Phi: feature matrix.
    key: jax rng key.
    lissa_iterations: how many states to sample to estimate.
    lissa_kappa: The lissa parameter (gets further divided by feature norm
      squared).
    feature_norm: The squared norm of the largest feature vector (possibly
      estimated). If None, computed directly from the feature matrix Phi.
    sample_with_replacement: Whether to sample with replacement.

  Returns:
    A tuple of (lissa estimate of inverse covariance matrix, rng key).
  """
  num_states, d = Phi.shape

  # Determine the largest feaure vector norm, square it.
  # TODO(bellemare): This uses the whole feature matrix; compare with just
  # batch.
  if feature_norm is None:
    feature_norm = utils.compute_max_feature_norm(Phi)

  I = np.eye(d)  # pylint: disable=invalid-name
  kappa = lissa_kappa / feature_norm
  estimate = kappa * I

  states, key = utils.draw_states(
      num_states, lissa_iterations, key, replacement=sample_with_replacement)
  sampled_Phis = Phi[states, :]  # pylint: disable=invalid-name

  for t in range(lissa_iterations):
    phi = sampled_Phis[t, :]

    # Here we make some optimizations. The equation for updating the lissa
    # estimate is:
    #     E_{t} = kI + (I - k * outer(phi, phi)) @ E_{t-1}
    # However, the matrix multiply is an O(n^3) operation. We can do better.
    # Rearrange as follows:
    #    E_{t} = k * (I - phi @ (phi^T @ E_{t-1})) + E_{t-1}
    # Each term has O(n^2) time complexity.
    # The trick is to replace the outer product followed by a matrix multiply
    # (outer(phi, phi) @ E_{t - 1}) with two vector-matrix multiplies
    # (phi @ (phi^T @ E_{t-1})).

    # Here, einsum is just giving us vector-matrix multiplication
    # syntactic sugar without needing reshapes. 🍬
    outer_prod_e = jnp.outer(phi, jnp.einsum('i,ij->j', phi, estimate))
    estimate = kappa * (I - outer_prod_e) + estimate

  return estimate, key
