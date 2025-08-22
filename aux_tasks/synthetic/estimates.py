# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np

from aux_tasks.synthetic import utils


def naive_inverse_covariance_matrix(
    compute_phi,
    sample_states,
    key,
    d,
    covariance_batch_size):
  """Estimates the covariance matrix naively.

  We want to return a covariance matrix whose norm is "equivalent" to a single
  data point, so we multiply by the covariance_batch_size.

  Args:
    compute_phi: A function that takes states and returns a matrix of phis.
    sample_states: A function that takes an rng key and a number of states
      to sample, and returns the sampled states.
    key: jax rng key.
    d: The number of features in phi.
    covariance_batch_size: how many states to sample to estimate.

  Returns:
    A tuple containing (the naive inverse covariance matrix, updated rng).
  """
  states, key = sample_states(key, covariance_batch_size)
  phi = compute_phi(states)
  matrix_estimate = jnp.linalg.solve(phi.T @ phi, jnp.eye(d))

  return matrix_estimate * covariance_batch_size, key  # pytype: disable=bad-return-type  # jax-ndarray


def lissa_inverse_covariance_matrix(
    compute_phi,
    sample_states,
    key,
    d,
    lissa_iterations,
    lissa_kappa,
    feature_norm = None):
  """Estimates the covariance matrix by LISSA.

  By default this method returns a covariance matrix whose norm is equivalent
  to a single data point, no need to multiply.

  Args:
    compute_phi: A function that takes states and returns a matrix of phis.
    sample_states: A function that takes an rng key and a number of states
      to sample, and returns the sampled states.
    key: jax rng key.
    d: The number of features in phi.
    lissa_iterations: how many states to sample to estimate.
    lissa_kappa: The lissa parameter (gets further divided by feature norm
      squared).
    feature_norm: The squared norm of the largest feature vector (possibly
      estimated). If None, computed directly from the feature matrix Phi.

  Returns:
    A tuple containing:
      (the lissa esimate of the inverse covariance matrix, updated rng).
  """
  states, key = sample_states(key, lissa_iterations)
  sampled_phis = compute_phi(states)

  if feature_norm is None:
    feature_norm = utils.compute_max_feature_norm(sampled_phis)

  I = np.eye(d)  # pylint: disable=invalid-name
  kappa = lissa_kappa / feature_norm
  estimate = kappa * I

  for t in range(lissa_iterations):
    phi = sampled_phis[t, :]

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
