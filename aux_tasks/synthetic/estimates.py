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

import jax.numpy as jnp
import numpy as np

from aux_tasks.synthetic import utils


def naive_inverse_covariance_matrix(Phi, key, covariance_batch_size):  # pylint: disable=invalid-name
  """Estimates the covariance matrix naively.

  We want to return a covariance matrix whose norm is "equivalent" to a single
  data point, so we multiply by the covariance_batch_size.

  Args:
    Phi: feature matrix.
    key: jax rng key.
    covariance_batch_size: how many states to sample to estimate.

  Returns:
    array: naive inverse covariance
  """
  num_states, d = Phi.shape

  states, key = utils.draw_states(num_states, covariance_batch_size, key)
  matrix_estimate = jnp.linalg.solve(Phi[states, :].T @ Phi[states, :],
                                     jnp.eye(d))

  return matrix_estimate * covariance_batch_size, key


def lissa_inverse_covariance_matrix(  # pylint: disable=invalid-name
    Phi,
    key,
    lissa_iterations,
    lissa_kappa,
    feature_norm=None):
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

  Returns:
    array: lissa estimate of inverse covariance
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

  states, key = utils.draw_states(num_states, lissa_iterations, key)  # pylint: disable=invalid-name
  sampled_Phis = Phi[states, :]  # pylint: disable=invalid-name

  for t in range(lissa_iterations):
    # Construct a rank-one multiplier.
    multiplier = I - kappa * sampled_Phis[t, :] @ sampled_Phis[t, :].T
    # Add one more term to the LISSA sequence.
    estimate = kappa * I + multiplier @ estimate

  return estimate, key
