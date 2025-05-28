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

"""Sampling functions needed to implement privacy mechanisms."""

import numpy as np
from scipy import optimize


def acg_potential(omega, x):
  """Computes the (unnormalized) angular central gaussian (ACG) density at x.

  The ACG distribution is obtained by sampling a gaussian random vector with
  covariance Omega and normalizing it to the unit sphere. The density is
  1/(x.T omega *x)**(-d/2).

  Args:
    omega: Covariance matrix (d x d)
    x: d-dimension vector.

  Returns:
    Density evaluated at x.
  """
  d = omega.shape[0]
  return (x.T * omega * x)**(-d / 2)


def sample_acg(diag, eigen_vec):
  """Draws a sample from the angular central gaussian.

  Distribution is parametrized by Omega = eigen_vec* np.diag(diag) *eigen_vec.T.
  Args:
    diag: np.ndarray of eigenvalues of the matrix Omega.
    eigen_vec: Eigenvectors of the matrix Omega.

  Returns:
    Vector sampled from ACG distribution.
  """
  d = eigen_vec.shape[0]
  transform_matrix = eigen_vec * np.diag(np.power(diag, -1 / 2))
  v = transform_matrix * np.random.randn(d, 1)
  return v / np.linalg.norm(v)


def shift_eigvals(mat):
  """Returns mat - lambda_min(mat)*I.

  Args:
    mat: A matrix.

  Returns:
    mat - lambda_min(mat) * I
  """
  mat = np.copy(mat)
  mat -= np.eye(mat.shape[0]) * np.min(np.linalg.eigvalsh(mat))
  return np.matrix(mat)


def bingham_potential(bingham_matrix, x):
  """Computes the (unnormalized) bingham density at x.

  Args:
    bingham_matrix: Matrix defining the Bingham distribution
    x: Vector to evaluate

  Returns:
    Unnormalized Bingham density at x.
  """
  return np.exp(-x.T * bingham_matrix * x)


def sample_bingham(bingham_matrix, b):
  """Draws a sample from the Bingham distribution with parameter bingham_matrix.

  Uses rejection sampling with proposal distribution given by the ACG
  distribution with parameter I + 2*bingham_matrix/b.

  Assumes that the minimum eigenvalue of bingham_matrix is 0.
  Args:
    bingham_matrix: Matrix defining the bingham density.
    b: Parameter used to define the covariance matrix of the ACG distribution.

  Returns:
    A vector sampled from the Bingham distribution.
  """
  d = bingham_matrix.shape[0]
  omega = np.eye(d) + 2 * bingham_matrix / b
  acg_matrix = np.exp(-(d - b) / 2) * (d / b)**(d / 2)
  diag, sing_vectors = np.linalg.eig(omega)
  while True:
    acg_sample = sample_acg(diag, sing_vectors)
    unif = np.random.rand()
    ratio = (
        bingham_potential(bingham_matrix, acg_sample) / acg_matrix /
        acg_potential(omega, acg_sample))
    if unif < ratio:
      break
  return acg_sample


# Tools to find the optimal value b in the above sampler to minimize the
# number of samples required by rejection sampling.


def b_objective(b, lambdas):
  """The optimal b parameter is the root of this function.

  Args:
   b: A scalar
   lambdas: Eigenvalues of a matrix.

  Returns:
    Objective function evaluated at b.
  """
  return np.sum([1. / (b + 2 * l) for l in lambdas]) - 1


def b_objective_derivative(b, lambdas):
  """The derivative of b_objective.

  Args:
    b: A scalar
    lambdas: Eigenvalues of a matrix

  Returns:
    Derivate evaluated at b.
  """
  return np.sum([-1. / (b + 2 * l)**2 for l in lambdas])


def optimal_b(bingham_matrix):
  """Uses Newton's method to find the root of b_objective.

  Args:
    bingham_matrix: Parameter defining the Bingham distribution

  Returns:
    Vector as np.ndarray optimizing b_objective
  """
  lambdas = np.linalg.eigvalsh(bingham_matrix)
  return optimize.newton(
      b_objective, 1, b_objective_derivative, args=(lambdas,))


def sample_n_bingham_opt(bingham_matrix, num_samples):
  """Draws num_samples from the Bingham distribution.

  This function does not assume that the minimum eigenvalue of bingham_matrix is
  zero, and computes the optimal parameter b for the rejection sampling
  procedure.
  Args:
    bingham_matrix:  Parameter defining the Bingham distribution
    num_samples: Num of samples desired.

  Returns:
    Samples from the Bingham distribution.
  """
  d = bingham_matrix.shape[0]
  bingham_matrix = shift_eigvals(bingham_matrix)
  b = optimal_b(bingham_matrix)
  result = np.empty((d, num_samples))
  for i in range(num_samples):
    result[:, i] = sample_bingham(bingham_matrix, b).T
  return np.asmatrix(result)


def sample_bingham_opt(bingham_matrix):
  """Same as sample_n_bingham, except outputs only one sample."""
  return sample_n_bingham_opt(bingham_matrix, 1)
