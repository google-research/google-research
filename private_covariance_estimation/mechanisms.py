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

"""Implementation of DP mechanisms for covariance matrices."""

import numpy as np
from scipy import optimize
from scipy import stats

from private_covariance_estimation import sampling_utils as samp


# Functions to generate a valid sigma for the Gaussian mechanism.
# Implementation based on:
# [1]Balle, B., & Wang, Y. X. (2018). Improving the gaussian mechanism for
# differential privacy: Analytical calibration and optimal denoising.
# arXiv preprint arXiv:1805.06530.
def phi(x):
  """Cumulative distribution function of a Gaussian.

  Args:
     x: A value

  Returns:
     CDF of a standard gaussian.
  """
  return stats.norm(0, 1).cdf(x)


def b_plus(v, epsilon):
  """Implementation helper.

  Implements BPlus from [1] above.

  Args:
    v : A value
    epsilon: The epsilon of differential privacy

  Returns:
    Value of B_plus from [1] above.
  """
  prob_1 = phi(np.sqrt(v * epsilon))
  prob_2 = phi(-np.sqrt(epsilon * (v + 2)))
  return prob_1 - np.exp(epsilon) * prob_2


def b_minus(v, epsilon):
  """Implementation helper.

  Implements Bminus from [1] above.

  Args:
    v : A value
    epsilon: The epsilon of differential privacy

  Returns:
        Value of Bminus from [1] above.
  """
  prob_1 = phi(-np.sqrt(v * epsilon))
  prob_2 = phi(-np.sqrt(epsilon * (v + 2)))
  return prob_1 - np.exp(epsilon) * prob_2


#  Useful constant for computing the inverse  below using a bracket method.
MAX_BRACKET = 10000


def compute_inverse(b_func, epsilon, delta):
  """Returns the solution to B(v, epsilon) = delta.

  Args:
    b_func: A function (b_plus or b_minus)
    epsilon: The epsilon of differential privacy
    delta: The delta of differential privacy

  Returns:
    Solution to b_func(v, epsilon) = delta
  """
  return optimize.root_scalar(
      lambda v: b_func(v, epsilon) - delta, bracket=(0, MAX_BRACKET)).root


def find_alpha(epsilon, delta):
  """Helper function.

  Returns the value of alpha from the implementation in [1] above.

  Args:
     epsilon: The epsilon of differential privacy
     delta: The delta of differential privacy

  Returns:
     Alpha from the implementation in [1] above.
  """
  delta_0 = phi(0) - np.exp(epsilon) * phi(-np.sqrt(2 * epsilon))
  b_func = b_minus
  if delta_0 < delta:
    b_func = b_plus
  v = compute_inverse(b_func, epsilon, delta)
  alpha = np.sqrt(1 + v / 2) + np.sqrt(v / 2)
  if delta_0 < delta:
    alpha = np.sqrt(1 + v / 2) - np.sqrt(v / 2)
  return alpha


def find_sigma(epsilon, delta, l2_sensitivity):
  """Returns the best sigma for the Gaussian mechanism.

  Args:
    epsilon: The epsilon of differential privacy
    delta: The delta of differential privacy
    l2_sensitivity: The L2 sensitivity

  Returns:
    Best choice of sigma for the Gaussian mechanism.
  """
  alpha = find_alpha(epsilon, delta)
  return alpha * l2_sensitivity / np.sqrt(2 * epsilon)


def gaussian_mechanism(cov, epsilon, delta, truncation_level=2**32):
  """Implementation of the Gaussian mechanism.

  Implements the mechanism introduced in
  http://kunaltalwar.org/papers/PrivatePCA.pdf. Given a covariance matrix cov
  returns cov + M where M is a symmetric matrix with Gaussian noise. The
  implementation assumes all data points used to generate the covariance matrix
  cov have norm less than 1.

  Args:
    cov: The true covariance matrix. This a numpy.Matrix object
    epsilon: The epsilon of (epsilon, delta) differential privacy
    delta: The delta of (epsilon, delta) differential privacy
    truncation_level: The eigenvalues of the output matrix will be rescaled so
      the max eigenvalue matches truncation_level. Useful if there is an a
      priori bound on the eigenvalues of the covariance matrix.

  Returns:
    An (epsilon, delta) differentially private estimate of C.

  """
  d = cov.shape[0]
  # Compute the noise matrix
  sigma = find_sigma(epsilon, delta, 1.0)
  empty = np.empty((d, d))
  rows, cols = np.triu_indices_from(empty, 0)
  empty[rows, cols] = np.random.randn(len(rows)) * sigma
  empty[cols, rows] = empty[rows, cols]
  # Compute the noised covariance matrix
  hat_cov = cov + empty
  # Force hat_cov to be PSD
  ev, evec = np.linalg.eig(hat_cov)
  evec = np.matrix(evec)
  ev = np.maximum(ev, 0)
  hat_cov = evec * np.diag(ev) * evec.T
  # also compute the maximum eigenvalue of hat_cov
  hat_cov_max_eigval = np.max(ev)
  # If maximum eigenvalue of hat_cov is larger than truncation_level, then
  # rescale so it is truncation_level.
  if hat_cov_max_eigval > truncation_level:
    hat_cov *= truncation_level / hat_cov_max_eigval
  return hat_cov


# Laplace Mechanism


def laplace_mechanism(cov, epsilon, truncation_level=2**32):
  """Runs the Laplace mechanism on matrix cov.

  Returns an estimate of the covariance matrix C while preserving
  (epsilon, 0)-differential privacy using the Laplace mechanism.
  The implementation assumes all data points used to generate the covariance
  matrix cov have norm less than 1.

  Args:
    cov : Numpy matrix
    epsilon: From (epsilon, 0) differential privacy
    truncation_level: The eigenvalues of the output matrix will be rescaled so
    the max eigenvalue matches truncation_level. Useful if there is an a priori
    bound on the eigenvalues of the covariance matrix.

  Returns:
    cov + M where M is symmetric with entries having a Laplace
      distribution.
  """
  d = cov.shape[0]
  # Compute the noise matrix
  noise_scale = 2 * d / epsilon
  empty = np.empty((d, d))
  rows, cols = np.triu_indices_from(empty, 0)
  empty[rows, cols] = np.random.laplace(0, noise_scale, len(rows))
  empty[cols, rows] = empty[rows, cols]
  # Compute the noised covariance matrix
  hat_cov = cov + empty
  # Force hat_cov to be PSD
  ev, evec = np.linalg.eig(hat_cov)
  evec = np.matrix(evec)
  ev = np.maximum(ev, 0)
  hat_cov = evec * np.diag(ev) * evec.T
  # also compute the maximum eigenvalue of hat_cov
  hat_cov_max_eigval = np.max(ev)

  # If maximum eigenvalue of hat_cov is larger than n, then rescale so it is n
  if hat_cov_max_eigval > truncation_level:
    hat_cov *= truncation_level / hat_cov_max_eigval
  return hat_cov


# Iterative sampling mechanism


def split_privacy_between_eigenvalues_and_vectors_uniformly(n, d, epsilon):
  """Split privacy budget between eigenvalues and eigenvectors equally."""
  print("Ignoring parameters in budget splitting n:%d  d:%d" % (n, d))
  return (epsilon / 2.0, epsilon / 2.0)


def get_vector_privacy_parameters_uniformly(eigvals, epsilon):
  d = len(eigvals)
  return np.ones(d) * epsilon / float(d)


# Utility bound as a function of dimension and number of points.
def utility_bound(t, d, n, epsilon):
  dim_sum = np.sum(np.sqrt(d - np.arange(0, d, 1)))
  return np.sqrt(d) / t + dim_sum * np.sqrt(n) / np.sqrt(epsilon - t)


def split_privacy_between_eigenvalues_and_vectors(n, d, epsilon):
  epsilon_val = optimize.minimize_scalar(
      lambda x: utility_bound(x, d, n, epsilon),
      method="bounded",
      bounds=[epsilon / d**2, epsilon - epsilon / d**2])
  epsilon_val = epsilon_val.x
  epsilon_matrix = epsilon - epsilon_val
  return (epsilon_val, epsilon_matrix)


def get_vector_privacy_parameters(eigvals, epsilon):
  d = len(eigvals)
  weights = np.sqrt(eigvals * reversed(np.array(range(d))))
  weights = weights / np.sum(weights)
  return weights * epsilon


def orthogonal_complement_householder(v):
  """Returns the orthogonal complement (as a matrix) of a vector v.

  Method uses Householder reflections.
  Args:
    v: Vector to compute the orthogonal complement from.

  Returns:
    Orthogonal complement as matrix of vector v.
  """
  d = v.shape[0]
  e1 = np.asmatrix(np.zeros((d, 1)))
  e1[0] = 1
  u = v - e1
  u = u / np.linalg.norm(u)
  proj = np.eye(d) - 2 * u * u.T
  return proj[:, 1:]


def orthogonal_complement(v):
  return orthogonal_complement_householder(v)


def iterative_mechanism(
    cov,
    n,
    epsilon,
    debug=False,
    privacy_splitting=split_privacy_between_eigenvalues_and_vectors,
    vector_privacy_splitting=get_vector_privacy_parameters):
  """Implementation of the iterative sampling mechanism from [1].

  [1] Kamin et al. Differentially Private Covariance Estimation.
  The implementation assumes all data points used to generate the covariance
  matrix have norm less than 1.
  Args:
      cov: The true covariance matrix
      n: The number of data points used to generate the covariance matrix
      epsilon: From (epsilon, 0) differential privacy
      debug: Whether to print debug information
      privacy_splitting: Function that decides how to split privacy budget
        between eigenvalues and eigenvectors. This function takes as argument
        the dimension d of the covariance matrix, the number of data points n
        and privacy paramter epsilon. See for instance
        split_privacy_between_eigenvalues_and_vectors
      vector_privacy_splitting: Function that decides how to split a privacy
        budget among all eigenvectors of the covariance matrix. Takes as input
        the (private) eigenvalues of the matrix cov as well as epsilon.

  Returns:
      Private covariance matrix.
  """
  d = cov.shape[0]
  # Estimate the eigenvalues of C using the Laplace mechanism and a 1/d fraction
  # of the privacy budget
  epsilon_eigvals, epsilon_matrix = privacy_splitting(n, d, epsilon)
  eigvals = np.linalg.eigvalsh(cov)
  eigvals += np.random.laplace(0, 1.0 / epsilon_eigvals, d)
  if debug:
    print("Eigvals epsilon: %f matrix epsilon %f" %
          (epsilon_eigvals, epsilon_matrix))
    print("Eigen-values:")
    print(eigvals)
  # Note: We sort the eigenvalues after adding noise to ensure that our
  #       estimated eigenvalues are in decreasing order.
  eigvals = reversed(np.sort(eigvals))
  eigvals = np.maximum(0, np.minimum(eigvals, n))  # clamp eigenvalues to [0,n]

  # Calculate the privacy budget for each round
  epsilon_per_round = vector_privacy_splitting(eigvals, epsilon_matrix)
  # Iteratively sample the eigenvectors

  # The reconstruction matrix maps from the lower dimensional subspaces we are
  # sampling in back into the original space. For the first round, it is just
  # the identity matrix
  reconstruction_matrix = np.eye(d)
  # eigvecs is a matrix whose columns are the estimated eigenvectors.
  eigvecs = np.zeros((d, d))

  for i in range(d):
    # If the eigenvalue is too small skip sampling that eigenvector. Since the
    # eigenvalues are in decreasing order, we can skip the rest.
    if eigvals[i] < 1e-10:
      break
    # Sample an approximate eigenvector from the Bingham distribution capturing
    # approximately maximal variance from the covariance matrix C. Note: in
    # round i, the covariance matrix C is a (d-i) by (d-i) matrix describing
    # the variance of the data in the subspace orthogonal to the vectors sampled
    # so far.
    eigvec = samp.sample_bingham_opt(-2 * epsilon_per_round[i] * cov)
    # Reconstruct that vector in the original d dimensional space and add it to
    # the eigenvector matrix
    eigvecs[:, i] = np.squeeze(reconstruction_matrix * eigvec)
    # Compute a basis for the orthogonal subspace of eigvec.
    complement = orthogonal_complement(eigvec)
    # Update the covariance matrix to be the remaining variance in that
    # orthogonal subspace.
    cov = complement.T * cov * complement
    # Update the reconstruction matrix for mapping from the subspace back to the
    # original space
    reconstruction_matrix = reconstruction_matrix * complement

  return eigvecs * np.asmatrix(np.diag(eigvals)) * eigvecs.T


def iterative_mechanism_talwar(cov, n, epsilon, debug=False):
  """Implements the iterative covariance estimation mechanism of [2].

  [2] Michael Kapralov and Kunal Talwar. On differentially private low rank
  approximation.
  The implementation assumes all data points used to generate the covariance
  matrix cov have norm less than 1.
  Args:
      cov: The input covariance matrix
      n: Number of data points used to generate this covariance matrix
      epsilon: From epsilon differential privacy
      debug: Whether to print debug information

  Returns:
      Private covariance matrix.
  """
  cov = np.asmatrix(cov)
  d = cov.shape[0]
  output = np.zeros_like(cov)
  for _ in range(d):
    eigvec = np.asmatrix(samp.sample_bingham_opt(-4 * epsilon / d * cov))
    eigval = np.squeeze(eigvec.T * cov * eigvec).item()
    eigval += np.random.laplace(0, 2 * d / epsilon)
    eigval = np.maximum(0, np.minimum(eigval, n))
    low_rank = eigval * np.dot(eigvec, eigvec.T)
    cov = cov - low_rank
    output += low_rank
    if debug:
      print("Current covariance")
      print(cov)
  return output
