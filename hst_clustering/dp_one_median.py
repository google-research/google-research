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

"""Library to solve the DP 1-median problem."""

from absl import logging
import numpy as np
from scipy import optimize

from hst_clustering import experiment_config


def smooth_norm_conj(t, lambd=0.01):
  """Smooth approximation to the l2 norm.

  Uses a quadratic approximation to the norm when the true norm is
  less than 2 * lambd. Uses the true norm - lambd otherwise.

  Args:
    t: A numpy array.
    lambd: A smoothing parameter. The larger lambd is, the closer the
      approximation is to the square l2 norm.

  Returns:
    A smooth approximation to the l2 norm.
  """
  smooth_approx = np.linalg.norm(t) ** 2 / 4 / lambd
  if np.linalg.norm(t) < 2 * lambd:
    return smooth_approx
  return np.linalg.norm(t) - lambd


def smooth_norm_grad(t, lambd=0.01):
  """Gradient of the smooth approximation to the norm.

  Args:
    t: A numpy array.
    lambd: A smoothing parameter.

  Returns:
    The gradient of smooth_norm_conj.
  """
  return 1 / 2 / lambd * t if np.linalg.norm(
      t) < 2 * lambd else t / np.linalg.norm(t)


def smooth_norm_hess_p(t, p, lambd=0.01):
  """Returns the Hessian of smooth_approx_conj applied to a point p.

  Args:
    t: Numpy array. Point where the Hessian is calculated.
    p: Numpy array. Point to which the Hessian is applied.
    lambd: Smoothing parameter.

  Returns:
    H(t) * p.
  """
  if np.linalg.norm(t) < 2 * lambd:
    return p / 2 / lambd
  return (p * np.linalg.norm(t)**2 - t * np.dot(t, p)) / np.linalg.norm(t)**3


def private_k_med_objective(lambd, epsilon, delta, p, gamma, n):
  """Objective function for private k medians.

  Objective is based on the parameters for DP convex optimization defined here
  https://www.uvm.edu/~jnear/papers/TPDPCO.pdf

  Args:
    lambd: Smoothing parameter for the objective.
    epsilon: Desired epsilon privacy guarantee.
    delta: Desired delta privacy guarantee.
    p: dimension of the problem.
    gamma: Accuracy upper bound of the solution. Norm of gradient MUST be less
      than gamm in order for DP guarantees to be valid. n : Number of points in
      the dataset.
    n: Number of points in the dataset.

  Returns:
    objective: Function that takes a center c and dataset X as input. It
      evaluates the smooth distance of the data to the center c.
    jac: The gradient of the above function with respect to c.
    hess_p: Function that takes as input c, X, and p and return H(c,X)*p
        where H(c, X) is the hessian of the objective.
    b2: Correction factor of the objective perturbation method to account
        for gradient not being zero.
  """
  e1 = epsilon / 2.0
  e2 = e1
  d1 = delta / 2.0
  d2 = d1
  e3 = np.maximum(e1 - 1.0, e1 / 2)
  beta = 1 / 2 / lambd

  regularizer = p * beta / (e1 - e3)

  sigma1 = 2 / n * (1 + np.sqrt(2 * np.log(1 / d1))) / e3
  b1 = np.random.normal(0, sigma1, p)

  sigma2 = n * gamma / regularizer * (1 + np.sqrt(2 * np.log(1 / d2))) / e2
  b2 = np.random.normal(0, sigma2, p)

  def objective(c, data):
    c_repeated = np.repeat(c.reshape(1, -1), n, 0)
    smooth_obj = np.average(
        np.apply_along_axis(lambda t: smooth_norm_conj(t, lambd), 1,
                            data - c_repeated))
    return (
        smooth_obj
        + 1 / n * regularizer / 2 * np.linalg.norm(c) ** 2
        + np.dot(b1, c)
    )

  def jac(c, data):
    c_repeated = np.repeat(c.reshape(1, -1), n, 0)
    smooth_jac = np.average(
        np.apply_along_axis(lambda t: -smooth_norm_grad(t, lambd), 1,
                            data - c_repeated),
        axis=0)
    return smooth_jac + 1 / n * regularizer * c + b1

  def hess_p(c, p, data):
    c_repeated = np.repeat(c.reshape(1, -1), n, 0)
    smooth_hess = np.average(
        np.apply_along_axis(lambda t: smooth_norm_hess_p(t, p, lambd), 1,
                            data - c_repeated),
        axis=0)
    return smooth_hess + 1 / n * regularizer * p

  return objective, jac, hess_p, b2


class PrivacyParams:
  lambd: float  # The smoothing factor lambda used in
  # https://dl.acm.org/doi/pdf/10.1145/3534678.3539409. It is used to generate
  # a smooth approximation to the euclidean norm (bottom of page 227). As
  # this value goes to zero, the smooth approximation approaches the norm.
  epsilon: float
  delta: float
  gamma: float  # Accuracy expected to be satisfied by the optimizer of the
  # one median.


def get_private_kmed_center(data,
                            lambd,
                            epsilon,
                            delta,
                            gamma,
                            max_samples=None):
  """Gets private kmedian center.

  Args:
    data: Data set each row is a data point.
    lambd: Smoothing parameter of the norm function.
    epsilon: Privacy parameter.
    delta: Privacy parameter.
    gamma: Expected accuracy of the solver. Gradient norm must be less than
      gamma to satisfy privacy.
    max_samples: The maximun number of points the algorithm handles. If None it
      uses the full dataset otherwise it samples max_samples entries from the
      dataset.

  Returns:
    a) "Private" solution assuming that the optimizer converged
    b) Truly private solution based on gamma.
  """
  n, p = data.shape
  if n == 0:
    error = """There are no points assigned to a center. Returning a
    random center in [0,1]^p"""
    logging.warning(error)

    return np.random.uniform(0, 1, p), np.random.uniform(0, 1, p)
  if max_samples is None:
    max_samples = n
  sampling_fraction = np.minimum(max_samples / n, 1.0)
  if sampling_fraction < 1.0:
    ix = np.random.binomial(1, sampling_fraction, n) == 1
    data = data[ix, :]

  private_objective, private_jac, hess_p, b2 = private_k_med_objective(
      lambd, epsilon, delta, p, gamma, n)
  sol = optimize.minimize(
      lambda c: private_objective(c, data),
      np.zeros(p),
      method="trust-ncg",
      jac=lambda c: private_jac(c, data),
      hessp=lambda c, p: hess_p(c, p, data),
      options={"gtol": gamma})
  return sol.x, sol.x + b2


### Functions for evaluating the true kmedians objective
def kmed_objective(data, mu):
  """K medians objective.

  Args:
    data: Dataset. Each row is a datapoint.
    mu: Center to evaluate.

  Returns:
    sum ||x_i - mu||.
  """
  n, _ = data.shape
  mu_mat = np.repeat(mu.reshape(1, -1), n, 0)
  return _objective_repeated(data, mu_mat)


def _objective_repeated(data, mu_mat):
  return np.sum(np.linalg.norm(data - mu_mat, axis=1))


def kmed_gradient(data, mu):
  """Gradient of the kmedians objective.

  Args:
    data: Numpy array with one row per example.
    mu: Center to be evaluated.

  Returns:
    Gradient of the kemdian objective.
  """
  n, _ = data.shape
  mu_mat = np.repeat(mu.reshape(1, -1), n, 0)
  return np.sum((data - mu_mat) / _objective_repeated(data, mu_mat), axis=0)


def get_one_median_as_center_selector(
    config,
):
  """Returns a center selector function based on OneMedianDpConfig.

  Wrapper function to be used with beam. A center selector is a function that
  takes as input a tuple and outputs a (key, center).
  The input tuple has a first entry that corresponds to a key. The second entry
  is an iterable collection of arrays where each array corresponds to a data
  point.

  A center selector is to be called to the output of GroupByKey in a
  PCollection.

  Args:
    config: A OneMedianDpConfig.

  Returns:
    A center selector.
  """
  eps = config.epsilon
  delta = config.delta
  gamma = config.gamma
  lambd = config.smoothing_factor

  def one_median_dp_beam(arrays):
    nonlocal eps
    nonlocal delta
    nonlocal gamma
    nonlocal lambd
    in_memory_arrays = list(arrays[1])
    data = np.vstack(in_memory_arrays)
    n, d = data.shape
    gamma_scaled = gamma / n * np.sqrt(d)
    logging.info("Calling private k center with params lambd: %f gamma: %f",
                 lambd, gamma_scaled)
    solution = get_private_kmed_center(data, lambd, eps, delta, gamma_scaled)
    return (arrays[0], solution[1])

  return one_median_dp_beam
