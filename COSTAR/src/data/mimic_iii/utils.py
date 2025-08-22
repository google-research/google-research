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

"""MIMIC-III utilities."""

import numpy as np
import scipy.interpolate
import scipy.special
import sklearn.gaussian_process
import sklearn.kernel_approximation

splev = scipy.interpolate.splev
logsumexp = scipy.special.logsumexp
GaussianProcessRegressor = sklearn.gaussian_process.GaussianProcessRegressor
RBFSampler = sklearn.kernel_approximation.RBFSampler


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


class RandomFourierFeaturesFunction:
  """Random function, sampled from RFF approximation of Gaussian process."""

  def __init__(self, input_dim, gamma, scale):
    self.rbf_sampler = RBFSampler(gamma=gamma)
    self.rbf_sampler.fit(np.random.normal(size=(1, input_dim)))
    self.w = np.random.normal(
        scale=scale / np.sqrt(self.rbf_sampler.n_components),
        size=(self.rbf_sampler.n_components, 1),
    )

  def __call__(self, bigx):
    phi_bigx = self.rbf_sampler.transform(bigx)
    return phi_bigx @ self.w


class DiscretizedRandomGPFunction:
  """Random function, sampled from Gaussian process."""

  def __init__(self, kernels):
    self.gp_sampler = GaussianProcessRegressor(kernel=sum(kernels))

  def __call__(self, bigx, n_samples):
    bigx = bigx.reshape(-1, 1)
    return self.gp_sampler.sample_y(bigx, n_samples=n_samples).T


class SplineTrendsMixture:
  """Random spline, sampled from 3 cubic splines."""

  class BSplines:
    """B splines."""

    def __init__(
        self, low, high, num_bases, degree, x=None, boundaries='stack'
    ):
      self._low = low
      self._high = high
      self._num_bases = num_bases
      self._degree = degree

      use_quantiles_as_knots = x is not None

      if use_quantiles_as_knots:
        knots = SplineTrendsMixture._quantile_knots(
            low, high, x, num_bases, degree
        )
      else:
        knots = SplineTrendsMixture._uniform_knots(low, high, num_bases, degree)

      if boundaries == 'stack':
        self._knots = SplineTrendsMixture._stack_pad(knots, degree)
      elif boundaries == 'space':
        self._knots = SplineTrendsMixture._space_pad(knots, degree)

      self._tck = (self._knots, np.eye(num_bases), degree)

    @property
    def dimension(self):
      return self._num_bases

    def design(self, x):
      return np.array(splev(np.atleast_1d(x), self._tck)).T

  @staticmethod
  def _uniform_knots(low, high, num_bases, degree):
    num_interior_knots = num_bases - (degree + 1)
    knots = np.linspace(low, high, num_interior_knots + 2)
    return np.asarray(knots)

  @staticmethod
  def _quantile_knots(low, high, x, num_bases, degree):
    num_interior_knots = num_bases - (degree + 1)
    clipped = x[(x >= low) & (x <= high)]
    knots = np.percentile(clipped, np.linspace(0, 100, num_interior_knots + 2))
    knots = [low] + list(knots[1:-1]) + [high]
    return np.asarray(knots)

  @staticmethod
  def _stack_pad(knots, degree):
    knots = list(knots)
    knots = ([knots[0]] * degree) + knots + ([knots[-1]] * degree)
    return knots

  @staticmethod
  def _space_pad(knots, degree):
    knots = list(knots)
    d1 = knots[1] - knots[0]
    b1 = np.linspace(knots[0] - d1 * degree, knots[0], degree + 1)
    d2 = knots[-1] - knots[-2]
    b2 = np.linspace(knots[-1], knots[-1] + d2 * degree, degree + 1)
    return list(b1) + knots[1:-1] + list(b2)

  class PopulationModel:
    """Population."""

    def __init__(self, basis, class_prob, class_coef):
      self.basis = basis
      self.n_classes = len(class_coef)

      self.class_prob = np.array(class_prob)
      self.class_coef = np.array(class_coef)

    def sample_class_prob(self, rng):
      logits = rng.normal(size=self.n_classes)
      self.class_prob[:] = np.exp(logits - logsumexp(logits))

    def sample_class_coef(self, mean, cov, rng):
      mvn_rvs = rng.multivariate_normal
      self.class_coef[:] = mvn_rvs(mean, cov, size=self.n_classes)

    def sample(self, size=1):
      z = np.random.choice(self.n_classes, size=size, p=self.class_prob)
      w = self.class_coef[z]
      return z, w

  def __init__(self, n_patients, max_time):
    class_coef = np.array([
        [1.0, 0.9, 0.0, -0.5, -1.0],  # rapidly decline
        [1.0, 1.0, 0.5, 0.2, 0.2],  # mild decline
        [-0.3, -0.2, -0.2, -0.3, -0.2],  # stable
    ])
    low, high, n_bases, degree = 0.0, max_time, class_coef.shape[1], 4
    self.basis = SplineTrendsMixture.BSplines(
        low, high, n_bases, degree, boundaries='space'
    )
    self.population = SplineTrendsMixture.PopulationModel(
        self.basis, [0.35, 0.35, 0.3], class_coef
    )
    self.classes, self.coefs = self.population.sample(size=n_patients)

  def __call__(self, time_range):
    return np.dot(self.coefs, self.basis.design(time_range).T)
