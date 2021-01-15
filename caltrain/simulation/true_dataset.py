# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Simulation datasets to compute ground truth calibration error."""
import numpy as np
import scipy.integrate as integrate
from scipy.special import beta as betafn  # pylint: disable=no-name-in-module


class TrueDataset:
  """Base class for implementing simulated dataset."""

  def __init__(self, seed=47, n_samples=None):
    self.seed = seed
    self.reset(n_samples)

  def reset(self, n_samples=None):
    self.next_seed = self.seed
    if n_samples is not None:
      self.n_samples = n_samples

  def dataset(self):
    """Generate simulated dataset."""
    # use current state to set seed. The goal is to put the rnd num gen in
    # the same place when this function is called for the k'th time, regardless
    # of how many random numbers we generate
    np.random.seed(self.next_seed)
    self.next_seed = int(10000000 * np.random.rand())

    # generate examples one at a time so that a small n_samples set would
    # be included in a larger n_samples set
    self.fx = np.zeros((self.n_samples))
    yflip = np.zeros((self.n_samples))
    for i in range(self.n_samples):
      self.fx[i] = self.sample_fx()
      yflip[i] = np.random.rand()
    self.yprob = self.eval(self.fx)
    self.y = yflip < self.yprob

    return self.fx, self.yprob, self.y

  def sample_fx(self):
    """Generate single sample from f(X) distribution."""
    raise NotImplementedError

  def pdf_fx(self, x):
    """PDF for f(X)."""
    raise NotImplementedError

  def eval(self, fx):
    """True form of E[Y | f(X)]."""
    raise NotImplementedError

  def true_calib_error(self, norm='L2'):
    """True calibration error."""
    if norm == 'L2':
      integrand = self.integrand_l2
      return np.sqrt(
          integrate.quad(integrand, 0., 1.)[0] * (1 - self.p1) +
          np.square(1. - self.eval(1.)) * self.p1)
    elif norm == 'L1':
      integrand = self.integrand_l1
      return integrate.quad(integrand, 0., 1.)[0] * (
          1 - self.p1) + np.abs(1. - self.eval(1.)) * self.p1
    else:
      raise NotImplementedError

  def integrand_l2(self, x):
    return np.square(x - self.eval(x)) * self.pdf_fx(x)

  def integrand_l1(self, x):
    return np.abs(x - self.eval(x)) * self.pdf_fx(x)


class TrueDatasetBetaPrior(TrueDataset):
  """Assumes f(x) ~ Beta(alpha, beta)."""

  def __init__(self, seed=47, alpha=1, beta=1, n_samples=None, p1=None):
    super(TrueDatasetBetaPrior, self).__init__(seed, n_samples)
    self.alpha = alpha
    self.beta = beta
    self.p1 = p1

  def sample_fx(self):
    if self.p1 is None:
      return np.random.beta(self.alpha, self.beta)
    else:
      if np.random.rand() < self.p1:
        return 1.
      else:
        return np.random.beta(self.alpha, self.beta)

  def pdf_fx(self, x):
    return pow(x, self.alpha - 1.) * pow(1. - x, self.beta - 1.) / betafn(
        self.alpha, self.beta)
