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

"""Polynomial simulation datasets to compute ground truth calibration error.

Simulated datasets in this file assume that E[Y | f(x)] is polynomial in
either f(x) or (1-f(x)).
"""
import numpy as np
from scipy.special import beta as betafn  # pylint: disable=no-name-in-module

from caltrain.simulation.true_dataset import TrueDatasetBetaPrior


class TruePolynomial(TrueDatasetBetaPrior):
  """E[Y | f(X)] = f(X)^d, f(x) ~ Beta(alpha, beta)."""

  def __init__(self, seed=47, alpha=1, beta=1, d=2., n_samples=None, p1=0):
    super(TruePolynomial, self).__init__(seed, alpha, beta, n_samples, p1=p1)
    self.d = d

  def eval(self, fx):
    return pow(fx, self.d)

  # Mike did this integral by hand
  def true_calib_error(self):
    ans = (betafn(self.alpha + 2, self.beta) -
           2. * betafn(self.alpha + self.d + 1, self.beta) +
           betafn(self.alpha + 2 * self.d, self.beta)) / betafn(
               self.alpha, self.beta)
    return np.sqrt(ans)


class TrueTwoParamPolynomial(TrueDatasetBetaPrior):
  """Assumes E[Y | f(X)] = a*f(X)^d and f(x) ~ Beta(alpha, beta)."""

  def __init__(self,
               seed=47,
               alpha=1,
               beta=1,
               b=2.,
               a=1.,
               n_samples=None,
               p1=0):
    super(TrueTwoParamPolynomial, self).__init__(
        seed, alpha, beta, n_samples, p1=p1)
    self.a = a
    self.b = b

  def eval(self, fx):
    return np.exp(self.a) * pow(fx, self.b)


class TrueFlipPolynomial(TrueDatasetBetaPrior):
  """Assumes E[Y | f(X)] = 1-(1-f(X))^d and f(x) ~ Beta(alpha, beta)."""

  def __init__(self, seed=47, alpha=1, beta=1, d=2., n_samples=None, p1=0):
    super(TrueFlipPolynomial, self).__init__(
        seed, alpha, beta, n_samples, p1=p1)
    self.d = d

  def eval(self, fx):
    return 1. - pow(1. - fx, self.d)


class TrueTwoParamFlipPolynomial(TrueDatasetBetaPrior):
  """Assumes E[Y | f(X)] = 1-a*(1-f(X))^b and f(x) ~ Beta(alpha, beta).

  Corresponds to glm "logflip_logflip_b0_b1".
  """

  def __init__(self,
               seed=47,
               alpha=1,
               beta=1,
               b=2.,
               a=1.0,
               n_samples=None,
               p1=0):
    super(TrueTwoParamFlipPolynomial, self).__init__(
        seed, alpha, beta, n_samples, p1=p1)
    self.a = a
    self.b = b

  def eval(self, fx):
    return 1. - np.exp(self.a) * pow(1. - fx, self.b)
