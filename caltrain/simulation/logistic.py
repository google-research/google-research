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

"""Simulated datasets from the logistic family."""
import numpy as np

from caltrain.simulation.true_dataset import TrueDataset
from caltrain.simulation.true_dataset import TrueDatasetBetaPrior


class TrueLogisticUniform(TrueDataset):
  """Assumes E[Y | f(x)] = 1/(1+exp(-a*(fx-b))) and f(x)~Uniform(0,1)."""

  def __init__(self, seed=47, a=10., b=-5., n_samples=None):
    super(TrueLogisticUniform, self).__init__(seed, n_samples)
    self.a = a
    self.b = b

  def dataset(self):
    # use current state to set seed. The goal is to put the rnd num gen in
    # the same place when this function is called for the k'th time, regardless
    # of how many random numbers we generate
    np.random.seed(self.next_seed)
    self.next_seed = int(10000000 * np.random.rand())

    # generate random inputs and output coin flips to ensure that the larger
    # n_samples set will include the smaller n_samples set
    r = np.random.rand((2 * self.n_samples)).reshape(self.n_samples, 2)
    self.fx = r[:, 0]
    self.yprob = self.eval(self.fx)
    self.y = r[:, 1] < self.yprob

    return self.fx, self.yprob, self.y

  def sample_fx(self, x):
    return np.random.rand()

  def pdf_fx(self, x):
    return 1.

  def eval(self, fx):
    return 1. / (1. + np.exp(-self.a * (fx - self.b)))


class TrueLogisticBeta(TrueDatasetBetaPrior):
  """Assumes E[Y | f(x)] = 1/(1+exp(-a*(fx-b))) and f(x)~Beta(alpha, beta)."""

  def __init__(self,
               seed=47,
               a=10.,
               b=-5.,
               alpha=1,
               beta=1,
               n_samples=None,
               p1=0):
    super(TrueLogisticBeta, self).__init__(seed, alpha, beta, n_samples, p1=p1)
    self.a = a
    self.b = b

  def eval(self, fx):
    return 1. / (1. + np.exp(-self.a * (fx - self.b)))


class TrueLogisticLogOdds(TrueDatasetBetaPrior):
  """E[Y | f(x)] = 1/(1+exp(-(a+b*(log(fx/1-fx))))) and f(x)~Beta(alpha, beta).

  The conditional probability E[Y | f(x)] is a two parameter logistic function
  of the log odds, i.e. log(fx/(1-fx)). Corresponds to the glm fit
  "logit_logit_b0_b1".
  """

  def __init__(self,
               seed=47,
               a=10.,
               b=-5.,
               alpha=1,
               beta=1,
               n_samples=None,
               p1=0):
    super(TrueLogisticLogOdds, self).__init__(
        seed, alpha, beta, n_samples, p1=p1)
    self.a = a
    self.b = b

  def eval(self, fx):
    if isinstance(fx, float):
      if fx == 1.0:
        fx = fx - 1E-16
    else:
      fx[fx == 1.0] = 1.0 - 1E-16
    return 1. / (1. + np.exp(-(self.a + self.b * np.log(fx / (1. - fx)))))


class TrueLogisticTwoParamFlipPolynomial(TrueDatasetBetaPrior):
  """E[Y | f(x)] = 1/(1+exp(-(a+b*(log(fx/1-fx))))) and f(x)~Beta(alpha, beta).

  The conditional probability E[Y | f(x)] is a two parameter logistic function
  of the log odds, i.e. log(fx/(1-fx)). Corresponds to the glm fit
  "logit_logit_b0_b1".
  """

  def __init__(self,
               seed=47,
               a=10.,
               b=-5.,
               alpha=1,
               beta=1,
               n_samples=None,
               p1=0):
    super(TrueLogisticTwoParamFlipPolynomial, self).__init__(
        seed, alpha, beta, n_samples, p1=p1)
    self.a = a
    self.b = b

  def eval(self, fx):
    if isinstance(fx, float):
      if fx == 1.0:
        fx = fx - 1E-16
    else:
      fx[fx == 1.0] = 1.0 - 1E-16
    return 1. / (1. + np.exp(-(self.a + self.b * np.log(1. - fx))))
