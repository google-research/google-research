# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Scaling law estimator M3.
"""

import numpy as np
import sklearn.linear_model

from revisiting_neural_scaling_laws.methods import base


class Estimator(base.M):
  """Scaling law estimator M3."""

  def __init__(self,
               loss_values,
               c = -0.5,
               gamma = 0.,
               update_gamma = True
               ):
    """Constructor.

    Args:
      loss_values: a dictionary {x: y}, where x is the data size and y is the
        error/loss of the model (lower is better).
      c: initial value of the scaling exponent.
      gamma: initial value of the gamma parameter.
      update_gamma: set to True if gamma is learnable.
    """
    super(Estimator, self).__init__(loss_values,
                                    c,
                                    err_inf=0,
                                    update_err_inf=False,
                                    update_c=True)
    self.gamma = gamma
    self.update_gamma = update_gamma
    # pre-compute
    self.ones = np.ones_like(self.x)

  def _f(self, data_size):
    """RHS."""
    return self.beta * ((1. / data_size + self.gamma) ** self.c)

  def _fv(self):
    """Vectorized implementation of _f()."""
    return self.beta * ((1 / self.x + self.gamma) ** self.c)

  def _g(self, err):
    """LHS: exess loss = err - err_inf."""
    return err

  def _gv(self):
    """Vectorized implementation of _g()."""
    return self.y

  def predict_loss(self, data_size):
    """Estimate the loss given the data size x."""
    return self._f(data_size)

  def _update_c_and_beta(self):
    """Update estimate of beta and c using linear regression."""
    # we don't precompute in M3 because gamma changes.
    logx = np.log(1 / self.x + self.gamma)
    ones_logx = np.stack([self.ones, logx], axis=1)

    # estimate using linear regression
    labels = np.log(self._gv())
    clf = sklearn.linear_model.LinearRegression(
        fit_intercept=False).fit(ones_logx, labels)
    self.beta = np.exp(clf.coef_[0])
    self.c = max(0, clf.coef_[1])  # exponent has to be >= 0

  def _update_gamma_fnct(self):
    """Update gamma using a grid search.

    Because the scale of gamma is unknown in advance, we generate a grid of
    values by the solving the equation exactly for each single data size x. We
    then pick the best value among those in the grid.
    """
    gamma_vals = (self.y / self.beta) ** (1 / self.c) - 1 / self.x
    best_gamma = self.gamma  # current estimate
    best_loss = self._get_objt()

    # pick the one that minimizes the objective among positive values
    for gamma in gamma_vals:
      if gamma >= 0:
        # calculate the new objective function
        self.gamma = gamma
        loss = self._get_objt()
        # compare with best
        if loss < best_loss:
          best_loss = loss
          best_gamma = gamma

    self.gamma = best_gamma

  def estimate_scaling_params(self,
                              max_iterations = 10_000,
                              verbose = True,
                              stop = 1e-10
                              ):
    """Estimate scaling law parameters.

    We iterate between solving for beta & c in closed-form using linear
    regression and optimizing gamma using grid search.

    Args:
      max_iterations: maximum number of times where we iterate between
        optimizing c and beta using least squares and optimizing gamma using
        grid search. You can keep this large since we also monitor progress and
        stop early if we reach convergence.
      verbose: set to True to display progress.
      stop: stop optimizing params when progress in beta is below this value.
    """
    old_beta = self.beta  # we use progress on beta as a stopping criterion
    for k in range(max_iterations):
      self._update_c_and_beta()
      if self.update_gamma:
        self._update_gamma_fnct()

      if verbose:
        obj = self._get_objt()  # objective function; for reporting only.
        print('iter, obj, c, beta, gamma: ', k, obj, self.c,
              self.beta, self.gamma)

      gap_beta = self.beta - old_beta
      old_beta = self.beta
      if abs(gap_beta) < stop:
        break
