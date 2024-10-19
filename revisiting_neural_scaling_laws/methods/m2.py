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

"""Scaling law estimator M2.
"""
from typing import Optional

import numpy as np
import sklearn.linear_model

from revisiting_neural_scaling_laws.methods import base


class Estimator(base.M):
  """Scaling law estimator M2."""

  def __init__(self,
               loss_values,
               c = -0.5,
               err_inf = None,
               update_c = True,
               update_err_inf = True,
               lo_bound = 0.
               ):
    """Constructor.

    Args:
      loss_values: a dictionary {x: y}, where x is the data size and y is the
        error/loss of the model (lower is better).
      c: initial value of the scaling exponent.
      err_inf: initial estimate of the limiting error rate with infinite data.
        If None, we use the ~ minimum observed loss as an initial estimate.
      update_c: set to True if the exponent c is learnable.
      update_err_inf: set to True if the err_inf is learnable.
      lo_bound: lower bound on error/loss. Default is zero.
    """
    super(Estimator, self).__init__(loss_values,
                                    c,
                                    err_inf,
                                    update_c,
                                    update_err_inf,
                                    lo_bound)
    # pre-compute
    self.ones = np.ones_like(self.x)
    self.ones_logx = np.stack([self.ones, np.log(self.x)], axis=1)

  def _f(self, data_size):
    """RHS: power law f(x) = beta * x^c."""
    return self.beta * (data_size ** self.c)

  def _fv(self):
    """Vectorized implementation of _f()."""
    return self.beta * (self.x ** self.c)

  def _g(self, err):
    """LHS: exess loss = err - err_inf."""
    return err - self.err_inf

  def _gv(self):
    """Vectorized implementation of _g()."""
    return self.y - self.err_inf

  def predict_loss(self, data_size):
    """Estimate the loss given the data size."""
    return self._f(data_size) + self.err_inf

  def _hessian(self):
    """Hessian w.r.t. err_inf, used for faster optimization."""
    log_g = np.log(self._gv())
    log_f = np.log(self._fv())
    h_inf = np.mean((1 - log_g + log_f) / (self.y - self.err_inf) ** 2)
    return h_inf

  def _update_errs(self, lr = 1e-8, epochs = 1000):
    """Update err_inf using gradient descent."""
    if self.update_err_inf:
      log_f = np.log(self._fv())
      for _ in range(epochs):
        residual = np.log(self._gv()) - log_f
        grad = - np.mean(residual / (self.y - self.err_inf))  # gradient
        # now: descent + projection steps
        self.err_inf -= lr * grad
        self._project()

  def _update_errs_hessian(self, lr = 0.1, epochs = 100):
    """Update estimate of err_inf using Newton's method."""
    if self.update_err_inf:
      log_f = np.log(self._fv())
      for _ in range(epochs):
        residual = np.log(self._gv()) - log_f
        grad = - np.mean(residual / (self.y - self.err_inf))
        hess = self._hessian()
        # now: descent + projection steps
        self.err_inf -= lr * grad / hess
        self._project()

  def _update_c_and_beta(self):
    """Update estimate of beta and c using linear regression."""
    labels = np.log(self._gv())
    if self.update_c:
      clf = sklearn.linear_model.LinearRegression(
          fit_intercept=False).fit(self.ones_logx, labels)
      log_beta = clf.coef_[0]
      self.c = clf.coef_[1]
    else:
      log_beta = np.mean(labels - self.ones_logx[:, 1] * self.c)
    self.beta = np.exp(log_beta)

  def estimate_scaling_params(self,
                              max_iterations = 10_000,
                              verbose = True,
                              lr = 1e-8,
                              epochs = 1000,
                              grad_iters = 100,
                              stop = 1e-10
                              ):
    """Estimate scaling law parameters.

    We first estimate parameters using gradient descent for a few epochs before
    switching to Newton's method to accelerate convergence (since we are close
    to the optimal solution by then).

    Args:
      max_iterations: maximum number of times where we iterate between
        optimizing c and beta using least squares and optimizing err_inf using
        gradient descent / Newton's method. You can keep this large since we
        also monitor progress and stop early if we reach convergence.
      verbose: set to True to display progress.
      lr: learning rate used to optimize err_inf using gradient descent. This
        should be very small. A larger learning rate is used when switching to
        Newton's method.
      epochs: number of epochs when optimizing err_inf using gradient descent.
      grad_iters: number of iterations used for gradient descent before
        switching to Newton's method.
      stop: stop optimizing params when progress in beta is below this value.
    """
    if not self.update_err_inf:  # e.g. use M1
      max_iterations = 1

    old_beta = self.beta  # we use progress on beta as a stopping criterion
    k = 0
    for k in range(max_iterations):
      self._update_c_and_beta()
      if self.update_err_inf:
        if k < grad_iters:  # use gradient descent
          self._update_errs(lr=lr, epochs=epochs)
        else:  # switch to Newton's method now that we are close to optimal sol
          self._update_errs_hessian()

      if verbose:
        obj = self._get_objt()  # objective function; for reporting only.
        print('iter, obj, c, beta, err_inf: ', k, obj, self.c,
              self.beta, self.err_inf)

      gap_beta = self.beta - old_beta
      old_beta = self.beta

      # use early stopping only after switching to Newton's method
      if abs(gap_beta) < stop and k > grad_iters + 1:
        break
