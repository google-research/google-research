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

"""Scaling law estimator M5.
"""
from typing import Optional

import numpy as np
import sklearn.linear_model

from revisiting_neural_scaling_laws.methods import base


EPS = 1e-5


class Estimator(base.M):
  """Scaling law estimator M4."""

  def __init__(self,
               loss_values,
               c = -0.5,
               err_inf = None,
               err_0 = 1.0,
               update_c = True,
               update_err_inf = True,
               update_err_0 = True,
               update_alpha = True,
               lo_bound = 0.,
               up_bound = None
               ):
    """Constructor.

    Args:
      loss_values: a dictionary {x: y}, where x is the data size and y is the
        error/loss of the model (lower is better).
      c: initial value of the scaling exponent.
      err_inf: initial estimate of the limiting error rate with infinite data.
        If None, we use the ~ minimum observed loss as an initial estimate.
      err_0: initial estimate of the random-guessing error/loss. If None, we use
        the ~ maximum observed loss as an initial estimate.
      update_c: set to True if the exponent c is learnable.
      update_err_inf: set to True if the err_inf is learnable.
      update_err_0: set to True if err_0 is learnable.
      update_alpha: set to True if alpha is learnable.
      lo_bound: lower bound on error/loss. Default is zero.
      up_bound: upper bound on error/loss. Default is None.
    """
    super(Estimator, self).__init__(loss_values,
                                    c,
                                    err_inf,
                                    update_c,
                                    update_err_inf,
                                    lo_bound,
                                    up_bound)

    self.err_0 = err_0 if err_0 is not None else self.max_err
    # in M4, we initialize err_0 using the maximum observed loss in the data
    # because we take the logarithm, we drop the corresponding size
    self.update_alpha = update_alpha
    self.update_err_0 = update_err_0

    # precompute
    # pylint: disable=invalid-name
    num_examples = len(self.x)
    if update_alpha:
      A = np.zeros((num_examples, 3))  # we use A to solve lienar regression
      for i in range(num_examples):
        xi = self.x[i]
        A[i, 0] = 1.0
        A[i, 1] = -np.log(xi)
        A[i, 2] = np.log(self.err_0 - loss_values[xi])
      self.A = A
    else:  # for ablation, we compare with fixed alpha (e.g. alpha=1 in paper)
      A = np.zeros((num_examples, 2))
      for i in range(num_examples):
        xi = self.x[i]
        A[i, 0] = 1.0
        A[i, 1] = -np.log(xi)
      self.A = A

  def _f(self, data_size):
    """RHS: power law f(x) = beta * x^c."""
    return self.beta * (data_size ** self.c)

  def _fv(self):
    """Vectorized implementation of _f()."""
    return self.beta * (self.x ** self.c)

  def _g(self, loss):
    """LHS: normalized ecess loss/error."""
    return (loss - self.err_inf) / ((self.err_0 - loss) ** self.alpha)

  def _gv(self):
    """Vectorized implementation of g."""
    return (self.y - self.err_inf) / ((self.err_0 - self.y) ** self.alpha)

  def predict_loss(self, data_size, prec = 10_000):
    """Estimate the loss given the data size.

    Here is how we calculate it:
    We have (err_x - err_inf) / (err_0 - err_x)^alpha = f(x).
    If we denote b = err_0 - err_inf and write err_x = w + err_inf, the eq above
    becomes: w / (b - w) ** alpha = f. Rearranging terms: w = f (b-w)^alpha.
    We choose w that minimizes the difference between both sides of the last eq.

    Args:
      data_size: size of data.
      prec: desired precision of predicted loss; e.g. prec=1000 means that the
        predicted loss is accurate up to 3 decimal places.

    Returns:
      predicted_loss: predicted loss using the M4 estimator.
    """
    if abs(self.alpha) < EPS:
      return self.err_inf + self._f(data_size)
    else:
      f = self._f(data_size)
      b = self.err_0 - self.err_inf
      w = np.linspace(0, b, prec)
      loss = np.abs(w - f * (b - w) ** self.alpha)
      indx_min = np.argmin(loss)
      return w[indx_min] + self.err_inf

  def _hessian(self):
    """Calculate 2nd derviatives w.r.t. (err_inf, err_0)."""
    log_g = np.log(self._gv())
    log_f = np.log(self._fv())
    # second derivative w.r.t. err_inf
    h_inf = np.mean((1 - log_g + log_f) / (self.y - self.err_inf) ** 2)
    # second derivative w.r.t. err_0
    h_0 = self.alpha * np.mean(
        (self.alpha + log_g - log_f) / (self.err_0 - self.y) ** 2)
    return h_inf, h_0

  def _grad(self):
    """Calculate gradient w.r.t. (err_inf, err_0)."""
    log_f = np.log(self._fv())
    log_g = np.log(self._gv())
    residual = log_g - log_f
    # gradients
    grad_inf = - np.mean(residual / (self.y - self.err_inf))
    grad_0 = - self.alpha * np.mean(residual / (self.err_0 - self.y))
    return grad_inf, grad_0

  def _update_errs(self, lr = 1e-8, epochs = 1000):
    """Update err_inf and err_0 using gradient descent."""
    for _ in range(epochs):
      grad_inf, grad_0 = self._grad()
      # descent + projection steps
      if self.update_err_inf:
        self.err_inf -= lr * grad_inf
      if self.update_err_0:
        self.err_0 -= lr * grad_0
      self._project()

  def _update_errs_hessian(self, lr = 0.1, epochs = 100):
    """Update estimate of err_inf and err_0 using Newton's method."""
    for _ in range(epochs):
      grad_inf, grad_0 = self._grad()
      h_inf, h_0 = self._hessian()
      # descent + projection steps
      if self.update_err_inf and abs(h_inf) > EPS:
        self.err_inf -= lr * grad_inf / h_inf
      if self.update_err_0 and abs(h_0) > EPS:
        self.err_0 -= lr * grad_0 / h_0
      self._project()

  def _update_c_beta_alpha(self):
    """Update estimates of alpha, beta and c using linear regression."""
    clf = sklearn.linear_model.LinearRegression(
        fit_intercept=False, positive=True)

    if self.update_alpha:
      labels = np.log(self.y - self.err_inf)
      clf.fit(self.A, labels)
      # we multiply c by -1 (because of the non-negativity constraint)
      log_beta, self.c, self.alpha = clf.coef_[0], -clf.coef_[1], clf.coef_[2]
    else:
      labels = np.log(self.y - self.err_inf) - np.log(self.err_0 - self.y)
      clf.fit(self.A, labels)
      # we multiply c by -1 (because of the non-negativity constraint)
      log_beta, self.c = clf.coef_[0], -clf.coef_[1]

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
        optimizing (c, beta, alpha) using least squares and optimizing err_inf
        and err_0 using gradient descent / Newton's method. You can keep this
        large since we also monitor progress and stop early if we reach
        convergence.
      verbose: set to True to display progress.
      lr: learning rate used to optimize err_inf/err_0 using gradient descent.
        This should be very small. A larger learning rate is used when switching
        to Newton's method.
      epochs: number of epochs when optimizing err_inf/err_0 using grad descent.
      grad_iters: number of iterations used for gradient descent before
        switching to Newton's method.
      stop: stop optimizing params when progress in beta is below this value.
    """
    if not self.update_err_inf and not self.update_err_0:
      max_iterations = 1
    old_beta = self.beta  # stopping criterion
    for k in range(max_iterations):
      # update c, alpha, beta using least squares
      self._update_c_beta_alpha()

      # Initially, use grad descent. Then, swtich to Newton's method.
      if k < grad_iters:  # use gradient descent
        self._update_errs(lr=lr, epochs=epochs)
      else:
        self._update_errs_hessian()  # switch to Newton's method

      if verbose:
        obj = self._get_objt()  # objective function; for reporting only.
        print('iter, obj, c, beta, alpha, err_inf: ', k,
              obj, self.beta, self.c, self.alpha, self.err_inf)

      gap_beta = self.beta - old_beta
      old_beta = self.beta
      # use early stopping only after switching to Newton's method
      if abs(gap_beta) < stop and k > grad_iters + 1:
        break
