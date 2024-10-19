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

"""A collection of common methods for scaling law estimators.
"""
from typing import Optional

import numpy as np


ERR_MARGIN = 1e-3


class M:
  """A collection of common methods for scaling law estimators."""

  def __init__(self,
               loss_values,
               c = -0.5,
               err_inf = None,
               update_c = True,
               update_err_inf = True,
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
      update_c: set to True if the exponent c is learnable.
      update_err_inf: set to True if the err_inf is learnable.
      lo_bound: lower bound on error/loss. Default is zero.
      up_bound: upper bound on error/loss. Default is None.
    """
    if lo_bound is None or lo_bound < 0:
      raise ValueError("lo_bound must be a non-negative real number.")
    if up_bound is not None and up_bound <= lo_bound:
      raise ValueError("up_bound must be larger than lo_bound.")

    self.loss_values = loss_values
    self.x = np.array(sorted(list(self.loss_values.keys())))
    self.y = np.array([self.loss_values[x] for x in self.x])

    # params used for projection
    self.min_err = min(self.loss_values.values()) - ERR_MARGIN  # min loss
    self.max_err = max(self.loss_values.values()) + ERR_MARGIN  # max loss

    # initialize learnable parameters
    self.c = c
    self.beta = 1.0
    self.gamma = 0.  # used by M3
    self.alpha = 0.  # used by M4

    if err_inf is None and update_err_inf:
      err_inf = self.min_err
    elif err_inf is None:  # e.g. in M1
      err_inf = 0
    self.err_inf = err_inf  # initial value of scaled inf data limit
    self.err_0 = 1.0  # used by M4

    self.update_c = update_c
    self.update_err_inf = update_err_inf
    self.update_gamma = False  # used by M3
    self.update_alpha = False  # used by M4
    self.update_err_0 = False  # used by M4

    # these are used to project params and obtain more accurate estimates
    self.lo_bound = max(0., lo_bound - ERR_MARGIN)
    if up_bound is not None: up_bound = up_bound + ERR_MARGIN
    self.up_bound = up_bound

  def predict_loss(self, data_size):
    """Estimate the loss given the data size."""
    raise NotImplementedError

  def _fv(self):
    """Vectorized implementation of RHS: e.g. power law term in M2."""
    raise NotImplementedError

  def _gv(self):
    """Vectorized implementation of LHS: e.g. excess error in M2."""
    raise NotImplementedError

  def _get_objt(self):
    """Objective function to be optimized; for reporting/debugging purposes."""
    g = np.log(self._gv())
    f = np.log(self._fv())
    return 0.5 * np.linalg.norm(g - f)

  def _project(self):
    """Project optimization variables into the feasible set."""
    self.err_inf = max(self.err_inf, self.lo_bound)  # 0 <= err_inf <= min_err
    self.err_inf = min(self.err_inf, self.min_err)
    self.err_0 = max(self.err_0, self.max_err)  # err_0 >= max_err

    if self.up_bound is not None:
      self.err_0 = min(self.err_0, self.up_bound)

  def loss_curve(self,
                 min_data_size,
                 max_data_size,
                 num_points = 10_000
                 ):
    """Predict performance for data sizes in [min_data_size, max_data_size]."""
    if min_data_size < 1:
      raise ValueError("min_data_size must be larger than zero.")
    xn = np.linspace(min_data_size, max_data_size, num_points)
    yn = np.array([self.predict_loss(data_size) for data_size in xn])
    return xn, yn

  def err_limit(self):
    return self.err_inf
