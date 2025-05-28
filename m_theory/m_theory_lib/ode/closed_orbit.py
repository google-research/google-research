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

"""Example application: Finding closed orbits."""

import dataclasses
from typing import Callable, Optional

import numpy.typing
import ode_hessian
import scipy.optimize
import tensorflow as tf


_DEFAULT_USE_TF_FUNCTION = True


maybe_tf_function = ode_hessian.maybe_tf_function


@dataclasses.dataclass
class ClosedOrbitSpec:
  """Closed Orbit Specification."""
  ode_bp: ode_hessian.ODEBackpropProblem
  y0: numpy.ndarray
  y1: numpy.ndarray
  loss: float
  d_loss_d_y0: numpy.ndarray
  hessian: numpy.ndarray


def find_closed_orbit(
    *,
    tf_dy_dt,
    y0,
    tf_loss = None,
    t0_to_t1 = (0.0, 1.0),
    report = None,
    gtol=1e-8,
    use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Finds a closed orbit."""
  y0 = numpy.asarray(y0)
  odeint_kwargs = dict(method='DOP853', rtol=1e-10, atol=1e-10)
  #
  if tf_loss is None:
    @maybe_tf_function(use_tf_function)
    def tf_loss(t_y0, t_y1):
      return tf.math.reduce_sum(tf.math.square(t_y1 - t_y0))
  ode_bp = ode_hessian.ODEBackpropProblem(
      dim_y=y0.size,
      tf_dy_dt=tf_dy_dt,
      tf_L_y0y1=tf_loss)
  #
  def f(ys):
    y1, loss_at_t1, d_loss_d_y0, hessian = ode_bp.backprop(
        ys, t0_to_t1, want_order=0, odeint_kwargs=odeint_kwargs)
    del y1, d_loss_d_y0, hessian  # Unused.
    if report is not None:
      report(f'L: {loss_at_t1:18.12g}')
    return loss_at_t1
  def fprime(ys):
    y1, loss_at_t1, d_loss_d_y0, hessian = ode_bp.backprop(
        ys, t0_to_t1, want_order=1, odeint_kwargs=odeint_kwargs)
    del y1, loss_at_t1, hessian  # Unused.
    return d_loss_d_y0
  y0_opt = scipy.optimize.fmin_bfgs(f, y0,
                                    fprime=fprime,
                                    gtol=gtol)
  y1, loss, d_loss_d_y0, hessian = ode_bp.backprop(
      y0_opt, t0_to_t1, want_order=2, odeint_kwargs=odeint_kwargs)
  return ClosedOrbitSpec(ode_bp=ode_bp, y0=y0_opt,
                         y1=y1, loss=loss, d_loss_d_y0=d_loss_d_y0,
                         hessian=hessian)
