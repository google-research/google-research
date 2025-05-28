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

"""Basic tests for ODE-hessian code.

These tests cover basic functionality.
"""

import unittest

import numpy
import ode_hessian
import tensorflow as tf


def _tc(x):
  return tf.constant(x, dtype=tf.float64)


def _tf_sir_dy_dt(t_ys):
  # A SIR model for spread of an infectious disease.
  beta, gamma = 5, 1.25
  t_s, t_i, t_r = tf.unstack(t_ys)
  del t_r  # Unused.
  return tf.stack([
      -beta * t_i * t_s,
      beta * t_i * t_s - gamma * t_i,
      gamma * t_i], axis=0)


def _tf_sir_loss(t_ys_start, t_ys_final):
  del t_ys_start  # Unused.
  # In this example, we are going for 'recovered = 0.80' at-end.
  return tf.math.square(t_ys_final[2] - 0.8)


class HelpersTest(unittest.TestCase):
  """Basic tests for helper functions."""

  def test_maybe_tf_function(self):
    """Checks maybe_tf_function() behavior."""
    def f(x):
      return x
    self.assertIs(ode_hessian.maybe_tf_function(False)(f), f)
    self.assertIs(ode_hessian.maybe_tf_function(True), tf.function)

  def test_fd_grad(self):
    """Smoke-tests basic fd_grad behavior."""
    x0s = numpy.random.RandomState(seed=0).normal(size=(3, 10))
    fn_d4 = lambda v: ((v[:, numpy.newaxis] - x0s) ** 4).sum(axis=0)
    v0 = [1, 1.5, 1.7]
    grad_d4 = ode_hessian.fd_grad(fn_d4, v0)
    self.assertEqual(grad_d4.shape, (10, 3))
    self.assertTrue(
        numpy.allclose(
            grad_d4[1],
            [4 * (p-q)**3 for p, q in zip(v0, x0s[:, 1])],
            rtol=1e-3, atol=1e-3))

  def test_fd_hessian(self):
    """Smoke-tests basic fd_hessian behavior."""
    def fn(xs):
      return (xs[Ellipsis, 0] + 2 * xs[Ellipsis, 1])**2
    x0 = numpy.array([1.0, 10.0, 77.0])
    hessian = ode_hessian.fd_hessian(fn, x0)
    self.assertEqual(hessian.shape, (3, 3))
    self.assertTrue(
        numpy.allclose(
            hessian,
            [[2, 4, 0], [4, 8, 0], [0, 0, 0]],
            rtol=1e-3, atol=1e-3))

  def test_tf_jacobian(self):
    """Smoke-tests tf_jacobian behavior."""
    @tf.function
    def tf_f(t_xs):
      t_x0, t_x1, *_ = tf.unstack(t_xs)
      return tf.stack([2.5 * t_x1 + t_x0**2,
                       3 * t_x0**2 * t_x1 + 7,
                       t_x0 - t_x1
                       ], axis=0)
    tf_jf = ode_hessian.tf_jacobian(tf_f)
    jacobian = tf_jf(_tc([4.0, 5.0, 6.0, 2.0])).numpy()
    self.assertEqual(jacobian.shape, (3, 4))
    self.assertTrue(
        numpy.allclose(
            jacobian,
            [[8, 2.5, 0, 0],
             [120, 48, 0, 0],
             [1, -1, 0, 0]],
            rtol=1e-3, atol=1e-3))

  def test_tf_jac_vec(self):
    """Smoke-tests tf_jac_vec behavior."""
    @tf.function
    def tf_f(t_xs):
      t_x0, t_x1, *_ = tf.unstack(t_xs)
      return tf.stack([2.5 * t_x1 + t_x0**2,
                       3 * t_x0**2 * t_x1 + 7,
                       t_x0 - t_x1
                       ], axis=0)
    tf_jv = ode_hessian.tf_jac_vec(tf_f)
    jv = tf_jv(
        _tc([3.0, 5.0, 6.0, 2.0]),
        _tc([1, 2, 10])).numpy()
    self.assertEqual(jv.shape, (4,))
    self.assertTrue(
        numpy.allclose(jv, [196, 46.5, 0, 0], rtol=1e-3, atol=1e-3))

  def test_tf_grad(self):
    """Smoke-tests tf_grad behavior."""
    @tf.function
    def tf_f(t_xs):
      t_x0, t_x1, *_ = tf.unstack(t_xs)
      return 2 * t_x0 * t_x1**2 + 10
    tf_grad_f = ode_hessian.tf_grad(tf_f)
    grad_f = tf_grad_f(
        _tc([3.0, 5.0, 6.0])).numpy()
    self.assertEqual(grad_f.shape, (3,))
    self.assertTrue(
        numpy.allclose(grad_f, [50, 60, 0], rtol=1e-3, atol=1e-3))

  def test_tf_grad_hessian(self):
    """Smoke-tests tf_grad_hessian behavior."""
    @tf.function
    def tf_f(t_xs):
      t_x0, t_x1, *_ = tf.unstack(t_xs)
      return 2 * t_x0 * t_x1**2 + 10
    tf_grad_f, tf_hessian_f = ode_hessian.tf_grad_hessian(tf_f)
    grad_f = tf_grad_f(
        _tc([3.0, 5.0, 6.0])).numpy()
    self.assertEqual(grad_f.shape, (3,))
    self.assertTrue(
        numpy.allclose(grad_f, [50, 60, 0], rtol=1e-3, atol=1e-3))
    hessian_f = tf_hessian_f(_tc([3.0, 5.0, 6.0])).numpy()
    self.assertEqual(hessian_f.shape, (3, 3))
    self.assertTrue(
        numpy.allclose(hessian_f,
                       [[0, 20, 0],
                        [20, 12, 0],
                        [0, 0, 0]],
                       rtol=1e-3, atol=1e-3))

  def test_tf_backprop_ode(self):
    """Smoke-tests tf_backprop_ode behavior."""
    @tf.function
    def tf_dy_dt(t_ys):
      t_y0, t_y1, *_ = tf.unstack(t_ys)
      return tf.stack(
          [-t_y1, t_y0, t_y0 * t_y1, _tc(1)], axis=0)
    tf_dyext_dt = ode_hessian.tf_backprop_ode(tf_dy_dt, use_tf_function=False)
    d_dt_yext = tf_dyext_dt(
        _tc([2, 3, 1, 5, -1, -3, -6, -4])).numpy()
    self.assertTrue(
        numpy.allclose(d_dt_yext,
                       [-3, 2, 6, 1, 21, 11, 0, 0],
                       rtol=1e-3, atol=1e-3))

  def test_scipy_odeint(self):
    """Smoke-tests scipy_odeint behavior."""
    def f_dy_dt(ys):
      ret = numpy.zeros_like(ys)
      dist2 = ys.dot(ys)
      ret[0] = ys[1] / dist2
      ret[1] = -ys[0] / dist2
      return ret
    #
    ys_final = ode_hessian.scipy_odeint(
        f_dy_dt,
        (0, 2 * numpy.pi),
        [2, 0, 0, 0])
    ys_final_ivp = ode_hessian.scipy_odeint(
        f_dy_dt,
        (0, 2 * numpy.pi),
        [2, 0, 0, 0],
        method='DOP853')
    self.assertTrue(
        numpy.allclose(ys_final,
                       [0, -2, 0, 0],
                       rtol=1e-3, atol=1e-3))
    self.assertTrue(
        numpy.allclose(ys_final_ivp,
                       [0, -2, 0, 0],
                       rtol=1e-3, atol=1e-3))


class OBPTest(unittest.TestCase):
  """Basic tests for ODEBackpropProblem backpropagation."""

  def test_obp(self):
    """Smoke-tests ODEBackpropProblem.backprop() against finite differencing."""
    def tf_dy_dt(t_ys):
      t_y0, t_y1, t_y2 = tf.unstack(t_ys)
      # Some more or less generic 3d nonlinear dynamics.
      return tf.stack([
          _tc(2.0) * t_y0 * t_y1 * t_y1 - _tc(0.5) * t_y1 * t_y2,
          _tc(0.25) + t_y2 * t_y2 - _tc(0.75) * t_y1,
          _tc(1.25) * t_y2 * t_y0 * t_y0 - _tc(0.125) * t_y2 +
          _tc(0.125) * t_y1],
                      axis=0)
    def tf_loss(t_ys_start, t_ys_final):
      return tf.math.square(0.5 * t_ys_start[0] - t_ys_final[1])
    #
    t0_t1 = (5, 5.25)
    y0 = [1.0, 0.25, 0.75]
    obp = ode_hessian.ODEBackpropProblem(
        dim_y=3,
        tf_dy_dt=tf_dy_dt,
        tf_L_y0y1=tf_loss)
    backprop = obp.backprop(y0, t0_t1)
    y1, loss, grad_via_bp, hessian_via_bp = backprop
    del y1, loss  # Unused.
    # Evaluating the gradient and hessian with finite-differencing.
    def fn(ys):
      bp = obp.backprop(
          ys, t0_t1, want_order=0,
          odeint_kwargs=dict(rtol=1e-10, atol=1e-10))
      return bp[1]
    hessian_via_fd = ode_hessian.fd_hessian(fn, y0)
    grad_via_fd = ode_hessian.fd_grad(fn, y0)
    self.assertTrue(
        numpy.allclose(grad_via_fd, grad_via_bp, rtol=1e-3, atol=1e-3))
    self.assertTrue(
        numpy.allclose(hessian_via_fd, hessian_via_bp, rtol=1e-3, atol=1e-3))


class DifferentialProgrammingTest(unittest.TestCase):
  """Basic tests for ODEBackpropProblem differential programming."""

  def test_obp_dp(self):
    """Tests that ODEBackpropProblem.dp_backprop() agrees with .backprop()."""
    #
    t0_t1 = (0, 2.5)
    y0 = [0.99, 0.01, 0]
    obp = ode_hessian.ODEBackpropProblem(
        dim_y=3,
        tf_dy_dt=_tf_sir_dy_dt,
        tf_L_y0y1=_tf_sir_loss)
    dp_backprop = obp.dp_backprop(
        y0, t0_t1,
        odeint_kwargs=dict(rtol=1e-12, atol=1e-12),
        include_hessian_d2ydot_dy2_term=True)
    y1_via_dp, loss_via_dp, grad_via_dp, hessian_via_dp = dp_backprop
    backprop = obp.backprop(y0, t0_t1, want_order=2,
                            odeint_kwargs=dict(rtol=1e-12, atol=1e-12))
    y1, loss, grad, hessian = backprop
    self.assertTrue(
        numpy.allclose(y1, y1_via_dp, rtol=1e-3, atol=1e-3))
    self.assertTrue(
        numpy.allclose(loss, loss_via_dp, rtol=1e-3, atol=1e-3))
    self.assertTrue(
        numpy.allclose(grad, grad_via_dp, rtol=1e-3, atol=1e-3))
    self.assertTrue(
        numpy.allclose(hessian, hessian_via_dp, rtol=1e-3, atol=1e-3))

  def test_obp_dp_d2ydot_dy2(self):
    """Shows that ODEBackpropProblem.dp_backprop() needs the s_i F_i,kl term."""
    #
    t0_t1 = (0, 2.5)
    y0 = [0.99, 0.01, 0]
    obp = ode_hessian.ODEBackpropProblem(
        dim_y=3,
        tf_dy_dt=_tf_sir_dy_dt,
        tf_L_y0y1=_tf_sir_loss)
    dp_backprop = obp.dp_backprop(
        y0, t0_t1,
        odeint_kwargs=dict(rtol=1e-12, atol=1e-12),
        include_hessian_d2ydot_dy2_term=False)
    y1_via_dp, loss_via_dp, grad_via_dp, hessian_via_dp = dp_backprop
    backprop = obp.backprop(y0, t0_t1, want_order=2,
                            odeint_kwargs=dict(rtol=1e-12, atol=1e-12))
    y1, loss, grad, hessian = backprop
    self.assertTrue(
        numpy.allclose(y1, y1_via_dp, rtol=1e-3, atol=1e-3))
    self.assertTrue(
        numpy.allclose(loss, loss_via_dp, rtol=1e-3, atol=1e-3))
    self.assertTrue(
        numpy.allclose(grad, grad_via_dp, rtol=1e-3, atol=1e-3))
    self.assertFalse(
        numpy.allclose(hessian, hessian_via_dp, rtol=1e-3, atol=1e-3))


if __name__ == '__main__':
  unittest.main()
