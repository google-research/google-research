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

"""Backpropagating Hessians through ODEs."""

import functools
import numbers
import time
from typing import Callable, Optional

import numpy
import numpy.typing
import scipy.integrate
import scipy.optimize
import tensorflow as tf


# Internal variable names need to use terse abbreviations that
# closely follow the (somewhat involved) mathematics.
# pylint:disable=invalid-name

# Pylint is wrong when complaining about 'this is of type type'
# annotations where we precisely mean that. The type `type`
# is not a generic in such situations!
# pylint:disable=g-bare-generic

# Module-private switch to turn off autograph for debugging.
# This then simplifies inspecting tensors, as we can call
# .numpy() on intermediate quantities while in eager mode.
_DEFAULT_USE_TF_FUNCTION = True


def maybe_tf_function(use_tf_function):
  """Returns tf.function if use_tf_function is true, identity otherwise."""
  return tf.function if use_tf_function else lambda f: f


def fd_grad(f, x0, *, eps=1e-7):
  """Computes a gradient via finite-differencing.

  Args:
    f: The ArrayLike -> ArrayLike function to take the gradient of.
    x0: The position at which to take the gradient.
    eps: step size for symmetric differencing.

  Returns:
    The gradient of `f` at `x0`, computed by taking central differences.
    If `f` returns an array-like which as an ndarray would have shape `s`,
    then the shape of this gradient is `s + (len(x0),)`, with the final
    index indicating the coordinate `i` with respect to which the partial
    derivative was computed effectively as:
    `(f(x0 + delta_i]) - f(x0 - delta_i)) / (2 * eps)`, where `delta_i` is
    `numpy.array([eps if k == i else 0] for i in range(len(x0))]`.
  """
  x0 = numpy.asarray(x0)
  if x0.ndim != 1:
    raise ValueError(f'Need 1-index position-vector x0, got shape: {x0.shape}')
  x0_eps_type = type(x0[0] + eps)
  if not isinstance(x0[0], x0_eps_type):
    # If `eps` changes cannot be represented alongside x0-coordinates,
    # adjust the array to have suitable element-type.
    x0 = x0.astype(x0_eps_type)
  dim = x0.size
  f0 = numpy.asarray(f(x0))
  result = numpy.zeros(f0.shape + (dim,), dtype=f0.dtype)
  denominator = 2 * eps
  xpos = numpy.array(x0)
  for num_coord in range(dim):
    xpos[num_coord] = x0[num_coord] + eps
    f_plus = numpy.asarray(f(xpos))
    xpos[num_coord] = x0[num_coord] - eps
    f_minus = numpy.asarray(f(xpos))
    result[Ellipsis, num_coord] = (f_plus - f_minus) / denominator
    xpos[num_coord] = x0[num_coord]
  return result


def fd_hessian(f, x0, *, eps=1e-5):
  """Computes a hessian via iterated-finite-differencing."""
  grad_f = lambda x: fd_grad(f, x, eps=eps)
  return fd_grad(grad_f, x0, eps=eps)


def tf_jacobian(t_vec_func,
                use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a TF vector-valued function to its TF Jacobian-function."""
  # This is here only used to work out the hessian w.r.t. ODE initial-state
  # and final-state coordinates. Computing the costate-equation jacobian
  # with this would be wasteful.
  @maybe_tf_function(use_tf_function)
  def tf_j(t_xs):
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_xs)
      v = t_vec_func(t_xs)
    ret = tape.jacobian(v, t_xs,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return ret
  return tf_j


def tf_jac_vec(
    t_vec_func,
    use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a TF vector-function F to a "(x, sx) -> sx_j Fj,k(x)" function."""
  @maybe_tf_function(use_tf_function)
  def tf_j(t_xs, t_s_xs):
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_xs)
      t_v = t_vec_func(t_xs)
    return tape.gradient(t_v, t_xs,
                         output_gradients=[t_s_xs],
                         unconnected_gradients=tf.UnconnectedGradients.ZERO)
  return tf_j


def tf_jac_vec_v1(
    t_vec_func,
    use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a TF vector-function F to a "(x, sx) -> sx_j Fj,k(x)" function."""
  # See discussion in the accompanying article.
  @maybe_tf_function(use_tf_function)
  def tf_j(t_xs, t_s_xs):
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_xs)
      t_v = tf.tensordot(t_s_xs, t_vec_func(t_xs), axes=1)
    return tape.gradient(t_v, t_xs,
                         unconnected_gradients=tf.UnconnectedGradients.ZERO)
  return tf_j


def tf_grad(tf_f,
            use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a TF scalar-function to its TF gradient-function."""
  @maybe_tf_function(use_tf_function)
  def tf_grad_f(t_ys):
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_ys)
      t_loss = tf_f(t_ys)
    return tape.gradient(t_loss, t_ys,
                         unconnected_gradients=tf.UnconnectedGradients.ZERO)
  return tf_grad_f


def tf_grad_hessian(tf_f,
                    want_hessian=True,
                    use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a TF scalar-function to its `gradient` and `hessian` functions."""
  tf_fprime = tf_grad(tf_f, use_tf_function=use_tf_function)
  return tf_fprime, (tf_jacobian(tf_fprime, use_tf_function=use_tf_function)
                     if want_hessian else None)


def tf_backprop_ode(tf_dy_dt,
                    use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a (y -> dy/dt) ODE to the 'doubled sensitivity-backprop' ODE."""
  tf_jv = tf_jac_vec(tf_dy_dt, use_tf_function=use_tf_function)
  @maybe_tf_function(use_tf_function)
  def tf_back_dyext_dt(yext):
    ys, s_ys = tf.unstack(tf.reshape(yext, (2, -1)))
    return tf.concat([tf_dy_dt(ys), -tf_jv(ys, s_ys)], axis=0)
  return tf_back_dyext_dt


def tf_dp_backprop_ode(tf_dy_dt,
                       dim_y,
                       use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a (y -> dy/dt) ODE to the 'differential programming backprop' ODE.

  The returned callable cannot be used directly as a dy/dt for ODE-integration,
  but needs to be wrapped into a closure that sets the first bool argument
  to `True` if the sigma_i F_i,jk term is to be included, or `False` if not.
  The latter makes sense for backpropagating the hessian at a minimum.

  Args:
    tf_dy_dt: tf.Tensor -> tf.Tensor function that maps a state-space position
      to a velocity.
    dim_y: Dimension of state-space.
    use_tf_function: Whether to @tf.function wrap the result.

  Returns:
    A (bool, tf.Tensor) -> tf.Tensor function which, depending on whether the
    first argument is `True`, includes the sigma_i F_i,jk term in the
    computation of the rate-of-change of the extended state-space vector.
    This extended vector (always, irrespective of how the boolean is set)
    has structure tf.concat([ys, s_ys, tf.reshape(s2_ys, -1)], axis=0),
    where s_ys are the components of the gradient, and s2_ys are the
    components of the hessian.
  """
  tf_jv = tf_jac_vec(tf_dy_dt, use_tf_function=use_tf_function)
  # Jacobian of h_ij F_j, i.e. J_ik := h_ij F_j,k.
  @maybe_tf_function(use_tf_function)
  def tf_hF(t_h_ij, t_ys):
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_ys)
      t_v = tf.linalg.matvec(t_h_ij, tf_dy_dt(t_ys))
    return tape.jacobian(t_v, t_ys,
                         unconnected_gradients=tf.UnconnectedGradients.ZERO)
  # sigma_i F_i,kl:
  @maybe_tf_function(use_tf_function)
  def tf_sF2(t_xs, t_s_xs):
    tape0 = tf.GradientTape()
    with tape0:
      tape0.watch(t_xs)
      tape1 = tf.GradientTape()
      with tape1:
        tape1.watch(t_xs)
        t_sF = tf.tensordot(t_s_xs, tf_dy_dt(t_xs), axes=1)
      t_grad_sF = tape1.gradient(
          t_sF, t_xs,
          unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return tape0.jacobian(
        t_grad_sF, t_xs,
        unconnected_gradients=tf.UnconnectedGradients.ZERO)
  #
  @maybe_tf_function(use_tf_function)
  def tf_back_dyext_dt(include_d2f, yext):
    ys = yext[:dim_y]
    s_ys = yext[dim_y: 2 * dim_y]
    s2_ys = tf.reshape(yext[2 * dim_y:], (dim_y, dim_y))
    ddt_ys = tf_dy_dt(ys)
    ddt_s_ys = tf_jv(ys, s_ys)
    hF = tf_hF(s2_ys, ys)
    ddt_s2_ys_linear = hF + tf.transpose(hF)
    if include_d2f:
      ddt_s2_ys = ddt_s2_ys_linear + tf_sF2(ys, s_ys)
    else:
      ddt_s2_ys = ddt_s2_ys_linear
    return tf.concat([ddt_ys,
                      -ddt_s_ys,
                      -tf.reshape(ddt_s2_ys, (-1,))], axis=0)
  return tf_back_dyext_dt


def scipy_odeint(f_dy_dt,
                 t0_t1,
                 y0,
                 *args,
                 method = None,
                 **kwargs):
  """Wraps scipy's `odeint` to use a (y -> dy/dt) ODE-function.

  Args:
    f_dy_dt: Function mapping a state-vector numpy.ndarray y to its
      "velocity" dy/dt.
    t0_t1: Pair of `(starting_time, final_time)`.
    y0: State-vector at `starting_time`.
    *args: Further arguments, forwarded to the scipy ode-integrator.
    method: If `None`, this function internally uses `scipy.integrate.odeint()`
     for ODE-integration. Otherwise, it uses `scipy.integrate.solve_ivp()`,
     forwarding the `method`-parameter.
    **kwargs: Further keyword arguments, forwarded to the scipy ode-integrator.

  Returns:
    numpy.ndarray, state-vector at `final_time`.
  """
  def f_wrapped(t, y):
    del t  # Unused.
    return f_dy_dt(y)
  if method is None:  # Use .odeint()
    kwargs['tfirst'] = True
    kwargs['full_output'] = False
    ret = scipy.integrate.odeint(f_wrapped, y0, t0_t1, *args, **kwargs)
    return ret[1, :]
  else:  # Use .solve_ivp()
    ret = scipy.integrate.solve_ivp(f_wrapped, t0_t1, y0, method=method,
                                    *args, **kwargs)
    return ret.y[:, -1]


def _check_and_symmetrize(m, opt_rtol_atol):
  """Optionally symmetrizes `m`, with tolerance thresholds."""
  if opt_rtol_atol is None:
    return m
  rtol, atol = opt_rtol_atol
  if not numpy.allclose(m, m.T, rtol=rtol, atol=atol):
    return None
  return 0.5 * (m + m.T)


class ODEBackpropProblem:
  """An ODE problem on which we want to do back-propagation.

  Attributes:
    numpy_dy_dt: The {state} -> {rate of change} numpy-array-valued
      function describing the dynamics. Must not be modified.

  """
  # The deeper reason for this to be a class (rather than having just
  # some functions with interfaces similar to
  # scipy.integrate.odeint()) is that we want to use some intermediate
  # quantities, such as TensorFlow-functions that have been wrapped up
  # to hand them to an ODE-solver, across multiple calls. While such
  # quantities are not dynamical state (which is what class instances
  # are typically used for), they are cached optionally-computed
  # properties (generated when needed first), and since they are
  # needed/used across multiple calls, having a class instance to own
  # them and manage lazy creation simplifies the logic. Given that the
  # general structure of this code is rather functional (since we
  # naturally pass around functions describing various aspects of an
  # ODE problem), this is an example for object-oriented and
  # functional design properly complementing one another.

  def __init__(self,
               *,
               dim_y,
               with_timings=False,
               tf_dy_dt,
               tf_L_y0y1,
               tf_dtype = tf.float64,
               use_tf_function = _DEFAULT_USE_TF_FUNCTION):
    """Initializes the instance.

    Args:
      dim_y: state-space dimension.
      with_timings: whether callbacks should record timing measurements.
      tf_dy_dt: tf.Tensor -> tf.Tensor state-space-velocity function,
        as a function of state-space position.
      tf_L_y0y1: (tf.Tensor, tf.Tensor) -> tf.Tensor 'loss' as a function
        of initial and final state-space position.
      tf_dtype: Numerical type to use as dtype= for the calculation.
      use_tf_function: whether to wrap TensorFlow functions into @tf.function.
    """
    self._dim_y = dim_y
    self._tf_dy_dt = tf_dy_dt
    self._tf_L_y0y1 = tf_L_y0y1
    self._tf_dtype = tf_dtype
    self._use_tf_function = use_tf_function
    self._with_timings = with_timings
    # Timings
    self._t_numpy_dy_dt = []
    self._t_numpy_back_ode = []
    self._t_numpy_back2_ode = []
    self._t_numpy_dp_back_ode = []
    #
    if with_timings:
      def numpy_dy_dt(ys):
        t0 = time.monotonic()
        result = tf_dy_dt(tf.constant(ys, dtype=tf_dtype)).numpy()
        self._t_numpy_dy_dt.append(time.monotonic() - t0)
        return result
    else:
      def numpy_dy_dt(ys):
        return tf_dy_dt(tf.constant(ys, dtype=tf_dtype)).numpy()
    self.numpy_dy_dt = numpy_dy_dt
    #
    def tf_L_y01(tf_y01):
      y0, y1 = tf.unstack(tf.reshape(tf_y01, (2, -1)))
      return tf_L_y0y1(y0, y1)
    self._tf_L_y01 = tf_L_y01
    # Gradient and hessian of the loss, as compilation-cached callables:
    self._opt_tf_dL_dys_and_d2L_dys_2 = [None, None]

  @functools.cached_property
  def _loss_has_direct_y0_dependency(self):
    """Checks whether the loss has an explicit dependency on `y0`.

    If the TensorFlow-generated graph sees an unconnected gradient
    when backpropagating from the loss into the `y0` argument,
    this returns `False`.
    """
    tc_y0 = tf.constant([0.0] * self._dim_y, dtype=self._tf_dtype)
    tc_y1 = tf.constant([0.0] * self._dim_y, dtype=self._tf_dtype)
    tape_for_checking_y0_dependency = tf.GradientTape()
    with tape_for_checking_y0_dependency:
      tape_for_checking_y0_dependency.watch(tc_y0)
      t_loss_for_checking_y0_dependency = self._tf_L_y0y1(tc_y0, tc_y1)
      dL_dy0_or_None = (
          tape_for_checking_y0_dependency.gradient(
              t_loss_for_checking_y0_dependency, tc_y0,
              unconnected_gradients=tf.UnconnectedGradients.NONE))
    return dL_dy0_or_None is not None

  @functools.cached_property
  def _tf_back_ode(self):
    """The backpropagation-extended ODE."""
    return tf_backprop_ode(self._tf_dy_dt,
                           use_tf_function=self._use_tf_function)

  @functools.cached_property
  def _numpy_back_ode(self):
    """NumPy wrapper for tf_back_ode()."""
    tf_dtype = self._tf_dtype
    tf_back_ode = self._tf_back_ode.get_concrete_function(
        tf.zeros(2 * self._dim_y, dtype=tf_dtype))
    if self._with_timings:
      def fn_ydot(yexts):
        t0 = time.monotonic()
        result = tf_back_ode(tf.constant(yexts, dtype=tf_dtype)).numpy()
        self._t_numpy_back_ode.append(time.monotonic() - t0)
        return result
    else:
      def fn_ydot(yexts):
        return tf_back_ode(tf.constant(yexts, dtype=tf_dtype)).numpy()
    return fn_ydot

  @functools.cached_property
  def _tf_back2_ode(self):
    """The twice-backpropagation-extended ODE."""
    return tf_backprop_ode(self._tf_back_ode,
                           use_tf_function=self._use_tf_function)

  @functools.cached_property
  def _numpy_back2_ode(self):
    """NumPy wrapper for tf_back2_ode()."""
    tf_dtype = self._tf_dtype
    tf_back2_ode = self._tf_back2_ode.get_concrete_function(
        tf.zeros(4 * self._dim_y, dtype=tf_dtype))
    if self._with_timings:
      def fn_ydot(ye2):
        t0 = time.monotonic()
        result = tf_back2_ode(tf.constant(ye2, dtype=tf_dtype)).numpy()
        self._t_numpy_back2_ode.append(time.monotonic() - t0)
        return result
    else:
      def fn_ydot(ye2):
        return tf_back2_ode(tf.constant(ye2, dtype=tf_dtype)).numpy()
    return fn_ydot

  @functools.cached_property
  def _tf_dp_back_ode(self):
    """The differential-programming backpropagation-extended ODE."""
    return tf_dp_backprop_ode(self._tf_dy_dt,
                              self._dim_y,
                              use_tf_function=self._use_tf_function)

  def _numpy_dp_back_ode(self, include_sF_term):
    """NumPy wrapper for tf_dp_back_ode()."""
    tf_dtype = self._tf_dtype
    tf_dp_back_ode = self._tf_dp_back_ode.get_concrete_function(
        include_sF_term,
        tf.zeros(self._dim_y * (2 + self._dim_y),
                 dtype=tf_dtype))
    if self._with_timings:
      def fn_ydot(ye):
        t0 = time.monotonic()
        result = tf_dp_back_ode(tf.constant(ye, dtype=tf_dtype)).numpy()
        self._t_numpy_dp_back_ode.append(time.monotonic() - t0)
        return result
    else:
      def fn_ydot(ye):
        return tf_dp_back_ode(tf.constant(ye, dtype=tf_dtype)).numpy()
    return fn_ydot

  @functools.cached_property
  def _numpy_dp_back_ode_with_sF_term(self):
    """NumPy wrapper for tf_dp_back_ode(), including the s_i*F_i,jk-term."""
    return self._numpy_dp_back_ode(True)

  @functools.cached_property
  def _numpy_dp_back_ode_without_sF_term(self):
    """NumPy wrapper for tf_dp_back_ode(), without the s_i*F_i,jk-term."""
    return self._numpy_dp_back_ode(False)

  def collect_timers(self):
    """Resets and collects timers."""
    result = (
        self._t_numpy_dy_dt,
        self._t_numpy_back_ode,
        self._t_numpy_back2_ode,
        self._t_numpy_dp_back_ode)
    self._t_numpy_dy_dt = []
    self._t_numpy_back_ode = []
    self._t_numpy_back2_ode = []
    self._t_numpy_dp_back_ode = []
    return result

  def backprop(
      self,
      y0,
      t0_to_t1,
      *,
      odeint = scipy_odeint,
      odeint_args=(),
      odeint_kwargs=(),
      symmetrize_hessian_rtol_atol=None,
      want_order = 2,
      use_reconstructed_y0=False):
    """Computes the loss-value, and optionally its gradient and hessian.

    Args:
      y0: array-like real vector, the starting position.
      t0_to_t1: array-like real vector, initial and final time.
      odeint: Callable to use for ODE-integration.
        Must have same calling signature as the default,
        which is `scipy_odeint`.
      odeint_args: Extra arguments to provide to `odeint` for
        each ODE-integration.
      odeint_kwargs: Extra keyword arguments to provide to `odeint` for
        each ODE-integration, will get converted to dict().
      symmetrize_hessian_rtol_atol: Optional pair `(rtol, atol)`. If absent,
        the un-symmetrized hessian will be returned. If present,
        the discrepancy between the computed hessian and its transpose
        is checked against these relative/absolute tolerance thresholds,
        and the symmetrized hessian is returned.
      want_order: If 0, only the function's value will be computed.
        If 1, the gradient will also be computed and returned.
        If 2, the hessian will be added.
      use_reconstructed_y0: Whether backpropagation-of-backpropagation
        should use the numerically-noisy reconstructed starting position y0
        from the first backpropagation. This parameter only exists to
        gauge the impact of this improvement vs. what autogenerated
        code that does not know about this optimization would do.

    Returns:
      A 4-tuple `(y1, val_loss, opt_grad, opt_hessian)` of numerical data,
      where `y1` is the ODE-integration final-state vector as a numpy.ndarray,
      `val_loss` is the corresponding value of the loss-function,
      `opt_grad` is `None` or the gradient as a numpy.ndarray,
      depending on `want_order`, and `opt_hessian` is `None` or the hessian
      as a numpy.ndarray, depending on `want_order`.

    Raises:
      ValueError, if symmetrization was asked for and the Hessian's degree
      of asymmetry violates thresholds.
    """
    if not 0 <= want_order <= 2:
      raise ValueError(f'Invalid {want_order=}')
    y0 = numpy.asarray(y0, dtype=self._tf_dtype.as_numpy_dtype())
    if y0.size != self._dim_y:
      raise ValueError(
          f'Expected y0 to have shape [{self._dim_y}], got: {list(y0.shape)}')
    t0_to_t1 = numpy.asarray(t0_to_t1)
    odeint_kwargs = dict(odeint_kwargs)
    dim = y0.size
    dim_zeros = numpy.zeros_like(y0)
    #
    # Forward-propagate ODE.
    #
    y1 = odeint(self.numpy_dy_dt, t0_to_t1, y0, *odeint_args, **odeint_kwargs)
    tc_y0 = tf.constant(y0, dtype=self._tf_dtype)
    tc_y1 = tf.constant(y1, dtype=self._tf_dtype)
    T_val = self._tf_L_y0y1(tc_y0, tc_y1).numpy()
    if want_order == 0:
      # We only want the function value.
      return y1, T_val, None, None
    #
    # Backprop ODE to get d {loss T} / d {y0}.
    #
    tc_y01 = tf.concat([tc_y0, tc_y1], axis=0)
    # Let TensorFlow work out the gradient and hessian w.r.t. inputs
    # of the loss-function, or take cached callables if this was already
    # done. Note that if we ask for a hessian and an earlier calculation
    # done on the same instance compiled the gradient-function only,
    # we redo that compilation. This little waste of effort is generally benign.
    opt_tf_dL_dys_and_d2L_dys_2 = self._opt_tf_dL_dys_and_d2L_dys_2
    if want_order >= 2 and opt_tf_dL_dys_and_d2L_dys_2[1] is None:
      opt_tf_dL_dys_and_d2L_dys_2[:] = tf_grad_hessian(self._tf_L_y01,
                                                       want_hessian=True)
    elif want_order >= 1 and opt_tf_dL_dys_and_d2L_dys_2[0] is None:
      opt_tf_dL_dys_and_d2L_dys_2[:] = tf_grad_hessian(self._tf_L_y01,
                                                       want_hessian=False)
    # pylint:disable=not-callable
    # We do know that these actually are callable now.
    opt_tf_dL_dys, opt_tf_d2L_dys_2 = opt_tf_dL_dys_and_d2L_dys_2
    sL_y01 = opt_tf_dL_dys(tc_y01).numpy()
    # From here on, variable naming and also tags `F{n}` align
    # with the text of the paper.
    s_T_y_start, s_T_y_final = sL_y01.reshape(2, -1)  # F3, F4
    obp1_start = numpy.concatenate([y1, s_T_y_final], axis=0)  # F5
    # The full vector for step F7 in the paper.
    # Retaining this allows us to do experiments that illustrate
    # the impact on result-quality of starting the 2nd
    # ODE-backpropagation from an unnecessarily noisy starting point.
    s_T_y_start_via_y_final_full = odeint(  # F6
        self._numpy_back_ode,
        t0_to_t1[::-1],  # Reversed!
        obp1_start,
        *odeint_args, **odeint_kwargs)
    s_T_y_start_via_y_final = s_T_y_start_via_y_final_full[dim:]
    s_T_y_start_total = s_T_y_start_via_y_final + s_T_y_start
    if want_order == 1:
      # We only want the function value and sensitivity w.r.t. y0.
      return y1, T_val, s_T_y_start_total, None
    #
    # Start of the actual backpropagation of the Hessian.
    #
    # The paper discusses computing an individual
    # row of the hessian. Here, we then have to assemble these rows
    # into a matrix.
    #
    result_hessian = numpy.zeros([dim, dim], dtype=y0.dtype)
    d2L_dys_2 = opt_tf_d2L_dys_2(tc_y01).numpy()
    T00 = d2L_dys_2[:dim, :dim]
    T01 = d2L_dys_2[:dim, dim:]
    T11 = d2L_dys_2[dim:, dim:]
    #
    if use_reconstructed_y0:
      y0_for_back2 = s_T_y_start_via_y_final_full[:dim]
    else:
      y0_for_back2 = y0
    for row_j in range(dim):
      onehot_j = numpy.arange(dim) == row_j
      zj_obp1_start = odeint(
          self._numpy_back2_ode,
          t0_to_t1,  # Reversed-reversed
          numpy.concatenate(
              [y0_for_back2,
               s_T_y_start_via_y_final,
               dim_zeros,
               onehot_j],
              axis=0),
          *odeint_args, **odeint_kwargs)[2*dim:]
      zj_s_T_y_final = zj_obp1_start[dim:]
      zj_y_final = (
          zj_obp1_start[:dim] +
          # Paper:
          ## # from F4
          ## &es(T11(pos0[:]=y_start[:], pos1[:]=y_final[:]) @ a, b;
          ##     z_s_T_y_final[:] @ b -> a) +
          # Python variant:
          T11.dot(zj_s_T_y_final) +
          # Paper:
          ## # from F3
          ## &es(T01(pos0[:]=y_start[:], pos1[:]=y_final[:]) @ a, b;
          ##     z_s_T_y_start[:] @ b -> a))
          # Python variant, using `zj_s_T_y_start = onehot_j`:
          T01[row_j, :])
      result_hessian[row_j, :] = (
          # Paper:
          ## # from F3
          ## &es(T00(pos0[:]=y_start[:], pos1[:]=y_final[:]) @ a, b;
          ##     z_s_T_y_start @ a -> b) +
          # Python variant, using `zj_s_T_y_start = onehot_j`:
          T00[row_j, :] +
          # Paper:
          ## # from F4
          ## &es(T01(pos0[:]=y_start[:], pos1[:]=y_final[:]) @ a, b;
          ##      z_s_T_y_start @ a -> b) +
          # Python variant:
          T01.dot(zj_s_T_y_final) +
          # Paper: ODE(...)
          odeint(
              self._numpy_back_ode,
              t0_to_t1[::-1],  # Reversed!
              numpy.concatenate([y1, zj_y_final], axis=0),
              *odeint_args, **odeint_kwargs)[dim:])
    opt_symmetrized_hessian = _check_and_symmetrize(
        result_hessian, symmetrize_hessian_rtol_atol)
    if opt_symmetrized_hessian is None:
      raise ValueError('Hessian violates symmetry expectations '
                       '(likely due to numerical noise).')
    return (y1, T_val,
            s_T_y_start_total,
            opt_symmetrized_hessian)

  def dp_backprop(self,
                  y0,
                  t0_to_t1,
                  *,
                  odeint = scipy_odeint,
                  odeint_args=(),
                  odeint_kwargs=(),
                  include_hessian_d2ydot_dy2_term = True):
    """ODE-backpropagates a hessian via 'differential programming'.

    Args:
      y0: array-like real vector, the starting position.
      t0_to_t1: array-like real vector, initial and final time.
      odeint: Callable to use for ODE-integration.
        Must have same calling signature as the default,
        which is `scipy_odeint`.
      odeint_args: Extra arguments to provide to `odeint` for
        each ODE-integration.
      odeint_kwargs: Extra keyword arguments to provide to `odeint` for
        each ODE-integration, will get converted to dict().
      include_hessian_d2ydot_dy2_term: whether the s_i F_i,kl term should
        be included in the calculation. When backpropagating
        a hessian around a critical point, the F_i,kl factor
        multiplies a zero, and so we can skip this computation.
    Returns:
      A 4-tuple `(y1, val_loss, grad, hessian)` of numerical data,
      where `y1` is the ODE-integration final-state vector as a numpy.ndarray,
      `val_loss` is the corresponding value of the loss-function,
      `grad` is the gradient as a numpy.ndarray, and
      `opt_hessian` is the hessian as a numpy.ndarray.

    Raises:
      NotImplementedError, if the loss has an actual dependency on the
      initial-state. The version of this code accompanying the publication
      illustrates the value of having a formalism to keep track of
      complicated dependencies via the generic `backprop` method, but
      tries to keep the corresponding complication out of the discussion of
      dynamic 'differential programming'.
    """
    if self._loss_has_direct_y0_dependency:
      raise NotImplementedError('Loss has a dependency on initial-state.')
    t0_to_t1 = numpy.asarray(t0_to_t1)
    odeint_kwargs = dict(odeint_kwargs)
    dim = self._dim_y
    y1 = odeint(self.numpy_dy_dt, t0_to_t1, y0, *odeint_args, **odeint_kwargs)
    tc_y01 = tf.concat([tf.constant(y0, dtype=self._tf_dtype),
                        tf.constant(y1, dtype=self._tf_dtype)], axis=0)
    loss = self._tf_L_y01(tc_y01).numpy()
    opt_tf_dL_dys_and_d2L_dys_2 = self._opt_tf_dL_dys_and_d2L_dys_2
    if opt_tf_dL_dys_and_d2L_dys_2[1] is None:
      opt_tf_dL_dys_and_d2L_dys_2[:] = tf_grad_hessian(self._tf_L_y01,
                                                       want_hessian=True)
    # pylint:disable=not-callable
    # We do know that these actually are callable now.
    opt_tf_dL_dys, opt_tf_d2L_dys_2 = opt_tf_dL_dys_and_d2L_dys_2
    d2L_dys_2 = opt_tf_d2L_dys_2(tc_y01).numpy()
    T1 = opt_tf_dL_dys(tc_y01).numpy()[dim:]
    T11 = d2L_dys_2[dim:, dim:]
    y0_ext_re = odeint(
        self._numpy_dp_back_ode_with_sF_term if include_hessian_d2ydot_dy2_term
        else self._numpy_dp_back_ode_without_sF_term,
        t0_to_t1[::-1],
        numpy.concatenate([y1, T1, T11.ravel()], axis=0),
        *odeint_args, **odeint_kwargs)
    return (y1,
            loss,
            y0_ext_re[dim : 2 * dim],
            y0_ext_re[2 * dim:].reshape(dim, dim))
