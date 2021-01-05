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

"""General utility functions for M-Theory investigations."""

import numpy
import scipy.optimize
import tensorflow as tf


def get_symmetric_traceless_basis(n):
  """Computes a basis for symmetric-traceless matrices."""
  num_matrices = n * (n + 1) // 2 - 1
  # Basis for symmetric-traceless n x n matrices.
  b = numpy.zeros([num_matrices, n, n])
  # First (n-1) matrices are diag(1, -1, 0, ...), diag(0, 1, -1, 0, ...).
  # These are not orthogonal to one another.
  for k in range(n - 1):
    b[k, k, k] = 1
    b[k, k + 1, k + 1] = -1
  i = n - 1
  for j in range(n):
    for k in range(j + 1, n):
      b[i, j, k] = b[i, k, j] = 1
      i += 1
  return b


def tf_grad(t_scalar_func):
  """Maps a TF scalar-function to its TF gradient-function."""
  def f_grad(t_pos):
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_pos)
      t_val = t_scalar_func(t_pos)
    grad = tape.gradient(t_val, t_pos)
    assert grad is not None, '`None` gradient.'
    return grad
  return f_grad


def tf_stationarity(t_scalar_func):
  """Maps a TF scalar-function to its TF gradient-length-squared function."""
  def f_stat(t_pos):
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_pos)
      t_val = t_scalar_func(t_pos)
    grad = tape.gradient(t_val, t_pos)
    assert grad is not None, '`None` gradient (for stationarity).'
    return tf.reduce_sum(tf.math.square(grad))
  return f_stat


def tf_jacobian(t_vec_func):
  """Maps a TF vector-function to its TF Jacobian-function."""
  def f_jac(t_pos):
    tape = tf.GradientTape(persistent=True)
    with tape:
      tape.watch(t_pos)
      v_components = tf.unstack(t_vec_func(t_pos))
    gradients = [tape.gradient(v_component, t_pos)
                 for v_component in v_components]
    assert all(g is not None for g in gradients), 'Bad Gradients for Jacobian.'
    jacobian = tf.stack(gradients, axis=1)
    return jacobian
  return f_jac


def tf_hessian(t_scalar_func):
  """Maps a TF scalar-function to its TF hessian-function."""
  t_grad = tf_grad(t_scalar_func)
  return tf_jacobian(tf_grad(t_scalar_func))


def tf_mdnewton_step(tf_vec_func, t_pos, tf_jacobian_func=None):
  """Performs a MDNewton-iteration step.

  Args:
    tf_vec_func: A R^m -> R^m TF-tensor-to-TF-tensor function.
    t_pos: A R^m position TF-tensor.
    tf_jacobian_func: Optional Jacobian-function (if available).
  Returns:
    A pair (tf_tensor_new_pos, residual_magnitude)
  """
  if tf_jacobian_func is None:
    tf_jacobian_func = tf_jacobian(t_vec_func)
  jacobian = tf_jacobian_func(t_pos)
  residual = tf_vec_func(t_pos)
  update = tf.linalg.lstsq(tf_jacobian_func(t_pos),
                           residual[:, tf.newaxis],
                           fast=False)
  return t_pos - update[:, 0], tf.reduce_sum(tf.abs(residual))


def tf_mdnewton(tf_scalar_func,
                t_pos,
                tf_grad_func=None,
                tf_jacobian_func=None,
                maxsteps=None,
                debug_func=print):
  if tf_grad_func is None:
    tf_grad_func = tf_grad(tf_scalar_func)
  if tf_jacobian_func is None:
    tf_jacobian_func = tf_jacobian(tf_grad_func)
  num_step = 0
  last_residual = numpy.inf
  while True:
    num_step += 1
    t_pos_next, t_residual = tf_mdnewton_step(tf_grad_func, t_pos,
                                              tf_jacobian_func=tf_jacobian_func)
    residual = t_residual.numpy()
    if residual > last_residual or (maxsteps is not None
                                    and num_step >= maxsteps):
      yield t_pos  # Make sure we do yield the position before stopping.
      return
    t_pos = t_pos_next
    last_residual = residual
    if debug_func is not None:
      debug_func('[MDNewton step=%d] val=%s' % (
          num_step,
          tf_scalar_func(t_pos).numpy()))
    yield t_pos


def _fixed_fmin_bfgs(f_opt, x0, **kwargs):
  # Fixes a wart in scipy.optimize.fmin_bfgs() behavior.
  last_seen = [(numpy.inf, None)]
  def f_opt_wrapped(xs):
    val_opt = f_opt(xs)
    if last_seen[0][0] > val_opt:
      last_seen[0] = (val_opt, xs.copy())
    return val_opt
  #
  ret = scipy.optimize.fmin_bfgs(f_opt_wrapped, x0, **kwargs)
  # Always return the smallest value encountered during minimization,
  # not the actual result from fmin_bfgs()
  if kwargs.get('full_output'):
    return (last_seen[0][1],) + ret[1:]
  return last_seen[0][1]


def tf_minimize(tf_scalar_func, x0,
                tf_grad_func=None,
                precise=False,
                optimize_imprecise_first=True,
                dtype=tf.float64,
                gtol=1e-5, maxiter=10**4, mdnewton_maxsteps=7):
  """Minimizes a TensorFlow function."""
  if tf_grad_func is None:
    tf_grad_func = tf_grad(tf_scalar_func)
  def f_opt(params):
    return tf_scalar_func(tf.constant(params, dtype=dtype)).numpy()
  def fprime_opt(params):
    return tf_grad_func(tf.constant(params, dtype=dtype)).numpy()
  if not optimize_imprecise_first:
    opt_xs = x0
  else:
    opt_info = _fixed_fmin_bfgs(f_opt,
                                numpy.array(x0),
                                fprime=fprime_opt,
                                gtol=gtol,
                                maxiter=maxiter,
                                disp=0,
                                full_output=True)
    # TODO(tfish): Check full output for convergence.
    # Not much of a problem currently, since we are always
    # checking stationarity.
    opt_xs = opt_info[0]
  if not precise:
    return f_opt(opt_xs), opt_xs
  *_, t_ret_xs = tf_mdnewton(
    tf_scalar_func,
    tf.constant(opt_xs, dtype=tf.float64),
    tf_grad_func=tf_grad_func,
    maxsteps=mdnewton_maxsteps)
  ret_xs = t_ret_xs.numpy()
  return f_opt(ret_xs), ret_xs
