# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Alternative complex matrix exponentiation.

Provides a basic complex matrix exponentiation function `cexpm` for TensorFlow.

Complex Matrix Exponentiation that allows holomorphic backpropagation.
We need this here because TensorFlow <=1.13's tf.linalg.expm() does not
support taking Hessians. (In newer versions, it does support taking
gradients.)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import operator
import tensorflow as tf


def _get_taylor_strategy(n_max, eye, m, prod=operator.mul):
  """Finds out how to build x**N with low depth, given all the lower powers."""
  depth_and_tensor_power_by_exponent = [None] * (n_max + 1)
  depth_and_tensor_power_by_exponent[0] = (0, eye)
  depth_and_tensor_power_by_exponent[1] = (0, m)
  for n in range(2, n_max + 1):
    best_depth, best_k = min(
        (1 + max(depth_and_tensor_power_by_exponent[k][0],
                 depth_and_tensor_power_by_exponent[n - k][0]),
         k) for k in range(1, n))
    depth_and_tensor_power_by_exponent[n] = (
        best_depth, prod(depth_and_tensor_power_by_exponent[best_k][1],
                         depth_and_tensor_power_by_exponent[n - best_k][1]))
  return depth_and_tensor_power_by_exponent


def _complex_matmul(p, q):
  """Implements complex matrix multiplication using 'C = R + R' embedding."""
  return tf.stack([
      # {real1} * {real2} - {imag1} * {imag2}
      tf.matmul(p[0, :, :], q[0, :, :]) - tf.matmul(p[1, :, :], q[1, :, :]),
      # {real1} * {imag2} + {imag1} * {real2}
      tf.matmul(p[0, :, :], q[1, :, :]) + tf.matmul(p[1, :, :], q[0, :, :])])


def tf_shallow_expm_taylor(t_m, n_max=40, name_scope='expm'):
  """Computes Taylor polynomial for matrix exponentiation via shallow graph."""
  shape = t_m.shape.as_list()
  dim = shape[-1]
  if len(shape) == 3:
    if shape[0] != 2 or shape[1] != shape[2]:
      # Leading index must be for choosing between real/imaginary coefficients.
      raise ValueError(
          'Complex matrix must be shape [2, N, N], observed: {}'.format(shape))
    eye = numpy.stack([numpy.eye(dim, dtype=numpy.float64),
                       numpy.zeros([dim, dim], dtype=numpy.float64)])
    fn_product = _complex_matmul
  else:
    eye = numpy.eye(dim, dtype=numpy.float64)
    fn_product = tf.matmul
  #
  factorials = [1] * (n_max + 1)
  for n in range(2, n_max + 1):
    factorials[n] = n * factorials[n - 1]
  with tf.name_scope(name='', values=[t_m]) as scope:
    factorials_factors = tf.constant([1.0 / v for v in factorials],
                                     dtype=tf.float64)
    taylor_strategy = _get_taylor_strategy(n_max,
                                           tf.constant(eye),
                                           t_m, prod=fn_product)
    return tf.einsum('c,cimn->imn' if len(shape) == 3 else 'c,cmn->mn',
                     factorials_factors,
                     tf.stack([m for _, m in taylor_strategy]))


def _c64(x):
  """Wraps up a float64 constant-array as a TF constant."""
  return tf.constant(x, tf.float64)


def _get_num_squarings(t_m):
  """Computes the number of squarings to use for exponentiating a tensor."""
  shape = t_m.shape.as_list()
  dim = shape[0]
  t_l2 = (tf.einsum('iab,iab->', t_m, t_m)  # Complex t_m.
          if len(shape) == 3
          else tf.einsum('ab,ab->', t_m, t_m))  # Real t_m.
  return tf.maximum(
      _c64(0.0), tf.ceil(tf.log(t_l2) / numpy.log(2)))


def _get_squaring_cascade(t_num_squarings, t_m, prod=operator.mul,
                         max_squarings=100):
  """Gets the TF graph cascade of squaring operations."""
  def get_cascade_tail(t_m_squared_n_times, n):
    if n == max_squarings:
      return t_m_squared_n_times
    def false_fn():
      t_m_squared_n_plus_one_times = prod(t_m_squared_n_times,
                                          t_m_squared_n_times)
      return get_cascade_tail(t_m_squared_n_plus_one_times, n + 1)
    #
    return tf.cond(tf.equal(_c64(n), t_num_squarings),
                   true_fn=lambda: t_m_squared_n_times,
                   false_fn=false_fn)
  return get_cascade_tail(t_m, 0)


def cexpm(t_m_complex, max_squarings=20, taylor_n_max=20, complex_arg=True):
  """Drop-in replacement for tf.linalg.expm(), optionally for complex arg."""
  if complex_arg:
    t_m = tf.stack([tf.real(t_m_complex),
                    tf.imag(t_m_complex)])
  else:
    t_m = t_m_complex
  fn_product = _complex_matmul if complex_arg else tf.matmul
  t_num_squarings = tf.minimum(_c64(max_squarings),
                               _get_num_squarings(t_m))
  t_m_scaled = t_m * tf.pow(_c64(0.5), t_num_squarings)
  exp_t_m_scaled = tf_shallow_expm_taylor(t_m_scaled, n_max=taylor_n_max)
  ret = _get_squaring_cascade(t_num_squarings, exp_t_m_scaled,
                                    prod=fn_product,
                                    max_squarings=max_squarings)
  if complex_arg:
    return tf.complex(ret[0], ret[1])
  else:
    return ret
