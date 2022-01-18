# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Potential and Stationarity computation: tensorflow interface.

This module allows running the potential computation and also discovery
on GPU hardware.
"""


import numpy
import scipy.optimize
import tensorflow as tf

from dim4.so8_supergravity_extrema.code import algebra
from dim4.so8_supergravity_extrema.code import scalar_sector


_evaluator = [None]


def get_tf_scalar_evaluator():
  if _evaluator[0] is not None:
    return _evaluator[0]
  _evaluator[0] = scalar_sector.get_scalar_manifold_evaluator(
    to_scaled_constant=(
      lambda x, scale=1: tf.constant(numpy.array(x) * scale)),
    expm=tf.linalg.expm,
    einsum=tf.einsum,
    eye=lambda n: tf.constant(numpy.eye(n), dtype=tf.complex128),
    trace=tf.linalg.trace,
    concatenate=lambda ts: tf.concat(ts, 0),
    complexify=lambda a: tf.cast(a, tf.complex128),
    re=tf.math.real,
    im=tf.math.imag,
    conjugate=tf.math.conj)
  return _evaluator[0]


def S_id(v):
  """Fingerprints a potential value to a string identifier."""
  # This mostly "trims" the cosmological constant (so we can
  # always disambiguate by just adding more digits and keeping
  # substring identity), but nevertheless makes e.g.
  # 11.99999999999972 ==> S1200000.
  return 'S{:07d}'.format(int(-v * 1e5 + 1e-4))


def combine_loss(t_stationarity, t_rpow_factor, t_susy_contrib):
  return tf.math.asinh(t_stationarity + t_susy_contrib) * t_rpow_factor


def scan(
    output_filename,
    seed=0, scale=0.1,
    maxiter=1000,
    rpow=None,
    susy_regulator=None,
    combine_loss_func=combine_loss,
    stationarity_threshold=1e-7):
  """Obtains a basic TensorFlow-based scanner for extremal points."""
  tf_scalar_evaluator = get_tf_scalar_evaluator()
  tc0 = tf.constant(0.0, dtype=tf.float64)
  tc1 = tf.constant(1.0, dtype=tf.float64)
  tc_inv_neg6 = tf.constant(-1 / 6.0, dtype=tf.float64)
  #
  if rpow is not None:
    tc_onb = tf.constant(algebra.e7.v70_from_v70o, dtype=tf.float64)
  else:
    tc_onb = None
  def get_loss(t_v70, t_pot, t_stat, t_a1):
    if rpow is None:
      t_rpow_factor = tc1
    else:
      t_rpow_factor = tf.pow(
        tf.reduce_sum(tf.math.square(tf.einsum('ab,b->a', tc_onb, t_v70))),
        rpow)
    if susy_regulator is None:
      t_susy_contrib = tc0
    else:
      t_a1_sq = tf.einsum('ij,ik->jk', t_a1, tf.math.conj(t_a1))
      # We want this matrix to have an eigenvector (1, 0, ..., 0)
      # with eigenvalue V / 6g^2.
      t_delta = (
        t_a1_sq[:, 0] - tf.complex(tf.pad(tf.reshape(t_pot * tc_inv_neg6, (1,)), [(0, 7)]),
                                   tc0))
      t_susy_contrib = (
        susy_regulator * tf.math.real(
            tf.reduce_sum(t_delta * tf.math.conj(t_delta))))
    return combine_loss_func(t_stat, t_rpow_factor, t_susy_contrib)
  #
  def tf_f_info(t_v70):
    return tf_scalar_evaluator(
      tf.complex(t_v70, tf.zeros(70, dtype=tf.float64)))
  #
  def f_loss(v70):
    t_v70 = tf.constant(v70, dtype=tf.float64)
    info = tf_f_info(t_v70)
    loss = get_loss(t_v70, info.potential, info.stationarity, info.a1).numpy()
    return loss
  #
  def fprime_loss(v70):
    t_v70 = tf.constant(v70, dtype=tf.float64)
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_v70)
      info = tf_f_info(t_v70)
      loss = get_loss(t_v70, info.potential, info.stationarity, info.a1)
    grad = tape.gradient(
      loss,
      t_v70)
    return grad.numpy()
  #
  rng = numpy.random.RandomState(seed)
  while True:
    v70 = rng.normal(scale=scale, size=[70])
    opt_v70 = scipy.optimize.fmin_bfgs(
      f_loss, v70, fprime=fprime_loss, maxiter=maxiter,
      disp=0)
    info = tf_f_info(tf.constant(opt_v70, dtype=tf.float64))
    opt_stat = info.stationarity.numpy()
    if not opt_stat <= stationarity_threshold:
      # Optimization did not produce near-zero stationarity-violation.
      print(
          'Stationarity-minimum at non-critical point, stat=%.3g, pot=%.5f' % (
              opt_stat, info.potential.numpy()))
      continue
    opt_pot = info.potential.numpy()
    if output_filename is not None:
      with open(output_filename, 'at') as h:
        h.write('%.12g,%.12g,%s\n' %
                (opt_pot, opt_stat, ','.join(map(repr, opt_v70))))
    yield opt_pot, opt_stat, opt_v70
