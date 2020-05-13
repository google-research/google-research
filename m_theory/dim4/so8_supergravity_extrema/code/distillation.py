# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Distills precision data out of approximate critical point locations."""


import ast  # For ast.literal_eval() only.
import collections
import contextlib
import glob
import itertools
import math
import mpmath
import numpy
import opt_einsum
import os
import pprint
import re
import scipy.linalg
import scipy.optimize
import tensorflow as tf

from dim4.so8_supergravity_extrema.code import algebra
from dim4.so8_supergravity_extrema.code import scalar_sector
from dim4.so8_supergravity_extrema.code import scalar_sector_mpmath
from dim4.so8_supergravity_extrema.code import scalar_sector_tensorflow
from dim4.so8_supergravity_extrema.code import symmetries
from m_theory_lib import m_util


# A 'model' of a critical point is a coordinate description that has been
# rotated in such a way that many entries can be set to zero.
LowDimensionalModel = collections.namedtuple(
    'LowDimensionalModel',
    ['v70_from_params', 'params'])


# Model-parameter ratios that will be automatically recognized.
_NICE_RATIOS = {}
for pq, f in [((p, q), p / float(q))
              for p in range(1, 12) for q in range(1, 9)]:
  if f not in _NICE_RATIOS:
    _NICE_RATIOS[f] = pq


_RE_MPF = re.compile(r'mpf\((?P<number>[^)]+)\)')


def mpfmt(a, num_digits=6):
  """Formats a mpmath float to given number of decimals."""
  log10_a = mpmath.log(abs(a), 10)
  scaling = int(math.floor(log10_a))
  a0 = float(a * mpmath.mpf('10')**(-scaling))
  return "mpf('%se%d')" % (round(a0, num_digits), scaling)


def mpf_zeros(dims):
  """Like numpy.zeros(), but generates an array of mpmath.mpf(0) instead.

  This is a work-around for numpy.zeros(dims, dtype=mpmath.mpf) producing
  a dtype=object array with entries the number zero (which loses information
  about mpmath precision).
  """
  size = 1
  for d in dims:
    size *= d
  return numpy.array([mpmath.mpf(0)] * size, dtype=mpmath.mpf).reshape(*dims)


def v70_from_model(low_dim_model):
  """Computes a coordinate 70-vector from a model."""
  if len(low_dim_model.params) == 0:
    # Fix for numpy not handling [N, 0]-arrays well.
    return mpf_zeros([70])
  return numpy.dot(low_dim_model.v70_from_params, low_dim_model.params)


# === CANONICALIZATION ===


def product_decompose_rotation(rot):
  """Decomposes a rotation matrix into Davenport chained rotations."""
  dim = rot.shape[0]
  factors = []
  work_rot = rot.copy()
  debug_step = 0
  for axis0 in range(dim):
    for axis1 in sorted(range(1 + axis0, dim),
                        # Resolve largest-to-be-zeroed-out
                        # index first.
                        key=lambda n: -abs(work_rot[axis0, n])):
      ea0 = work_rot[axis0, axis0]
      ea1 = work_rot[axis1, axis0]
      angle = math.atan2(ea1, ea0)
      if angle != 0:
        factors.append((axis0, axis1, angle))
      ca = math.cos(angle)
      sa = math.sin(angle)
      row0 = work_rot[axis0].copy()
      row1 = work_rot[axis1].copy()
      work_rot[axis0, :] = +ca * row0 + sa * row1
      work_rot[axis1, :] = -sa * row0 + ca * row1
  return list(reversed(factors))


def resynthesize_rotation_for_rep(output_dim,
                                  input_dim,
                                  decomposed_rotation,
                                  einsum_spec,
                                  rep_action):
  """Builds a rotation from a chain of [i,j]-plane rotations in a given rep."""
  def get_part(ija):
    i, j, angle = ija
    gen = numpy.zeros([input_dim, input_dim])
    gen[i, j] = -angle
    gen[j, i] = +angle
    return scipy.linalg.expm(numpy.einsum(einsum_spec, gen, rep_action))
  partial = numpy.eye(output_dim)
  for num_part, ija in enumerate(decomposed_rotation):
    partial = numpy.einsum('ij,jk->ik', get_part(ija), partial)
  return partial


def get_diagonalizing_rotation(m8x8):
  """Finds a SO(8) rotation that diagonalizes a symmetric 8x8 matrix."""
  eigvals, basis = scipy.linalg.eigh(m8x8)
  eigvecs = basis.T[[i for i, _ in sorted(enumerate(eigvals),
                                          key=lambda iv: -iv[1])]]
  det = numpy.linalg.det(eigvecs)
  assert abs(abs(det) - 1.0) < 1e-6, 'Eigenbasis not orthonormal'
  if det < 0:
    # Make eigenvector basis right-handed by reversing last eigenvector.
    eigvecs[-1] *= -1
  return eigvecs.T


def _diagonalize_half_of_v70(v70, diagonalize_35s=True):
  """Canonicalizes v70-vector by applying a suitable Spin(8) rotation."""
  # Diagonalizes either the 35s or 35c, as requested.
  m_35s = numpy.einsum('Aij,A->ij', algebra.su8.m_35_8_8.real, v70[:35])
  m_35c = numpy.einsum('Aij,A->ij', algebra.su8.m_35_8_8.real, v70[35:])
  rot = get_diagonalizing_rotation(m_35s if diagonalize_35s else m_35c)
  decomposed_rot = product_decompose_rotation(rot)
  resynthesized_rot = resynthesize_rotation_for_rep(
      8, 8, decomposed_rot, 'ab,->ab', numpy.ones([]))
  if not numpy.allclose(rot, resynthesized_rot, rtol=1e-3, atol=1e-5):
    raise ValueError('Resynthesized rotation does not match original rotation.')
  generator_mapping_spec = 'sS,sScC->cC' if diagonalize_35s else 'cC,sScC->sS'
  rep_action = 0.25 * algebra.spin8.gamma_sscc
  rot_other_rep = resynthesize_rotation_for_rep(
          8, 8, decomposed_rot, generator_mapping_spec, rep_action)
  (rot35s, rot35c) = ((rot, rot_other_rep) if diagonalize_35s
                      else (rot_other_rep, rot))
  return (numpy.dot(rot35s.T, numpy.dot(m_35s, rot35s)),
          numpy.dot(rot35c.T, numpy.dot(m_35c, rot35c)))


def _get_residual_symmetry_of_matrix_diagonal(diag_entries, max_deviation=1e-4):
  """Finds all generators that leave a given matrix diagonal invariant."""
  # Max deviation of 1e-4 emperically makes sense for most solutions.
  seen_entries_and_indices = []
  for index, v in enumerate(diag_entries):
    for v0, indices in seen_entries_and_indices:
      if abs(v - v0) < max_deviation:
        indices.append(index)
        break
    else:  # Reached end of seen_entries_and_indices.
      seen_entries_and_indices.append((v, [index]))
  generators = []
  for _, index_group in seen_entries_and_indices:
    generators.extend([(i1, i2)
                       for i1 in index_group for i2 in index_group
                       if i1 < i2])
  return generators


def _get_generators_for_reducing_second_m35(diag_entries_first_m35,
                                            einsum_spec, rep_action):
  """Builds the generators for canonicalizing the 2nd 35-irrep."""
  gens = _get_residual_symmetry_of_matrix_diagonal(diag_entries_first_m35)
  m = numpy.zeros([len(gens), 8, 8])
  for i, (a, b) in enumerate(gens):
    m[i, a, b] = +1
    m[i, b, a] = -1
  return numpy.einsum(einsum_spec, m, rep_action)


def _reduce_second_m35(m35s, m35c, is_diagonal_35s, seed=0):
  """Reduces the 2nd 35-irrep."""
  diag = numpy.diagonal(m35s if is_diagonal_35s else m35c)
  gens = _get_generators_for_reducing_second_m35(
      diag,
      'gsS,sScC->gcC' if is_diagonal_35s else 'gcC,sScC->gsS',
      algebra.spin8.gamma_sscc)
  num_gens = len(gens)
  if num_gens == 0:
    return m35s, m35c  # No residual symmetry to exploit.
  # This residual symmetry is typically rather small.
  # So, doing a direct minimization is perhaps appropriate.
  rng = numpy.random.RandomState(seed=seed)
  v_coeffs_initial = rng.normal(
      scale=1e-3, size=(num_gens,))  # Break symmetry with noise.
  #
  # @tf.function  # TODO(tfish): Activate once compilation is fast.
  def tf_get_m35_rotated(t_coeffs):
    tc_gens = tf.constant(gens, dtype=tf.float64)
    tc_m35 = tf.constant(m35c if is_diagonal_35s else m35s,
                         dtype=tf.float64)
    t_rot = tf.linalg.expm(tf.einsum('i,iab->ab', t_coeffs, tc_gens))
    return tf.einsum(
      'Ab,Bb->AB',
      tf.einsum('ab,Aa->Ab', tc_m35, t_rot), t_rot)
  #
  # @tf.function  # TODO(tfish): Activate once compilation is fast.
  def tf_get_loss(t_coeffs):
    t_m35_rotated = tf_get_m35_rotated(t_coeffs)
    # Our 'loss' is the sum of magnitudes of the off-diagonal parts after
    # rotation.
    return (tf.norm(t_m35_rotated, ord=1) -
            tf.norm(tf.linalg.diag_part(t_m35_rotated), ord=1))
  opt_val, opt_pos = m_util.tf_minimize(
    tf_get_loss, tf.constant(v_coeffs_initial, dtype=tf.float64), precise=False)
  m_diag = tf_get_m35_rotated(tf.constant(opt_pos, dtype=tf.float64)).numpy()
  return (m35s, m_diag) if is_diagonal_35s else (m_diag, m35c)


def canonicalize_v70(v70,
                     do_diagonalize_35s=(True, False),
                     still_good=lambda v70: True):
  """Canonicalizes a 70-vector.

  Finds and applies a suitable SO(8) rotation that minimizes the number of
  entries of the 35s and 35c symmetric traceless matrices.

  Args:
    v70: The vector describing a point on the scalar manifold.
    do_diagonalize_35s: Sequence of values to try for the boolean parameter that
      controls whether to diagonalize 35s or 35c.
    still_good: Function checking whether the canonicalized form
      still describes a physically equivalent point.

  Returns:
    A dict with keys 'diag35s' and 'diag35c', and entries each of the form
    {'35s: <8x8 matrix>, '35c': <8x8 matrix>, 'v70': <reduced 70-vector>,
     'potential': <the potential>, 'stationarity': <stationarity-value>}
    or None (if no longer 'good') that describe the outcome of diagonalizing
    the 35s or 35c.
  """
  ret = {}
  for diagonalize_35s in do_diagonalize_35s:
    m35s0, m35c0 = _diagonalize_half_of_v70(
        v70, diagonalize_35s=diagonalize_35s)
    m35s, m35c = _reduce_second_m35(m35s0, m35c0, diagonalize_35s)
    v70_diag = algebra.e7.v70_from_35s35c(m35s, m35c)
    sinfo = scalar_sector.numpy_scalar_manifold_evaluator(v70_diag)
    if not still_good(v70_diag):
      # We tried to canonicalize but failed - i.e. we changed the solution
      # to something different.
      ret['diag35s' if diagonalize_35s else 'diag35c'] = None
    else:
      ret['diag35s' if diagonalize_35s else 'diag35c'] = {
          '35s': m35s, '35c': m35c, 'v70': v70_diag,
          'potential': sinfo.potential,
          'stationarity': sinfo.stationarity}
  return ret


# === MODELING ===


def _model35(m, registered_coeffs_prev=(), digits=4):
  """Helper for modeling a symmetric traceless matrix.

  This helper scans the entries of a symmetric traceless matrix,
  extracting almost-zero, almost-identical, and
  almost-in-simple-arithmetic-ratio entries. This allows us to build
  low-dimensional models for approximate 70-vectors that describe
  a solution.

  We use two calls to this in succession, one modeling the 35s,
  the other one modeling the 35c. This accumulates coefficients.


  Args:
    m: [8, 8]-array of approximate numerical data. Expected to be a symmetric
      traceless matrix.
    registered_coeffs_prev: List of coefficients extracted for another such
      matrix. Used to build a combined low-dimensional model for the relevant
      degrees of freedom from both the 35s and 35c representations.
    digits: How many decimal digits to take into account when determining
      whether two matrix entries are almost-identical.
  """
  # TODO(tfish): Improve handling of known linear constraints between
  # model-parameters.
  registered_coeffs = list(registered_coeffs_prev)  # Copy.
  threshold = 10**(-digits)
  nice_ratio_low, nice_ratio_high = 1 - 5 * threshold, 1 + 5 * threshold
  entry_by_row_col = {}
  # Scan upper-triagonal part only.
  for row in range(8):
    for col in range(row, 8):
      v = m[row, col]
      if abs(v) >= threshold:
        entry_by_row_col[(row, col)] = v
  # Try to find nice identities.
  # Let us start from the smallest-in-magnitude entries
  # and proceed by magnitude.
  def register_coeff(c):
    """Registers coeff and returns (registered_index, factor)"""
    abs_c = abs(c)
    for num_seen_coeff, d in enumerate(registered_coeffs):
      ratio = abs_c / d
      candidate_ratios = [
          r_pq for r_pq in _NICE_RATIOS.items()
          if nice_ratio_low <= r_pq[0] / ratio <= nice_ratio_high]
      if not candidate_ratios:
        # Observed coefficient cannot be explained with this registered one.
        continue  # with next registered coefficient.
      assert len(candidate_ratios) == 1, 'Expected only one viable ratio.'
      p, q = candidate_ratios[0][1]
      return (num_seen_coeff, (p if c > 0 else -p, q))
    # We did not find an entry that explains this one.
    registered_coeffs.append(abs_c)
    return (len(registered_coeffs) - 1, (1 if c > 0 else -1, 1))
  #
  canonicalized = []
  for row_col, coeff in sorted(entry_by_row_col.items(),
                               # Proceed by increasing magnitude.
                               key=lambda kv: abs(kv[1])):
    canonicalized_id, numer_denom = register_coeff(coeff)
    canonicalized.append((row_col, canonicalized_id, numer_denom))
  return registered_coeffs, canonicalized


def get_low_dimensional_model(v70, digits=4):
  """Builds a few-parameters model for a canonicalized 70-vector."""
  m8x8s, m8x8c = opt_einsum.contract(
      'v,vsab->sab', v70, algebra.e7.v70_as_sc8x8)
  coeffs_m8x8s, model_m8x8s = _model35(m8x8s, digits=digits)
  coeffs_all, model_m8x8c = _model35(
      m8x8c, registered_coeffs_prev=coeffs_m8x8s, digits=digits)
  model_dim = len(coeffs_all)
  m8x8s8x8c_from_model = mpf_zeros([2, 8, 8, model_dim])
  for num_m8x8, m8x8_entries in ((0, model_m8x8s), (1, model_m8x8c)):
    for (row, col), num_model_coeff, (numer, denom) in m8x8_entries:
      if row == col == 7:
        continue  # Skip last diagonal entry.
      m8x8s8x8c_from_model[num_m8x8, row, col, num_model_coeff] = (
          mpmath.mpf(numer) / mpmath.mpf(denom))
      if row != col:  # Add symmetrized entry.
        m8x8s8x8c_from_model[num_m8x8, col, row, num_model_coeff] = (
            m8x8s8x8c_from_model[num_m8x8, row, col, num_model_coeff])
      # Just as we added the symmetrized entries, we also need to
      # make the not-last diagonal entries contribute to the last
      # diagonal entry that is also determined my other matrix entries
      # (due to tracelessness).
      else:  #  Diagonal, but not-last entry.
        m8x8s8x8c_from_model[num_m8x8, 7, 7, num_model_coeff] -= (
            mpmath.mpf(numer) / mpmath.mpf(denom))
  v70_from_params = opt_einsum.contract('vsab,sabm->vm',
                                        algebra.e7.v70_from_sc8x8,
                                        m8x8s8x8c_from_model)
  return LowDimensionalModel(
      v70_from_params=v70_from_params,
      params=numpy.array([mpmath.mpf(x) for x in coeffs_all]))


def find_simple_low_dimensional_model(v70s, min_digits=3,
                                      still_good=lambda v70: True):
  """Iteratively searches for a simple low-dimensional model.

  For a collection of alternative v70s describing the same critical point,
  tries to find a simple linear model that still captures the critical point.

  Args:
    v70s: Sequence of candidate 70-vectors describing the same critical point
      (canonicalized differently).
    still_good: f(v70) -> bool, criterion function determining if a
      reconstructed 70-vector still approximately describes the same solution.

  Yields:
    LowDimensionalModel. Results are filtered by acceptability, and ordered by
      increasing complexity.
  """
  eff_min_digits = min(13, min_digits)
  for digits in (3, 4, 5, 6, 7, eff_min_digits):
    if digits < eff_min_digits:
      continue
    models = sorted([get_low_dimensional_model(v70, digits=digits)
                     for v70 in v70s], key=lambda m: len(m.params))
    models_and_v70s = [(m, numpy.dot(m.v70_from_params, m.params))
                       for m in models]
    for n, (model, mv70) in enumerate(models_and_v70s):
      is_good = still_good(mv70)
      if is_good:
        yield model


def refine_model_nelder_mead(
    low_dimensional_model,
    xtol=1e-7,
    ftol=1e-12,
    step0=0.5**17,
    evaluator=scalar_sector.numpy_scalar_manifold_evaluator,
    # TODO(tfish): Make mpmath-enabled scipy.optimize Nelder-Mead
    # the default.
    fmin=scipy.optimize.fmin):
  """Refines a low-dimensional parameter model via Nelder-Mead optimization.

  Args:

    low_dimensional_model: A LowDimensionalModel object, the model to refine.
    xtol: Coordinate-parameter tolerance threshold.
    ftol: Function value (i.e. stationarity) tolerance threshold.
    step0: Initial step size.
    evaluator: scalar_sector.numpy_scalar_manifold_evaluator or any other such
      evaluator with the same signature.
    fmin: Nelder-Mead minimizing function adhering to the interface of
      scipy.optimize.fmin().
  """
  if len(low_dimensional_model.params) == 0:
    return low_dimensional_model
  nn = [0]
  best = [numpy.inf, None]
  def f_off(model_vec):
    nn[0] += 1
    if nn[0] % 20 == 0:
      print('[Nelder-Mead] model-vec', model_vec.dtype, list(model_vec))
    v70 = numpy.dot(low_dimensional_model.v70_from_params,
                    numpy.array([x for x in model_vec], dtype=mpmath.mpf))
    sinfo = evaluator(v70)
    if nn[0] % 100 == 0:
      print('[Nelder-Mead]\n potential %s\n stationarity %s' % (
          sinfo.potential, sinfo.stationarity))
      if not sinfo.stationarity < best[0]:
        raise NoImprovement()
      best[0] = sinfo.stationarity
      best[1] = model_vec
    return sinfo.stationarity
  x0 = low_dimensional_model.params
  initial_simplex = numpy.array(
      [[x - step0 / len(x0) for x in x0]] +
      [[x + step0 * (j == k) for j, x in enumerate(x0)]
       for k in range(len(x0))], dtype=mpmath.mpf)
  try:
    opt = fmin(
      f_off,
      x0,
      initial_simplex=initial_simplex,
      ftol=ftol,
      xtol=xtol,
      maxiter=10**7,
      maxfun=10**7,
      # dtype=mpmath.mpf  # TODO(tfish): Enable `dtype` argument.
    )
  except NoImprovement:
    opt = best[1]
  return LowDimensionalModel(
      v70_from_params=low_dimensional_model.v70_from_params, params=opt)


def refine_model_nelder_mead_mpmath(
    low_dimensional_model,
    xtol=mpmath.mpf('1e-20'),
    ftol=mpmath.mpf('1e-40'),
    step0=mpmath.mpf('1e-8'),
    evaluator=scalar_sector_mpmath.mpmath_scalar_manifold_evaluator,
    fmin=scipy.optimize.fmin):
  """Refines like refine_model_nelder_mead(), but by default uses mpmath."""
  # TODO(tfish): This is not very sensible with a non-mpmath-enabled default
  # `fmin` implementation. Make a mpmath-enabled implementation the default.
  if len(low_dimensional_model.params) == 0:
    return low_dimensional_model
  return refine_model_nelder_mead(
      low_dimensional_model,
      xtol=xtol,
      ftol=ftol,
      step0=step0,
      evaluator=evaluator,
      fmin=fmin)


def refine_model_mdnewton_mpmath(
    low_dimensional_model,
    newton_steps=5,
    expected_quality=(0.1, 0.01, 1e-3, 1e-5, 1e-8, 1e-12, 1e-18, 1e-24),
    log10_stop_quality=-130,
    still_good=lambda v70: True,
    norm=1):
  """Refines a model using mpmath's MDNewton() solver.

  For some solutions, this fails due to the critical point being not a 'generic'
  one.

  Args:
    low_dimensional_model: A LowDimensionalModel.
    still_good: f(v70) -> bool, determines if the solution is still good.
    newton_steps: The number of multidimensional Newton steps to take.
    expected_quality: List of expected quality thresholds for each step.
      (Last entry implicitly repeats.)
    log10_stop_quality: Stop early if this quality-threshold is reached.
    norm: The `norm` parameter for mpmath.calculus.optimization.MDNewton().

  Returns:
    A refined LowDimensionalModel.

  Raises:
    ValueError, if the solution is no longer good.
  """
  if len(low_dimensional_model.params) == 0:
    return low_dimensional_model
  def f_off(*model_vec):
    v70 = numpy.dot(low_dimensional_model.v70_from_params,
                    numpy.array(model_vec, dtype=mpmath.mpf))
    sinfo = scalar_sector_mpmath.mpmath_scalar_manifold_evaluator(v70)
    return tuple(sinfo.grad_potential)
  newton = mpmath.calculus.optimization.MDNewton(
      mpmath.mp, f_off, tuple(low_dimensional_model.params),
      verbose=1,
      norm=lambda x: mpmath.norm(x, norm))
  newton_iter = iter(newton)
  for num_step in range(1, newton_steps + 1):
    opt, stationarity = next(newton_iter)
    opt_v70  = numpy.dot(low_dimensional_model.v70_from_params, opt)
    if mpmath.log(stationarity, 10) <= log10_stop_quality:
      break
    expected_stationarity = expected_quality[
        min(len(expected_quality) - 1, num_step)]
    if stationarity > expected_stationarity:
      raise ValueError(
          'Stationarity does not improve as expected: '
          'seen=%.6g, wanted=%.6g, step=%d' % (
              stationarity, expected_stationarity, num_step))
    if not still_good(opt_v70):
      raise ValueError('Solution is no longer good.')
    print('[MDNewton Step %d, num_params=%d]: stat=%s' % (
      num_step, len(opt), mpfmt(stationarity)))
  return LowDimensionalModel(
      v70_from_params=low_dimensional_model.v70_from_params,
      params=numpy.array(opt, dtype=mpmath.mpf))


class BadStationarity(Exception):
  """A solution unexpectedly violates the stationarity-condition."""

class NoImprovement(Exception):
  """Optimization no longer improves a solution."""


def refine_model_gradient_descent(low_dimensional_model,
                                  log10_stop_quality=-24,
                                  report_on=lambda n: n % 1000 == 0):
  """Refines a low-dimensional model via basic gradient descent."""
  if len(low_dimensional_model.params) == 0:
    return low_dimensional_model
  target_stationarity = 10.0**log10_stop_quality
  v70 = numpy.dot(low_dimensional_model.v70_from_params,
                  numpy.array(low_dimensional_model.params,
                              dtype=mpmath.mpf)).astype(float)
  tf_scalar_evaluator = scalar_sector_tensorflow.get_tf_scalar_evaluator()
  sinfo0 = tf_scalar_evaluator(tf.constant(v70, tf.float64))
  pot0 = sinfo0.potential.numpy()
  def still_good(potential):
    return abs(pot0 - potential) < 1e-4
  def pot_stat_grad(v70):
    tape = tf.GradientTape()
    t_v70 = tf.constant(v70, dtype=tf.float64)
    with tape:
      tape.watch(t_v70)
      sinfo = tf_scalar_evaluator(t_v70)
    return (sinfo.potential.numpy(),
            sinfo.stationarity.numpy(),
            tape.gradient(sinfo.stationarity, t_v70).numpy())
  def do_gradient_steps(num_steps, v70_start,
                        learning_rate=1e-5,
                        max_acceptable_stationarity=numpy.inf,
                        report_on=lambda n: False):
    v70 = v70_start
    for n in range(num_steps):
      n_pot, n_stat, n_grad = pot_stat_grad(v70)
      if n_stat > max_acceptable_stationarity:
        raise BadStationarity()
      if report_on(n):
        print('[GradDesc] %3d: p=%.16g s=%.8g' % (n, n_pot, n_stat))
      v70 -= learning_rate * n_grad
    return n_pot, n_stat, v70
  v70_now = v70
  stat_now = sinfo0.stationarity.numpy()
  learning_rate = 1e-5
  can_increase_learning_rate = True
  while True:
    trial_performances = [(numpy.inf, 0.02)]
    trial_learning_rate_factors = (5.0, 2.0, 1.25, 1.0, 0.8, 0.5, 0.2)
    try:
      for learning_rate_factor in trial_learning_rate_factors:
        if not can_increase_learning_rate and learning_rate_factor > 1:
          continue
        trial_learning_rate = learning_rate * learning_rate_factor
        # Do 10 steps with the trial learning rate.
        pot_stat_pos_log = [
          do_gradient_steps(10, v70_now,
                            learning_rate=trial_learning_rate)]
        # Closely look at what happens over a few more steps.
        for n in range(10):
          pot_stat_pos_log.append(
            do_gradient_steps(1, pot_stat_pos_log[-1][-1],
                              learning_rate=trial_learning_rate))
        if not all(still_good(pot_stat_pos[0])
                   for pot_stat_pos in pot_stat_pos_log):
          continue  # with next learning_rate_factor.
        if not all(psp_prev[1] >= psp[1] for psp_prev, psp in zip(
            pot_stat_pos_log, pot_stat_pos_log[1:])):
          continue  # with next learning_rate_factor.
        trial_performances.append(
          (pot_stat_pos_log[-1][1], learning_rate_factor))
      trial_performances.sort()
      best_factor = trial_performances[0][1]
      learning_rate *= best_factor * 0.9  # Include safety fudge-factor.
      print('[GradDesc] Adjusted learning rate to: %g' % learning_rate)
      pot, stat, v70_next = do_gradient_steps(
        8000, v70_now,
        learning_rate=learning_rate,
        max_acceptable_stationarity=1.1 * stat_now, report_on=report_on)
      if stat <= target_stationarity or learning_rate < 1e-16:
        return pot, stat, v70_next
      if stat < stat_now:
        stat_now = stat
        v70_now = v70_next
        can_increase_learning_rate = True
      else:
        raise BadStationarity()
    except BadStationarity:
      can_increase_learning_rate = False
      "Gradient-descent failed. Reducing learning rate."
      learning_rate *= 0.75


def write_model(h, model, extra_info):
  """Writes a model to a filehandle.

  Args:
    h: Filehandle to write to. Must support .write({string}).
    model: The low-dimensional model to write.
    extra_info: Extra key/value entries to add to the model.
  """
  extended_dict = extra_info.copy()
  extended_dict['v70_from_params'] = v70_from_params_asdict(
      model.v70_from_params)
  extended_dict['params'] = [str(x) for x in model.params]
  h.write(pprint.pformat(extended_dict))


# === DISTILLATION ===

def distill_model(low_dimensional_model,
                  target_digits_position=7,
                  newton_steps=4,  # May be None.
                  skip_gradient_descent=False,
                  still_good=lambda _: True):
  """Distills a low-dimensional model to high precision."""
  xtol = 10**(-target_digits_position)
  ftol = 100 * xtol * xtol
  log10_stop_quality = -(2 * target_digits_position + 2)
  from_mdnewton = False
  # First, find out if MDNewton works for refining this model.
  # If it does, this is excellent.
  try:
    if newton_steps is None:
      print('Skipping MDNewton')
      raise StopIteration()
    print('[Distillation] Trying MDNewton, steps=%d, target_digits_position=%d,'
          ' dps=%d' % (newton_steps, target_digits_position, mpmath.mp.dps))
    refined_model, from_mdnewton = (
        refine_model_mdnewton_mpmath(
            low_dimensional_model,
            still_good=still_good,
            log10_stop_quality=log10_stop_quality,
            newton_steps=newton_steps), True)
  except Exception as exn:
    print('[Distillation] MDNewton failed:', exn)
    try:
      if skip_gradient_descent:
        print('Skipping Gradient Descent')
        raise StopIteration()
      refined_pot_stat_pos, from_mdnewton = (
          refine_model_gradient_descent(
              low_dimensional_model,
              log10_stop_quality=log10_stop_quality), False)
      refined_model = get_low_dimensional_model(
          refined_pot_stat_pos[-1],
          # In this case, i.e. when we had to use gradient-descent,
          # export a complete model, and do not allow making guesses.
          # (As these were problematic in the first place!)
          digits=15)
    except Exception as exn:
      print('[Distillation] Gradient descent failed:', exn)
      # Otherwise, try to refine the model using nelder-mead.
      # (May want to change this to use high-precision mpmath Nelder-Mead.)
      refined_model, from_mdnewton = (
          refine_model_nelder_mead(
              low_dimensional_model, xtol=xtol, ftol=ftol), False)
  refined_v70 = v70_from_model(refined_model)
  sinfo = scalar_sector.numpy_scalar_manifold_evaluator(
      refined_v70.astype(float))
  if sinfo.stationarity < 1e-5:
    return refined_model, from_mdnewton
  return None, False


def distill(v70, target_digits_position=7, newton_steps=4,
            skip_gradient_descent=False,
            min_model_digits=None,
            allowed_forms=('diag35s', 'diag35c')):
  """Distills a raw v70 into high-precision few-parameters form."""
  xtol = 10**(-target_digits_position)
  ftol = 100 * xtol * xtol
  sinfo0 = scalar_sector.numpy_scalar_manifold_evaluator(v70)
  def still_good(v70):
    sinfo = scalar_sector.numpy_scalar_manifold_evaluator(v70)
    return (abs(sinfo0.potential - sinfo.potential) < 1e-3 and
            (sinfo.stationarity < 20 * sinfo0.stationarity or
             sinfo.stationarity < 0.01))
  canonicalized = canonicalize_v70(v70, still_good=still_good)
  canon_v70s = [v['v70'] for k, v in canonicalized.items()
                if v is not None and k in allowed_forms]
  iter_models = find_simple_low_dimensional_model(
      canon_v70s,
      min_digits=min_model_digits or int(
          -math.log(sinfo0.stationarity, 10)) // 2 - 2,
      still_good=still_good)
  from_mdnewton = False
  for num_model, model in enumerate(iter_models):
    refined_model, from_mdnewton = distill_model(
        model,
        target_digits_position=target_digits_position,
        newton_steps=newton_steps,
        skip_gradient_descent=skip_gradient_descent)
    if refined_model is not None:
      return refined_model, from_mdnewton
  return None, False


def v70_from_params_asdict(v70_from_params):
  """Converts a sparse `v70_from_params` array to a dict."""
  dict_v70_from_params = {}
  for i in range(v70_from_params.shape[0]):
    for j in range(v70_from_params.shape[1]):
      v = v70_from_params[i, j]
      if v:
        dict_v70_from_params[(i, j)] = str(v)
  return dict_v70_from_params


def read_distillate(distillate_filename):
  """Reads a distillation result file as dict."""
  with open(distillate_filename, 'r') as h:
    txt = h.read()
    # It may be that there are still some accidental mpf(...) entries
    # in the data which literal_eval cannot process. remove these.
    cleaned_txt = _RE_MPF.sub(lambda m: m.group('number') , txt)
    return ast.literal_eval(cleaned_txt)


def get_distillate_model(distilled_data):
  """Converts the data in a distillation result file to LowDimensionalModel."""
  params = [mpmath.mpf(s) for s in distilled_data['params']]
  v70 = mpf_zeros([70])
  v70_from_params = mpf_zeros([70, len(params)])
  for (j, k), s_coeff in distilled_data['v70_from_params'].items():
    v70_from_params[j, k] = mpmath.mpf(s_coeff)
  return LowDimensionalModel(v70_from_params=v70_from_params,
                             params=params)


def read_distillate_model(distillate_filename):
  """Reads a distillation result file as LowDimensionalModel."""
  return get_distillate_model(read_distillate(distillate_filename))


# First distillation will be only roughly good enough to find the correct
# few-parameters model. Redistill to "purify" to high accuracy.
def redistill(distillate_filename, out_filename, expected_dps=60,
              newton_steps=7,
              min_model_digits=None):
  """Second-distills a distillate."""
  if mpmath.mp.dps < expected_dps:
    raise RuntimeError(
        'Precision setting for mpmath is below the expected dps '
        'for this calculation: %s < %s' % (mpmath.mp.dps, expected_dps))
  # Tries to find out if there are further opportunities that reduce the number
  # of parameters which have been missed in 1st distillation, and if so,
  # performs the corresponding reduction.
  # In any case, adds basic information about physics.
  distillate_model = read_distillate_model(distillate_filename)
  v70 = v70_from_model(distillate_model)
  if expected_dps > 15:
    sinfo0 = scalar_sector_mpmath.mpmath_scalar_manifold_evaluator(v70)
  else:
    sinfo0 = scalar_sector.numpy_scalar_manifold_evaluator(v70)
  threshold_deviation = (max(3 * sinfo0.stationarity, 1e-7)
                         if expected_dps >= 40 else 1e-3)
  # First-distillation produced a high-accuracy form of the numerical
  # solution, so we should use a stricter check here.
  def still_good(v70_trial):
    sinfo = scalar_sector.numpy_scalar_manifold_evaluator(v70_trial)
    return (abs(sinfo0.potential - sinfo.potential) < threshold_deviation and
            sinfo.stationarity < threshold_deviation)
  # This is strongly expected to work, given that we already had highly
  # accurate data.
  low_dim_model_take2 = iter(find_simple_low_dimensional_model(
      [v70],
      min_digits=max(3, min_model_digits or int(
          -mpmath.log(sinfo0.stationarity, 10)) // 2 - 2),
      still_good=still_good)).next()
  if len(low_dim_model_take2.params) < len(distillate_model.params):
    print('Note: Could further reduce the number of model parameters '
          '({} -> {}).'.format(distillate_model.params,
                               low_dim_model_take2.params))
  model_take2, mdnewton_ok = distill_model(
      low_dim_model_take2,
      newton_steps=newton_steps,
      still_good=still_good,
      target_digits_position=expected_dps)
  v70_accurate = v70_from_model(model_take2)
  sinfo_accurate = scalar_sector_mpmath.mpmath_scalar_manifold_evaluator(
      v70_accurate)
  stationarity = sinfo_accurate.stationarity
  log10_stationarity = math.floor(mpmath.log(stationarity, 10))
  approx_stationarity_str = (
      '%.3fe%d' %
      (float(stationarity *
             mpmath.mpf(10)**(-log10_stationarity)),
       log10_stationarity))
  with open(out_filename, 'w') as h:
    write_model(h, model_take2,
                dict(mdnewton_ok=mdnewton_ok,
                     potential=str(sinfo_accurate.potential),
                     stationarity=approx_stationarity_str))


def get_susy_from_a2(a2, tol=1e-6):
  """Determines residual supersymmetry from A2."""
  su, ss, svh = numpy.linalg.svd(
      a2.astype(numpy.complex128).reshape(8, -1))
  susy_raw = [u for u, s in zip(su, ss)
              if abs(s) <= tol]
  susy = [r / r[numpy.argmax(abs(r))] for r in susy_raw]
  assert all(numpy.allclose(u, u.real) for u in susy)
  return [tuple(u) for u in susy]


def get_gravitino_mass_eigenstates_from_a1(a1, potential, tolerance=1e-6):
  """Extracts gravitino m^2/m0^2 from A1."""
  na1 = a1.astype(numpy.complex128)
  normalized_a1_sq = numpy.einsum(
      'ij,ik->jk',
      na1, na1.conjugate()) * (-6 / float(potential))
  assert numpy.allclose(normalized_a1_sq, normalized_a1_sq.T.conjugate())
  eigvals, eigvecs = numpy.linalg.eigh(normalized_a1_sq)
  assert numpy.allclose(eigvals, eigvals.real)
  aggregated = symmetries.aggregate_eigenvectors(eigvals,
                                                 eigvecs, tolerance=tolerance)
  spaces = [numpy.stack(eigvecs, axis=-1) for _, eigvecs in aggregated]
  assert all(numpy.allclose(
      numpy.eye(space.shape[-1]),
      numpy.einsum('ij,ik->jk', space, space.conjugate())) for space in spaces)
  return aggregated


def get_fermion_mass_eigenstates_from_a2(a2, potential, tolerance=1e-6):
  """Extracts spin-1/2 fermion m^2/m0^2 from A2."""
  a3 = scalar_sector.get_a3_56x56_from_a2(
      a2,
      sqrt2=mpmath.sqrt(mpmath.mpf(2)),
      einsum=opt_einsum.contract,
      conjugate=lambda a:numpy.array([z.conjugate()
                                      for z in a.reshape(-1)]).reshape(a.shape))
  na3 = a3.astype(numpy.complex128)
  normalized_a3_sq = numpy.einsum(
      'ij,ik->jk',
      na3, na3.conjugate()) * (-6 / float(potential))  # Real-Symmetric.
  assert numpy.allclose(normalized_a3_sq, normalized_a3_sq.T.conjugate())
  eigvals, eigvecs = numpy.linalg.eigh(normalized_a3_sq)
  assert numpy.allclose(eigvals, eigvals.real)
  assert numpy.allclose(numpy.dot(eigvecs, eigvecs.T.conjugate()),
                        numpy.eye(56))
  aggregated = symmetries.aggregate_eigenvectors(eigvals, eigvecs,
                                                 tolerance=tolerance)
  spaces = [numpy.stack(eigvecs, axis=-1) for _, eigvecs in aggregated]
  assert all(numpy.allclose(
      numpy.eye(space.shape[-1]),
      numpy.einsum('ij,ik->jk', space, space.conjugate())) for space in spaces)
  return aggregated


def get_scalar_mass_eigenstates_from_mass_matrix(mass_matrix, tolerance=1e-6):
  """Extracts aggregated scalar masses-squared from the mass matrix."""
  eigvals, eigvecs = numpy.linalg.eigh(mass_matrix)
  return symmetries.aggregate_eigenvectors(eigvals, eigvecs,
                                           tolerance=tolerance)


def call_with_scalar_mass_matrix_evaluator(f, *args):
  """Returns f(evaluator, *args), with `evaluator` a mass matrix evaluator.

  Here, `evaluator` is a TensorFlow-based evaluator function
  that maps a 70-vector to a scalar mass matrix.

  We are then passing this on to a function that gets run in TensorFlow
  session context. This way, we can bulk-process solutions.
  """
  graph = tf.Graph()
  with graph.as_default():
    tf_scalar_evaluator = scalar_sector_tensorflow.get_tf_scalar_evaluator()
    t_left_onb = tf.Variable(  # Uses orthonormal basis.
        initial_value=numpy.zeros([70]), trainable=False, dtype=tf.float64)
    t_input = tf.placeholder(tf.float64, shape=[70])
    t_v70 = tf.Variable(
        initial_value=numpy.zeros([70]), trainable=False, dtype=tf.float64)
    op_assign_input = tf.assign(t_v70, t_input)
    sinfo = tf_scalar_evaluator(
        tf.cast(t_v70, tf.complex128),
        t_left=tf.cast(tf.einsum('vV,V->v',
                                 tf.constant(algebra.e7.v70_from_v70o),
                                 t_left_onb),
                       tf.complex128))
    t_potential = sinfo.potential
    t_scalar_mass_matrix = (
        tf.real(tf.hessians([t_potential], [t_left_onb])[0]) *
        # Factor -3/8 (rather than -3/4) is due to normalization of
        # our orthonormal basis.
        (-3.0 / 8) / t_potential)
    with tf.compat.v1.Session() as sess:
      sess.run([tf.global_variables_initializer()])
      def evaluator(v70):
        sess.run([op_assign_input], feed_dict={t_input: v70})
        ret = sess.run([t_scalar_mass_matrix])[0]
        return ret
      return f(evaluator, *args)


def get_scalar_mass_matrix(v70):
  """Returns the spin-0 mass matrix at position v70."""
  tf_scalar_evaluator = scalar_sector_tensorflow.get_tf_scalar_evaluator()
  t_left_onb = tf.constant(numpy.zeros(70), dtype=tf.float64)
  t_v70 = tf.constant(v70, dtype=tf.complex128)
  sinfo0 = tf_scalar_evaluator(t_v70)
  tc_v70_from_v70o = tf.constant(algebra.e7.v70_from_v70o)
  def tf_potential_relative(t_left_onb):
    sinfo = tf_scalar_evaluator(
      t_v70,
      t_left=tf.cast(tf.einsum('vV,V->v', tc_v70_from_v70o, t_left_onb),
                     tf.complex128))
    return sinfo.potential
  scalar_mm = m_util.tf_hessian(tf_potential_relative)(
    tf.constant(numpy.zeros(70), dtype=tf.float64)).numpy()
  return (scalar_mm /
          # Factor -3/8 (rather than -3/4) is due to normalization of
          # our orthonormal basis.
          (-3.0 / 8) / sinfo0.potential.numpy())


def explain_physics(distillate_filename):
  """Explains residual SUSY and particle masses for a solution."""
  distillate = read_distillate(distillate_filename)
  model = get_distillate_model(distillate)
  v70 = v70_from_model(model)
  sinfo = scalar_sector_mpmath.mpmath_scalar_manifold_evaluator(v70)
  a1 = sinfo.a1
  a2 = sinfo.a2
  gravitinos = get_gravitino_mass_eigenstates_from_a1(a1, sinfo.potential)
  fermions = get_fermion_mass_eigenstates_from_a2(a2, sinfo.potential)
  scalar_mass_matrix = get_scalar_mass_matrix(v70)
  scalars = get_scalar_mass_eigenstates_from_mass_matrix(scalar_mass_matrix)
  return dict(potential=str(sinfo.potential),
              susy=get_susy_from_a2(a2),
              gravitinos=gravitinos,
              fermions=fermions,
              scalars=scalars)
