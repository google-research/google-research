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

import base64
import collections
import dataclasses
import hashlib
import itertools
import math
import os
import pprint
import sys
import time
from typing import Optional
import warnings

import numpy
import scipy.linalg
import scipy.optimize
import tensorflow as tf

# For this maths-heavy project that needs to be open-sourceable,
# the style guide sometimes pulls us away from the most concise notation.
# In these cases: ignore pylint.
# pylint: disable=invalid-name
# pylint: disable=g-complex-comprehension
# pylint: disable=redefined-outer-name


_str_ignore_einsum_annotations = str.maketrans(
    {'^': None, '_': None, '|': None, ';': None})


def nsum(spec, *arrays, optimize='greedy'):
  """Numpy-Einsum Convenience wrapper.

  This uses "greedy" contraction as the default contraction-strategy.
  Also, it will strip the characters '^', '_', '|', and ';' from the
  contraction-spec, which hence can be used to document index-placement
  in the underlying physics formulas.

  Args:
    spec: string, the contraction-specification.
    *arrays: The arrays to contract over.
    optimize: The `optimize` parameter for numpy.einsum().

  Returns:
    The generalized-einstein-summation contraction result.
  """
  translated_spec = spec.translate(_str_ignore_einsum_annotations)
  try:
    return numpy.einsum(translated_spec, *arrays, optimize=optimize)
  # If something goes wrong, re-write the exception to a more telling one.
  except Exception as exn:  # pylint:disable=broad-except
    shapes_dtypes = [(x.shape, x.dtype) for x in arrays]
    raise ValueError(
        f'nsum failure, spec: {spec!r}, pieces: {shapes_dtypes!r}, '
        f'exception: {exn!r}')


def asymm2(a, *einsum_specs):
  """Antisymmetrizes an array."""
  for einsum_spec in einsum_specs:
    a = 0.5 * (a - nsum(einsum_spec, a))
  return a


# Convenience short-hands.
rng = numpy.random.RandomState
expm = scipy.linalg.expm
tf_expm = tf.linalg.expm


def rm(filename):
  """Removes a file, returning error or `None` if successful."""
  try:
    os.unlink(filename)
    return None
  except OSError as exn:
    return exn


def home_relative(path):
  """Returns a path relative to $HOME."""
  return os.path.join(os.getenv('HOME'), path)


def arg_enabled(name, *enabling_tags):
  """Helper for selectively enabling parts of a semi-interactive script."""
  return name == '__main__' and (
      set(sys.argv[1:]) & (set(enabling_tags) | {'all'}))


def final(iterator):
  """Returns the final element of an iterator."""
  ret = None
  for x in iterator:
    ret = x
  return ret


def evstats(m, d=6):
  """Returns eigenvalue statistics for a matrix."""
  if numpy.allclose(m, m.T):
    eigvals = numpy.linalg.eigh(m)[0]
    return sorted(collections.Counter(eigvals.round(d)).items())
  eigvals = numpy.linalg.eigvals(m)
  return sorted(collections.Counter(eigvals.round(d)).items(),
                key=lambda zn: (zn[0].real, zn[0].imag))


def get_gramian_onb(gramian, eigenvalue_threshold=1e-7):
  """Computes orthogonalizing transform for a gramian.

  Args:
    gramian: [N, N]-array G.
    eigenvalue_threshold: Eigenvalues smaller than this
      are considered to be equal to zero.

  Returns:
    A pair of matrices (R, R_inv) such that R @ R_inv = numpy.eye(N) and
    einsum('Aa,Bb,ab->AB', R, R, gramian) is diagonal with entries
    in (0, 1, -1).
    Example: If gramian=numpy.array([[100.0, 0.1], [0.1, 1.0]])
    then R.round(6) == numpy.array([[0.00101, -1.00005], [-0.1, -0.000101]])
    and R[0, :] as well as R[1, :] are orthonormal unit vectors w.r.t.
    the scalar product given by the gramian.
  """
  gramian = numpy.asarray(gramian)
  sprods_eigvals, sprods_eigvecsT = numpy.linalg.eigh(gramian)
  abs_evs = abs(sprods_eigvals)
  onbi_scaling = numpy.where(abs_evs <= eigenvalue_threshold,
                             1.0,
                             numpy.sqrt(abs_evs))
  onbi = numpy.einsum('WZ,Z->WZ',
                      sprods_eigvecsT, onbi_scaling)
  onb = numpy.einsum('WZ,Z->WZ',
                     sprods_eigvecsT, 1.0 / onbi_scaling).T
  assert numpy.allclose(onb @ onbi, numpy.eye(onb.shape[0]))
  return onb, onbi


def dstack(*pieces):
  """Assembles a matrix from blocks-on-the-diagonal."""
  a_pieces = [numpy.asarray(piece) for piece in pieces]
  dtype = numpy.find_common_type([a.dtype for a in a_pieces],
                                 [numpy.float64])
  piece_shapes = [x.shape for x in a_pieces]
  if not all(len(s) == 2 and s[0] == s[1] for s in piece_shapes):
    raise ValueError(f'Invalid diagonal-piece shapes: {piece_shapes!r}')
  block_sizes = [s[0] for s in piece_shapes]
  block_ranges = [0] + numpy.cumsum(block_sizes).tolist()
  result = numpy.zeros([block_ranges[-1]] * 2, dtype=dtype)
  for num_block, block in enumerate(a_pieces):
    start_idx, end_idx = block_ranges[num_block:num_block + 2]
    result[start_idx:end_idx, start_idx:end_idx] = block
  return result


def numpy_fingerprint(a, digits=3):
  """Fingerprints a numpy-array."""
  # Hack to ensure that -0.0 gets consistently shown as 0.0.
  minus_zero_hack = 1e-100+1e-100j
  return base64.b64encode(
      hashlib.sha256(
          str(
              (a.shape, ','.join(map(repr, numpy.round(a + minus_zero_hack,
                                                       digits).flat))))
          .encode('utf-8'))
      .digest()).decode('utf-8').strip('\n=')


def nonzero_entries(array, eps=1e-7):
  """Extracts sparse [(value, *indices), ...] array representation.

  Args:
    array: The numpy array to obtain a sparse representation for,
    eps: Threshold magnitude. Entries <= `eps` are skipped.
  Returns:
    List of (coefficient, index0, ..., indexN) tuples representing
    the non-zero entries of the array.
  """
  entries = []
  for indices in itertools.product(*[range(n) for n in array.shape]):
    v = array[indices]
    if abs(v) <= eps:
      continue
    entries.append((v,) + indices)
  return entries


def numpy_from_nonzero_entries(shape, entries, dtype=None):
  """Produces a numpy array from a sparse representation.

  Args:
    shape: The shape of the array.
    entries: The entries as an iterable of (value, index0, ..., indexN)
      tuples, e.g. as produced by nonzero_entries().
    dtype: The array-dtype. Defaults to the dtype of the sum of all values.

  Returns:
    The numpy array.
  """
  if dtype is None:
    dtype = type(sum(z[0] for z in entries))
  ret = numpy.zeros(shape, dtype=dtype)
  for v, *indices in entries:
    ret[tuple(indices)] += v
  return ret


def as_code(array, func_name, eps=1e-7):
  """Prints Python code that synthesizes a given numpy array."""
  # This is mostly useful in intermediate stages of research,
  # when we temporarily want to make some definition that was found
  # to work directly part of the code, for reproducibility.
  entries = nonzero_entries(array, eps=eps)
  print(f'\ndef {func_name}():\n'
        f'  data = [\n {pprint.pformat(entries, compact=True, indent=4)[1:]}\n'
        f'  return numpy_from_nonzero_entries({array.shape}, data)\n\n')


def sparse_dict_from_array(array, magnitude_threshold=0):
  """Converts a array to a dict of nonzero-entries keyed by index-tuple."""
  ret = {}
  for index_tuple in itertools.product(*(map(range, array.shape))):
    v = array[index_tuple]
    if abs(v) > magnitude_threshold:
      ret[index_tuple] = v
  return ret


def permutation_sign(p):
  """Determines the sign of a permutation, given as a sequence of integers."""
  q = list(p)  # Copy to list.
  sign = 1
  for n in range(len(p)):
    while n != q[n]:
      qn = q[n]
      q[n], q[qn] = q[qn], q[n]  # Flip to make q[qn] = qn.
      sign = -sign
  return sign


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


def symmetric_svd(m):
  """Computes the 'symmetric SVD' of a symmetric (complex) matrix.

  Args:
   m: [n, n]-ndarray, the symmetric matrix to be decomposed.

  Returns:
   (q, s), where `q` is a [n, n]-complex-ndarray, and `s` is a [n]-real-ndarray,
   and q.T @ m @ q == numpy.diag(s).
  """
  # Conceptually, this algorithm is about 'regarding the 1j that
  # multiplies the imaginary part of the matrix as a different
  # imaginary unit than the one we get in eigenvalues',
  # i.e. unraveling the complex-symmetric matrix into a real-symmetric
  # one by splitting real and imaginary parts (where we have to be careful,
  # as the imaginary piece does not become an antisymmetric contribution,
  # as is usual).
  dim = m.shape[0]
  if m.shape[1] != dim or not numpy.allclose(m, m.T):
    raise ValueError('Matrix is not symmetric!')
  m_re = m.real
  m_im = m.imag
  mb = numpy.zeros([2 * dim, 2 * dim], dtype=m_re.dtype)
  mb[:dim, :dim] = m_re
  mb[dim:, dim:] = -m_re
  mb[dim:, :dim] = mb[:dim, dim:] = m_im
  mb_eigvals, mb_eigvecsT = numpy.linalg.eigh(mb)
  # We need that half of the eigenvectors that is associated
  # with 'large' eigenvalues.
  # Let us call these the 'plus eigenvectors'.
  eigvals_sorting_indices = numpy.argsort(mb_eigvals)
  plus_eigvecs_re_im = mb_eigvecsT[:, eigvals_sorting_indices[dim:]]
  plus_eigvecs = plus_eigvecs_re_im[:dim, :] - 1j * plus_eigvecs_re_im[dim:, :]
  diagonalized = plus_eigvecs.T @ m @ plus_eigvecs
  diag = numpy.diag(diagonalized)
  diag_re = diag.real
  assert numpy.allclose(diag_re, diag)
  assert numpy.allclose(numpy.diag(diag), diagonalized)
  return plus_eigvecs, diag_re


def undiskify(z):
  """Maps SL(2)/U(1) poincare disk coord to Lie algebra generator-factor."""
  # Conventions match (2.13) in https://arxiv.org/abs/1909.10969
  return 2* numpy.arctanh(abs(z)) * numpy.exp(1j * numpy.angle(z))


def diskify(z):
  """Maps Lie algebra generator-factor to SL(2)/U(1) poincare disk coord."""
  # Conventions match (2.13) in https://arxiv.org/abs/1909.10969
  return numpy.tanh(0.5 * abs(z)) * numpy.exp(1j * numpy.angle(z))


def aligning_rotation(v_target, v_in):
  """Returns some SO(N) rotation that turns v_in into v_target."""
  v_target = numpy.asarray(v_target)
  v_in = numpy.asarray(v_in)
  dim = v_target.shape[0]
  vn_target = v_target / numpy.sqrt(v_target.dot(v_target))
  vn_in = v_in / numpy.sqrt(1 / numpy.finfo(numpy.float64).max +
                            v_in.dot(v_in))
  cos_angle = vn_target.dot(vn_in)
  v_parallel = vn_target * cos_angle
  v_perp = vn_in - v_parallel
  v_perp_len = sin_angle = numpy.sqrt(v_perp.dot(v_perp))
  if v_perp_len < 100 * numpy.finfo(numpy.float64).resolution:
    # The rotation that we would need to apply is very close to the
    # identity matrix. Just return that instead.
    return numpy.eye(dim)
  # Otherwise, we can normalize the perpendicular vector.
  vn_perp = v_perp / v_perp_len
  return (numpy.eye(dim) -
          numpy.outer(vn_target, vn_target) -
          numpy.outer(vn_perp, vn_perp) +
          sin_angle * (numpy.outer(vn_target, vn_perp) -
                       numpy.outer(vn_perp, vn_target)) +
          cos_angle * numpy.outer(vn_target, vn_target) +
          cos_angle * numpy.outer(vn_perp, vn_perp))


def _aitken_accelerated_inner(iterable):
  # See: https://en.wikipedia.org/wiki/Aitken%27s_delta-squared_process
  # Also: "Structure and Interpretation of Computer Programs",
  # 3.5.3 "Exploiting the Stream Paradigm".
  z_prev2, z_prev = numpy.nan, numpy.nan
  for z in iterable:
    if numpy.isfinite(z_prev2):
      yield z - (z - z_prev)**2 / (z + z_prev2 - 2 * z_prev)
    z_prev2, z_prev = z_prev, z


def aitken_accelerated(iterable, order=1):
  """Convergence-accelerates an iterable."""
  if order == 0:
    return iterable
  return aitken_accelerated(_aitken_accelerated_inner(iterable), order - 1)


# [t_a, t_b] = f_abc t_c
def get_f_abc(t_abc, filename=None, sanitize=None, verbose=False):
  """Computes structure constants, [t_a, t_b] = f_abc t_c.

  Args:
    t_abc: [dim_ad, D, D]-ndarray, generator matrices.
      When reading t_abc entries as matrices, the matrix that
      corresponds to generator #0 is t_abc[0].T.
    filename: Optional filename to save the computation's result to,
      in NumPy's compressed .npz format. Must end with '.npz'.
      If `filename` is provided and a corresponding file exists,
      this file is de-serialized instead of re-doing the computation.
    sanitize: Optional ndarray->ndarray function to remove numerical
      noise on f_abC.
    verbose: Whether to announce (via print()) re-computation of structure
      constants (rather than re-use of serialized data).

  Returns:
    A pair (f_abc, f_abC), where `f_abc` are the structure constants,
      computed as tr(t_a [t_b, t_c]), and `f_abC` have the last index
      converted to a co-adjoint index via an extra 'inverse Cartan-Killing
      metric' factor.
  """
  if filename is not None and not filename.endswith('.npz'):
    raise ValueError(f'Filename should end with ".npz": {filename!r}')
  if sanitize is None:
    sanitize = lambda x: x
  k_ab = nsum('_aN^M,_bM^N->ab', t_abc, t_abc)
  try:
    if filename is None:
      raise IOError('')
    f_abc = numpy.load(filename)['f_abc']
  except (IOError, OSError):
    if verbose:
      print('Computing f_abc.')
    commutators_ab = 2 * asymm2(nsum('_aP^M,_bN^P->_abN^M', t_abc, t_abc),
                                '_abN^M->_baN^M')
    f_abc = nsum('_abM^N,_cN^M->abc', commutators_ab, t_abc)
    if filename is not None:
      numpy.savez_compressed(filename, f_abc=f_abc)
  f_abC = sanitize(nsum('_abc,^cC->_ab^C', f_abc, numpy.linalg.inv(k_ab)))
  return f_abc, f_abC


def get_commutant(f_abC, g_an, space=None, ev_threshold=1e-7):
  """Returns the commutant of a set of generators inside a Lie algebra.

  Args:
    f_abC: [D, D, D]-array, structure constants that define the enclosing
      Lie algebra, [G_a, G_b] = f_ab^c G_c.
    g_an: [D, N]-array, generators for which we want to determine the commutant.
    space: Optional [D, M]-array. If provided, determine commutant within this
      subspace of the Lie algebra.
    ev_threshold: Eigenvalue threshold for commutant.

  Returns:
    [D, P]-array, P generators which all commute wit the g_an.
  """
  # For a Lie algebra given in terms of its structure constants f_abC,
  # as well as a collection of generators g_na, find the generators
  # h_mb that commute with the g_na.
  dim = f_abC.shape[0]
  subspace = numpy.eye(dim) if space is None else space
  for gen in g_an.T:
    gen_ad_on_subspace = nsum('abc,a,bm->cm', f_abC, gen, subspace)
    svd_u, svd_s, svd_vh = numpy.linalg.svd(gen_ad_on_subspace,
                                            full_matrices=False)
    del svd_u  # Unused here.
    sub_subspace = svd_vh[svd_s <= ev_threshold, :len(svd_s)]
    subspace = nsum('an,mn->am', subspace, sub_subspace)
  return subspace


def lie_algebra_derivative(f0_mnp, fj_mnp):
  """Computes Lie algebra commutators [L, Ln]."""
  # Idea: f_mnp, f0_mnp are structure constants embedded in the same irrep,
  # so every f_m, f0_m corresponds to a generator-matrix.
  # We want to decompose the commutators in terms of the original generators.
  dim = f0_mnp.shape[0]
  comms = 2 * asymm2(nsum('MPQ,NRP->MNRQ', f0_mnp, fj_mnp), 'MNRQ->NMRQ')
  decomposed, residuals, *_ = numpy.linalg.lstsq(
      f0_mnp.reshape(dim, dim * dim).T,
      comms.reshape(dim * dim, dim * dim).T)
  del residuals  # Unused, named only for documentation purposes.
  return decomposed.reshape(f0_mnp.shape)


def spot_check_t_aMN(t_aMN, num_checks=100, seed=1):
  """Spot-checks closure of a matrix Lie algebra."""
  rng = numpy.random.RandomState(seed=seed)
  for num_check in range(num_checks):
    n1 = rng.randint(0, t_aMN.shape[0])
    n2 = rng.randint(0, t_aMN.shape[0])
    g1 = t_aMN[n1].T
    g2 = t_aMN[n2].T
    g12 = g1 @ g2 - g2 @ g1
    # The claim is that `g12` is always expressible in terms of t_aMN.T
    _, residue = numpy.linalg.lstsq(t_aMN.reshape(t_aMN.shape[0], -1).T,
                                    g12.T.ravel())[:2]
    if not numpy.allclose(0, residue):
      raise ValueError(
          f'Failed (n={num_check}): [T{n1}, T{n2}], '
          f'max_residue={max(residue):.6g}')
  return True


def spot_check_f_abC(t_aMN, f_abC, num_checks=1000, seed=1):
  """Spot-checks structure constants of a matrix Lie algebra."""
  rng = numpy.random.RandomState(seed=seed)
  for num_check in range(num_checks):
    n1 = rng.randint(0, t_aMN.shape[0])
    n2 = rng.randint(0, t_aMN.shape[0])
    g1 = t_aMN[n1].T
    g2 = t_aMN[n2].T
    g12 = g1 @ g2 - g2 @ g1
    f_g12 = nsum('_aM^N,_a->^N_M', t_aMN, f_abC[n1, n2])
    if not numpy.allclose(g12, f_g12):
      tprint(g12, name='g12')
      tprint(f_g12, name='f_g12')
      raise RuntimeError(f'Failed (n={num_check}): [T{n1}, T{n2}]')
  return True


def tff64(x):
  return tf.constant(x, dtype=tf.float64)


def tfc128(re, im=0.0):
  return tf.complex(tff64(re), tff64(im))


def numpy_func_from_tf(tf_func,
                       dtype=tf.float64,
                       allow_extra_args=False,
                       debug_tag=None):
  """Wraps up a tf.Tensor->tf.Tensor function as ndarray->ndarray."""
  def f_np(pos, *extra_args):
    if not allow_extra_args and extra_args:
      raise ValueError('Unexpected extra arguments to function.')
    ret = tf_func(tf.constant(pos, dtype=dtype)).numpy()
    if debug_tag is not None:
      print('DEBUG %s(%r): %s' % (debug_tag, pos, ret))
    return ret
  return f_np


_DEFAULT_USE_TF_FUNCTION = False


def tf_grad(t_scalar_func, use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a TF scalar-function to its TF gradient-function."""
  maybe_tf_function = tf.function if use_tf_function else lambda x: x
  @maybe_tf_function
  def f_grad(t_pos):
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_pos)
      t_val = t_scalar_func(t_pos)
    grad = tape.gradient(t_val, t_pos)
    assert grad is not None, '`None` gradient.'
    return grad
  return f_grad


def tf_ext_grad(t_scalar_func, use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a TF scalar-function to its gradient-extended variant."""
  maybe_tf_function = tf.function if use_tf_function else lambda x: x
  @maybe_tf_function
  def f_grad(t_pos):
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_pos)
      t_val = t_scalar_func(t_pos)
    grad = tape.gradient(t_val, t_pos)
    assert grad is not None, '`None` gradient.'
    return t_val, grad
  return f_grad


def tf_batch_grad(tb_scalar_func, use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a TF batched-scalar-function to its TF batched-gradient-function."""
  # TODO(tfish): Properly document and test-cover this.
  # This so far has only seen explorative use.
  maybe_tf_function = tf.function if use_tf_function else lambda x: x
  @maybe_tf_function
  def f_batch_grad(tb_pos):
    tape = tf.GradientTape()
    with tape:
      tape.watch(tb_pos)
      tb_val = tb_scalar_func(tb_pos)[:, tf.newaxis]
    tb_grad_raw = tape.batch_jacobian(tb_val, tb_pos)
    tb_grad_raw_shape = tf.shape(tb_grad_raw).as_list()
    # We have to remove the extra batch-index on `tb_val` that was introduced
    # above.
    tb_grad = tf.reshape(tb_grad_raw,
                         [tb_grad_raw_shape[0]] + tb_grad_raw_shape[2:])
    assert tb_grad is not None, '`None` gradient.'
    return tb_grad
  return f_batch_grad


def tf_stationarity(t_scalar_func, use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a TF scalar-function to its TF gradient-length-squared function."""
  maybe_tf_function = tf.function if use_tf_function else lambda x: x
  @maybe_tf_function
  def f_stat(t_pos):
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_pos)
      t_val = t_scalar_func(t_pos)
    grad = tape.gradient(t_val, t_pos)
    assert grad is not None, '`None` gradient (for stationarity).'
    return tf.reduce_sum(tf.math.square(grad))
  return f_stat


def _tf_jacobian_v1(t_vec_func, use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a TF vector-function to its TF Jacobian-function."""
  maybe_tf_function = tf.function if use_tf_function else lambda x: x
  @maybe_tf_function
  def f_jac(t_pos, *further_unwatched_args):
    tape = tf.GradientTape(persistent=True)
    with tape:
      tape.watch(t_pos)
      v_components = tf.unstack(t_vec_func(t_pos, *further_unwatched_args))
    gradients = [tape.gradient(v_component, t_pos)
                 for v_component in v_components]
    assert all(g is not None for g in gradients), 'Bad Gradients for Jacobian.'
    # The gradient's index must come last, so we have to stack along axis 0.
    jacobian = tf.stack(gradients, axis=0)
    return jacobian
  return f_jac


def _tf_jacobian_v2(t_vec_func, use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a TF vector-function to its TF Jacobian-function."""
  maybe_tf_function = tf.function if use_tf_function else lambda x: x
  @maybe_tf_function
  def tf_j(t_xs, *further_unwatched_args):
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_xs)
      v = t_vec_func(t_xs, *further_unwatched_args)
    ret = tape.jacobian(v, t_xs)
    return ret
  return tf_j


# TODO(tfish): Change to using _v2 once this supports all the weird graph Ops
# that we are using.
tf_jacobian = _tf_jacobian_v1
# tf_jacobian = _tf_jacobian_v2


def tf_hessian(t_scalar_func, use_tf_function=_DEFAULT_USE_TF_FUNCTION):
  """Maps a TF scalar-function to its TF hessian-function."""
  return tf_jacobian(
      tf_grad(t_scalar_func,
              use_tf_function=use_tf_function),
      use_tf_function=use_tf_function)


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
    tf_jacobian_func = tf_jacobian(tf_vec_func)
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
  """Finds a zero of a tf.Tensor R^N->R function via MDNewton({gradient})."""
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


def grid_scan(f, index_ranges, maybe_prev_f=None):
  """Iterates over grid-positions and function-values.

  The common situation is: We want to map out values of a function
  that involves e.g. optimization, so whenever we compute that
  function, it is useful to know the value at a neighboring point.
  This function iterates over grid-points in such a way that each
  function-evaluation gets to see the value at a neighboring point.

  Args:
    f: (Tuple[int, ...], Optional[T]) -> T: Function to be scanned.
    index_ranges: Sequence of pairs (low_end, high_end) that specify
      ranges for each index.
    maybe_prev_f: Optional neighbor-value to previded to
       f(starting_point).

  Yields:
    Pair of (indices, f_value), where f_value is obtained by calling
    f(indices, {value of f at a neighboring point,
                or maybe_prev_f for the starting point})
  """
  # Each `index_ranges` entry is (end_neg, end_pos),
  # so [-10..10] would get encoded as (-11, 11).
  num_indices = len(index_ranges)
  max_index = num_indices - 1
  ipos0 = (0,) * num_indices
  f_now = f(ipos0, maybe_prev_f)
  yield ipos0, f_now
  if num_indices == 0:
    return
  stack = [(+1, 0, ipos0, f_now), (-1, 0, ipos0, f_now)]
  while stack:
    direction, icursor, ipos_now, f_now = stack.pop()
    ipos_c = ipos_now[icursor]
    ipos_c_end = index_ranges[icursor][(1 + direction) // 2]
    # Subtle: We must make sure that we do not recurse-down into
    # scanning the next-index range from both the increasing and
    # decreasing branch. The `and not` part of the condition below
    # ensures this.
    if icursor < max_index and not (ipos_c == 0 and direction == -1):
      stack.extend(
          [(+1, icursor + 1, ipos_now, f_now),
           (-1, icursor + 1, ipos_now, f_now)])
    if ipos_c == ipos_c_end - direction:
      continue  # Reached the end for this stride - done.
    ipos_next = tuple(idx + direction * (k == icursor)
                      for k, idx in enumerate(ipos_now))
    f_next = f(ipos_next, f_now)
    yield ipos_next, f_next
    stack.append((direction, icursor, ipos_next, f_next))


def _fixed_fmin(f_opt, x0, minimizer_func, **kwargs):
  """Internal - Fixes a wart in scipy.optimize.fmin_bfgs() behavior."""
  # Always return the smallest value encountered during the entire
  # minimization procedure, not the actual result from fmin_bfgs().
  last_seen = [(numpy.inf, None)]
  def f_opt_wrapped(xs):
    val_opt = f_opt(xs)
    if last_seen[0][0] > val_opt:
      last_seen[0] = (val_opt, xs.copy())
    return val_opt
  #
  ret = minimizer_func(f_opt_wrapped, x0, **kwargs)
  if kwargs.get('full_output'):
    return (last_seen[0][1],) + ret[1:]
  return last_seen[0][1]


class OptimizationError(Exception):
  """Optimization failed."""


class LineSearchError(Exception):
  """Line search failed."""


def line_search_wolfe12(f, fprime, pos_start, direction, grad_at_start,
                        f_at_start, f_at_previous_point,
                        **kwargs):
  """Runs line_search_wolfe1, falling back to line_search_wolfe2 as needed.

  Args:
    f: numpy.ndarray -> float function, the function to line-search over.
    fprime: The gradient function of `f`.
    pos_start: numpy.ndarray, the starting position for line-search.
    direction: numpy.ndarray, the direction of the line.
    grad_at_start: numpy.ndarray, gradient at start.
    f_at_start: Value of the function at the starting point.
    f_at_previous_point: (Estimated) value of the function at the
      previous point.
    **kwargs: Other keyword arguments to pass on to
      scipy.optimize.linesearch.line_search_wolfe1().
      Only the key-value pairs with keys 'c1', 'c2', 'amax' will get passed on
      to scipy.optimize.linesearch.line_search_wolfe2() in case we fall back
      to that other function.

  Returns:
    The result of calling scipy.optimize.linesearch.line_search_wolfe1(),
    respectively scipy.optimize.linesearch.line_search_wolfe2().

  Raises:
    LineSearchError: line search failed.
  """
  ret = scipy.optimize.linesearch.line_search_wolfe1(
      f, fprime, pos_start, direction, grad_at_start,
      f_at_start, f_at_previous_point,
      **kwargs)
  if ret[0] is not None:
    return ret
  # Otherwise, line search failed, and we try _wolfe2.
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', scipy.optimize.linesearch.LineSearchWarning)
    wolfe2_kwargs = {key: val for key, val in kwargs.items()
                     if key in ('c1', 'c2', 'amax')}
    ret = scipy.optimize.linesearch.line_search_wolfe2(
        f, fprime, pos_start, direction, grad_at_start,
        f_at_start, f_at_previous_point,
        **wolfe2_kwargs)
    if ret[0] is None:
      raise LineSearchError()
    return ret


@dataclasses.dataclass(frozen=True)
class PreliminaryOptimum:
  """A preliminary proposal for a minimization problem solution."""
  num_step: int
  val: numpy.ndarray
  norm_grad: float
  pos: numpy.ndarray
  args: 'Any'  # TODO(tfish): Be more specific here.
  grad: numpy.ndarray
  inv_hessian: Optional[numpy.ndarray]

  def __repr__(self):
    """Returns a string representation of the instance."""
    return (f'<PreliminaryOptimum, num_step={self.num_step}, '
            f'norm_grad={self.norm_grad:.6g}, val={self.val:.6g}, '
            f'pos={self.pos.round(3).tolist()}>')


def bfgs(f, xs_start, fprime, args=(),
         rho_k_max=1000,
         norm=numpy.inf,
         gtol=-numpy.inf,
         inv_hessian=None):
  """Yields PreliminaryOptimum instances for BFGS-optimization.

  This algorithm is a generator re-implemenetation of
  scipy.optimize.fmin_bfgs().

  Args:
    f: (x_pos, *args) -> float function, the objective function.
    xs_start: numpy.typing.ArrayLike, the starting position for minimization.
    fprime: (x_pos, *args) -> grad: numpy.ndarray, the gradient of `f`.
    args: extra arguments to be provided to `f` and `fprime`.
    rho_k_max: Maximum value of the rho_k parameter.
      Normally does not need any adjustments.
    norm: The vector norm to use when reporting gradient-norms.
    gtol: Tolerable value for the gradient-norm.
      Optimization will terminate once gradient-norm gets smaller than this
      threshold.
    inv_hessian: Optional[numpy.ndarray], the initial guess for the inverse
      Hessian. If not provided, the identity matrix will be used.

  Yields:
    PreliminaryOptimum, an intermediate point during optimization.
  """
  # This has been checked manually on a particular supergravity
  # equilibrium to zoom down to the equilibrium just exactly
  # like scipy.optimize.fmin_bfgs does.
  args = tuple(args)
  xs_start = numpy.asarray(xs_start).ravel()
  dim = xs_start.size
  if not args:
    fx = f
    fxprime = fprime
  else:
    fx = lambda xs: f(xs, *args)
    fxprime = lambda xs: fprime(xs, *args)
  def vnorm(xs):
    return numpy.linalg.norm(xs, ord=norm)
  f_current = float(fx(xs_start))
  grad_current = fxprime(xs_start)
  identity = numpy.eye(dim)
  inv_hessian_current = (
      numpy.asarray(inv_hessian) if inv_hessian is not None else identity)
  # Sets the initial step guess to dx ~ 1
  f_previous = f_current + 0.5 * numpy.linalg.norm(grad_current)
  xk = xs_start
  for k in itertools.count():
    pk = -numpy.dot(inv_hessian_current, grad_current)
    try:
      alpha_k, _, _, f_current, f_previous, grad_next = (
          line_search_wolfe12(
              fx, fxprime, xk, pk, grad_current,
              f_current, f_previous, amin=1e-100, amax=1e100))
    except LineSearchError:
      # Line search failed to find a better solution. We are done.
      return
    xk_next = xk + alpha_k * pk
    sk = xk_next - xk
    xk = xk_next
    if grad_next is None:
      grad_next = fxprime(xk_next)
    yk = grad_next - grad_current
    grad_current = grad_next
    yk_sk = numpy.dot(yk, sk)
    if yk_sk == 0:  # Weird, but as in scipy.optimize.fmin_bfgs.
      rho_k = rho_k_max
    else:
      rho_k = 1 / yk_sk
    a1 = identity - sk[:, numpy.newaxis] * yk[numpy.newaxis, :] * rho_k
    a2 = identity - yk[:, numpy.newaxis] * sk[numpy.newaxis, :] * rho_k
    inv_hessian_current = numpy.dot(a1, numpy.dot(inv_hessian_current, a2)) + (
        rho_k * sk[:, numpy.newaxis] * sk[numpy.newaxis, :])
    norm_grad = vnorm(grad_current)
    yield PreliminaryOptimum(num_step=k,
                             val=f_current,
                             norm_grad=norm_grad,
                             pos=xk_next,
                             args=args,
                             grad=grad_current,
                             inv_hessian=inv_hessian_current)
    if norm_grad < gtol:
      return


def bfgs_scan(
        f, fprime, xs_start,
        param_ranges,
        param_range_mid_indices=None,
        extract_result_fn=(
            lambda prelim_opt: prelim_opt.pos.tolist() + [prelim_opt.val]),
        verbose=True,
        forward_positions=True,
        forward_hessians=False,
        report=print,
        gtol=1e-10):
  """BFGS-scans a function."""
  # `forward_positions=True` generally has a dramatic positive effect.
  # `forward_hessians=True` may well have a negative effect.
  param_ranges = [numpy.asarray(prange) for prange in param_ranges]
  if param_range_mid_indices is None:
    param_range_mid_indices = [prange.size // 2 for prange in param_ranges]
  grid_scan_index_ranges = [
      (-mid_index - 1, prange.size - mid_index)
      for mid_index, prange in zip(param_range_mid_indices, param_ranges)]
  result = numpy.full([prange.size for prange in param_ranges], None)
  def f_scan(indices, f_prev):
    f_arg = numpy.array(
        [prange[mid + k]
         for prange, mid, k in zip(param_ranges,
                                   param_range_mid_indices,
                                   indices)])
    prev_pos = (xs_start if f_prev or forward_positions is not None
                else f_prev.pos)
    inv_hessian = (None if f_prev is None or not forward_hessians
                   else f_prev.inv_hessian)
    optimum = None
    for prelim_opt in bfgs(f, prev_pos, fprime, args=(f_arg,), gtol=gtol,
                           inv_hessian=inv_hessian):
      optimum = prelim_opt
      if verbose and optimum.num_step % 50 == 0:
        report(f'BFGS n={optimum.num_step}, val={optimum.val:.6g}, '
               f'norm_grad={optimum.norm_grad:.6g}')
    return optimum
  t_start = time.time()
  t_prev = t_start
  for num_step, (indices, f_scan_val) in enumerate(
      grid_scan(f_scan, grid_scan_index_ranges)):
    result[indices] = val_here = extract_result_fn(f_scan_val)
    if verbose:
      t_now = time.time()
      report(f'N={num_step:6d}, T={t_now - t_start:.3f}s '
             f'(+{t_now - t_prev:.3f} s): {indices} -> {val_here[-1]}')
      t_prev = t_now
  return result.tolist()


def tf_minimize(tf_scalar_func, x0,
                tf_grad_func=None,
                # Sequence of letters from set('BGN'), B=BFGS, G=Gradient,
                # N=Newton.
                strategy='B',
                # Default is to do 100 gradient-steps at learning rate 1e-3.
                gradient_steps=((100, 1e-3),),
                show_gradient_loss_every_n_steps=None,
                dtype=tf.float64,
                fail_on_nan=True,
                # Only works if tf_grad_func is not given.
                cache_gradients=True,
                gtol=1e-5, maxiter=10**4, mdnewton_maxsteps=7):
  """Minimizes a TensorFlow function."""
  tf_ext_grad_func = None
  if tf_grad_func is None:
    # pylint:disable=function-redefined
    tf_ext_grad_func = tf_ext_grad(tf_scalar_func)
    @tf.function
    def tf_grad_func(t_params):
      _, t_grad = tf_ext_grad_func(t_params)
      return t_grad
  if not cache_gradients:
    def f_opt(params):
      ret = tf_scalar_func(tf.constant(params, dtype=dtype)).numpy()
      if fail_on_nan and numpy.isnan(ret):
        raise OptimizationError('NaN in objective function.')
      return ret
    def fprime_opt(params):
      ret = tf_grad_func(tf.constant(params, dtype=dtype)).numpy()
      if fail_on_nan and numpy.any(numpy.isnan(ret)):
        raise OptimizationError('NaN in gradient function.')
      return ret
  else:
    # The way scipy's optimizers work, they will usually compute the function
    # and its gradient once each per location. Let's speed this
    # up by caching the last value, by input position.
    last_cached = [numpy.zeros_like(x0) + numpy.nan,
                   numpy.nan,
                   numpy.zeros_like(x0) + numpy.nan]
    def fprime_opt(params):
      if numpy.all(params == last_cached[0]):
        return last_cached[2].copy()
      t_val, t_grad = tf_ext_grad_func(tf.constant(params, dtype=dtype))
      last_cached[0][:] = params
      last_cached[1] = t_val.numpy()
      last_cached[2] = t_grad.numpy()
      if fail_on_nan and numpy.any(numpy.isnan(last_cached[2])):
        raise OptimizationError('NaN in gradient function.')
      return last_cached[2].copy()
    def f_opt(params):
      if numpy.all(params == last_cached[0]):
        return last_cached[1]
      t_val, t_grad = tf_ext_grad_func(tf.constant(params, dtype=dtype))
      last_cached[0][:] = params
      last_cached[1] = t_val.numpy()
      last_cached[2] = t_grad.numpy()
      if fail_on_nan and numpy.isnan(last_cached[1]):
        raise OptimizationError('NaN in objective function.')
      return last_cached[1]
  xs_now = x0
  num_gradient_descent_stages_done = 0
  for strategy_step in strategy:
    if strategy_step == 'B':  # BFGS
      opt_info = _fixed_fmin(f_opt,
                             numpy.array(xs_now),
                             minimizer_func=scipy.optimize.fmin_bfgs,
                             fprime=fprime_opt,
                             gtol=gtol,
                             maxiter=maxiter,
                             disp=0,
                             full_output=True)
      # TODO(tfish): Check full output for convergence.
      # Not much of a problem currently, since we are always
      # checking stationarity.
      xs_now = opt_info[0]
    elif strategy_step == 'N':  # Multi-Dimensional (MD)Newton
      *_, t_ret_xs = tf_mdnewton(
          tf_scalar_func,
          tf.constant(xs_now, dtype=tf.float64),
          tf_grad_func=tf_grad_func,
          maxsteps=mdnewton_maxsteps)
      xs_now = t_ret_xs.numpy()
    elif strategy_step == 'G':  # Gradient Descent.
      num_gradient_steps, learning_rate = gradient_steps[
          min(len(gradient_steps)-1, num_gradient_descent_stages_done)]
      num_gradient_descent_stages_done += 1
      for num_gradient_step in range(num_gradient_steps):
        xs_now -= learning_rate * fprime_opt(xs_now)
        if (show_gradient_loss_every_n_steps is not None and
            num_gradient_step % show_gradient_loss_every_n_steps == 0):
          print('[gradient, lr=%.6g, step=%4d] L=%.6g' % (
              learning_rate, num_gradient_step, f_opt(xs_now)))
    else:
      raise RuntimeError('Unknown strategy step: %r' % (strategy_step,))
  return f_opt(xs_now), xs_now


def tf_minimize_v2(
    tf_scalar_func, x0,
    tf_grad_func=None,
    # 'Strategy' is a sequence of pairs (strategy_name, *details, opt_kwargs),
    # where opt_kwargs are different for each strategy.
    # BFGS, CG: args are forwarded as kwargs to
    #   scipy.optimize.fmin_bfgs() / scipy.optimize.fmin_cg().
    #   Relevant args are: gtol, maxiter.
    # CUSTOM: `details` has length-1 and provides the actual optimizer-function
    #   to call. Call signature must be compatible with how this
    #   minimizer-wrapper
    #   calls scipy.optimize.fmin_bfgs.
    #   So, ('CUSTOM', scipy.optimize.fmin_bfgs, kwargs) is equivalent to
    #   ('BFGS', kwargs).
    # GD: gradient-descent. kwargs are:
    #  schedule=[(num_steps, learning_rate), ...]
    #  show_loss_every_n_steps.
    # MDNewton: kwargs are: maxsteps.
    strategy=(('BFGS', None),),
    dtype=tf.float64,
    fail_on_nan=True,
    cache_gradients=True,  # Only works if tf_grad_func is not given.
    use_tf_function=True,
    zoom=1.0,  # We need the 'zoom' argument for aligning docstrings.
    default_gtol=1e-7, default_maxiter=10**4, default_mdnewton_maxsteps=3,
    default_gd_schedule=((100, 3e-4),),):
  """Minimizes a TensorFlow function."""
  # TODO(tfish): Document properly. We currently have some code that already
  # uses the improved _v2, but still some code using the original interface.
  # Small details about how the optimizer should behave are still changing,
  # and the docstring should be finalized once these are resolved.
  #
  # TODO(tfish): Add args-argument for parameters that get passed on to
  # the scalar function and its gradient-function.
  if zoom != 1.0:
    # This was used in some code but ultimately deemed to not be a useful idea.
    raise ValueError('Deprecated legacy argument `zoom` has non-default value.')
  maybe_tf_function = tf.function if use_tf_function else lambda x: x
  tf_ext_grad_func = None
  if tf_grad_func is None:
    # pylint:disable=function-redefined
    tf_ext_grad_func = tf_ext_grad(tf_scalar_func)
    @maybe_tf_function
    def tf_grad_func(t_params):
      _, t_grad = tf_ext_grad_func(t_params)
      return t_grad
  if not cache_gradients:
    def f_opt(params):
      ret = tf_scalar_func(tf.constant(params, dtype=dtype)).numpy()
      if fail_on_nan and numpy.isnan(ret):
        raise OptimizationError('NaN in objective function.')
      return ret
    def fprime_opt(params):
      ret = tf_grad_func(tf.constant(params, dtype=dtype)).numpy()
      if fail_on_nan and numpy.any(numpy.isnan(ret)):
        raise OptimizationError('NaN in gradient function.')
      return ret
  else:
    # The way scipy's optimizers work, they will usually compute the function
    # and its gradient once each per location. Let's speed this
    # up by caching the last value, by input position.
    last_cached = [
        numpy.zeros_like(x0) + numpy.nan,  # Cached position.
        numpy.nan,  # Cached value.
        numpy.zeros_like(x0) + numpy.nan]  # Cached gradient.
    # Note: SciPy has a weird bug: When we return a gradient,
    # it assumes that it receives ownership of the gradient
    # vector-object and be free to modify the object from
    # there on. This breaks if we just return the cached vector -
    # SciPy would modify that object and break the invariant
    # that for the given last-evaluated cached position x0,
    # the cached-gradient vector holds the value of the gradient
    # at x0. We hence have to .copy() the gradient that we return.
    def fprime_opt(params):
      if numpy.all(params == last_cached[0]):
        return last_cached[2].copy()
      t_val, t_grad = tf_ext_grad_func(
          tf.constant(params, dtype=dtype))
      last_cached[0][:] = params
      last_cached[1] = t_val.numpy()
      last_cached[2][:] = t_grad.numpy()
      if fail_on_nan and not numpy.all(numpy.isfinite(last_cached[2])):
        raise OptimizationError('NaN in gradient function.')
      return last_cached[2].copy()
    def f_opt(params):
      if numpy.all(params == last_cached[0]):
        return last_cached[1]
      t_val, t_grad = tf_ext_grad_func(tf.constant(params, dtype=dtype))
      last_cached[0][:] = params
      last_cached[1] = t_val.numpy()
      last_cached[2][:] = t_grad.numpy()
      if fail_on_nan and not numpy.isfinite(last_cached[1]):
        raise OptimizationError('NaN in gradient function.')
      return last_cached[1]
  xs_now = numpy.array(x0)
  for strategy_step, *strategy_details, strategy_kwargs in strategy:
    if strategy_step in ('BFGS', 'CG'):
      kwargs = dict(gtol=default_gtol,
                    maxiter=default_maxiter,
                    disp=0)
      if strategy_kwargs is not None:
        kwargs.update(strategy_kwargs)
      if strategy_step == 'BFGS':
        minimizer_func = scipy.optimize.fmin_bfgs
      elif strategy_step == 'CG':
        minimizer_func = scipy.optimize.fmin_cg
      # TODO(tfish): Check full output for convergence.
      # Not much of a problem currently, since we are always
      # checking stationarity.
      xs_now = _fixed_fmin(f_opt,
                           xs_now,
                           fprime=fprime_opt,
                           minimizer_func=minimizer_func,
                           **kwargs)
    elif strategy_step == 'CUSTOM':
      minimizer_func = strategy_details[0]
      opt_info = _fixed_fmin(f_opt,
                             xs_now,
                             fprime=fprime_opt,
                             minimizer_func=minimizer_func)
      # TODO(tfish): Check full output for convergence.
      # Not much of a problem currently, since we are always
      # checking stationarity.
      xs_now = opt_info[0]
    elif strategy_step == 'MDNewton':  # Multi-Dimensional Newton
      kwargs = dict(maxsteps=default_mdnewton_maxsteps)
      kwargs.update(strategy_kwargs)
      *_, t_xs_opt = tf_mdnewton(
          tf_scalar_func,
          tf.constant(xs_now, dtype=dtype),
          tf_grad_func=tf_grad_func,
          **kwargs)
      xs_now = t_xs_opt.numpy()
    elif strategy_step == 'GD':  # Gradient Descent.
      kwargs = dict(schedule=default_gd_schedule,
                    show_gradient_loss_every_n_steps=None)
      kwargs.update(strategy_kwargs)
      show_gradient_loss_every_n_steps = kwargs.get(
          'show_gradient_loss_every_n_steps')
      for num_steps, learning_rate in kwargs['schedule']:
        for num_gradient_step in range(num_steps):
          xs_new = xs_now - learning_rate * fprime_opt(xs_now)
          if f_opt(xs_new) > f_opt(xs_now):
            # Gradient-step did not reduce the function-value.
            break  # Do not proceed with this learning-rate.
          else:
            xs_now = xs_new  # Accept new position.
          if (show_gradient_loss_every_n_steps is not None and
              num_gradient_step % show_gradient_loss_every_n_steps == 0):
            print('[gradient, lr=%.6g, step=%4d] L=%.6g' % (
                learning_rate, num_gradient_step, f_opt(xs_now)))
    else:
      raise ValueError('Unknown strategy step: %r' % strategy_step)
  return f_opt(xs_now), xs_now


def tf_reshaped_to_1_batch_index(t_x, num_non_batch_indices):
  """Reshapes tf-tensor `t_x` to a single batch-index."""
  return tf.reshape(
      t_x,
      tf.concat([tf.constant([-1], dtype=tf.int32),
                 tf.shape(t_x)[-num_non_batch_indices:]], axis=0))


def tf_restore_batch_indices(tb1_x, t_ref, num_non_batch_indices_ref):
  """Restores the batch indices on `t_ref` to `tb1_x`."""
  return tf.reshape(
      tb1_x,
      tf.concat(
          [tf.shape(t_ref)[:-num_non_batch_indices_ref],
           tf.shape(tb1_x)[1:]], axis=0))


def gramian_eigenspaces(gramian, eps=1e-5):
  """Returns eigenspaces of a Gramian matrix."""
  # .eigh() will give us an orthonormal basis, while .eigvals() typically would
  # not (even for a hermitean matrix).
  eigvals, eigvecsT = numpy.linalg.eigh(gramian)
  eigenspaces = []  # [(eigval, [eigvec1, eigvec2, ...]), ...]
  for eigval, eigvec in zip(eigvals, eigvecsT.T):
    matching_eigenspaces = (
        espace for espace in eigenspaces if abs(espace[0] - eigval) <= eps)
    try:
      espace = next(matching_eigenspaces)
      espace[1].append(eigvec)
    except StopIteration:  # Did not have a matching eigenvalue.
      eigenspaces.append((eigval, [eigvec]))
  return [(eigval, numpy.vstack(eigvecs))
          for eigval, eigvecs in eigenspaces]


def decompose_residual_symmetry(v_scalars, f_abC,
                                gg_generators,
                                threshold=1e-3):
  """Decomposes the unbroken gauge symmetry into {semisimple}+{u1s} pieces."""
  dim_ad = f_abC.shape[-1]
  comm_v_scalars = numpy.einsum('abc,a->cb',
                                f_abC[:len(v_scalars), :, :],
                                v_scalars)
  gg_comms = numpy.einsum('na,ba->nb',
                          gg_generators, comm_v_scalars)
  svd_u, svd_s, svd_vh = numpy.linalg.svd(gg_comms)
  del svd_vh  # Unused.
  # TODO(tfish): Perhaps use .null_space()?
  unbroken_gg_gens = [
      numpy.einsum('ng,n->g', gg_generators, u)
      for u in svd_u.T[abs(svd_s) <= threshold]]
  assert all(abs(numpy.dot(comm_v_scalars, v)).max() < 1e-3
             for v in unbroken_gg_gens)
  ugg = (numpy.vstack(unbroken_gg_gens) if unbroken_gg_gens
         else numpy.zeros([0, dim_ad]))
  # We need the geometric object that codifies taking the commutator
  # with an unbroken gauge-group generator.
  comm_ugg = nsum('ma,abc->mcb', ugg, f_abC)
  # We can use this to determine the derivative.
  d_ugg = nsum('nb,mcb->mnc', ugg, comm_ugg)
  svd_du, svd_ds, svd_dvh = numpy.linalg.svd(d_ugg.reshape(-1, dim_ad))
  del svd_du  # Unused.
  d_ugg_gens = svd_dvh[:len(svd_ds)][svd_ds > threshold, :]
  # Also, those unbroken-gauge-group generators that commute
  # with all unbroken-gauge-group generators give us the U(1)
  # generators (for groups that are "semisimple plus U1's").
  if ugg.size == 0:
    u1_gens = numpy.zeros([0, dim_ad])
  else:
    svd_cu, svd_cs, svd_cvh = numpy.linalg.svd(d_ugg.reshape(ugg.shape[0], -1))
    del svd_cvh  # Unused.
    u1_gens = nsum('pn,pa->na', svd_cu[:, svd_cs <= threshold], ugg)
  return ugg, d_ugg_gens, u1_gens


def gen_x(gens_a, *rest):
  """Computes the action of symmetry generators on a tensor product space.

  Args:
    gens_a: [dim_ad, dim_a, dim_a]-ndarray, action of the Lie algebra
      generators on an a-dimensional representation.
    *rest: Further generator-actions for other representations, in the same form
      as above.

  Returns:
    [dim_ad, dim_b, dim_b]-ndarray, action of the Lie algebra generators on the
      b-dimensional tensor product representation.
  """
  if not rest:
    return gens_a
  gens_b, *rest_rest = rest
  dim_ad = gens_a.shape[0]
  if dim_ad != gens_b.shape[0]:
    raise ValueError(
        'Lie algebra dimension mismatch: '
        f'{gens_a.shape[0]} vs. {gens_b.shape[0]}.')
  dim_a = gens_a.shape[1]
  dim_b = gens_b.shape[1]
  dim_ab = dim_a * dim_b
  gens_ab = (
      # The Leibniz rule tells us how to transform the tensor product.
      numpy.einsum(
          'aMN,mn->aMmNn',
          gens_a, numpy.eye(dim_b)).reshape(dim_ad, dim_ab, dim_ab) +
      numpy.einsum(
          'amn,MN->aMmNn',
          gens_b, numpy.eye(dim_a)).reshape(dim_ad, dim_ab, dim_ab))
  return gen_x(gens_ab, *rest_rest)


def a_act(reps, actions):
  """Computes the Lie algebra action on a product representation.

  Higher-level convenience interface on top of the `gen_x` primitive.

  Example: a_act('vvv', dict(v=eps3)) are the
    generators, as a [3, 27, 27]-array, of SO(3) acting on
    the (vector) x (vector) x (vector) representation.

  Args:
    reps: str, sequence of letters describing the representation.
    actions: Mapping from representation-names to [dim_ad, d, d]-actions.

  Returns:
    [dim_ad, product(d), product(d)]-ndarray, action on the product space.
  """
  return gen_x(*(actions[rep] for rep in reps))


def get_lie_algebra_rank(gens, f_ABC, threshold=1e-3):
  """Numerically determines the rank of a Lie algebra."""
  # Strategy: Take a random element of the unbroken symmetry group
  # (which we can think of as being a random linear combination
  # of Cartan generators) and find the largest subalgebra
  # that commutes with this random element.
  rng = numpy.random.RandomState(seed=0)
  v_random = numpy.einsum('n,na->a',
                          rng.normal(size=gens.shape[0]),
                          gens)
  comm_v_random = numpy.einsum('abc,a->cb', f_ABC, v_random)
  comms = numpy.einsum('na,ba->nb', gens, comm_v_random)
  svd_s = numpy.linalg.svd(comms)[1]
  return (svd_s <= threshold).sum()


def get_simultaneous_eigenspace(linear_ops, eigenvalues, threshold=1e-6):
  """Computes a simultaneous eigenspace for multiple linear operators.

  Args:
    linear_ops: Sequence of k [N, N]-ndarrays, the k linear mappings.
    eigenvalues: sequence of eigenvalues.
    threshold: Maximal magnitude of the SVD-generalized eigenvalue.

  Returns:
    A basis for the D-dimensional eigenspace, as a [D, N]-ndarray.
  """
  dim = linear_ops[0].shape[-1]
  null_ops = numpy.stack([
      lin_op - eigval * numpy.eye(dim)
      for lin_op, eigval in zip(linear_ops, eigenvalues)], axis=0)
  svd_u, svd_s, svd_vh = numpy.linalg.svd(null_ops.reshape(-1, dim))
  del svd_u  # Unused.
  return svd_vh[svd_s <= threshold, :]


def weight_decompose(linear_ops, space=None, inner_product=None, d=2):
  """Weight-decomposes a collection of operators acting on some space."""
  # `linear_ops` may be a sequence or a numpy ndarray. In the latter case,
  # implicit length test won't work.
  # pylint:disable=g-explicit-length-test
  if len(linear_ops) == 0:
    return {(): space}
  dim = linear_ops[0].shape[1]
  if space is None:
    space = numpy.eye(dim)
  if inner_product is None:
    inner_product = numpy.eye(dim)
  sub_decomposition = weight_decompose(linear_ops[1:], d=d,
                                       space=space, inner_product=inner_product)
  op = linear_ops[0]
  ret = {}
  for sub_weight, sub_weightspace in sub_decomposition.items():
    sub_weightspace_inner_product = nsum(
        'ma,nb,ab->mn', sub_weightspace.conj(), sub_weightspace, inner_product)
    op_on_sub_weightspace, *_ = (
        numpy.linalg.lstsq(
            sub_weightspace_inner_product,
            nsum('mc,cb,ba,na->mn',
                 sub_weightspace.conj(),
                 inner_product, op,
                 sub_weightspace)))
    # Subtle problem: We cannot use numpy.linalg.eigh() even despite
    # knowing that all eigenvalues will be real, since
    # `op_on_sub_weightspace` might fail to be hermitean nevertheless.
    # Unfortunately, numpy.linalg.eig() does not in general return
    # a good basis for highly degenerate eigenspaces. Practically,
    # this means that for the 7-th or so Cartan generator,
    # we may well see relative deviations of the order of ~1%
    # in eigenvalue.
    op_eigvals, op_eigvecsT = numpy.linalg.eig(op_on_sub_weightspace)
    eigvecs_by_eigval = {}
    for eigval, eigvec in zip(op_eigvals, op_eigvecsT.T):
      r_eigval = numpy.round(eigval, d)
      eigvecs_by_eigval.setdefault(r_eigval, [])
      eigvecs_by_eigval[r_eigval].append(eigvec)
    for eigval, eigvecs in eigvecs_by_eigval.items():
      weight = (eigval,) + sub_weight
      ret[weight] = nsum('na,mn->ma',
                         sub_weightspace,
                         numpy.stack(eigvecs, axis=0))
  return ret


def product_decompose_rotation(rot):
  """Decomposes a rotation matrix into Davenport chained rotations."""
  dim = rot.shape[0]
  factors = []
  work_rot = rot.copy()
  for axis0 in range(dim):
    for axis1 in sorted(
        range(1 + axis0, dim),
        # Resolve largest-to-be-zeroed-out
        # index first.
        key=lambda n: -abs(work_rot[axis0, n])):  # pylint:disable=cell-var-from-loop
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
  for ija in decomposed_rotation:
    partial = numpy.einsum('ij,jk->ik', get_part(ija), partial)
  return partial


def get_residual_symmetry_of_matrix_diagonal(diag_entries, max_deviation=1e-4):
  """Finds all generators that leave a given matrix diagonal invariant."""
  # Max deviation of 1e-4 empirically makes sense for most solutions.
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
    generators.extend((i1, i2)
                      for i1 in index_group for i2 in index_group
                      if i1 < i2)
  return generators


def get_random_so_n_z_rotation(n, rng=None):
  """Generates a random SO(N, Z) element."""
  if rng is None:
    rng = numpy.random.RandomState()
  perm = list(range(n))
  rng.shuffle(perm)
  signs = rng.choice((-1, 1), size=n)
  is_odd = bool(((permutation_sign(perm) == -1) +
                 sum(x == -1 for x in signs)) % 2)
  if is_odd:
    # If overall parity is 'odd', flip one sign to fix this.
    signs[0] *= -1
  so_n_z_rotation = numpy.zeros([n, n])
  for k in range(n):
    so_n_z_rotation[perm[k], k] = signs[k]
  return so_n_z_rotation


def get_generators_for_post_diagonalization_reduction(
    diag_entries,
    einsum_spec, rep_action):
  """Builds the generators for canonicalization after 1st diagonalization."""
  gens = get_residual_symmetry_of_matrix_diagonal(diag_entries)
  m = numpy.zeros([len(gens), 8, 8])
  for i, (a, b) in enumerate(gens):
    m[i, a, b] = +1
    m[i, b, a] = -1
  return numpy.einsum(einsum_spec, m, rep_action)


# Model-parameter ratios that will be automatically recognized.
_NICE_RATIOS = {}
for pq, f in [((p, q), p / float(q))
              for p in range(1, 12) for q in range(1, 9)]:
  if f not in _NICE_RATIOS:
    _NICE_RATIOS[f] = pq


def model_vector(vec, registered_coeffs_prev=(), digits=4):
  """Helper for modeling a vector in terms of almost-identical entries.

  This helper scans the entries of a vector,
  extracting almost-zero, almost-identical, and
  almost-in-simple-arithmetic-ratio entries. This allows us to build
  low-dimensional models for approximate vectors that describe
  a solution.

  Args:
    vec: array of approximate numerical data.
    registered_coeffs_prev: List of coefficients extracted for another such
      vector.
    digits: How many decimal digits to take into account when determining
      whether two matrix entries are almost-identical.

  Returns:
    A pair (registered_coeffs, canonicalized), where `registered_coeffs`
    is a list of known coefficients, and `canonicalized` is a list of triplets
    of the form (index, canonicalized_id, (numerator, denominator)).
  """
  # TODO(tfish): Improve handling of known linear constraints between
  # model-parameters.
  registered_coeffs = list(registered_coeffs_prev)  # Copy.
  threshold = 10**(-digits)
  nice_ratio_low, nice_ratio_high = 1 - 5 * threshold, 1 + 5 * threshold
  entry_by_idx = {}
  for idx, v in enumerate(vec):
    if abs(v) >= threshold:
      entry_by_idx[idx] = v
  # Try to find nice identities.
  # Let us start from the smallest-in-magnitude entries
  # and proceed by magnitude.
  def register_coeff(c):
    """Registers coeff and returns (registered_index, factor)."""
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
  for idx, coeff in sorted(entry_by_idx.items(),
                           # Proceed by increasing magnitude.
                           key=lambda kv: abs(kv[1])):
    canonicalized_id, numer_denom = register_coeff(coeff)
    canonicalized.append((idx, canonicalized_id, numer_denom))
  return registered_coeffs, canonicalized


def tformat(array,
            name=None,
            d=None,
            filter=lambda x: abs(x) > 1e-8,  # pylint:disable=redefined-builtin
            format='%s',  # pylint:disable=redefined-builtin
            nmax=numpy.inf,
            ncols=120):
  """Formats a numpy-array in human readable table form."""
  # Leading row will be replaced if caller asked for a name-row.
  if d is not None:
    array = array.round(d)
  dim_widths = [
      max(1, int(math.ceil(math.log(dim + 1e-100, 10))))
      for dim in array.shape]
  format_str = '%s: %s' % (' '.join('%%%dd' % w for w in dim_widths), format)
  rows = []
  for indices in itertools.product(*[range(dim) for dim in array.shape]):
    v = array[indices]
    if filter(v):
      rows.append(format_str % (indices + (v,)))
  num_entries = len(rows)
  if num_entries > nmax:
    rows = rows[:nmax]
  if ncols is not None:
    width = max(map(len, rows)) if rows else 80
    num_cols = max(1, ncols // (3 + width))
    num_xrows = int(math.ceil(len(rows) / num_cols))
    padded = [('%%-%ds' % width) % s
              for s in rows + [''] * (num_cols * num_xrows - len(rows))]
    table = numpy.array(padded, dtype=object).reshape(num_cols, num_xrows).T
    xrows = [' | '.join(row) for row in table]
  else:
    xrows = rows
  if name is not None:
    return '\n'.join(
        ['=== %s, shape=%r, %d%s / %d non-small entries ===' % (
            name, array.shape,
            num_entries,
            '' if num_entries == len(rows) else ' (%d shown)' % len(rows),
            array.size)] +
        [r.strip() for r in xrows])
  return '\n'.join(r.strip() for r in xrows)


def tprint(array, sep=' ', end='\n', file=sys.stdout, **tformat_kwargs):
  """Prints a numpy array in human readable table form."""
  print(tformat(array, **tformat_kwargs), sep=sep, end=end, file=file)


def csv_numdata(filename, numdata_start_column=0):
  """Yields the row-vectors of an all-numerical CSV file."""
  with open(filename, 'rt') as h:
    # We do not `yield` from within a with-context.
    rows = list(h)
  for row in rows:
    rawdata = row.split(',')
    yield rawdata[:numdata_start_column] + [
        float(x) for x in rawdata[numdata_start_column:]]


def csv_save(filename, array, magnitude_threshold=0):
  """Saves an array as CSV data."""
  with open(filename, 'wt') as h:
    for indices, value in sorted(
        sparse_dict_from_array(
            array, magnitude_threshold=magnitude_threshold).items()):
      h.write('%r,%s\n' % (value, ','.join(map(str, indices))))


def hdf5_fetch(hdf5_filename, path):
  """Fetches array data plus .attrs from a hdf5 file."""
  # HDF5 is slightly esoteric, and should not be a general dependency.
  # Only import it if this function actually does get used.
  import h5py  # pylint:disable=g-import-not-at-top
  with h5py.File(hdf5_filename, 'r') as h_in:
    data = h_in[path]
    return numpy.array(data), dict(data.attrs.items())
