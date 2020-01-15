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

"""Generic implementation of the potential+stationarity computation.

Depending on how this is integrated, it can run the computation with
double-float numpy arrays, mpmath numpy arrays, or TensorFlow 1.x graph-objects.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import numpy
import scipy.linalg

from dim4.so8_supergravity_extrema.code import algebra


ScalarInfo = collections.namedtuple(
    'ScalarInfo',
    ['stationarity',
     'potential',
     'potential_a1',
     'potential_a2',
     'generator',
     'vielbein',
     't_tensor',
     'a1',
     'a2',
     # 70-vector, with entries corresponding to the orthonormal bases of
     # (anti)self-dual [ijkl] +/- {complement}.
     'grad_potential',
    ])


def get_unscaled_proj_35_8888_sd_asd(dtype=numpy.int32):
  """Computes the [35, 8, 8, 8, 8]-projector to the (anti)self-dual 4-forms."""
  # In principle, we could use the so8.gamma_vvvvss and so8.gamma_vvvvcc here,
  # but we opt for instead constructing some explicit orthonormal bases for the
  # 35-dimensional irreps as this makes it easier to stay exact despite using
  # floating point numbers.
  #
  # We first need some basis for the 35 self-dual 4-forms.
  # Our convention is that we lexicographically list those 8-choose-4
  # combinations that contain the index 0.
  ret_sd = numpy.zeros([35, 8, 8, 8, 8], dtype=dtype)
  ret_asd = numpy.zeros([35, 8, 8, 8, 8], dtype=dtype)
  #
  def get_complementary(ijkl):
    mnpq = tuple(n for n in range(8) if n not in ijkl)
    return (algebra.permutation_sign(ijkl + mnpq), ijkl, mnpq)
  #
  complements = [get_complementary(ijkl)
                 for ijkl in itertools.combinations(range(8), 4)
                 if 0 in ijkl]
  for num_sd, (sign, ijkl, mnpq) in enumerate(complements):
    for abcd in itertools.permutations(range(4)):
      sign_abcd = algebra.permutation_sign(abcd)
      for r, part2_sign in ((ret_sd, 1), (ret_asd, -1)):
        r[num_sd,
          ijkl[abcd[0]], ijkl[abcd[1]],
          ijkl[abcd[2]], ijkl[abcd[3]]] = sign_abcd
        r[num_sd,
          mnpq[abcd[0]], mnpq[abcd[1]], mnpq[abcd[2]],
          mnpq[abcd[3]]] = sign_abcd * sign * part2_sign
  return ret_sd, ret_asd


def get_scalar_manifold_evaluator(
    frac=lambda p, q: p / float(q),
    to_scaled_constant=lambda x, scale=1: numpy.array(x) * scale,
    expm=scipy.linalg.expm,
    einsum=numpy.einsum,
    eye=lambda n: numpy.eye(n, dtype=numpy.complex128),
    # We need tracing-over-last-two-indices as a separate operation, as
    # tf.einsum() cannot trace.
    # Caution: numpy.trace() by default traces over the first two indices,
    # but tf.trace() traces over the last two indices.
    trace=lambda x: numpy.trace(x, axis1=-1, axis2=-2),
    concatenate=numpy.concatenate,
    complexify=lambda a: a.astype(numpy.complex128),
    re=lambda z: z.real,
    im=lambda z: z.imag,
    conjugate=lambda z: z.conj()):
  """Wraps up the potential and stationarity computation.

  This allows writing the potential and stationarity expressions only once,
  but going through the computation in different ways, producing e.g.
  - numpy arrays (complex-valued).
  - numpy arrays (mpmath.mpc high-precision complex-valued)
  - TensorFlow 1.x 'tensor' graph components.

  As this needs to first wrap up some constants, we return a function
  that maps an object representing a 70-vector location to a pair of
  (potential, stationarity).

  Args:
    frac: Function mapping two integers to a fraction (to be overridden for
      high-precision computations).
    to_scaled_constant: Function mapping an array-like to a scaled constant.
      (To be overridden with tf.constant() for TensorFlow).
    expm: Matrix exponentiation function.
    einsum: Generalized Einstein summation. A function that is compatible
      with tf.einsum() or numpy.einsum() calling conventions.
    eye: Generalized `numpy.eye()` function returning a complex identity-matrix.
    trace: Function mapping a rank-(N+2) array to a rank-N array by tracing
      over the last two indices (needed as tf.einsum() does not support this
      operation.)
    concatenate: Concatenation function compatible with numpy.concatenate.
    complexify: A function that maps a real array to a corresponding complex
      array.
    re: Function mapping a number to its real part. Needed as
      numpy.array([mpmath.mpc(1+1j)]).imag does not propagate the `.imag`
      method to the dtype=object numbers.
    im: Like `re`, but extracts the imaginary part.

  Returns: A function f(v70) -> ScalarInfo that maps a 70-vector of
    e7/su(8) generator-coefficients in the (35s+35c)-basis to a `ScalarInfo`
    object with information about the corresponding point on the scalar
    manifold.
  """
  t_28_8_8 = to_scaled_constant(algebra.su8.m_28_8_8.astype(numpy.complex128))
  t_e7_a_ij_kl = to_scaled_constant(algebra.e7.t_a_ij_kl[:70, :, :])
  uproj_35_8888_sd, uproj_35_8888_asd = get_unscaled_proj_35_8888_sd_asd()
  proj_35_8888_sd = to_scaled_constant(uproj_35_8888_sd, scale=frac(1, 24))
  proj_35_8888_asd = to_scaled_constant(uproj_35_8888_asd, scale=frac(1, 24))
  #
  def expand_ijkl(t_ab):
    """Index-expands 28, 28 -> [8, 8] [8, 8]."""
    return frac(1, 2) * einsum(
        'ijB,BIJ->ijIJ',
        einsum('AB,Aij->ijB', t_ab, t_28_8_8), t_28_8_8)
  #
  def evaluator(t_v70, t_left=None, t_right_vielbein=None):
    # t_left, if provided, is exponentiated to 2nd degree in the Taylor series
    # and multiplies the V-matrix from the left. Taking the 2nd derivative
    # w.r.t. `t_left` hence gives us the scalar mass matrix as appropriate for
    # a canonically normalized kinetic term.
    # t_right_vielbein, if provided, is a Vielbein matrix that overrides
    # the exponentiated Vielbein. This is useful for quickly exploring around
    # a background (but we cannot do mass matrices then).
    t_gen_56_56 = None
    if t_right_vielbein is None:
      t_gen_56_56 = einsum('v,vIJ->JI', complexify(t_v70), t_e7_a_ij_kl)
      t_mid_vielbein = expm(t_gen_56_56)
      t_mr_vielbein = t_mid_vielbein
    else:
      t_mr_vielbein = t_right_vielbein  # Actually override.
    if t_left is None:
      t_vielbein = t_mr_vielbein
    else:
      t_gen_left = einsum('v,vIJ->JI', t_left, t_e7_a_ij_kl)
      t_left_degree2 = (
          eye(56) + t_gen_left
          + 0.5 * einsum('ab,bc->ac', t_gen_left, t_gen_left))
      t_vielbein = einsum('ab,bc->ac', t_left_degree2, t_mr_vielbein)
    t_u_ijIJ = expand_ijkl(t_vielbein[:28, :28])
    t_u_klKL = conjugate(t_u_ijIJ)
    t_v_ijKL = expand_ijkl(t_vielbein[:28, 28:])
    t_v_klIJ = conjugate(t_v_ijKL)
    t_uv = t_u_klKL + t_v_klIJ
    t_uuvv = (
        einsum('lmJK,kmKI->lkIJ', t_u_ijIJ, t_u_klKL) -
        einsum('lmJK,kmKI->lkIJ', t_v_ijKL, t_v_klIJ))
    t_t = einsum('ijIJ,lkIJ->lkij', t_uv, t_uuvv)
    t_a1 = frac(-4, 21) * trace(einsum('mijn->ijmn', t_t))
    t_a2 = frac(-4, 3 * 3) * (
        # Antisymmetrize in last 3 indices, but using antisymmetry in last 2.
        # Note factor 1/3 above (in -4/(3*3) rather than -4/3).
        t_t + einsum('lijk->ljki', t_t) + einsum('lijk->lkij', t_t))
    t_a1_real = re(t_a1)
    t_a1_imag = im(t_a1)
    t_a2_real = re(t_a2)
    t_a2_imag = im(t_a2)
    t_potential_a1 = frac(-3, 4) * (
        einsum('ij,ij->', t_a1_real, t_a1_real) +
        einsum('ij,ij->', t_a1_imag, t_a1_imag))
    t_potential_a2 = frac(1, 24) * (
        einsum('ijkl,ijkl->', t_a2_real, t_a2_real) +
        einsum('ijkl,ijkl->', t_a2_imag, t_a2_imag))
    t_potential = t_potential_a1 + t_potential_a2
    t_x0 = (+frac(4, 1) * einsum('mi,mjkl->ijkl', t_a1, t_a2)
            -frac(3, 1) * einsum('mnij,nklm->ijkl', t_a2, t_a2))
    t_x0_real = re(t_x0)
    t_x0_imag = im(t_x0)
    t_x0_real_sd = einsum('aijkl,ijkl->a', proj_35_8888_sd, t_x0_real)
    t_x0_imag_asd = einsum('aijkl,ijkl->a', proj_35_8888_asd, t_x0_imag)
    t_stationarity = frac(1, 2) * (
        einsum('a,a->', t_x0_real_sd, t_x0_real_sd) +
        einsum('a,a->', t_x0_imag_asd, t_x0_imag_asd))
    return ScalarInfo(
        stationarity=t_stationarity,
        potential=t_potential,
        # It makes sense to also expose the a1-potential and a2-potential.
        potential_a1=t_potential_a1,
        potential_a2=t_potential_a2,
        generator=t_gen_56_56,
        vielbein=t_vielbein,
        t_tensor=t_t,
        a1=t_a1,
        a2=t_a2,
        grad_potential=concatenate([t_x0_real_sd,
                                    t_x0_imag_asd]))
  return evaluator


numpy_scalar_manifold_evaluator = get_scalar_manifold_evaluator()


def get_a3_56x56_from_a2(a2,
                         sqrt2=2**.5,
                         einsum=numpy.einsum,
                         conjugate=lambda z: z.conj()):
  """Computes the spin-1/2 fermion "naive" mass matrix A3 from A2.

  Args:
    a2: The A2^i_[jkl] tensor.
    sqrt2: The square root of 2.
      (Overridable for high-precision numerical computations.)
    einsum: `einsum` operation to use.
      (Overridable for high-precision numerical computations.)
    conjugate: array-conjugation function to use.
      (Overridable for high-precision numerical computations.)

  Returns: The [56, 56]-array of (complex) fermion masses.
  """
  # complex-conjugate this to get the same conventions as (4.30) in
  # https://arxiv.org/pdf/0705.2101.pdf
  a2_nP = einsum('nijk,Pijk->nP', conjugate(a2),
                 algebra.su8.m_56_8_8_8)
  return (sqrt2 / 24.0) * (
      einsum('Almn,Blmn->AB',
             einsum('APlm,nP->Almn',
                    algebra.su8.eps_56_56_8_8, a2_nP),
             algebra.su8.m_56_8_8_8))
