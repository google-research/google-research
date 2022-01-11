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

"""Generic A1,A2,A3 code for all maximal gauged D=4 supergravities.

Computations that depend on the four-dimensional fermion shifts only.
This includes:

A3 from A2, the 'gradient in the local base' stationarity criterion,
and the superpotential.
"""

import itertools

from m_theory_lib import algebra
from m_theory_lib import m_util as mu

import numpy
import tensorflow as tf

# Naming deviates from PEP-8 conventions where this makes mathematics easier
# to read. Also, local variables may name-match module-global definitions.
# pylint:disable=invalid-name
# pylint:disable=redefined-outer-name


def _get_35sd_asd_bases():
  """Obtains orthonormal bases for the 35-dim sd/asd 4-forms."""
  # We could use the spin8.gamma_vvvvss / spin8.gamma_vvvvcc to construct
  # these bases, but then, naively orthonormalizing the inner product
  # on the 35-irrep would leave us with weird square roots. Hence,
  # we rebuild this directly from combinatorics here.
  mapping_sd = numpy.zeros([8, 8, 8, 8, 35])
  mapping_asd = numpy.zeros([8, 8, 8, 8, 35])
  s8 = set(range(8))
  for a, ijkl in enumerate(itertools.combinations(range(8), 4)):
    if 0 not in ijkl:
      # Lexicographic order ensures we get "all entries with 0" first.
      break
    mnpq = tuple(sorted(s8 - set(ijkl)))
    sign44 = mu.permutation_sign(ijkl + mnpq)
    for p0123 in itertools.permutations(range(4)):
      p0, p1, p2, p3 = p0123
      sign_p = mu.permutation_sign(p0123)
      ijkl_a = (ijkl[p0], ijkl[p1], ijkl[p2], ijkl[p3], a)
      mnpq_a = (mnpq[p0], mnpq[p1], mnpq[p2], mnpq[p3], a)
      mapping_sd[ijkl_a] = sign_p
      mapping_sd[mnpq_a] = sign_p * sign44
      mapping_asd[ijkl_a] = sign_p
      mapping_asd[mnpq_a] = -sign_p * sign44
  return mapping_sd, mapping_asd


@tf.function
def tf_dwn_stationarity_vec(t_A1, t_A2):
  """Computes stationarity-violation 70-vector 'in the local frame'."""
  # Formula:
  # arXiv: https://arxiv.org/pdf/1302.6219.pdf, formula (3.2);
  # originally: https://inspirehep.net/literature/191530
  # (2.20) - (2.22).
  _m_8888sd_35ortho, _m_8888asd_35ortho = _get_35sd_asd_bases()
  t_x0 = (tf.einsum('mi,mjkl->ijkl', t_A1, t_A2)
          + mu.tfc128(-0.75, 0.0) * tf.einsum('mnij,nklm->ijkl', t_A2, t_A2))
  t_x0_real = tf.math.real(t_x0)
  t_x0_imag = tf.math.imag(t_x0)
  # The self-dual part must be zero.
  t_x0_re_sd = tf.einsum(
      'ijkl,ijklX->X', t_x0_real, mu.tff64(_m_8888sd_35ortho))
  t_x0_im_asd = tf.einsum(
      'ijkl,ijklX->X', t_x0_imag, mu.tff64(_m_8888asd_35ortho))
  return tf.concat([t_x0_re_sd, t_x0_im_asd], axis=0)


@tf.function
def tf_dwn_stationarity(t_A1, t_A2):
  """Computes stationarity-violation 'in the local frame'."""
  return (
      tf.math.reduce_sum(tf.math.square(
          tf_dwn_stationarity_vec(t_A1, t_A2))) / mu.tff64(48.0))


@tf.function
def tf_A3_from_A2(t_A2):
  """Computes the A3-tensor from the A2-tensor."""
  tc_56_8_8_8 = tf.constant(
      algebra.g.su8.m_56_8_8_8.astype(numpy.complex128),
      dtype=tf.complex128)
  tc_eps_56_56_8_8 = tf.constant(
      algebra.g.su8.eps_56_56_8_8.astype(numpy.complex128),
      dtype=tf.complex128)
  t_A2_nP = tf.einsum('nijk,Pijk->nP',
                      tf.math.conj(t_A2),
                      tc_56_8_8_8)
  return tf.einsum('nP,APlm,Blmn->AB',
                   t_A2_nP,
                   tc_eps_56_56_8_8,
                   tc_56_8_8_8) * mu.tfc128(2**.5 / 24.0)


@tf.function
def tf_fermion_massmatrix(t_A3, t_potential, tc_masses_factor):
  """Computes the spin-1/2 mass matrix from the A3-tensor."""
  # The extra factor 2.0 relative to https://arxiv.org/abs/1906.00207
  # makes the fermion masses align with the way particle states are
  # grouped into SUSY multiplets in appendix (B.2) of:
  # https://arxiv.org/abs/1909.10969
  return mu.tfc128(2.0) * tf.einsum(
      'ij,ik->jk',
      t_A3, tf.math.conj(t_A3)) * (
          tc_masses_factor /
          tf.cast(t_potential, tf.complex128))


@tf.function
def tf_vector_massmatrix(t_A2, t_potential, tc_masses_factor):
  """Computes the spin-1 mass matrix from the A2-tensor."""
  # This is most readily available in https://arxiv.org/pdf/1103.2785.pdf,
  # Eq. (5.27) and (5.28), setting the scaling-symmetry ('trombone')
  # contributions B to zero.
  # With the conventions from arXiv:1103.2785, we reproduce
  # Table B.3 of arXiv:1909.10969.
  # Note that we get 56 masses, 28 of which are spurious and will always be zero
  # in any valid gauging.
  tc_56_8_8_8 = tf.constant(
      algebra.g.su8.m_56_8_8_8.astype(numpy.complex128),
      dtype=tf.complex128)
  tc_28_8_8 = tf.constant(
      algebra.g.su8.m_28_8_8.astype(numpy.complex128),
      dtype=tf.complex128)
  tc_eps_56_56_8_8 = tf.constant(
      algebra.g.su8.eps_56_56_8_8.astype(numpy.complex128),
      dtype=tf.complex128)
  t_A2_8_56 = mu.tfc128(1 / 6.0) * tf.einsum(
      'ipqr,Xpqr->iX', t_A2, tc_56_8_8_8)
  t_A2c_8_56 = tf.math.conj(t_A2_8_56)
  M_ij_KL = (
      mu.tfc128(-1 / 4.0) * tf.einsum(
          'iX,lX,jk,Iij,Kkl->IK',
          t_A2_8_56, t_A2c_8_56,
          tf.eye(8, dtype=tf.complex128),
          tc_28_8_8, tc_28_8_8) +
      mu.tfc128(1 / 8.0) * tf.einsum(
          'ipqk,ljpq,Iij,Kkl->IK',
          t_A2, tf.math.conj(t_A2), tc_28_8_8, tc_28_8_8))
  M_ijkl = mu.tfc128(1 / 4.0) * tf.einsum(
      'iX,lY,XYjk,Iij,Kkl->IK',
      t_A2_8_56, t_A2_8_56,
      tc_eps_56_56_8_8,
      tc_28_8_8, tc_28_8_8)
  M_vec = tf.reshape(
      tf.einsum('abAB->aAbB',
                tf.reshape(
                    tf.stack([M_ij_KL, M_ijkl,
                              tf.math.conj(M_ijkl), tf.math.conj(M_ij_KL)]),
                    [2, 2, 28, 28])),
      [56, 56])
  return (M_vec * tc_masses_factor /
          tf.cast(t_potential, tf.complex128))


def superpotential(A1, direction=None):
  """Computes the superpotential from A1.

  Args:
    A1: [8, 8]-complex128-ndarray, the A1-tensor.
    direction: Optional [8]-ndarray, the direction along which to evaluate A1.
      If this is not provided, the mass matrix eigendirection corresponding
      to the lightest gravitino is picked.

  Returns:
    The superpotential, as a complex number.
  """
  A1_sq = A1.dot(A1.conjugate())
  if direction is None:
    a1_sq_eigvals, a1_sq_eigvecsT = numpy.linalg.eigh(A1_sq)
    direction = a1_sq_eigvecsT[:, numpy.argmin(a1_sq_eigvals)]
  return direction.dot(A1.dot(direction))
