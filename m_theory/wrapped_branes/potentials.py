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

"""Scalar potential definitions for wrapped-branes models.

This includes definitions from:

  - https://arxiv.org/abs/1906.08900  ("cgr")
  - https://arxiv.org/abs/1009.3805   ("dgkv")

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import dataclasses
import numpy
import tensorflow.compat.v1 as tf

# TensorFlow based matrix exponentiation that supports higher derivatives.
# Will not be needed for TensorFlow1.15+.
from m_theory_lib import tf_cexpm
from m_theory_lib import util


@dataclasses.dataclass(frozen=True)
class Problem(object):
  num_scalars: float  # The number of scalars.
  tf_potential: callable  # TensorFlow function computing the potential.
  tf_potential_kwargs: dict  # Extra keyword args for tf_potential.


### The Scalar Potentials ###


def dim7_potential(scalars):
  """Implements the D=7 supergravity potential, (2.3) in arXiv:1906.08900."""
  # This can be regarded as a warm-up exercise.
  basis = tf.constant(util.get_symmetric_traceless_basis(5), dtype=tf.float64)
  g = tf.einsum('aAB,a->AB', basis, scalars)
  T = tf_cexpm.cexpm(g, complex_arg=False)
  trT = tf.linalg.trace(T)
  return tf.einsum('ij,ij->', T, T) - 0.5 * trT * trT


def cgr_potential(scalars, compactify_on='S2'):
  """Implements the potential (3.20) of arXiv:1906.08900."""
  basis5 = tf.constant(util.get_symmetric_traceless_basis(3), dtype=tf.float64)
  L = tf.constant(dict(S2=1, R2=0, H2=-1)[compactify_on], dtype=tf.float64)
  # Psi_a_alpha, a in (0,1), alpha in (0,1,2).
  psi = tf.reshape(scalars[:6], [2, 3])
  psi2 = tf.einsum('aA,aA->', psi, psi)
  # Parameters of a symmetric traceless 3x3 matrix.
  # (This is slightly silly here, given that we can use a SO(3) rotation
  # to diagonalize the generator.)
  tau33 = tf.einsum('aAB,a->AB', basis5, scalars[6: 11])
  s_phi, s_lambda = scalars[11], scalars[12]  # Scalars phi and lambda.
  p = 3 * s_phi - s_lambda  # Scalar phi3.
  s = tf.exp(-s_phi - 3 * s_lambda)  # Scalar sigma.
  sinv2 = tf.exp(2 * s_phi + 6 * s_lambda)
  # Here, we try to keep the arithmetic graph shallow by hand-expanding powers.
  # (We would not have to do that.)
  s2 = s * s
  s4 = s2 * s2
  e = tf.exp(-p)
  e2 = e * e
  e3 = e2 * e
  e4 = e2 * e2
  e6 = e2 * e4
  T = tf_cexpm.cexpm(tau33, complex_arg=False)
  trT = T[0, 0] + T[1, 1] + T[2, 2]  # Simplistic way to get trace.
  Tinv = tf_cexpm.cexpm(-tau33, complex_arg=False)
  psi_Tinv_psi = tf.einsum(
      'aB,cB->ac', tf.einsum('aA,AB->aB', psi, Tinv), psi)
  psi_T_psi = tf.einsum(
      'aB,cB->ac', tf.einsum('aA,AB->aB', psi, T), psi)
  ptp_ptp = (  # Manually expanded the \epsilon-factors here.
      psi_Tinv_psi[0, 0] * psi_Tinv_psi[1, 1] +   # a=0, c=0 => b=1, d=1.
      psi_Tinv_psi[1, 1] * psi_Tinv_psi[0, 0] -   # a=1, c=1 => b=0, d=0.
      psi_Tinv_psi[0, 1] * psi_Tinv_psi[1, 0] -   # a=0, c=1 => b=1, d=0.
      psi_Tinv_psi[1, 0] * psi_Tinv_psi[0, 1])    # a=1, c=0 => b=0, d=1.
  return (  # Formula (3.20) from the paper, sans factors g^2 and vol5.
      s4 * (
          - e4 * ptp_ptp
          - e2 * (psi_Tinv_psi[0, 0] + psi_Tinv_psi[1, 1])) +
      sinv2 * (
          -0.5 * e6 * (L - psi2) * (L - psi2)
          -e4 * (psi_T_psi[0, 0] + psi_T_psi[1, 1])
          +e2 * (0.5 * trT * trT - tf.einsum('AB,BA->', T, T))) +
      2 * s * (e3 * (L + psi2) + e * trT))



def dgkv_potential(scalars, compactify_on='S3'):
  """Implements the potential (3.8) of arXiv:1009.3805."""
  basis2 = tf.constant(util.get_symmetric_traceless_basis(2), dtype=tf.float64)
  L = tf.constant(dict(S3=1, R3=0, H3=-1)[compactify_on], dtype=tf.float64)
  s_phi, s_lambda, s_beta = scalars[0], scalars[1], scalars[2]
  tau22 = tf.einsum('aAB,a->AB', basis2, scalars[3:5])
  T = tf_cexpm.cexpm(tau22, complex_arg=False)
  Tinv = tf_cexpm.cexpm(-tau22, complex_arg=False)
  theta = scalars[5:7]
  chi = scalars[7:9]
  #
  tr = tf.linalg.trace
  esum = tf.einsum
  sq = tf.math.square
  exp = tf.exp
  trT = tr(T)
  th_Tinv_th = esum('a,ab,b->', theta, Tinv, theta)
  return (
      3 * L * exp(-10 * s_phi)
      -3/8 * exp(8 * s_lambda - 14 * s_phi) * sq(
          L - 2 * s_beta**2 - 2 * tf.einsum('a,a->', theta, theta))
      +0.5 * exp(-6 * s_phi) * (
          3 * exp(-8 * s_lambda)
          + exp(12 * s_lambda) * (sq(trT)
                                  - 2 * esum('ab,ab->', T, T))
          + 6 * exp(2 * s_lambda) * trT)
      -1.5 * exp(-10 * s_phi) * (
          exp(10 * s_lambda) * esum('a,ab,b->', theta, T, theta)
          - 2 * esum('a,a->', theta, theta)
          + exp(-10 * s_lambda) * th_Tinv_th)
      -6 * exp(-2 * s_lambda -14 * s_phi) * sq(s_beta) * th_Tinv_th
      -0.5 * exp(6 * s_lambda - 18 * s_phi) * esum('a,ab,b->', chi, T, chi))


### Problem definitions ###


# Problems are referred to by name (i.e. 'dim7', 'cgr-S2', etc.)
# in cgr_theory.py.
PROBLEMS = {
    # D=7 maximal supergravity.
    'dim7': Problem(num_scalars=14,
                    tf_potential=dim7_potential,
                    tf_potential_kwargs={}),  # No extra args.
    ###
    'cgr-S2': Problem(num_scalars=13,
                      tf_potential=cgr_potential,
                      tf_potential_kwargs=dict(compactify_on='S2')),
    'cgr-R2': Problem(num_scalars=13,
                      tf_potential=cgr_potential,
                      tf_potential_kwargs=dict(compactify_on='R2')),
    'cgr-H2': Problem(num_scalars=13,
                      tf_potential=cgr_potential,
                      tf_potential_kwargs=dict(compactify_on='H2')),
    ###
    'dgkv-S3': Problem(num_scalars=9,
                       tf_potential=dgkv_potential,
                       tf_potential_kwargs=dict(compactify_on='S3')),
    'dgkv-R3': Problem(num_scalars=9,
                       tf_potential=dgkv_potential,
                       tf_potential_kwargs=dict(compactify_on='R3')),
    'dgkv-H3': Problem(num_scalars=9,
                       tf_potential=dgkv_potential,
                       tf_potential_kwargs=dict(compactify_on='H3')),
}
