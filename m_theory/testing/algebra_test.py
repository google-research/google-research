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

"""Tests for m_theory_lib/algebra.py.

Usage: (from the m_theory/ directory):
  python3 -m testing.algebra_test

"""

import unittest

from m_theory_lib import algebra
from m_theory_lib import m_util as mu
import numpy
import scipy.linalg


class AlgebraTests(unittest.TestCase):
  """Tests Algebra code."""

  def test_sl2x7(self):
    # TODO(tfish): Parametrize by Spin(8)-conventions.
    e7 = algebra.g.e7
    for scv in range(3):
      self.assertTrue(
          numpy.allclose(0, mu.nsum('ma,nb,abC->mnC',
                                    e7.sl2x7[scv],
                                    e7.sl2x7[scv],
                                    e7.f_abC)))
    # The (s, c)-commutators must produce the v-generators.
    comms_sc = mu.nsum('ma,nb,abC->mnC',
                       e7.sl2x7[0], e7.sl2x7[1],
                       e7.f_abC)
    d777 = numpy.zeros([7, 7, 7])  # 'diagonal-promoter'.
    for i in range(7):
      d777[i, i, i] = 1
    comms_expected = 2 * numpy.pad(mu.nsum('Aab,nab,AB,npq->pqB',
                                           e7.su8.m_35_8_8,
                                           e7.sl2x7_88v, e7.su8.ginv35, d777),
                                   [(0, 0), (0, 0), (70, 28)])
    self.assertTrue(numpy.allclose(comms_sc, comms_expected))

  def test_get_normalizing_subalgebra_generic(self):
    """Normalizing subalgebra is as expected in a generic situation."""
    e7 = algebra.g.e7
    su8 = e7.su8
    so8_algebra = e7.f_abC[-28:, :70, :70]
    # If we pick the subspace of (8s x 8s)_sym_traceless matrices that
    # are block-diagonal with entries only in the top-left 3x3 block,
    # then there is an obvious SO(5) centralizer and an obvious
    # SO(3)xSO(5) normalizer.
    the_subspace = mu.numpy_from_nonzero_entries(
        [70, 5],
        [(1.0, 0, 0), (1.0, 1, 1),  # The diagonal-traceless parts.
         (1.0, 7 + su8.inv_ij_map[(0, 1)], 2),
         (1.0, 7 + su8.inv_ij_map[(0, 2)], 3),
         (1.0, 7 + su8.inv_ij_map[(1, 2)], 4)])
    # Let us actually rotate around this subspace with some
    # randomly-picked generic small SO(8)-rotation.
    rotated_subspace = mu.nsum(
        'an,ba->bn',
        the_subspace,
        scipy.linalg.expm(
            mu.nsum('abc,a->cb',
                    so8_algebra,
                    mu.rng(0).normal(size=(28,), scale=0.1))))
    normalizer = algebra.get_normalizing_subalgebra(
        so8_algebra,
        rotated_subspace)
    self.assertEqual((28, 3 + 10), normalizer.shape)  # dim(SO(3)xSO(5)) = 13.

  def test_get_normalizing_subalgebra_full(self):
    """Case: the normalizing subalgebra is the full algebra."""
    e7 = algebra.g.e7
    so8_algebra = e7.f_abC[-28:, :70, :70]
    # We could simply have used the identity, but let us make this
    # slightly more generic here.
    the_subspace = numpy.eye(70)[::-1, :]
    normalizer = algebra.get_normalizing_subalgebra(
        so8_algebra,
        the_subspace)
    self.assertEqual((28, 28), normalizer.shape)

  def test_get_normalizing_subalgebra_empty(self):
    """Case: the normalizing subalgebra is empty."""
    e7 = algebra.g.e7
    so8_algebra = e7.f_abC[-28:, :70, :70]
    the_subspace = mu.numpy_from_nonzero_entries(
        [70, 7],
        [(1.0, n, n) for n in range(7)])
    normalizer = algebra.get_normalizing_subalgebra(
        so8_algebra,
        the_subspace)
    self.assertEqual((28, 0), normalizer.shape)


if __name__ == '__main__':
  unittest.main()
