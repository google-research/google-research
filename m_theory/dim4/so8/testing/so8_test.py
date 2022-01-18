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

"""Tests for SO(8) supergravity using the 'class Supergravity' framework.

Usage: (from the m_theory/ directory):
  python3 -m dim4.so8.testing.so8_test

"""

import collections
import unittest

from dim4.generic import a123
from dim4.so8.src import analysis
from m_theory_lib import algebra
from m_theory_lib import m_util as mu
import numpy


# Variable names sometimes need to align with established physics
# terminology, even where this clashes with PEP-8 expectations.
# pylint:disable=invalid-name


# An actual example solution as it came out of 70-parameter minimization.
# This corresponds to V/g^2 ~ -8.4721, is slightly off, and mildly
# pre-refined on coordinates.
_EXAMPLE_SOLUTION = numpy.array(
    [-0.100214461299605, 0.0003904655072480772, 0.10560168787732979,
     0.0001846206053331566, -0.10541201145321635, -0.00020018666494099892,
     0.10521795798135813, 0.03124890893586171, -1.3186624318670186e-05,
     -0.00012159476015296525, 0.2034907592393307, -4.1722619055066416e-05,
     1.1987025017822533e-05, -6.364101480769035e-05, 6.008193812140155e-05,
     -1.2743638928198551e-05, -0.03169211787594961, 0.004729360003582559,
     4.390781140220507e-06, 6.093818821243809e-05, -0.0003182209434802144,
     2.268956011916745e-05, -7.062301909941728e-06, -0.004694070931717473,
     0.0007347973859572082, 3.388293144457841e-05, -0.0001447659747066381,
     -5.7203596940211204e-05, 0.2059008958078604, -1.7661020749291478e-05,
     4.728083502229862e-06, 9.570985598295466e-05, -8.875894208108813e-06,
     0.00025235037216740153, -0.00038211142060267175, 4.734950000618651e-05,
     0.00010495713982161435, 0.00015705429526306094, -0.00023243492525036128,
     -6.65859892532683e-05, -0.20994356905706077, -0.0006075840604978061,
     8.969577811167928e-05, 4.039016252878569e-05, -0.0002007156952244193,
     -0.00010772633342406674, 0.003187293788394058, 0.00015519229201185292,
     -4.830330059394875e-06, -0.00024116554460627883, -0.013329151965635479,
     0.0003703606439157016, -0.0016604406208685796, -0.00011180172746819031,
     2.4986088977893217e-06, 6.739362507692481e-05, -0.0031395284112149405,
     -0.0003554023786373264, 0.0013674443340173374, -0.00010378625404556671,
     -0.2096249369780446, -0.015349653216948936, -4.916566305457451e-05,
     -4.4475956454649426e-05, 0.01569020129591563, 3.8618497566838915e-06,
     0.0008386892414183921, 0.0016501898425161717, -0.00032416799511314823,
     -0.012753884351514626])


SUGRA = analysis.SO8_SUGRA()


class SO8Tests(unittest.TestCase):
  """Tests for SO(8) Supergravity (using the `class Supergravity` framework)."""

  def test_origin_so8_solution(self):
    """Asserts properties of the solution at the origin."""
    t_v70 = mu.tff64([0] * 70)
    pot, tt, a1, a2, stat = [x.numpy()
                             for x in SUGRA.tf_ext_sugra_tensors(t_v70)]
    self.assertTrue(numpy.allclose(-6.0, pot))
    self.assertTrue(0 <= stat <= 1e-20)
    self.assertTrue(numpy.allclose(numpy.eye(8), a1))
    self.assertTrue(numpy.allclose(0, a2))
    tt_entries = sorted(collections.Counter(tt.ravel()).items())
    self.assertEqual([(-0.75, 56), (0, 3984), (0.75, 56)], tt_entries)

  def test_one_nontrivial_solution(self):
    """Asserts numpy stationarity of a nontrivial solution."""
    t_v70 = mu.tff64(_EXAMPLE_SOLUTION)
    tensors_stat = SUGRA.tf_ext_sugra_tensors(t_v70, with_stationarity=True)
    tensors_nostat = SUGRA.tf_ext_sugra_tensors(t_v70, with_stationarity=False)
    self.assertTrue(0 <= tensors_stat[-1].numpy() <= 1e-3)
    self.assertTrue(-8.473 <= tensors_stat[0].numpy() <= -8.472)
    self.assertTrue(numpy.isnan(tensors_nostat[-1].numpy()))
    self.assertTrue(numpy.allclose(tensors_stat[0].numpy(),
                                   tensors_nostat[0].numpy()))

  def test_canonicalize_equilibrium(self):
    """Asserts that canonicalization does simplify a solution."""
    rng = numpy.random.RandomState(seed=0)
    canonicalized = SUGRA.canonicalize_equilibrium(
        numpy.array(_EXAMPLE_SOLUTION),
        rng=rng, verbose=False)
    self.assertIsNotNone(canonicalized)
    # Canonicalization must have removed at least the 28 parameters
    # that correspond to either the off-diagonal 35s part
    # or the off-diagonal 35c part.
    self.assertTrue(all(abs(x) < 1e-5 for x in canonicalized[7:35]) or
                    all(abs(x) < 1e-5 for x in canonicalized[35 + 7:]))
    self.assertTrue(
        numpy.allclose(
            [-8.4721, 0.0],
            SUGRA.potential_and_stationarity(canonicalized), atol=1e-3))

  def test_generic_canonicalize_equilibrium(self):
    """Asserts that default canonicalization method does simplify a solution."""
    canonicalized = SUGRA.generic_canonicalize_equilibrium(
        numpy.array(_EXAMPLE_SOLUTION), verbose=True,
        # We use a fairly high tolerance here, as _EXAMPLE_SOLUTION
        # deliberately has been picked to be a bit off,
        # so stationarity will be not quite zero.
        tolerance=1e-3)
    self.assertIsNotNone(canonicalized)
    # Canonicalization must have removed at least the 28 parameters
    # that correspond to the off-diagonal 35s part.
    self.assertTrue(
        numpy.allclose(
            [-8.4721, 0.0],
            SUGRA.potential_and_stationarity(canonicalized), atol=1e-3))
    # The generic canonicalization must at least have removed 28 parameters
    # that correspond to performing an arbitrary SO(8)-rotation.
    self.assertLessEqual(sum(abs(x) >= 1e-5 for x in canonicalized), 70 - 28)

  def test_known_so8_solutions(self):
    """Asserts numpy stationarity of the known SO(8) solutions."""
    for row in mu.csv_numdata('dim4/so8/equilibria/SO8_SOLUTIONS.csv'):
      table_potential = row[0]
      potential, stationarity = SUGRA.potential_and_stationarity(row[2:])
      self.assertTrue(0.0 <= stationarity <= 1e-15)
      self.assertTrue(numpy.isclose(potential, table_potential, atol=1e-8))

  def test_known_so8c_solutions(self):
    """Asserts numpy stationarity of the known SO(8)c omega=pi/8 solutions."""
    tc_omega = mu.tff64(numpy.pi / 8)
    for row in mu.csv_numdata('dim4/so8/equilibria/SO8C_PI8_SOLUTIONS.csv'):
      table_potential = row[0]
      potential, stationarity = SUGRA.potential_and_stationarity(
          row[2:], t_omega=tc_omega)
      self.assertTrue(0.0 <= stationarity <= 1e-7)
      self.assertTrue(numpy.isclose(potential, table_potential, atol=1e-8))

  def test_hamming_superpotential(self):
    """Asserts that the superpotential has 'Hamming Code' form."""
    # Here, we fix the specific conventions we expect for the Hamming code.
    # Note that these match the usual 'octonionic multiplication table'
    # conventions for the Fano plane, which unfortunately differ from (7.5)
    # in https://arxiv.org/pdf/1909.10969.pdf.
    #
    # Note that these particular conventions should work with both "gsw"
    # and "octonionic" Spin8 gamma matrices.
    hamming_code = (',124,156,137,235,267,346,457,'
                    '1236,1257,1467,3567,2456,1345,2347,1234567')
    direction_A1 = numpy.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hamming_terms = [tuple(int(n) - 1 for n in indices)
                     for indices in hamming_code.split(',')]
    hamming_selectors = [numpy.array([x in term for x in range(7)], dtype=bool)
                         for term in hamming_terms]
    def holomorphic_w_hamming(zs):
      holomorphic_part = sum(zs[sel].prod() for sel in hamming_selectors)
      return holomorphic_part
    def holomorphic_w_sugra(zs):
      gs = mu.undiskify(zs) * 0.25
      v70 = mu.nsum('zna,zn->a',
                    algebra.g.e7.sl2x7[:2, :, :70],
                    numpy.stack([gs.real, gs.imag], axis=0))
      A1 = SUGRA.tf_ext_sugra_tensors(mu.tff64(v70),
                                      with_stationarity=False)[2].numpy()
      kaehler_factor = numpy.sqrt((1 - zs * zs.conjugate()).prod())
      return a123.superpotential(A1, direction_A1) * kaehler_factor
    probe_zs = numpy.random.RandomState(seed=0).normal(
        size=(1000, 7, 2),
        scale=0.2).dot([1, 1j])
    # Consistency requirement for the test to make sense at all.
    self.assertTrue((abs(probe_zs) < 1).all())
    for zs in probe_zs:
      hw_h = holomorphic_w_hamming(zs)
      hw_s = holomorphic_w_sugra(zs)
      self.assertLess(abs(hw_h - hw_s), 1e-10)


if __name__ == '__main__':
  unittest.main()
