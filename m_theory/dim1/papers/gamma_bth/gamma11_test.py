# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Basic tests for D=11 Supergravity quartic-fermion-terms code.

These tests cover basic functionality.

Background/Rationale:

  - We thoroughly test-cover properties of the Gamma-matrix
    infrastructure.

  - Somewhat unusually, as expectations often come in the form of
    algebraic invariants, tests will generally have to involve some
    logic to run the underlying calculation that tells us what must
    hold.

  - Some objects, such as the PsiSubstitutions, and GammaPsiTable
    instances, should be thought of as mostly-opaque building blocks
    in the overall setup. Here, all that matters is that we can
    instantiate them without things falling over. The really important
    bit then is about running actual calculations, which is
    test-covered separately.

"""

import hashlib
import itertools
import unittest

import gamma11
import numpy


# For testing a calculation, we do need a term-generator.
def _add_term(accumulator):
  # Term: (-3) * (\bar\Psi^a Gamma^0b Psi^c)(\bar\Psi^a Gamma^0b Psi^c).
  accumulator.collect(
      term_factor=-3,
      ci=(
          (+1, (a, (0, b), c), (a, (0, b), c))
          for a, b, c in itertools.product(*[range(1, 11)]*3)))


class Gamma11Test(unittest.TestCase):
  """Basic correctness validation tests for Gamma-matrix related objects."""

  def test_permutation_sign_d(self):
    """Tests that the permutation-sign computes permutation signs."""
    # Empty permutation is even.
    self.assertEqual(+1, gamma11.permutation_sign_d([]))
    self.assertEqual(-1, gamma11.permutation_sign_d([1, 0]))
    self.assertEqual(+1, gamma11.permutation_sign_d([1, 2, 0]))
    self.assertEqual(+1, gamma11.permutation_sign_d([1, 0, 3, 2]))

  def test_index_canonicalization(self):
    """Tests that we canonicalize index-tuples with expected sign."""
    get_sc = gamma11.get_sign_and_canonicalized
    self.assertEqual((-1, [1, 2, 3]), get_sc([3, 2, 1]))
    self.assertEqual((+1, []), get_sc([]))
    self.assertEqual((-1, [100, 101, 102]), get_sc([100, 102, 101]))

  def test_gammas11d_form(self):
    """Tests that gammas are an array with 11 32x32 Majorana Gamma-matrices."""
    gammas11d = gamma11.get_gammas11d()
    self.assertEqual((11, 32, 32), gammas11d.shape)
    self.assertEqual({-1, 0, 1}, set(gammas11d.ravel()))

  def test_gammas11d_clifford(self):
    """Tests that our Gamma matrices satisfy the Clifford algebra."""
    gammas11d = gamma11.get_gammas11d()
    gamma_gamma = numpy.einsum('iAC,jCB->ijAB', gammas11d, gammas11d)
    deviation = (gamma_gamma + numpy.einsum('ijAB->jiAB', gamma_gamma) -
                 2 * numpy.einsum('ij,AB->ijAB',
                                  numpy.diag([-1] + [1]*10),
                                  numpy.eye(32)))
    # Entries of the deviation must be all-zero. This must be exact.
    self.assertEqual({0}, set(deviation.ravel()))

  def test_gammas11d_as_in_literature(self):
    """Tests that our Gamma matrices precisely match Freedman-Van-Proeyen."""
    gammas11d = gamma11.get_gammas11d()
    # Shape and entries also belong to the properties we must check here.
    self.assertEqual((11, 32, 32), gammas11d.shape)
    self.assertEqual({-1, 0, 1}, set(gammas11d.ravel()))
    fingerprint = (
        hashlib.sha256(gammas11d.astype(numpy.int8).data.tobytes()).hexdigest())
    self.assertEqual('79c0154c481ecfa002f4076194dc3442'
                     '3419a0e46cb147d685a2fd81b15c60d2',
                     fingerprint)

  def test_gamma11family_arithmetics(self):
    """Tests that Gamma11Family multiplication matches matrix multiplication."""
    # We actually ask the Gamma11-family to use a different definition of
    # Gamma matrices than the default one here. Specifically, we go with
    # index-reversal on the spinors.
    gammas11d = gamma11.get_gammas11d()
    gammas11d_rev = gammas11d[:, ::-1, ::-1]
    g11family = gamma11.Gamma11Family(gamma_IAB=gammas11d_rev)
    some_spinor = numpy.arange(100, 132)
    sample_gamma_spinor_v1 = (
        gammas11d_rev[3] @ gammas11d_rev[0] @ gammas11d_rev[7]).dot(some_spinor)
    sample_gamma_spinor_v2 = (
        gammas11d_rev[7] @ gammas11d_rev[3] @ gammas11d_rev[0]).dot(some_spinor)
    sample_gamma_spinor_nonrev = (
        gammas11d[7] @ gammas11d[3] @ gammas11d[0]).dot(some_spinor)
    permuted, signs = g11family.apply_right((3, 0, 7), some_spinor)
    self.assertTrue(numpy.allclose(sample_gamma_spinor_v1,
                                   sample_gamma_spinor_v2))
    self.assertTrue(numpy.allclose(sample_gamma_spinor_v1,
                                   permuted * signs))
    self.assertFalse(numpy.allclose(sample_gamma_spinor_nonrev,
                                    permuted * signs))

  def test_gamma11family_as_matrix(self):
    """Tests that Gamma11Family matrix-extraction gives back good gammas."""
    g11 = gamma11.get_gammas11d()[:, ::-1, ::-1]
    g11family = gamma11.Gamma11Family(gamma_IAB=g11)
    m30489 = g11family.as_matrix([3, 0, 4, 8, 9])
    self.assertTrue(numpy.allclose(
        m30489, g11[3] @ g11[0] @ g11[4] @ g11[8] @ g11[9]))

  def test_gamma11family_as_sparse(self):
    """Tests that Gamma11Family sparse-extraction gives good data."""
    # Our strategy is to populate a full matrix from the .as_sparse()
    # data and compare.
    g11 = gamma11.get_gammas11d()[:, ::-1, ::-1]
    g11family = gamma11.Gamma11Family(gamma_IAB=g11)
    m_sparse = g11family.as_sparse([2, 4, 7, 3])
    m_full = g11[2] @ g11[4] @ g11[7] @ g11[3]
    resynthesized = numpy.zeros([32, 32])
    for coeff, index01 in m_sparse:
      # There must not already be a nonzero coefficient at this place.
      self.assertEqual(0, resynthesized[index01])
      resynthesized[index01] = coeff
    self.assertTrue(numpy.allclose(resynthesized, m_full))


class ComputationalIngredientsTest(unittest.TestCase):
  """Validation for basic instantiability of some building blocks."""

  def test_instantiate_gamma_psi_table(self):
    """Validates that we can instantiate a GammaPsiTable.

    We also check whether the instance has the expected attributes.
    """
    obj = gamma11.GammaPsiTable(gamma11_family=gamma11.Gamma11Family(),
                                psis=gamma11.PSIS,
                                psi_names=gamma11.PSI_NAME_BY_TAG)
    # issubclass() checks may look weird, but basically assess that
    # the above callable is indeed a class-object.
    self.assertTrue(issubclass(type(obj), gamma11.GammaPsiTable))
    self.assertTrue(issubclass(type(obj.gamma11_family),
                               gamma11.Gamma11Family))
    self.assertEqual((11, 32), obj.psis.shape)
    # Psi-names must be lexicographic.
    self.assertLess(obj.psi_names[obj.psis.ravel()[-2]],
                    obj.psi_names[obj.psis.ravel()[-1]])
    self.assertIsNotNone(obj.qbar_psis)
    self.assertIsNotNone(obj.phis)
    self.assertIsNotNone(obj.qbar_phis)

  def test_instantiate_psi_substitution_mapping(self):
    """Validates that we can instantiate a PsiSubstitutionMapping."""
    # Given that the only relevant method, .to_ffi_form() returns
    # an opaque object, there is not really much to do here beyond
    # checking that instantiation succeeds.
    mapping = gamma11.PsiSubstitutionMapping(
        new_variables=['p0', 'p1'],
        psi_variables=gamma11.PSI_NAME_BY_TAG[1:],
        # It is not invalid to fly with a ranking polynomial of the form 0/1,
        # which aggregates everything into index zero. In fact,
        # this is meaningful and useful for speed measurements,
        # to determine how much the calculation is memory I/O bound -
        # with this ranking, every coefficient will get accumulated into
        # bucket zero.
        ranking_polynomial=numpy.array([1] + [0]*11, dtype=numpy.int64),
        rules={'psi_v01s01': [(+1, 'p0'), (-1, 'p1')],
               'psi_v01s32': [(-1, 'p0'), (+1, 'p1')]})
    self.assertTrue(issubclass(type(mapping),
                               gamma11.PsiSubstitutionMapping))


class ComputationTest(unittest.TestCase):
  """Validation of some basic calculations."""

  def test_dict_accumulator_computation(self):
    """Validates a basic dict-accumulator calculation."""
    accumulator = gamma11.QuarticTermDictAccumulator()
    # for the dict-accumulator, this takes some seconds.
    _add_term(accumulator)
    first_entries = list(
        itertools.islice(accumulator.collected(scale_by=0.125),
                         0, 3))
    expected = [
        (-1.5, ('psi_v01s01', 'psi_v01s02', 'psi_v02s01', 'psi_v02s02')),
        (-3.0, ('psi_v01s01', 'psi_v01s02', 'psi_v02s09', 'psi_v02s10')),
        (3.0, ('psi_v01s01', 'psi_v01s02', 'psi_v02s11', 'psi_v02s12'))]
    self.assertEqual(expected, first_entries)

  def test_fast_accumulator_computation(self):
    """Validates a basic fast-accumulator calculation."""
    accumulator = gamma11.QuarticTermFastAccumulator()
    _add_term(accumulator)
    first_entries = list(
        itertools.islice(accumulator.collected(scale_by=0.25),
                         0, 3))
    expected = [
        (-3.0, ('psi_v01s01', 'psi_v01s02', 'psi_v02s01', 'psi_v02s02')),
        (-6.0, ('psi_v01s01', 'psi_v01s02', 'psi_v02s09', 'psi_v02s10')),
        (+6.0, ('psi_v01s01', 'psi_v01s02', 'psi_v02s11', 'psi_v02s12'))]
    self.assertEqual(expected, first_entries)

  def test_fast_accumulator_trivial_substitution_computation(self):
    """Validates that 1:1 substitution does not affect the result."""
    subs_mapping = gamma11.PsiSubstitutionMapping(
        new_variables=[name.upper() for name in gamma11.PSI_NAME_BY_TAG[1:]],
        psi_variables=gamma11.PSI_NAME_BY_TAG[1:],
        # This ranking polynomial is appropriate for substitutions
        # with 12 remaining parameters.
        ranking_polynomial=gamma11.RANKING_COEFFS_320_4,
        rules={psi: [(+1, psi.upper())] for psi in gamma11.PSI_NAME_BY_TAG[1:]}
    )
    subs_accumulator = gamma11.QuarticTermFastAccumulator(
        psi_substitutions=subs_mapping)
    nonsubs_accumulator = gamma11.QuarticTermFastAccumulator()
    _add_term(subs_accumulator)
    _add_term(nonsubs_accumulator)
    first_entries = list(zip(range(1000),
                             subs_accumulator.collected(),
                             nonsubs_accumulator.collected()))
    for _, left, right in first_entries:
      self.assertEqual(repr(left).lower(), repr(right))

  def test_fast_accumulator_substitution_computation(self):
    """Validates a basic fast-accumulator calculation with substitution."""
    subs_mapping = gamma11.PsiSubstitutionMapping(
        new_variables=[f'p{n:02d}' for n in range(1, 13)],
        psi_variables=gamma11.PSI_NAME_BY_TAG[1:],
        # This ranking polynomial is appropriate for substitutions
        # with 12 remaining parameters.
        ranking_polynomial=[
            24, -1608, 24, 252, -12, 1196,
            -120, 4, 3382, -539, 38, -1],
        rules={'psi_v01s01': [(+1, 'p05')],
               'psi_v01s02': [(+3, 'p06')],
               'psi_v02s01': [(+1, 'p07')],
               'psi_v02s02': [(+1, 'p08')],
               }
    )
    accumulator = gamma11.QuarticTermFastAccumulator(
        psi_substitutions=subs_mapping)
    _add_term(accumulator)
    all_entries = list(accumulator.collected(scale_by=-0.125))
    expected = [(+4.5, ('p05', 'p06', 'p07', 'p08'))]
    self.assertEqual(expected, all_entries)


if __name__ == '__main__':
  unittest.main()
