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

"""Tests for the BondLengthDistribution(s) classes."""

import os
import itertools
import numpy as np
import pandas as pd

from absl.testing import absltest
from absl.testing import parameterized

from smu import dataset_pb2
from smu.geometry import bond_length_distribution


TESTDATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'testdata')

# Some shortcuts to write less characters below
ATOM_C = dataset_pb2.BondTopology.ATOM_C
ATOM_N = dataset_pb2.BondTopology.ATOM_N
ATOM_O = dataset_pb2.BondTopology.ATOM_O
ATOM_F = dataset_pb2.BondTopology.ATOM_F
BOND_UNDEFINED = dataset_pb2.BondTopology.BOND_UNDEFINED
BOND_SINGLE = dataset_pb2.BondTopology.BOND_SINGLE
BOND_DOUBLE = dataset_pb2.BondTopology.BOND_DOUBLE
BOND_TRIPLE = dataset_pb2.BondTopology.BOND_TRIPLE


class FixedWindowTest(absltest.TestCase):

  def test_simple(self):
    dist = bond_length_distribution.FixedWindow(3, 5, None)
    self.assertAlmostEqual(dist.pdf(2.9), 0.0)
    self.assertAlmostEqual(dist.pdf(5.1), 0.0)
    self.assertAlmostEqual(dist.pdf(3.456), 0.5)
    self.assertEqual(dist.min(), 3)
    self.assertEqual(dist.max(), 5)

  def test_right_tail(self):
    dist = bond_length_distribution.FixedWindow(
        3, 5, right_tail_mass=0.8)
    # 0.2 of the mass is in the window, divded by 2 (size of the window)
    self.assertAlmostEqual(dist.pdf(3.456), 0.1)
    self.assertAlmostEqual(dist.pdf(5), 0.1)
    # Test slightly above the maximum to make sure we got the left side of the
    # right tail correct.
    self.assertAlmostEqual(dist.pdf(5.00000001), 0.1)
    self.assertAlmostEqual(dist.pdf(6), 0.08824969)
    self.assertEqual(dist.min(), 3)
    self.assertEqual(dist.max(), np.inf)


class GaussianTest(absltest.TestCase):

  def test_simple(self):
    dist = bond_length_distribution.Gaussian(10, 2, 3)
    # 4 stddev away shoudl be 0
    self.assertEqual(dist.pdf(2), 0)
    self.assertEqual(dist.pdf(18), 0)
    # 1 stddev away
    self.assertAlmostEqual(dist.pdf(8), 0.12131288)
    self.assertAlmostEqual(dist.pdf(12), 0.12131288)
    # mean
    self.assertAlmostEqual(dist.pdf(10), 0.200011129)
    self.assertEqual(dist.min(), 4)
    self.assertEqual(dist.max(), 16)

  def test_integral(self):
    dist = bond_length_distribution.Gaussian(2, .5, 2)
    # The precision/range here was empirical. GOing to higher does get closer and closer to 1.
    xs = np.arange(.99, 3.01, .0005)
    self.assertAlmostEqual(
      np.trapz(y=[dist.pdf(x) for x in xs], x=xs),
      1.0)

  def test_bad_inputs(self):
    with self.assertRaises(ValueError):
      dist = bond_length_distribution.Gaussian(10, -1, 1)
    with self.assertRaises(ValueError):
      dist = bond_length_distribution.Gaussian(10, 1, 0)
    with self.assertRaises(ValueError):
      dist = bond_length_distribution.Gaussian(10, 1, -1)



class EmpiricalTest(absltest.TestCase):

  def test_from_file(self):
    data = """1.0,1
1.1,2
1.2,3
1.3,4"""
    tmpfile = self.create_tempfile(content=data)
    dist = bond_length_distribution.Empirical.from_file(
        tmpfile, None)
    self.assertAlmostEqual(dist.bucket_size, 0.1)

    # Since the bucket sizes are 0.1, the pdfs are 10x the probability mass in
    # each bin.
    self.assertAlmostEqual(dist.pdf(1.0001), 1.0)
    self.assertAlmostEqual(dist.pdf(1.1001), 2.0)
    self.assertAlmostEqual(dist.pdf(1.2001), 3.0)
    self.assertAlmostEqual(dist.pdf(1.3001), 4.0)

  def test_from_sparse_dataframe(self):
    df_input = pd.DataFrame.from_dict({
        'length_str': ['1.234', '1.235', '1.236'],
        'count': [2, 3, 5]
    })
    got = (
        bond_length_distribution.Empirical
        .from_sparse_dataframe(df_input, right_tail_mass=0, sig_digits=3))
    self.assertAlmostEqual(got.pdf(1.2335), 0.0)
    self.assertAlmostEqual(got.pdf(1.2345), 200)
    self.assertAlmostEqual(got.pdf(1.2355), 300)
    self.assertAlmostEqual(got.pdf(1.2365), 500)
    self.assertAlmostEqual(got.pdf(1.2405), 0.0)

  def test_from_sparse_dataframe_interpolation(self):
    df_input = pd.DataFrame.from_dict({
        'length_str': ['1.2', '1.3', '1.6'],
        'count': [4, 5, 8]
    })
    got = (
        bond_length_distribution.Empirical
        .from_sparse_dataframe(df_input, right_tail_mass=0, sig_digits=1))
    self.assertAlmostEqual(got.pdf(1.25), 4 / 30 * 10)
    self.assertAlmostEqual(got.pdf(1.35), 5 / 30 * 10)
    self.assertAlmostEqual(got.pdf(1.45), 6 / 30 * 10)
    self.assertAlmostEqual(got.pdf(1.55), 7 / 30 * 10)
    self.assertAlmostEqual(got.pdf(1.65), 8 / 30 * 10)

  def test_from_sparse_dataframe_sig_digit_error(self):
    df_input = pd.DataFrame.from_dict({
        'length_str': ['1.234', '1.235'],
        'count': [2, 3]
    })
    with self.assertRaisesRegex(ValueError, 'Unexpected length_str'):
      _ = (
          bond_length_distribution.Empirical
          .from_sparse_dataframe(df_input, right_tail_mass=0, sig_digits=2))

  def test_from_arrays(self):
    dist = bond_length_distribution.Empirical.from_arrays(
        [1.0, 1.1], [5, 15], None)

    # Since the bucket sizes are 0.1, the pdfs are 10x the probability mass in
    # each bin.
    self.assertAlmostEqual(dist.pdf(0.95), 0.0)
    self.assertAlmostEqual(dist.pdf(1.05), 2.5)
    self.assertAlmostEqual(dist.pdf(1.15), 7.5)

  def test_edges(self):
    dist = bond_length_distribution.Empirical.from_arrays(
        [1.0, 1.1, 1.2, 1.3], [1, 2, 3, 4], None)

    # Test just below and above the minimum
    self.assertAlmostEqual(dist.pdf(0.9999), 0.0)
    self.assertAlmostEqual(dist.pdf(1.0001), 1.0)

    # Test just below and above the maximum
    self.assertAlmostEqual(dist.pdf(1.3999), 4.0)
    self.assertAlmostEqual(dist.pdf(1.4001), 0.0)

    self.assertEqual(dist.min(), 1.0)
    self.assertEqual(dist.max(), 1.4)


  def test_right_tail(self):
    dist = bond_length_distribution.Empirical.from_arrays(
        [1.0, 1.1], [5, 15], right_tail_mass=0.8)
    self.assertAlmostEqual(dist.pdf(1.05), 0.5)
    self.assertAlmostEqual(dist.pdf(1.15), 1.5)
    # This is just barely into the right tail, so should have the same pdf
    self.assertAlmostEqual(dist.pdf(1.2000000001), 1.5)
    self.assertAlmostEqual(dist.pdf(1.3), 1.2435437)

    self.assertEqual(dist.min(), 1)
    self.assertEqual(dist.max(), np.inf)

  def test_different_bucket_sizes(self):
    with self.assertRaises(ValueError):
      bond_length_distribution.Empirical.from_arrays(
          [1.0, 1.1, 1.9], [1, 1, 1], None)


class MixtureTest(absltest.TestCase):

  def test_simple(self):
    dist = bond_length_distribution.Mixture()
    dist.add(bond_length_distribution.FixedWindow(0.5, 1.5, None), 75)
    dist.add(bond_length_distribution.FixedWindow(1, 2, None), 25)
    self.assertAlmostEqual(dist.pdf(0.25), 0)
    self.assertAlmostEqual(dist.pdf(0.75), .75)
    self.assertAlmostEqual(dist.pdf(1.25), 1)
    self.assertAlmostEqual(dist.pdf(1.75), .25)
    self.assertAlmostEqual(dist.pdf(2.25), 0)

  def test_bad_weight(self):
    dist = bond_length_distribution.Mixture()
    with self.assertRaises(ValueError):
      dist.add(bond_length_distribution.FixedWindow(1, 2, None), 0)
    with self.assertRaises(ValueError):
      dist.add(bond_length_distribution.FixedWindow(1, 2, None), 0)

  def test_empty_mixture(self):
    dist = bond_length_distribution.Mixture()
    with self.assertRaises(ValueError):
      dist.pdf(1)


class AtomPairLengthDistributionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.dists = bond_length_distribution.AtomPairLengthDistributions()
    self.dists.add(
        BOND_SINGLE,
        bond_length_distribution.FixedWindow(1.2, 1.8, None))
    self.dists.add(
        BOND_DOUBLE,
        bond_length_distribution.FixedWindow(1.0, 1.4, None))

  def test_simple(self):
    self.assertEqual(self.dists.probability_of_bond_types(0.5), {})

    self.assertEqual(
        self.dists.probability_of_bond_types(1.1),
        {BOND_DOUBLE: 1.0})

    got = self.dists.probability_of_bond_types(1.3)
    self.assertLen(got, 2)
    self.assertAlmostEqual(got[BOND_SINGLE], 0.4)
    self.assertAlmostEqual(got[BOND_DOUBLE], 0.6)

    self.assertEqual(
        self.dists.probability_of_bond_types(1.5),
        {BOND_SINGLE: 1.0})

    self.assertEqual(self.dists.probability_of_bond_types(2.5), {})


class AllAtomPairLengthDistributions(absltest.TestCase):

  def test_atom_ordering(self):
    all_dists = bond_length_distribution.AllAtomPairLengthDistributions()
    all_dists.add(
        ATOM_N, ATOM_O,
        BOND_SINGLE,
        bond_length_distribution.FixedWindow(1, 2, None))
    self.assertEqual(
        all_dists.pdf_length_given_type(ATOM_N,
                                        ATOM_O,
                                        BOND_SINGLE,
                                        1.5), 1)
    self.assertEqual(
        all_dists.pdf_length_given_type(ATOM_O,
                                        ATOM_N,
                                        BOND_SINGLE,
                                        1.5), 1)

    self.assertEqual(
        all_dists.pdf_length_given_type(ATOM_N,
                                        ATOM_O,
                                        BOND_SINGLE,
                                        999), 0)
    self.assertEqual(
        all_dists.pdf_length_given_type(ATOM_O,
                                        ATOM_N,
                                        BOND_SINGLE,
                                        999), 0)

    # Make sure subsequent additions work as well
    all_dists.add(
        ATOM_N, ATOM_O,
        BOND_DOUBLE,
        bond_length_distribution.FixedWindow(2, 3, None))
    self.assertEqual(
        all_dists.pdf_length_given_type(ATOM_N,
                                        ATOM_O,
                                        BOND_DOUBLE,
                                        2.5), 1)
    self.assertEqual(
        all_dists.pdf_length_given_type(ATOM_O,
                                        ATOM_N,
                                        BOND_DOUBLE,
                                        2.5), 1)

  def test_probability_bond_types(self):
    all_dists = bond_length_distribution.AllAtomPairLengthDistributions()
    all_dists.add(
        ATOM_N, ATOM_O,
        BOND_SINGLE,
        bond_length_distribution.FixedWindow(1, 4, None))
    all_dists.add(
        ATOM_N, ATOM_O,
        BOND_DOUBLE,
        bond_length_distribution.FixedWindow(1, 2, None))
    got = all_dists.probability_of_bond_types(ATOM_N,
                                              ATOM_O,
                                              1.5)
    self.assertLen(got, 2)
    self.assertAlmostEqual(got[BOND_SINGLE], 0.25)
    self.assertAlmostEqual(got[BOND_DOUBLE], 0.75)

  def test_missing_types(self):
    all_dists = bond_length_distribution.AllAtomPairLengthDistributions()
    all_dists.add(
        ATOM_N, ATOM_O,
        BOND_SINGLE,
        bond_length_distribution.FixedWindow(1, 2, None))

    with self.assertRaises(KeyError):
      all_dists.probability_of_bond_types(ATOM_C,
                                          ATOM_C, 1.0)

    with self.assertRaises(KeyError):
      all_dists.pdf_length_given_type(ATOM_C,
                                      ATOM_C,
                                      BOND_SINGLE, 1.0)

  def test_add_from_gaussians_file(self):
    all_dists = bond_length_distribution.AllAtomPairLengthDistributions()

    all_dists.add_from_gaussians_file(
      os.path.join(TESTDATA_PATH, 'example_gaussian_input.csv'), 1)

    # The example file has a few lines copied from the real exporte files.
    #
    # Note two things in the file that might not be obvious
    # * An line with "n/a" as Bond that is ignored
    # * A line with "N:N" but no values that is ignored
    #
    # The numbers in the test below come from looking at a few cases there.

    # Two enties for C-C, making sure they mix.
    self.assertGreater(
      all_dists.pdf_length_given_type(ATOM_C, ATOM_C, BOND_SINGLE, 1.513), 0)
    self.assertGreater(
      all_dists.pdf_length_given_type(
        ATOM_C, ATOM_C, BOND_SINGLE, 1.588 + .001), 0)
    self.assertEqual(
      all_dists.pdf_length_given_type(
        ATOM_C, ATOM_C, BOND_SINGLE, 1.8), 0)

    # Testing double bond
    self.assertGreater(
      all_dists.pdf_length_given_type(
        ATOM_C, ATOM_C, BOND_DOUBLE, 1.299 + .001), 0)
    self.assertEqual(
      all_dists.pdf_length_given_type(
        ATOM_C, ATOM_C, BOND_DOUBLE, 1.8), 0)

    # Testing triple bond
    self.assertGreater(
      all_dists.pdf_length_given_type(
        ATOM_C, ATOM_C, BOND_TRIPLE, 1.183 + .001), 0)
    self.assertEqual(
      all_dists.pdf_length_given_type(
        ATOM_C, ATOM_C, BOND_TRIPLE, 1.0), 0)

    # Aromatic are converted to both single and double.
    self.assertGreater(
      all_dists.pdf_length_given_type(ATOM_N, ATOM_N, BOND_SINGLE, 1.304), 0)
    self.assertGreater(
      all_dists.pdf_length_given_type(ATOM_N, ATOM_N, BOND_DOUBLE, 1.304), 0)
    with self.assertRaises(KeyError):
      all_dists.pdf_length_given_type(ATOM_N, ATOM_N, BOND_TRIPLE, 1.304)


  def test_add_from_files(self):
    data = """1.0,1
1.1,2
1.2,3
1.3,2
"""
    data_increasing = """1.0,1
1.1,2
1.2,3
1.3,4
1.4,5
"""

    tmpdir = self.create_tempdir()
    stem = os.path.join(tmpdir, 'BONDS')
    self.create_tempfile(f'{stem}.6.0.6', content=data_increasing)
    self.create_tempfile(f'{stem}.6.1.6', content=data)

    self.create_tempfile(f'{stem}.6.0.7', content=data_increasing)
    self.create_tempfile(f'{stem}.6.1.7', content=data)
    self.create_tempfile(f'{stem}.6.2.7', content=data)
    self.create_tempfile(f'{stem}.6.3.7', content=data)

    all_dists = bond_length_distribution.AllAtomPairLengthDistributions()
    all_dists.add_from_files(stem, unbonded_right_tail_mass=0.8)

    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_C, BOND_UNDEFINED, 0.99), 0.0)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_N, BOND_UNDEFINED, 0.99), 0.0)

    # The 3/15 is the counts in the data_increasing file.
    # * 10 is for the pdf because the bucket is 0.1 wide
    # * 0.2 is because of the right tail mass.
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_C, BOND_UNDEFINED, 1.25),
        3.0 / 15.0 * 10 * 0.2)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_N, BOND_UNDEFINED, 1.25),
        3.0 / 15.0 * 10 * 0.2)

    # Test the right tail mass for the unbonded
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_C, BOND_UNDEFINED, 1.5),
        0.66666667)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_N, BOND_UNDEFINED, 1.5),
        0.66666667)

    # Test the bonded inside the pdf.
    # 3/8 are the counts in the data file
    # * 10 is for the pdf because the bucket is 0.1 wide
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_C, BOND_SINGLE, 1.25),
        3.0 / 8.0 * 10)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_N, BOND_SINGLE, 1.25),
        3.0 / 8.0 * 10)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_N, BOND_DOUBLE, 1.25),
        3.0 / 8.0 * 10)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_N, BOND_TRIPLE, 1.25),
        3.0 / 8.0 * 10)

    # Check for no right tail mass for the bonded
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_C, BOND_SINGLE, 1.5), 0.0)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_N, BOND_SINGLE, 1.5), 0.0)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_N, BOND_DOUBLE, 1.5), 0.0)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_N, BOND_TRIPLE, 1.5), 0.0)

  def test_add_from_sparse_dataframe(self):
    df = pd.DataFrame.from_records([
        ('c', 'c', 1, '1.0', 10),
        ('c', 'c', 1, '1.1', 30),
        ('n', 'o', 2, '1.4', 50),
        ('n', 'o', 2, '1.5', 50),
        ('n', 'n', 0, '1.7', 100),
        ('n', 'n', 0, '1.8', 100),
    ],
                                   columns=[
                                       'atom_char_0', 'atom_char_1',
                                       'bond_type', 'length_str', 'count'
                                   ])
    all_dists = bond_length_distribution.AllAtomPairLengthDistributions()
    all_dists.add_from_sparse_dataframe(
        df, sig_digits=1, unbonded_right_tail_mass=0.8)

    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_C, BOND_SINGLE, 1.05), 2.5)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_C, ATOM_C, BOND_SINGLE, 999), 0.0)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_N, ATOM_O, BOND_DOUBLE, 1.55), 5.0)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(ATOM_N, ATOM_N, BOND_UNDEFINED, 1.85),
        1.0)
    # This makes sure the right tail mass was included
    self.assertGreater(
        all_dists.pdf_length_given_type(ATOM_N, ATOM_N, BOND_UNDEFINED, 2.0), 0.0)
    self.assertGreater(
        all_dists.pdf_length_given_type(ATOM_N, ATOM_N, BOND_UNDEFINED, 3.0), 0.0)


class AddFromSpecStringTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.bond_lengths = bond_length_distribution.make_fake_empiricals()

  def is_empirical(self, atom_a, atom_b, bond_type):
    length_dist = self.bond_lengths[(atom_a, atom_b)][bond_type]
    return isinstance(length_dist,
                      bond_length_distribution.Empirical)

  def fixed_window_min(self, atom_a, atom_b, bond_type):
    length_dist = self.bond_lengths[(atom_a, atom_b)][bond_type]
    if not isinstance(length_dist,
                      bond_length_distribution.FixedWindow):
      return None
    return length_dist.minimum

  def fixed_window_max(self, atom_a, atom_b, bond_type):
    length_dist = self.bond_lengths[(atom_a, atom_b)][bond_type]
    if not isinstance(length_dist,
                      bond_length_distribution.FixedWindow):
      return None
    return length_dist.maximum

  def fixed_window_right_tail_mass(self, atom_a, atom_b, bond_type):
    length_dist = self.bond_lengths[(atom_a, atom_b)][bond_type]
    if not isinstance(length_dist,
                      bond_length_distribution.FixedWindow):
      return None
    return length_dist.right_tail_mass

  def test_empty_bond_lengths(self):
    self.bond_lengths.add_from_string_spec(None)
    self.assertTrue(self.is_empirical(ATOM_C, ATOM_C, BOND_SINGLE))
    self.assertTrue(self.is_empirical(ATOM_N, ATOM_O, BOND_DOUBLE))

  def test_fully_specified(self):
    self.bond_lengths.add_from_string_spec('C#C:1.1-1.2,N=O:1.3-1.4,O-F:1.5-1.6,C.F:2.0-2.1')

    self.assertEqual(self.fixed_window_min(ATOM_C, ATOM_C, BOND_TRIPLE), 1.1)
    self.assertEqual(self.fixed_window_max(ATOM_C, ATOM_C, BOND_TRIPLE), 1.2)
    self.assertIsNone(
        self.fixed_window_right_tail_mass(ATOM_C, ATOM_C, BOND_TRIPLE))
    self.assertTrue(self.is_empirical(ATOM_C, ATOM_C, BOND_SINGLE))

    self.assertEqual(self.fixed_window_min(ATOM_N, ATOM_O, BOND_DOUBLE), 1.3)
    self.assertEqual(self.fixed_window_max(ATOM_N, ATOM_O, BOND_DOUBLE), 1.4)
    self.assertIsNone(
        self.fixed_window_right_tail_mass(ATOM_N, ATOM_O, BOND_DOUBLE))
    self.assertTrue(self.is_empirical(ATOM_N, ATOM_O, BOND_SINGLE))

    self.assertEqual(self.fixed_window_min(ATOM_O, ATOM_F, BOND_SINGLE), 1.5)
    self.assertEqual(self.fixed_window_max(ATOM_O, ATOM_F, BOND_SINGLE), 1.6)
    self.assertIsNone(
        self.fixed_window_right_tail_mass(ATOM_O, ATOM_F, BOND_SINGLE))

    self.assertEqual(self.fixed_window_min(ATOM_C, ATOM_F, BOND_UNDEFINED), 2.0)
    self.assertEqual(self.fixed_window_max(ATOM_C, ATOM_F, BOND_UNDEFINED), 2.1)
    self.assertIsNone(
        self.fixed_window_right_tail_mass(ATOM_C, ATOM_F, BOND_UNDEFINED))

  def test_bond_wildcard(self):
    self.bond_lengths.add_from_string_spec('C~C:1.1-1.2')

    self.assertEqual(self.fixed_window_min(ATOM_C, ATOM_C, BOND_SINGLE), 1.1)
    self.assertEqual(self.fixed_window_min(ATOM_C, ATOM_C, BOND_DOUBLE), 1.1)
    self.assertEqual(self.fixed_window_min(ATOM_C, ATOM_C, BOND_TRIPLE), 1.1)
    self.assertTrue(self.is_empirical(ATOM_C, ATOM_C, BOND_UNDEFINED))

  @parameterized.parameters(
      '*-N:1.1-1.2',
      'N-*:1.1-1.2',
  )
  def test_atom_wildcard(self, spec):
    self.bond_lengths.add_from_string_spec(spec)

    self.assertEqual(self.fixed_window_min(ATOM_N, ATOM_N, BOND_SINGLE), 1.1)
    self.assertEqual(self.fixed_window_min(ATOM_N, ATOM_C, BOND_SINGLE), 1.1)
    self.assertEqual(self.fixed_window_min(ATOM_N, ATOM_O, BOND_SINGLE), 1.1)
    self.assertEqual(self.fixed_window_min(ATOM_N, ATOM_F, BOND_SINGLE), 1.1)
    self.assertTrue(self.is_empirical(ATOM_N, ATOM_N, BOND_DOUBLE))
    self.assertTrue(self.is_empirical(ATOM_C, ATOM_C, BOND_SINGLE))

  def test_left_missing_dist(self):
    self.bond_lengths.add_from_string_spec('C-C:-1.2')
    self.assertEqual(self.fixed_window_min(ATOM_C, ATOM_C, BOND_SINGLE), 0)
    self.assertEqual(self.fixed_window_max(ATOM_C, ATOM_C, BOND_SINGLE), 1.2)

  def test_right_missing_dist(self):
    self.bond_lengths.add_from_string_spec('C-C:1.1-')
    self.assertEqual(self.fixed_window_min(ATOM_C, ATOM_C, BOND_SINGLE), 1.1)
    self.assertGreater(
        self.fixed_window_right_tail_mass(ATOM_C, ATOM_C, BOND_SINGLE), 0)

  @parameterized.parameters(
      'Nonsense',
      'Hi',
      ',',
      'Hi,C-C:-',
      'P-N:1.1-1.2,C-C:-',
      'N-P:1.1-1.2,C-C:-',
      'N%N:1.1-1.2,C-C:-',
      'N=N',
      'N=N:',
      'N=N:1.2',
      'N=N:Nonsense',
      'N=N:Nonsense-1.2',
      'N=N:1.2-Nonsense',
  )
  def test_parse_errors(self, spec):
    with self.assertRaises(bond_length_distribution.BondLengthParseError):
      self.bond_lengths.add_from_string_spec(spec)


class SparseDataframFromRecordsTest(absltest.TestCase):

  def test_simple(self):
    input_list = [
        (('n', 'o', 1, '3.456'), 30),
        (('c', 'c', 2, '2.345'), 20),
        (('c', 'c', 1, '1.234'), 10),
    ]
    got = bond_length_distribution.sparse_dataframe_from_records(input_list)
    self.assertCountEqual(
        got.columns,
        ['atom_char_0', 'atom_char_1', 'bond_type', 'length_str', 'count'])
    np.testing.assert_array_equal(got['atom_char_0'], ['c', 'c', 'n'])
    np.testing.assert_array_equal(got['atom_char_1'], ['c', 'c', 'o'])
    np.testing.assert_array_equal(got['bond_type'], [1, 2, 1])
    np.testing.assert_array_equal(got['length_str'],
                                  ['1.234', '2.345', '3.456'])
    np.testing.assert_array_equal(got['count'], [10, 20, 30])


class TestInterpolateOutZeros(absltest.TestCase):

  def test_no_action(self):
    inputs = np.array([1, 1])
    got = bond_length_distribution.interpolate_zeros(inputs)
    np.testing.assert_array_equal([1, 1], got)

  def test_insert_one(self):
    inputs = np.array([1, 0, 1])
    got = bond_length_distribution.interpolate_zeros(inputs)
    np.testing.assert_array_equal([1, 1, 1], got)

  def test_insert_many(self):
    inputs = np.array([1, 0, 0, 0, 0, 0, 1])
    got = bond_length_distribution.interpolate_zeros(inputs)
    np.testing.assert_array_equal(np.ones(len(inputs), dtype=np.int32), got)

  def test_insert_multiple_regions(self):
    inputs = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    got = bond_length_distribution.interpolate_zeros(inputs)
    np.testing.assert_array_equal(np.ones(len(inputs), dtype=np.int32), got)

  def test_do_actual_interpolation_one(self):
    inputs = np.array([1, 0, 2])
    got = bond_length_distribution.interpolate_zeros(inputs)
    np.testing.assert_almost_equal([1.0, 1.5, 2.0], got)

  def test_do_actual_interpolation_many(self):
    inputs = np.array([1, 0, 0, 0, 0, 6])
    got = bond_length_distribution.interpolate_zeros(inputs)
    np.testing.assert_almost_equal([1, 2, 3, 4, 5, 6], got)


if __name__ == '__main__':
  absltest.main()
