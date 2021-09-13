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

"""Tests for the BondLengthDistribution(s) classes."""

import numpy as np
import os
from absl.testing import absltest
import numpy as np
import pandas as pd

from smu import dataset_pb2
from smu.geometry import bond_length_distribution


class FixedWindowLengthDistributionTest(absltest.TestCase):

  def test_simple(self):
    dist = bond_length_distribution.FixedWindowLengthDistribution(3, 5, None)
    self.assertAlmostEqual(dist.pdf(2.9), 0.0)
    self.assertAlmostEqual(dist.pdf(5.1), 0.0)
    self.assertAlmostEqual(dist.pdf(3.456), 0.5)

  def test_right_tail(self):
    dist = bond_length_distribution.FixedWindowLengthDistribution(
        3, 5, right_tail_mass=0.8)
    # 0.2 of the mass is in the window, divded by 2 (size of the window)
    self.assertAlmostEqual(dist.pdf(3.456), 0.1)
    self.assertAlmostEqual(dist.pdf(5), 0.1)
    # Test slightly above the maximum to make sure we got the left side of the
    # right tail correct.
    self.assertAlmostEqual(dist.pdf(5.00000001), 0.1)
    self.assertAlmostEqual(dist.pdf(6), 0.08824969)


class EmpiricalLengthDistributionTest(absltest.TestCase):

  def test_from_file(self):
    data = """1.0,1
1.1,2
1.2,3
1.3,4"""
    tmpfile = self.create_tempfile(content=data)
    dist = bond_length_distribution.EmpiricalLengthDistribution.from_file(
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
        'length_str': ['1.234', '1.235', '1.239'],
        'count': [2, 3, 5]
    })
    got = (
        bond_length_distribution.EmpiricalLengthDistribution
        .from_sparse_dataframe(df_input, right_tail_mass=0, sig_digits=3))
    self.assertAlmostEqual(got.pdf(1.2335), 0.0)
    self.assertAlmostEqual(got.pdf(1.2345), 200)
    self.assertAlmostEqual(got.pdf(1.2355), 300)
    # this is the internal implicit 0 count
    self.assertAlmostEqual(got.pdf(1.2365), 0.0)
    self.assertAlmostEqual(got.pdf(1.2395), 500)
    self.assertAlmostEqual(got.pdf(1.2405), 0.0)

  def test_from_sparse_dataframe_sig_digit_error(self):
    df_input = pd.DataFrame.from_dict({
        'length_str': ['1.234', '1.235'],
        'count': [2, 3]
    })
    with self.assertRaisesRegex(ValueError, 'Unexpected length_str'):
      _ = (
          bond_length_distribution.EmpiricalLengthDistribution
          .from_sparse_dataframe(df_input, right_tail_mass=0, sig_digits=2))

  def test_from_arrays(self):
    dist = bond_length_distribution.EmpiricalLengthDistribution.from_arrays(
        [1.0, 1.1], [5, 15], None)

    # Since the bucket sizes are 0.1, the pdfs are 10x the probability mass in
    # each bin.
    self.assertAlmostEqual(dist.pdf(0.95), 0.0)
    self.assertAlmostEqual(dist.pdf(1.05), 2.5)
    self.assertAlmostEqual(dist.pdf(1.15), 7.5)

  def test_edges(self):
    dist = bond_length_distribution.EmpiricalLengthDistribution.from_arrays(
        [1.0, 1.1, 1.2, 1.3], [1, 2, 3, 4], None)

    # Test just below and above the minimum
    self.assertAlmostEqual(dist.pdf(0.9999), 0.0)
    self.assertAlmostEqual(dist.pdf(1.0001), 1.0)

    # Test just below and above the maximum
    self.assertAlmostEqual(dist.pdf(1.3999), 4.0)
    self.assertAlmostEqual(dist.pdf(1.4001), 0.0)

  def test_right_tail(self):
    dist = bond_length_distribution.EmpiricalLengthDistribution.from_arrays(
        [1.0, 1.1], [5, 15], right_tail_mass=0.8)
    self.assertAlmostEqual(dist.pdf(1.05), 0.5)
    self.assertAlmostEqual(dist.pdf(1.15), 1.5)
    # This is just barely into the right tail, so should have the same pdf
    self.assertAlmostEqual(dist.pdf(1.2000000001), 1.5)
    self.assertAlmostEqual(dist.pdf(1.3), 1.2435437)

  def test_different_bucket_sizes(self):
    with self.assertRaises(ValueError):
      bond_length_distribution.EmpiricalLengthDistribution.from_arrays(
          [1.0, 1.1, 1.9], [1, 1, 1], None)


class AtomPairLengthDistributionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.dists = bond_length_distribution.AtomPairLengthDistributions()
    self.dists.add(
        dataset_pb2.BondTopology.BOND_SINGLE,
        bond_length_distribution.FixedWindowLengthDistribution(1.2, 1.8, None))
    self.dists.add(
        dataset_pb2.BondTopology.BOND_DOUBLE,
        bond_length_distribution.FixedWindowLengthDistribution(1.0, 1.4, None))

  def test_simple(self):
    self.assertEqual(self.dists.probability_of_bond_types(0.5), {})

    self.assertEqual(
        self.dists.probability_of_bond_types(1.1),
        {dataset_pb2.BondTopology.BOND_DOUBLE: 1.0})

    got = self.dists.probability_of_bond_types(1.3)
    self.assertLen(got, 2)
    self.assertAlmostEqual(got[dataset_pb2.BondTopology.BOND_SINGLE], 0.4)
    self.assertAlmostEqual(got[dataset_pb2.BondTopology.BOND_DOUBLE], 0.6)

    self.assertEqual(
        self.dists.probability_of_bond_types(1.5),
        {dataset_pb2.BondTopology.BOND_SINGLE: 1.0})

    self.assertEqual(self.dists.probability_of_bond_types(2.5), {})


class AllAtomPairLengthDistributions(absltest.TestCase):

  def test_atom_ordering(self):
    all_dists = bond_length_distribution.AllAtomPairLengthDistributions()
    all_dists.add(
        dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_O,
        dataset_pb2.BondTopology.BOND_SINGLE,
        bond_length_distribution.FixedWindowLengthDistribution(1, 2, None))
    self.assertEqual(
        all_dists.pdf_length_given_type(dataset_pb2.BondTopology.ATOM_N,
                                        dataset_pb2.BondTopology.ATOM_O,
                                        dataset_pb2.BondTopology.BOND_SINGLE,
                                        1.5), 1)
    self.assertEqual(
        all_dists.pdf_length_given_type(dataset_pb2.BondTopology.ATOM_O,
                                        dataset_pb2.BondTopology.ATOM_N,
                                        dataset_pb2.BondTopology.BOND_SINGLE,
                                        1.5), 1)

    self.assertEqual(
        all_dists.pdf_length_given_type(dataset_pb2.BondTopology.ATOM_N,
                                        dataset_pb2.BondTopology.ATOM_O,
                                        dataset_pb2.BondTopology.BOND_SINGLE,
                                        999), 0)
    self.assertEqual(
        all_dists.pdf_length_given_type(dataset_pb2.BondTopology.ATOM_O,
                                        dataset_pb2.BondTopology.ATOM_N,
                                        dataset_pb2.BondTopology.BOND_SINGLE,
                                        999), 0)

    # Make sure subsequent additions work as well
    all_dists.add(
        dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_O,
        dataset_pb2.BondTopology.BOND_DOUBLE,
        bond_length_distribution.FixedWindowLengthDistribution(2, 3, None))
    self.assertEqual(
        all_dists.pdf_length_given_type(dataset_pb2.BondTopology.ATOM_N,
                                        dataset_pb2.BondTopology.ATOM_O,
                                        dataset_pb2.BondTopology.BOND_DOUBLE,
                                        2.5), 1)
    self.assertEqual(
        all_dists.pdf_length_given_type(dataset_pb2.BondTopology.ATOM_O,
                                        dataset_pb2.BondTopology.ATOM_N,
                                        dataset_pb2.BondTopology.BOND_DOUBLE,
                                        2.5), 1)

  def test_probability_bond_types(self):
    all_dists = bond_length_distribution.AllAtomPairLengthDistributions()
    all_dists.add(
        dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_O,
        dataset_pb2.BondTopology.BOND_SINGLE,
        bond_length_distribution.FixedWindowLengthDistribution(1, 4, None))
    all_dists.add(
        dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_O,
        dataset_pb2.BondTopology.BOND_DOUBLE,
        bond_length_distribution.FixedWindowLengthDistribution(1, 2, None))
    got = all_dists.probability_of_bond_types(dataset_pb2.BondTopology.ATOM_N,
                                              dataset_pb2.BondTopology.ATOM_O,
                                              1.5)
    self.assertLen(got, 2)
    self.assertAlmostEqual(got[dataset_pb2.BondTopology.BOND_SINGLE], 0.25)
    self.assertAlmostEqual(got[dataset_pb2.BondTopology.BOND_DOUBLE], 0.75)

  def test_missing_types(self):
    all_dists = bond_length_distribution.AllAtomPairLengthDistributions()
    all_dists.add(
        dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_O,
        dataset_pb2.BondTopology.BOND_SINGLE,
        bond_length_distribution.FixedWindowLengthDistribution(1, 2, None))

    with self.assertRaises(KeyError):
      all_dists.probability_of_bond_types(dataset_pb2.BondTopology.ATOM_C,
                                          dataset_pb2.BondTopology.ATOM_C, 1.0)

    with self.assertRaises(KeyError):
      all_dists.pdf_length_given_type(dataset_pb2.BondTopology.ATOM_C,
                                      dataset_pb2.BondTopology.ATOM_C,
                                      dataset_pb2.BondTopology.BOND_SINGLE, 1.0)

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

    carbon = dataset_pb2.BondTopology.AtomType.ATOM_C
    nitrogen = dataset_pb2.BondTopology.AtomType.ATOM_N
    unbonded = dataset_pb2.BondTopology.BondType.BOND_UNDEFINED
    single = dataset_pb2.BondTopology.BondType.BOND_SINGLE
    double = dataset_pb2.BondTopology.BondType.BOND_DOUBLE
    triple = dataset_pb2.BondTopology.BondType.BOND_TRIPLE

    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, carbon, unbonded, 0.99), 0.0)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, nitrogen, unbonded, 0.99), 0.0)

    # The 3/15 is the counts in the data_increasing file.
    # * 10 is for the pdf because the bucket is 0.1 wide
    # * 0.2 is because of the right tail mass.
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, carbon, unbonded, 1.25),
        3.0 / 15.0 * 10 * 0.2)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, nitrogen, unbonded, 1.25),
        3.0 / 15.0 * 10 * 0.2)

    # Test the right tail mass for the unbonded
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, carbon, unbonded, 1.5),
        0.66666667)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, nitrogen, unbonded, 1.5),
        0.66666667)

    # Test the bonded inside the pdf.
    # 3/8 are the counts in the data file
    # * 10 is for the pdf because the bucket is 0.1 wide
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, carbon, single, 1.25),
        3.0 / 8.0 * 10)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, nitrogen, single, 1.25),
        3.0 / 8.0 * 10)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, nitrogen, double, 1.25),
        3.0 / 8.0 * 10)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, nitrogen, triple, 1.25),
        3.0 / 8.0 * 10)

    # Check for no right tail mass for the bonded
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, carbon, single, 1.5), 0.0)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, nitrogen, single, 1.5), 0.0)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, nitrogen, double, 1.5), 0.0)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, nitrogen, triple, 1.5), 0.0)

  def test_add_from_sparse_dataframe(self):
    df = pd.DataFrame.from_records([
        ('c', 'c', 1, '1.0', 10),
        ('c', 'c', 1, '1.2', 30),
        ('n', 'o', 2, '1.0', 50),
        ('n', 'o', 2, '1.5', 50),
        ('n', 'n', 0, '1.5', 100),
        ('n', 'n', 0, '1.8', 100),
    ],
                                   columns=[
                                       'atom_char_0', 'atom_char_1',
                                       'bond_type', 'length_str', 'count'
                                   ])
    all_dists = bond_length_distribution.AllAtomPairLengthDistributions()
    all_dists.add_from_sparse_dataframe(
        df, sig_digits=1, unbonded_right_tail_mass=0.8)

    carbon = dataset_pb2.BondTopology.AtomType.ATOM_C
    nitrogen = dataset_pb2.BondTopology.AtomType.ATOM_N
    oxygen = dataset_pb2.BondTopology.AtomType.ATOM_O
    unbonded = dataset_pb2.BondTopology.BondType.BOND_UNDEFINED
    single = dataset_pb2.BondTopology.BondType.BOND_SINGLE
    double = dataset_pb2.BondTopology.BondType.BOND_DOUBLE

    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, carbon, single, 1.05), 2.5)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(carbon, carbon, single, 999), 0.0)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(nitrogen, oxygen, double, 1.55), 5.0)
    self.assertAlmostEqual(
        all_dists.pdf_length_given_type(nitrogen, nitrogen, unbonded, 1.85),
        1.0)
    # This makes sure the right tail mass was included
    self.assertGreater(
        all_dists.pdf_length_given_type(nitrogen, nitrogen, unbonded, 2.0), 0.0)
    self.assertGreater(
        all_dists.pdf_length_given_type(nitrogen, nitrogen, unbonded, 3.0), 0.0)


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


if __name__ == '__main__':
  absltest.main()
