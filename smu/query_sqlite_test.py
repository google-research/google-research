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

"""Tests for query_sqlite."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

from smu import dataset_pb2
from smu import query_sqlite
from smu import smu_sqlite
from smu.geometry import bond_length_distribution
from smu.parser import smu_parser_lib

TESTDATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'testdata')


class TopologyDetectionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    query_sqlite.GeometryData._singleton = None

  def tearDown(self):
    super().tearDown()
    query_sqlite.GeometryData._singleton = None

  def test_simple(self):
    db_filename = os.path.join(tempfile.mkdtemp(), 'query_sqlite_test.sqlite')
    db = smu_sqlite.SMUSQLite(db_filename, 'w')
    parser = smu_parser_lib.SmuParser(
        os.path.join(TESTDATA_PATH, 'pipeline_input_stage2.dat'))
    db.bulk_insert(x.SerializeToString() for (x, _) in parser.process_stage2())

    with flagsaver.flagsaver(
        bond_lengths_csv=os.path.join(TESTDATA_PATH,
                                      'minmax_bond_distances.csv'),
        bond_topology_csv=os.path.join(TESTDATA_PATH,
                                       'pipeline_bond_topology.csv')):
      got = list(query_sqlite.topology_query(db, 'COC(=CF)OC'))

    # These are just the two conformers that came in with this smiles, so no
    # interesting detection happened, but it verifies that the code ran without
    # error.
    self.assertEqual([c.conformer_id for c in got], [618451001, 618451123])
    self.assertLen(got[0].bond_topologies, 1)
    self.assertEqual(got[0].bond_topologies[0].bond_topology_id, 618451)
    self.assertLen(got[1].bond_topologies, 1)
    self.assertEqual(got[1].bond_topologies[0].bond_topology_id, 618451)


# Some shortcuts to write less characters below
ATOM_C = dataset_pb2.BondTopology.ATOM_C
ATOM_N = dataset_pb2.BondTopology.ATOM_N
ATOM_O = dataset_pb2.BondTopology.ATOM_O
ATOM_F = dataset_pb2.BondTopology.ATOM_F
BOND_UNDEFINED = dataset_pb2.BondTopology.BOND_UNDEFINED
BOND_SINGLE = dataset_pb2.BondTopology.BOND_SINGLE
BOND_DOUBLE = dataset_pb2.BondTopology.BOND_DOUBLE
BOND_TRIPLE = dataset_pb2.BondTopology.BOND_TRIPLE


class GeometryDataTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.bond_lengths_csv = os.path.join(TESTDATA_PATH,
                                         'minmax_bond_distances.csv')
    self.bond_topology_csv = os.path.join(TESTDATA_PATH,
                                          'pipeline_bond_topology.csv')

  def create(self, bond_lengths):
    self._geometry_data = query_sqlite.GeometryData(self.bond_lengths_csv,
                                                    bond_lengths,
                                                    self.bond_topology_csv)

  def is_empirical(self, atom_a, atom_b, bond_type):
    length_dist = self._geometry_data.bond_lengths[(atom_a, atom_b)][bond_type]
    return isinstance(length_dist,
                      bond_length_distribution.EmpiricalLengthDistribution)

  def fixed_window_min(self, atom_a, atom_b, bond_type):
    length_dist = self._geometry_data.bond_lengths[(atom_a, atom_b)][bond_type]
    if not isinstance(length_dist,
                      bond_length_distribution.FixedWindowLengthDistribution):
      return None
    return length_dist.minimum

  def fixed_window_max(self, atom_a, atom_b, bond_type):
    length_dist = self._geometry_data.bond_lengths[(atom_a, atom_b)][bond_type]
    if not isinstance(length_dist,
                      bond_length_distribution.FixedWindowLengthDistribution):
      return None
    return length_dist.maximum

  def fixed_window_right_tail_mass(self, atom_a, atom_b, bond_type):
    length_dist = self._geometry_data.bond_lengths[(atom_a, atom_b)][bond_type]
    if not isinstance(length_dist,
                      bond_length_distribution.FixedWindowLengthDistribution):
      return None
    return length_dist.right_tail_mass

  def test_empty_bond_lengths(self):
    self.create(None)
    self.assertTrue(self.is_empirical(ATOM_C, ATOM_C, BOND_SINGLE))
    self.assertTrue(self.is_empirical(ATOM_N, ATOM_O, BOND_DOUBLE))

  def test_fully_specified(self):
    self.create('C#C:1.1-1.2,N=O:1.3-1.4,O-F:1.5-1.6,C.F:2.0-2.1')

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
    self.create('C~C:1.1-1.2')

    self.assertEqual(self.fixed_window_min(ATOM_C, ATOM_C, BOND_SINGLE), 1.1)
    self.assertEqual(self.fixed_window_min(ATOM_C, ATOM_C, BOND_DOUBLE), 1.1)
    self.assertEqual(self.fixed_window_min(ATOM_C, ATOM_C, BOND_TRIPLE), 1.1)
    self.assertTrue(self.is_empirical(ATOM_C, ATOM_C, BOND_UNDEFINED))

  @parameterized.parameters(
      '*-N:1.1-1.2',
      'N-*:1.1-1.2',
  )
  def test_atom_wildcard(self, spec):
    self.create(spec)

    self.assertEqual(self.fixed_window_min(ATOM_N, ATOM_N, BOND_SINGLE), 1.1)
    self.assertEqual(self.fixed_window_min(ATOM_N, ATOM_C, BOND_SINGLE), 1.1)
    self.assertEqual(self.fixed_window_min(ATOM_N, ATOM_O, BOND_SINGLE), 1.1)
    self.assertEqual(self.fixed_window_min(ATOM_N, ATOM_F, BOND_SINGLE), 1.1)
    self.assertTrue(self.is_empirical(ATOM_N, ATOM_N, BOND_DOUBLE))
    self.assertTrue(self.is_empirical(ATOM_C, ATOM_C, BOND_SINGLE))

  def test_left_missing_dist(self):
    self.create('C-C:-1.2')
    self.assertEqual(self.fixed_window_min(ATOM_C, ATOM_C, BOND_SINGLE), 0)
    self.assertEqual(self.fixed_window_max(ATOM_C, ATOM_C, BOND_SINGLE), 1.2)

  def test_right_missing_dist(self):
    self.create('C-C:1.1-')
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
    with self.assertRaises(query_sqlite.BondLengthParseError):
      self.create(spec)


if __name__ == '__main__':
  absltest.main()
