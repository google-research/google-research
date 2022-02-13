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

from smu import dataset_pb2
from smu import query_sqlite
from smu import smu_sqlite
from smu.geometry import bond_length_distribution
from smu.parser import smu_parser_lib

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


class GeometryDataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.bond_lengths_csv = os.path.join(TESTDATA_PATH,
                                         'minmax_bond_distances.csv')

  def create(self, bond_lengths):
    self._geometry_data = query_sqlite.GeometryData(self.bond_lengths_csv,
                                                    bond_lengths)

  def is_empirical(self, atom_a, atom_b, bond_type):
    length_dist = self._geometry_data.bond_lengths[(atom_a, atom_b)][bond_type]
    return isinstance(length_dist,
                      bond_length_distribution.EmpiricalLengthDistribution)

  def test_empty_bond_lengths(self):
    self.create(None)
    self.assertTrue(self.is_empirical(ATOM_C, ATOM_C, BOND_SINGLE))
    self.assertTrue(self.is_empirical(ATOM_N, ATOM_O, BOND_DOUBLE))

  def test_simple(self):
    # The real testing for this is in bond_length_distribution. We're just
    # checking that it is hooked up
    self.create('N=O:1-2')
    self.assertTrue(self.is_empirical(ATOM_C, ATOM_C, BOND_SINGLE))
    self.assertFalse(self.is_empirical(ATOM_N, ATOM_O, BOND_DOUBLE))



if __name__ == '__main__':
  absltest.main()
