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

"""Tests for query_sqlite."""

import os
import tempfile

from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
from tensorflow.io import gfile

from smu import dataset_pb2
from smu import smu_sqlite
from smu.parser import smu_parser_lib

from smu import query_sqlite

TESTDATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'testdata')


class TopologyDetectionTest(absltest.TestCase):

  def setUp(self):
    query_sqlite.GeometryData._singleton = None

  def tearDown(self):
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

    self.assertEqual([c.conformer_id for c in got], [618451001, 618451123])

if __name__ == '__main__':
  absltest.main()
