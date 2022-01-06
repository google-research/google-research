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

"""Tests for smu_sqlite."""

import os
import tempfile

from absl.testing import absltest

from smu import dataset_pb2
from smu import smu_sqlite
from smu.parser import smu_utils_lib


class SmuSqliteTest(absltest.TestCase):
  """Tests for smu_sqlite.

  Some notes about the conformer and bond topoology ids used in the tests:
  In the real dataset, bond topologies are a number between 1 and ~900k.
  Conformer ids are a bond topology id * 1000 + id(between 000 and 999). So the
  smallest conformer id we could see is 1000 and the largest is ~900999.
  Therefore, throughout these tests, we use conformer ids in the low thousands
  with small number % 1000. For exemple, create_db uses 2001, 4001, 6001, 8001
  That creates single digit bond topology ids.
  """

  def setUp(self):
    super(SmuSqliteTest, self).setUp()
    self.db_filename = os.path.join(tempfile.mkdtemp(), 'smu_test.sqlite')

  def add_bond_topology_to_conformer(self, conformer, btid):
    # We'll use a simple rule for making smiles. The SMILES is just btid
    # number of Cs
    def make_connectivity_matrix(num_c):
      if num_c == 2:
        return '1'
      return '1' + ('0' * (num_c - 2)) + make_connectivity_matrix(num_c - 1)

    if btid == 1:
      bt = smu_utils_lib.create_bond_topology('C', '', '4')
    else:
      bt = smu_utils_lib.create_bond_topology('C' * btid,
                                              make_connectivity_matrix(btid),
                                              '3' + ('2' * (btid - 2)) + '3')
    bt.bond_topology_id = btid
    bt.smiles = 'C' * btid
    conformer.bond_topologies.append(bt)

  def make_fake_conformer(self, cid):
    conformer = dataset_pb2.Conformer()
    conformer.conformer_id = cid
    self.add_bond_topology_to_conformer(conformer, cid // 1000)
    return conformer

  def encode_conformers(self, conformers):
    return [c.SerializeToString() for c in conformers]

  def create_db(self):
    db = smu_sqlite.SMUSQLite(self.db_filename, 'c')
    db.bulk_insert(
        self.encode_conformers([
            self.make_fake_conformer(cid) for cid in range(2001, 10001, 2000)
        ]))
    return db

  def create_db_with_multiple_bond_topology(self):
    # We'll set up 3 CIDS with one or more btids associated with them.
    # cid: 1000 -> btid 1
    # cid: 2000 -> btid 2, 1
    # cid: 3000 -> btid 3, 1, 2
    conf1 = self.make_fake_conformer(1000)
    conf2 = self.make_fake_conformer(2000)
    self.add_bond_topology_to_conformer(conf2, 1)
    conf3 = self.make_fake_conformer(3000)
    self.add_bond_topology_to_conformer(conf3, 1)
    self.add_bond_topology_to_conformer(conf3, 2)

    db = smu_sqlite.SMUSQLite(self.db_filename, 'c')
    db.bulk_insert(self.encode_conformers([conf1, conf2, conf3]))

    return db

  def test_find_by_conformer_id(self):
    db = self.create_db()

    got = db.find_by_conformer_id(4001)
    self.assertEqual(got.conformer_id, 4001)
    self.assertEqual(got.bond_topologies[0].smiles, 'CCCC')

  def test_key_errors(self):
    db = self.create_db()

    with self.assertRaises(KeyError):
      db.find_by_conformer_id(999)

  def test_write(self):
    create_db = self.create_db()
    del create_db

    db = smu_sqlite.SMUSQLite(self.db_filename, 'w')
    # The create_db makes conformer ids ending in 001. We'll add conformer ids
    # ending in 005 as the extra written ones to make it clear that they are
    # different.
    db.bulk_insert(
        self.encode_conformers([
            self.make_fake_conformer(cid) for cid in range(50005, 60005, 2000)
        ]))
    # Check an id that was already there
    self.assertEqual(db.find_by_conformer_id(4001).conformer_id, 4001)
    # Check an id that we added
    self.assertEqual(db.find_by_conformer_id(52005).conformer_id, 52005)

  def test_read(self):
    create_db = self.create_db()
    del create_db

    db = smu_sqlite.SMUSQLite(self.db_filename, 'r')
    with self.assertRaises(smu_sqlite.ReadOnlyError):
      db.bulk_insert(self.encode_conformers([self.make_fake_conformer(9999)]))

    with self.assertRaises(KeyError):
      db.find_by_conformer_id(9999)

    self.assertEqual(db.find_by_conformer_id(4001).conformer_id, 4001)

  def test_iteration(self):
    db = self.create_db()

    got_cids = [conformer.conformer_id for conformer in db]
    self.assertCountEqual(got_cids, [2001, 4001, 6001, 8001])

  def test_find_by_bond_topology_id(self):
    db = self.create_db_with_multiple_bond_topology()

    # Test the cases with 1, 2, and 3 results
    got_cids = [
        conformer.conformer_id for conformer in db.find_by_bond_topology_id(3)
    ]
    self.assertCountEqual(got_cids, [3000])

    got_cids = [
        conformer.conformer_id for conformer in db.find_by_bond_topology_id(2)
    ]
    self.assertCountEqual(got_cids, [2000, 3000])

    got_cids = [
        conformer.conformer_id for conformer in db.find_by_bond_topology_id(1)
    ]
    self.assertCountEqual(got_cids, [1000, 2000, 3000])

    # and test a non existent id
    self.assertEmpty(list(db.find_by_bond_topology_id(999)))

  def test_find_by_smiles(self):
    db = self.create_db_with_multiple_bond_topology()

    # Test the cases with 1, 2, and 3 results
    got_cids = [
        conformer.conformer_id for conformer in db.find_by_smiles('CCC')
    ]
    self.assertCountEqual(got_cids, [3000])

    got_cids = [conformer.conformer_id for conformer in db.find_by_smiles('CC')]
    self.assertCountEqual(got_cids, [2000, 3000])

    got_cids = [conformer.conformer_id for conformer in db.find_by_smiles('C')]
    self.assertCountEqual(got_cids, [1000, 2000, 3000])

    # and test a non existent id
    self.assertEmpty(list(db.find_by_smiles('I do not exist')))

  def test_find_by_expanded_stoichiometry(self):
    db = smu_sqlite.SMUSQLite(self.db_filename, 'c')
    db.bulk_insert(
        self.encode_conformers(
            [self.make_fake_conformer(cid) for cid in [2001, 2002, 4004]]))

    got_cids = [
        conformer.conformer_id
        for conformer in db.find_by_expanded_stoichiometry('(ch2)2(ch3)2')
    ]
    self.assertCountEqual(got_cids, [4004])

    got_cids = [
        conformer.conformer_id
        for conformer in db.find_by_expanded_stoichiometry('(ch3)2')
    ]
    self.assertCountEqual(got_cids, [2001, 2002])

    self.assertEmpty(list(db.find_by_expanded_stoichiometry('(nh)')))

  def test_no_expanded_stoichiometry_on_ineligible(self):
    db = smu_sqlite.SMUSQLite(self.db_filename, 'c')
    conf = self.make_fake_conformer(2001)
    # This makes the conformer ineligible
    conf.properties.errors.status = 600
    db.bulk_insert(self.encode_conformers([conf]))
    got_cids = [
        conformer.conformer_id
        for conformer in db.find_by_expanded_stoichiometry('')
    ]
    self.assertCountEqual(got_cids, [2001])


if __name__ == '__main__':
  absltest.main()
