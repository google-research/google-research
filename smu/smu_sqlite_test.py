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
"""Tests for smu_sqlite."""

import os
import tempfile

from absl.testing import absltest

from smu import dataset_pb2
from smu import smu_sqlite
from smu.geometry import bond_length_distribution
from smu.parser import smu_utils_lib


class SmuSqliteTest(absltest.TestCase):
  """Tests for smu_sqlite.

  Some notes about the molecule and bond topoology ids used in the tests:
  In the real dataset, bond topologies are a number between 1 and ~900k.
  Molecule ids are a bond topology id * 1000 + id(between 000 and 999). So the
  smallest molecule id we could see is 1000 and the largest is ~900999.
  Therefore, throughout these tests, we use molecule ids in the low thousands
  with small number % 1000. For exemple, create_db uses 2001, 4001, 6001, 8001
  That creates single digit bond topology ids.
  """

  def setUp(self):
    super(SmuSqliteTest, self).setUp()
    self.db_filename = os.path.join(tempfile.mkdtemp(), 'smu_test.sqlite')

  def add_bond_topology_to_molecule(self, molecule, btid, source):
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
    bt.topo_id = btid
    bt.smiles = 'C' * btid
    bt.info = source
    molecule.bond_topo.append(bt)

  def make_fake_molecule(self, mid):
    molecule = dataset_pb2.Molecule()
    molecule.mol_id = mid
    self.add_bond_topology_to_molecule(molecule, mid // 1000,
                                       dataset_pb2.BondTopology.SOURCE_DDT)
    return molecule

  def encode_molecules(self, molecules):
    return [c.SerializeToString() for c in molecules]

  def create_db(self):
    db = smu_sqlite.SMUSQLite(self.db_filename, 'c')
    db.bulk_insert(
        self.encode_molecules(
            [self.make_fake_molecule(mid) for mid in range(2001, 10001, 2000)]))
    return db

  def create_db_with_multiple_bond_topology(self):
    # We'll set up 3 CIDS with one or more btids associated with them.
    # mid: 1000 -> btid 1
    # mid: 2000 -> btid 2, 1
    # mid: 3000 -> btid 3, 1, 2
    mol1 = self.make_fake_molecule(1000)
    mol2 = self.make_fake_molecule(2000)
    self.add_bond_topology_to_molecule(mol2, 1,
                                       dataset_pb2.BondTopology.SOURCE_DDT)

    mol3 = self.make_fake_molecule(3000)
    self.add_bond_topology_to_molecule(mol3, 1,
                                       dataset_pb2.BondTopology.SOURCE_DDT)
    self.add_bond_topology_to_molecule(mol3, 2,
                                       dataset_pb2.BondTopology.SOURCE_DDT)

    db = smu_sqlite.SMUSQLite(self.db_filename, 'c')
    db.bulk_insert(self.encode_molecules([mol1, mol2, mol3]))

    return db

  def test_find_topo_id_for_smiles(self):
    db = self.create_db()

    self.assertEqual(db.find_topo_id_for_smiles('CC'), 2)
    with self.assertRaises(KeyError):
      db.find_topo_id_for_smiles('DoesNotExist')

  def test_get_smiles_to_id_dict(self):
    db = self.create_db()

    self.assertEqual(db.get_smiles_id_dict(), {
        'CC': 2,
        'CCCC': 4,
        'CCCCCC': 6,
        'CCCCCCCC': 8,
    })

  def test_bulk_insert_smiles(self):
    db = self.create_db()

    with self.assertRaises(KeyError):
      db.find_topo_id_for_smiles('NewSmiles')

    db.bulk_insert_smiles([['FirstSmiles', 111], ['NewSmiles', 222]])
    self.assertEqual(db.find_topo_id_for_smiles('NewSmiles'), 222)

  def test_find_by_mol_id(self):
    db = self.create_db()

    got = db.find_by_mol_id(4001)
    self.assertEqual(got.mol_id, 4001)
    self.assertEqual(got.bond_topo[0].smiles, 'CCCC')

  def test_key_errors(self):
    db = self.create_db()

    with self.assertRaises(KeyError):
      db.find_by_mol_id(999)

  def test_write(self):
    create_db = self.create_db()
    del create_db

    db = smu_sqlite.SMUSQLite(self.db_filename, 'w')
    # The create_db makes molecule ids ending in 001. We'll add molecule ids
    # ending in 005 as the extra written ones to make it clear that they are
    # different.
    db.bulk_insert(
        self.encode_molecules([
            self.make_fake_molecule(mid) for mid in range(50005, 60005, 2000)
        ]))
    # Check an id that was already there
    self.assertEqual(db.find_by_mol_id(4001).mol_id, 4001)
    # Check an id that we added
    self.assertEqual(db.find_by_mol_id(52005).mol_id, 52005)

  def test_read(self):
    create_db = self.create_db()
    del create_db

    db = smu_sqlite.SMUSQLite(self.db_filename, 'r')
    with self.assertRaises(smu_sqlite.ReadOnlyError):
      db.bulk_insert(self.encode_molecules([self.make_fake_molecule(9999)]))

    with self.assertRaises(KeyError):
      db.find_by_mol_id(9999)

    self.assertEqual(db.find_by_mol_id(4001).mol_id, 4001)

  def test_vaccum(self):
    db = self.create_db()
    self.assertEqual(db.find_by_mol_id(2001).mol_id, 2001)
    db.vacuum()
    self.assertEqual(db.find_by_mol_id(2001).mol_id, 2001)

  def test_smiles_iteration(self):
    db = self.create_db()

    iterate = db.smiles_iter()
    self.assertEqual(('CC', 2), next(iterate))
    self.assertEqual(('CCCC', 4), next(iterate))
    self.assertEqual(('CCCCCC', 6), next(iterate))
    self.assertEqual(('CCCCCCCC', 8), next(iterate))
    with self.assertRaises(StopIteration):
      next(iterate)

  def test_iteration(self):
    db = self.create_db()

    got_mids = [molecule.mol_id for molecule in db]
    self.assertCountEqual(got_mids, [2001, 4001, 6001, 8001])

  def test_find_by_topo_id_list(self):
    db = self.create_db_with_multiple_bond_topology()

    # Test the cases with 1, 2, and 3 results
    got_mids = [
        mol.mol_id for mol in db.find_by_topo_id_list(
            [3], smu_utils_lib.WhichTopologies.ALL)
    ]
    self.assertCountEqual(got_mids, [3000])

    got_mids = [
        mol.mol_id for mol in db.find_by_topo_id_list(
            [2], smu_utils_lib.WhichTopologies.ALL)
    ]
    self.assertCountEqual(got_mids, [2000, 3000])

    got_mids = [
        mol.mol_id for mol in db.find_by_topo_id_list(
            [1], smu_utils_lib.WhichTopologies.ALL)
    ]
    self.assertCountEqual(got_mids, [1000, 2000, 3000])

    # and test a non existent id
    self.assertEmpty(
        list(db.find_by_topo_id_list([999], smu_utils_lib.WhichTopologies.ALL)))

  def test_find_by_topo_id_list_multi_arg(self):
    db = self.create_db()

    got_mids = [
        mol.mol_id for mol in db.find_by_topo_id_list(
            [2, 8], smu_utils_lib.WhichTopologies.ALL)
    ]
    # This is testing that we only get 55000 returned once
    self.assertCountEqual(got_mids, [2001, 8001])

  def test_find_by_topo_id_unique_only(self):
    db = self.create_db()

    mol = self.make_fake_molecule(55000)
    self.add_bond_topology_to_molecule(mol, 55,
                                       dataset_pb2.BondTopology.SOURCE_DDT)
    db.bulk_insert(self.encode_molecules([mol]))

    got_mids = [
        mol.mol_id for mol in db.find_by_topo_id_list(
            [55], smu_utils_lib.WhichTopologies.ALL)
    ]
    # This is testing that we only get 55000 returned once
    self.assertCountEqual(got_mids, [55000])

  def test_find_by_topo_id_source_filtering(self):
    db = smu_sqlite.SMUSQLite(self.db_filename, 'c')
    # We'll make 2 molecules
    # 2001 with bt id 10 (DDT, STARTING) and bt id 11 (MLCR)
    # 4001 with bt id 10 (DDT), bt id 11 (DDT, STARTING), bt id 12 (CSD)
    # 6001 with bt id 12 (MLCR)
    molecules = []
    molecules.append(dataset_pb2.Molecule(mol_id=2001))
    self.add_bond_topology_to_molecule(
        molecules[-1], 10, dataset_pb2.BondTopology.SOURCE_STARTING
        | dataset_pb2.BondTopology.SOURCE_DDT)
    self.add_bond_topology_to_molecule(molecules[-1], 11,
                                       dataset_pb2.BondTopology.SOURCE_MLCR)
    molecules.append(dataset_pb2.Molecule(mol_id=4001))
    self.add_bond_topology_to_molecule(molecules[-1], 10,
                                       dataset_pb2.BondTopology.SOURCE_DDT)
    self.add_bond_topology_to_molecule(
        molecules[-1], 11, dataset_pb2.BondTopology.SOURCE_STARTING
        | dataset_pb2.BondTopology.SOURCE_DDT)
    self.add_bond_topology_to_molecule(molecules[-1], 12,
                                       dataset_pb2.BondTopology.SOURCE_CSD)
    molecules.append(dataset_pb2.Molecule(mol_id=6001))
    self.add_bond_topology_to_molecule(molecules[-1], 12,
                                       dataset_pb2.BondTopology.SOURCE_MLCR)
    db.bulk_insert(self.encode_molecules(molecules))

    def ids_for(bt_id, which):
      return [c.mol_id for c in db.find_by_topo_id_list([bt_id], which)]

    self.assertEqual(
        ids_for(10, smu_utils_lib.WhichTopologies.ALL), [2001, 4001])
    self.assertEqual(
        ids_for(11, smu_utils_lib.WhichTopologies.ALL), [2001, 4001])
    self.assertEqual(
        ids_for(12, smu_utils_lib.WhichTopologies.ALL), [4001, 6001])

    self.assertEqual(
        ids_for(10, smu_utils_lib.WhichTopologies.STARTING), [2001])
    self.assertEqual(ids_for(11, smu_utils_lib.WhichTopologies.MLCR), [2001])
    self.assertEqual(ids_for(12, smu_utils_lib.WhichTopologies.CSD), [4001])

    self.assertEmpty(ids_for(12, smu_utils_lib.WhichTopologies.DDT))
    self.assertEmpty(ids_for(11, smu_utils_lib.WhichTopologies.CSD))

  def test_find_by_smiles_list(self):
    db = self.create_db_with_multiple_bond_topology()

    # Test the cases with 1, 2, and 3 results
    got_mids = [
        mol.mol_id for mol in db.find_by_smiles_list(
            ['CCC'], smu_utils_lib.WhichTopologies.ALL)
    ]
    self.assertCountEqual(got_mids, [3000])

    got_mids = [
        mol.mol_id for mol in db.find_by_smiles_list(
            ['CC'], smu_utils_lib.WhichTopologies.ALL)
    ]
    self.assertCountEqual(got_mids, [2000, 3000])

    got_mids = [
        mol.mol_id for mol in db.find_by_smiles_list(
            ['C'], smu_utils_lib.WhichTopologies.ALL)
    ]
    self.assertCountEqual(got_mids, [1000, 2000, 3000])

    # and test a non existent id
    self.assertEmpty(
        list(
            db.find_by_smiles_list(['I do not exist'],
                                   smu_utils_lib.WhichTopologies.ALL)))

  def test_find_by_smiles_list_multi_arg(self):
    db = self.create_db()

    got_mids = [
        mol.mol_id for mol in db.find_by_smiles_list(
            ['CC', 'CCCCCC'], smu_utils_lib.WhichTopologies.ALL)
    ]
    self.assertCountEqual(got_mids, [2001, 6001])

  def test_repeat_smiles_insert(self):
    db = smu_sqlite.SMUSQLite(self.db_filename, 'c')
    db.bulk_insert(
        self.encode_molecules(
            [self.make_fake_molecule(mid) for mid in [2001, 2002, 2003]]))
    got_mids = [
        molecule.mol_id for molecule in db.find_by_smiles_list(
            ['CC'], smu_utils_lib.WhichTopologies.ALL)
    ]
    self.assertCountEqual(got_mids, [2001, 2002, 2003])

  def test_find_by_expanded_stoichiometry_list(self):
    db = smu_sqlite.SMUSQLite(self.db_filename, 'c')
    db.bulk_insert(
        self.encode_molecules(
            [self.make_fake_molecule(mid) for mid in [2001, 2002, 4004]]))

    got_mids = [
        molecule.mol_id
        for molecule in db.find_by_expanded_stoichiometry_list(['(ch2)2(ch3)2'])
    ]
    self.assertCountEqual(got_mids, [4004])

    got_mids = [
        molecule.mol_id
        for molecule in db.find_by_expanded_stoichiometry_list(['(ch3)2'])
    ]
    self.assertCountEqual(got_mids, [2001, 2002])

    got_mids = [
        molecule.mol_id for molecule in db.find_by_expanded_stoichiometry_list(
            ['(ch2)2(ch3)2', '(ch3)2'])
    ]
    self.assertCountEqual(got_mids, [2001, 2002, 4004])

    self.assertEmpty(list(db.find_by_expanded_stoichiometry_list(['(nh)'])))

  def test_find_by_stoichiometry(self):
    db = smu_sqlite.SMUSQLite(self.db_filename, 'c')
    db.bulk_insert(
        self.encode_molecules(
            [self.make_fake_molecule(mid) for mid in [2001, 2002, 4004]]))

    got_mids = [
        molecule.mol_id for molecule in db.find_by_stoichiometry('c2h6')
    ]
    self.assertCountEqual(got_mids, [2001, 2002])

    got_mids = [
        molecule.mol_id for molecule in db.find_by_stoichiometry('c4h10')
    ]
    self.assertCountEqual(got_mids, [4004])

    self.assertEmpty(list(db.find_by_stoichiometry('c3')))

    with self.assertRaises(smu_utils_lib.StoichiometryError):
      db.find_by_stoichiometry('P3Li')

  def test_find_by_topology(self):
    db = smu_sqlite.SMUSQLite(self.db_filename, 'c')

    # We'll make a pretty fake molecule. N2O2H2 with
    # the O at 0,0
    # the Ns at 1.1,0 and 0,1.1
    # The Hs right night to the Ns
    # We'll given it the ring topology to start and the symetric ring broken
    # topologies should be found.

    molecule = dataset_pb2.Molecule(mol_id=9999)
    molecule.prop.calc.fate = dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW

    bt = molecule.bond_topo.add(smiles='N1NO1', topo_id=100)
    geom = molecule.opt_geo.atompos

    bt.atom.append(dataset_pb2.BondTopology.ATOM_O)
    geom.append(dataset_pb2.Geometry.AtomPos(x=0, y=0, z=0))

    bt.atom.append(dataset_pb2.BondTopology.ATOM_N)
    geom.append(dataset_pb2.Geometry.AtomPos(x=0, y=1.1, z=0))
    bt.bond.append(
        dataset_pb2.BondTopology.Bond(
            atom_a=0, atom_b=1, bond_type=dataset_pb2.BondTopology.BOND_SINGLE))
    bt.atom.append(dataset_pb2.BondTopology.ATOM_N)
    geom.append(dataset_pb2.Geometry.AtomPos(x=1.1, y=0, z=0))
    bt.bond.append(
        dataset_pb2.BondTopology.Bond(
            atom_a=0, atom_b=2, bond_type=dataset_pb2.BondTopology.BOND_SINGLE))
    bt.bond.append(
        dataset_pb2.BondTopology.Bond(
            atom_a=1, atom_b=2, bond_type=dataset_pb2.BondTopology.BOND_SINGLE))

    bt.atom.append(dataset_pb2.BondTopology.ATOM_H)
    geom.append(dataset_pb2.Geometry.AtomPos(x=0, y=1.2, z=0))
    bt.bond.append(
        dataset_pb2.BondTopology.Bond(
            atom_a=1, atom_b=3, bond_type=dataset_pb2.BondTopology.BOND_SINGLE))
    bt.atom.append(dataset_pb2.BondTopology.ATOM_H)
    geom.append(dataset_pb2.Geometry.AtomPos(x=1.2, y=0, z=0))
    bt.bond.append(
        dataset_pb2.BondTopology.Bond(
            atom_a=2, atom_b=4, bond_type=dataset_pb2.BondTopology.BOND_SINGLE))

    for pos in geom:
      pos.x /= smu_utils_lib.BOHR_TO_ANGSTROMS
      pos.y /= smu_utils_lib.BOHR_TO_ANGSTROMS
      pos.z /= smu_utils_lib.BOHR_TO_ANGSTROMS

    db.bulk_insert([molecule.SerializeToString()])
    db.bulk_insert_smiles([['N1NO1', 100], ['N=[NH+][O-]', 101]])

    bond_lengths = bond_length_distribution.make_fake_empiricals()

    # We'll query by the topology that was in the DB then the one that wasn't
    for query_smiles in ['N1NO1', 'N=[NH+][O-]']:

      got = list(db.find_by_topology(query_smiles, bond_lengths=bond_lengths))
      self.assertLen(got, 1)
      self.assertCountEqual([100, 101, 101],
                            [bt.topo_id for bt in got[0].bond_topo])

  def test_find_topo_id_by_smarts(self):
    db = self.create_db()

    # 5 carbons in a row will only match the 6 or 8 carbon bond topology
    self.assertEqual(list(db.find_topo_id_by_smarts('CCCCC')), [6, 8])

    with self.assertRaises(ValueError):
      _ = list(db.find_topo_id_by_smarts(']Broken)](Smarts'))


if __name__ == '__main__':
  absltest.main()
