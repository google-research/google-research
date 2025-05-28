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
"""Tests for smu_utils_lib."""

import copy
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from rdkit import Chem
from tensorflow.io import gfile

from google.protobuf import text_format
from smu import dataset_pb2
from smu.geometry import utilities
from smu.parser import smu_parser_lib
from smu.parser import smu_utils_lib

MAIN_DAT_FILE = 'x07_sample.dat'
STAGE1_DAT_FILE = 'x07_stage1.dat'
TESTDATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'testdata')


def str_to_bond_topology(s):
  bt = dataset_pb2.BondTopology()
  text_format.Parse(s, bt)
  return bt


def get_stage1_molecule():
  parser = smu_parser_lib.SmuParser(
      os.path.join(TESTDATA_PATH, STAGE1_DAT_FILE))
  molecule, _ = next(parser.process_stage1())
  return molecule


def get_stage2_molecule():
  parser = smu_parser_lib.SmuParser(os.path.join(TESTDATA_PATH, MAIN_DAT_FILE))
  molecule, _ = next(parser.process_stage2())
  return molecule


class GetOriginalLabelsTest(absltest.TestCase):

  def test_special_from_dat_id(self):
    self.assertIsNone(
        smu_utils_lib.special_case_bt_id_from_dat_id(123456, 'CC'))
    self.assertEqual(
        smu_utils_lib.special_case_bt_id_from_dat_id(999998, 'O'), 899650)
    self.assertEqual(
        smu_utils_lib.special_case_bt_id_from_dat_id(0, 'O'), 899650)
    with self.assertRaises(ValueError):
      smu_utils_lib.special_case_bt_id_from_dat_id(0, 'NotASpecialCaseSmiles')

  def test_special_from_bt_id(self):
    self.assertIsNone(smu_utils_lib.special_case_dat_id_from_bt_id(123456))
    self.assertEqual(
        smu_utils_lib.special_case_dat_id_from_bt_id(899651), 999997)

  def test_get_composition(self):
    bt = dataset_pb2.BondTopology()
    bt.atom.extend([
        dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_C,
        dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_H,
        dataset_pb2.BondTopology.ATOM_H, dataset_pb2.BondTopology.ATOM_H
    ])
    self.assertEqual('x03_c2nh3', smu_utils_lib.get_composition(bt))

  def test_get_original_label(self):
    molecule = dataset_pb2.Molecule()
    molecule.bond_topo.add()
    molecule.bond_topo[0].atom.extend([
        dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_C,
        dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_H,
        dataset_pb2.BondTopology.ATOM_H, dataset_pb2.BondTopology.ATOM_H
    ])
    molecule.mol_id = 123045
    self.assertEqual('x03_c2nh3.000123.045',
                     smu_utils_lib.get_original_label(molecule))

  def test_get_original_label_special_case(self):
    molecule = dataset_pb2.Molecule()
    molecule.bond_topo.add()
    molecule.bond_topo[0].atom.extend([
        dataset_pb2.BondTopology.ATOM_O,
        dataset_pb2.BondTopology.ATOM_H,
        dataset_pb2.BondTopology.ATOM_H,
    ])
    molecule.mol_id = 899650001

    self.assertEqual('x01_oh2.000000.001',
                     smu_utils_lib.get_original_label(molecule))


class ExpandedStoichiometryFromTopologyTest(absltest.TestCase):

  def test_cyclobutane(self):
    bt = smu_utils_lib.create_bond_topology('CCCC', '110011', '2222')
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometry_from_topology(bt), '(ch2)4')

  def test_ethylene(self):
    bt = smu_utils_lib.create_bond_topology('CC', '2', '22')
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometry_from_topology(bt), '(ch2)2')

  def test_acrylic_acid(self):
    bt = smu_utils_lib.create_bond_topology('CCCOO', '2000100210', '21001')
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometry_from_topology(bt),
        '(c)(ch)(ch2)(o)(oh)')

  def test_fluorine(self):
    bt = smu_utils_lib.create_bond_topology('OFF', '110', '000')
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometry_from_topology(bt), '(o)(f)2')

  def test_fully_saturated(self):
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometry_from_topology(
            smu_utils_lib.create_bond_topology('C', '', '4')), '(ch4)')
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometry_from_topology(
            smu_utils_lib.create_bond_topology('N', '', '3')), '(nh3)')
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometry_from_topology(
            smu_utils_lib.create_bond_topology('O', '', '2')), '(oh2)')
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometry_from_topology(
            smu_utils_lib.create_bond_topology('F', '', '1')), '(fh)')

  def test_nplus_oneg(self):
    bt = smu_utils_lib.create_bond_topology('NO', '1', '30')
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometry_from_topology(bt), '(nh3)(o)')


class ExpandedStoichiometryFromAtomListTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # shortcuts
    self._c = dataset_pb2.BondTopology.AtomType.ATOM_C
    self._n = dataset_pb2.BondTopology.AtomType.ATOM_N
    self._o = dataset_pb2.BondTopology.AtomType.ATOM_O
    self._f = dataset_pb2.BondTopology.AtomType.ATOM_F

  def test_basic(self):
    self.assertCountEqual(
        smu_utils_lib.expanded_stoichiometries_from_atom_list(
            [self._c, self._c], 4), ['(ch2)2', '(ch)(ch3)'])
    self.assertCountEqual(
        smu_utils_lib.expanded_stoichiometries_from_atom_list(
            [self._c, self._n], 4), ['(ch2)(nh2)', '(ch)(nh3)', '(ch3)(nh)'])
    self.assertCountEqual(
        smu_utils_lib.expanded_stoichiometries_from_atom_list(
            [self._c, self._c, self._n, self._o, self._f], 9),
        ['(ch3)2(nh3)(o)(f)', '(ch3)2(nh2)(oh)(f)', '(ch2)(ch3)(nh3)(oh)(f)'])
    self.assertCountEqual(
        smu_utils_lib.expanded_stoichiometries_from_atom_list(
            [self._c, self._c, self._n, self._o, self._f], 5),
        [
            '(c)(ch2)(nh2)(oh)(f)', '(ch)(ch3)(n)(oh)(f)',
            '(c)(ch3)(nh)(oh)(f)', '(c)(ch2)(nh3)(o)(f)', '(ch)(ch3)(nh)(o)(f)',
            '(c)(ch3)(nh2)(o)(f)', '(ch)2(nh2)(oh)(f)', '(ch2)2(n)(oh)(f)',
            '(ch)(ch2)(nh2)(o)(f)', '(c)(ch)(nh3)(oh)(f)', '(ch)2(nh3)(o)(f)',
            '(ch)(ch2)(nh)(oh)(f)', '(ch2)2(nh)(o)(f)', '(ch2)(ch3)(n)(o)(f)'
        ])


class ExpandedStoichiometryFromStoichiometry(absltest.TestCase):

  def test_basic(self):
    self.assertCountEqual(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry('N2'),
        ['(n)2'])
    self.assertCountEqual(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry('C2H2'),
        ['(ch)2', '(c)(ch2)'])
    self.assertCountEqual(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry('C2OH'),
        ['(c)2(oh)', '(c)(ch)(o)'])
    self.assertCountEqual(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry('CNOFH'),
        ['(ch)(n)(o)(f)', '(c)(n)(oh)(f)', '(c)(nh)(o)(f)'])

  def test_multi_char_digit(self):
    got = smu_utils_lib.expanded_stoichiometries_from_stoichiometry('C7H12')
    self.assertIn('(c)3(ch3)4', got)

  def test_special_cases(self):
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry('ch4'),
        {'(ch4)'})
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry('h4c'),
        {'(ch4)'})
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry('nh3'),
        {'(nh3)'})
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry('h3n'),
        {'(nh3)'})
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry('oh2'),
        {'(oh2)'})
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry('h2o'),
        {'(oh2)'})
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry('fh'),
        {'(fh)'})
    self.assertEqual(
        smu_utils_lib.expanded_stoichiometries_from_stoichiometry('hf'),
        {'(fh)'})

  def test_errors(self):
    with self.assertRaises(smu_utils_lib.StoichiometryError):
      smu_utils_lib.expanded_stoichiometries_from_stoichiometry('nonsense')
    with self.assertRaises(smu_utils_lib.StoichiometryError):
      smu_utils_lib.expanded_stoichiometries_from_stoichiometry('C2H42')


class ComputeBondedHydrogensTest(absltest.TestCase):

  def test_c2_single(self):
    bt = text_format.Parse(
        """
      atom: ATOM_C
      atom: ATOM_C
      bond {
        atom_b: 1
        bond_type: BOND_SINGLE
      }
      """, dataset_pb2.BondTopology())
    self.assertEqual(
        smu_utils_lib.compute_bonded_hydrogens(
            bt, smu_utils_lib.compute_adjacency_matrix(bt)), [3, 3])

  def test_cn_double(self):
    bt = text_format.Parse(
        """
      atom: ATOM_C
      atom: ATOM_N
      bond {
        atom_b: 1
        bond_type: BOND_DOUBLE
      }
      """, dataset_pb2.BondTopology())
    self.assertEqual(
        smu_utils_lib.compute_bonded_hydrogens(
            bt, smu_utils_lib.compute_adjacency_matrix(bt)), [2, 1])

  def test_cn_double_opposite_bond_order(self):
    bt = text_format.Parse(
        """
      atom: ATOM_C
      atom: ATOM_N
      bond {
        atom_a: 1
        bond_type: BOND_DOUBLE
      }
      """, dataset_pb2.BondTopology())
    self.assertEqual(
        smu_utils_lib.compute_bonded_hydrogens(
            bt, smu_utils_lib.compute_adjacency_matrix(bt)), [2, 1])

  def test_charged(self):
    bt = text_format.Parse(
        """
      atom: ATOM_NPOS
      atom: ATOM_ONEG
      bond {
        atom_b: 1
        bond_type: BOND_SINGLE
      }
      """, dataset_pb2.BondTopology())
    self.assertListEqual(
        smu_utils_lib.compute_bonded_hydrogens(
            bt, smu_utils_lib.compute_adjacency_matrix(bt)), [3, 0])

  def test_explicit_hs(self):
    bt = text_format.Parse(
        """
      atom: ATOM_C
      atom: ATOM_O
      atom: ATOM_H
      atom: ATOM_H
      bond {
        atom_b: 1
        bond_type: BOND_DOUBLE
      }
      bond {
        atom_b: 2
        bond_type: BOND_SINGLE
      }
      bond {
        atom_b: 3
        bond_type: BOND_SINGLE
      }
      """, dataset_pb2.BondTopology())
    self.assertEqual(
        smu_utils_lib.compute_bonded_hydrogens(
            bt, smu_utils_lib.compute_adjacency_matrix(bt)), [2, 0])

  def test_explicit_hs_opposite_bond_oder(self):
    bt = text_format.Parse(
        """
      atom: ATOM_C
      atom: ATOM_O
      atom: ATOM_H
      atom: ATOM_H
      bond {
        atom_a: 1
        bond_type: BOND_DOUBLE
      }
      bond {
        atom_a: 2
        bond_type: BOND_SINGLE
      }
      bond {
        atom_a: 3
        bond_type: BOND_SINGLE
      }
      """, dataset_pb2.BondTopology())
    self.assertEqual(
        smu_utils_lib.compute_bonded_hydrogens(
            bt, smu_utils_lib.compute_adjacency_matrix(bt)), [2, 0])


class ParseBondTopologyTest(absltest.TestCase):

  def test_4_heavy(self):
    num_atoms, atoms_str, matrix, hydrogens = smu_utils_lib.parse_bond_topology_line(
        ' 4  N+O O O-  010110  3000')
    self.assertEqual(num_atoms, 4)
    self.assertEqual(atoms_str, 'N+O O O-')
    self.assertEqual(matrix, '010110')
    self.assertEqual(hydrogens, '3000')

  def test_7_heavy(self):
    num_atoms, atoms_str, matrix, hydrogens = smu_utils_lib.parse_bond_topology_line(
        ' 7  N+O O O O-F F   001011101001000000000  1000000')
    self.assertEqual(num_atoms, 7)
    self.assertEqual(atoms_str, 'N+O O O O-F F ')  # Note the trailing space
    self.assertEqual(matrix, '001011101001000000000')
    self.assertEqual(hydrogens, '1000000')


class CreateBondTopologyTest(absltest.TestCase):

  def test_no_charged(self):
    got = smu_utils_lib.create_bond_topology('CNFF', '111000', '1200')
    expected_str = """
atom: ATOM_C
atom: ATOM_N
atom: ATOM_F
atom: ATOM_F
atom: ATOM_H
atom: ATOM_H
atom: ATOM_H
bond {
  atom_b: 1
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 2
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 3
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 4
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 1
  atom_b: 5
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 1
  atom_b: 6
  bond_type: BOND_SINGLE
}
"""
    expected = str_to_bond_topology(expected_str)
    self.assertEqual(str(expected), str(got))

  def test_charged(self):
    # This is actually C N N+O-
    got = smu_utils_lib.create_bond_topology('CNNO', '200101', '2020')
    expected_str = """
atom: ATOM_C
atom: ATOM_N
atom: ATOM_NPOS
atom: ATOM_ONEG
atom: ATOM_H
atom: ATOM_H
atom: ATOM_H
atom: ATOM_H
bond {
  atom_b: 1
  bond_type: BOND_DOUBLE
}
bond {
  atom_a: 1
  atom_b: 2
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 2
  atom_b: 3
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 4
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 5
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 2
  atom_b: 6
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 2
  atom_b: 7
  bond_type: BOND_SINGLE
}
"""
    expected = str_to_bond_topology(expected_str)
    self.assertEqual(str(expected), str(got))

  def test_one_heavy(self):
    got = smu_utils_lib.create_bond_topology('C', '', '4')
    expected_str = """
atom: ATOM_C
atom: ATOM_H
atom: ATOM_H
atom: ATOM_H
atom: ATOM_H
bond {
  atom_b: 1
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 2
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 3
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 4
  bond_type: BOND_SINGLE
}
"""
    expected = str_to_bond_topology(expected_str)
    self.assertEqual(str(expected), str(got))


class FromCSVTest(absltest.TestCase):

  def test_basic(self):
    infile = tempfile.NamedTemporaryFile(mode='w', delete=False)
    infile.write(
        'id,num_atoms,atoms_str,connectivity_matrix,hydrogens,smiles\n')
    infile.write('68,3,C N+O-,310,010,[NH+]#C[O-]\n')
    infile.write('134,4,N+O-F F ,111000,1000,[O-][NH+](F)F\n')
    infile.close()

    with gfile.GFile(infile.name, 'r') as fobj:
      out = smu_utils_lib.generate_bond_topologies_from_csv(fobj)

    bt = next(out)
    self.assertEqual(68, bt.topo_id)
    self.assertLen(bt.atom, 4)
    self.assertEqual(bt.smiles, '[NH+]#C[O-]')

    bt = next(out)
    self.assertEqual(134, bt.topo_id)
    self.assertLen(bt.atom, 5)
    self.assertEqual(bt.smiles, '[O-][NH+](F)F')


class BondTopologyToRDKitMoleculeTest(absltest.TestCase):

  def test_o2(self):
    bond_topology = str_to_bond_topology("""
atom: ATOM_O
atom: ATOM_O
bond {
  atom_b: 1
  bond_type: BOND_DOUBLE
}
""")
    got = smu_utils_lib.bond_topology_to_rdkit_molecule(bond_topology)
    self.assertEqual('O=O', Chem.MolToSmiles(got))

  def test_methane(self):
    bond_topology = str_to_bond_topology("""
atom: ATOM_C
atom: ATOM_H
atom: ATOM_H
atom: ATOM_H
atom: ATOM_H
bond {
  atom_b: 1
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 2
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 3
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 4
  bond_type: BOND_SINGLE
}
""")
    got = smu_utils_lib.bond_topology_to_rdkit_molecule(bond_topology)
    self.assertEqual('[H]C([H])([H])[H]', Chem.MolToSmiles(got))

  # This molecule is an N+ central atom, bonded to C (triply), O-, and F
  def test_charged_molecule(self):
    bond_topology = str_to_bond_topology("""
atom: ATOM_C
atom: ATOM_NPOS
atom: ATOM_ONEG
atom: ATOM_F
bond {
  atom_b: 1
  bond_type: BOND_TRIPLE
}
bond {
  atom_a: 1
  atom_b: 2
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 1
  atom_b: 3
  bond_type: BOND_SINGLE
}
""")
    got = smu_utils_lib.bond_topology_to_rdkit_molecule(bond_topology)
    self.assertEqual('C#[N+]([O-])F', Chem.MolToSmiles(got))


class IterateBondTopologiesTest(parameterized.TestCase):
  # This test covers molecules
  # * old: without source field and with is_starting_topology
  # * new: with source field

  def make_old_fake_molecule(self, mol_id, num_bts):
    molecule = dataset_pb2.Molecule(mol_id=mol_id)
    molecule.prop.calc.status = 1
    for bt_id in range(num_bts):
      molecule.bond_topo.add(topo_id=100 + bt_id)
    return molecule

  def make_new_fake_molecule(self, mol_id, sources):
    molecule = dataset_pb2.Molecule(mol_id=mol_id)
    molecule.prop.calc.status = 1
    for bt_id, source in enumerate(sources):
      molecule.bond_topo.add(topo_id=100 + bt_id, info=source)
    return molecule

  def test_old_all(self):
    got = list(
        smu_utils_lib.iterate_bond_topologies(
            self.make_old_fake_molecule(123, 3),
            smu_utils_lib.WhichTopologies.ALL))
    self.assertEqual([(0, 100), (1, 101), (2, 102)],
                     [(bt_idx, bt.topo_id) for bt_idx, bt in got])

  def test_new_all(self):
    molecule = self.make_new_fake_molecule(123, [
        dataset_pb2.BondTopology.SOURCE_DDT,
        dataset_pb2.BondTopology.SOURCE_MLCR,
        dataset_pb2.BondTopology.SOURCE_CSD
    ])
    got = list(
        smu_utils_lib.iterate_bond_topologies(
            molecule, smu_utils_lib.WhichTopologies.ALL))
    self.assertEqual([(0, 100), (1, 101), (2, 102)],
                     [(bt_idx, bt.topo_id) for bt_idx, bt in got])

  @parameterized.parameters([0, 1, 2])
  def test_old_starting(self, starting_idx):
    molecule = self.make_old_fake_molecule(123, 3)
    molecule.bond_topo[starting_idx].is_starting_topology = True
    got = list(
        smu_utils_lib.iterate_bond_topologies(
            molecule, smu_utils_lib.WhichTopologies.STARTING))
    self.assertLen(got, 1)
    self.assertEqual(starting_idx, got[0][0])
    self.assertEqual(100 + starting_idx, got[0][1].topo_id)

  @parameterized.parameters([0, 1, 2])
  def test_new_starting(self, starting_idx):
    molecule = self.make_new_fake_molecule(123, [
        dataset_pb2.BondTopology.SOURCE_DDT,
        dataset_pb2.BondTopology.SOURCE_DDT, dataset_pb2.BondTopology.SOURCE_DDT
    ])

    molecule.bond_topo[starting_idx].info |= (
        dataset_pb2.BondTopology.SOURCE_STARTING)
    got = list(
        smu_utils_lib.iterate_bond_topologies(
            molecule, smu_utils_lib.WhichTopologies.STARTING))
    self.assertLen(got, 1)
    self.assertEqual(starting_idx, got[0][0])
    self.assertEqual(100 + starting_idx, got[0][1].topo_id)

  def test_old_no_starting(self):
    molecule = self.make_old_fake_molecule(123, 3)
    got = list(
        smu_utils_lib.iterate_bond_topologies(
            molecule, smu_utils_lib.WhichTopologies.STARTING))
    self.assertEmpty(got)

  def test_new_no_starting(self):
    molecule = self.make_new_fake_molecule(123, [
        dataset_pb2.BondTopology.SOURCE_DDT,
        dataset_pb2.BondTopology.SOURCE_MLCR
    ])
    got = list(
        smu_utils_lib.iterate_bond_topologies(
            molecule, smu_utils_lib.WhichTopologies.STARTING))
    self.assertEmpty(got)

  def test_old_smu(self):
    molecule = self.make_old_fake_molecule(123, 2)
    got = [
        i for i, _ in smu_utils_lib.iterate_bond_topologies(
            molecule, smu_utils_lib.WhichTopologies.DDT)
    ]
    self.assertEqual(got, [0, 1])

  @parameterized.parameters([
      (smu_utils_lib.WhichTopologies.DDT, [0, 1]),
      (smu_utils_lib.WhichTopologies.CSD, [2, 3]),
      (smu_utils_lib.WhichTopologies.MLCR, [4, 5]),
  ])
  def test_subsets(self, which, expected):
    molecule = self.make_new_fake_molecule(123, [
        dataset_pb2.BondTopology.SOURCE_DDT,
        dataset_pb2.BondTopology.SOURCE_DDT,
        dataset_pb2.BondTopology.SOURCE_CSD,
        dataset_pb2.BondTopology.SOURCE_CSD,
        dataset_pb2.BondTopology.SOURCE_MLCR,
        dataset_pb2.BondTopology.SOURCE_MLCR
    ])
    got = [i for i, _ in smu_utils_lib.iterate_bond_topologies(molecule, which)]
    self.assertEqual(got, expected)

  def test_stage1(self):
    molecule = self.make_old_fake_molecule(123, 1)
    molecule.prop.calc.status = 600
    got = list(
        smu_utils_lib.iterate_bond_topologies(
            molecule, smu_utils_lib.WhichTopologies.STARTING))
    self.assertLen(got, 1)
    self.assertEqual(0, got[0][0])
    self.assertEqual(100, got[0][1].topo_id)

  def test_duplicated(self):
    molecule = self.make_old_fake_molecule(123, 1)
    molecule.prop.calc.status = -1
    molecule.duplicate_of = 456
    got = list(
        smu_utils_lib.iterate_bond_topologies(
            molecule, smu_utils_lib.WhichTopologies.STARTING))
    self.assertLen(got, 1)
    self.assertEqual(0, got[0][0])
    self.assertEqual(100, got[0][1].topo_id)


class MoleculeToRDKitMoleculeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._molecule = get_stage2_molecule()

    # We'll make a new initial_geometry which is just the current one with all
    # coordinates multiplied by 1000
    self._molecule.ini_geo.append(self._molecule.ini_geo[0])
    new_geom = self._molecule.ini_geo[1]
    for atom_pos in new_geom.atompos:
      atom_pos.x = atom_pos.x * 1000
      atom_pos.y = atom_pos.y * 1000
      atom_pos.z = atom_pos.z * 1000

    # For the extra bond_topology, we'll just copy the existing one and change
    # the id. Through the dumb luck of the molecule we picked there's not a
    # simple way to make this a new bond topology and still have it look valid
    # to RDKit
    self._molecule.bond_topo.append(self._molecule.bond_topo[0])
    self._molecule.bond_topo[1].topo_id = 99999
    self._molecule.bond_topo[1].info = dataset_pb2.BondTopology.SOURCE_DDT

  def test_all_outputs(self):
    mols = list(smu_utils_lib.molecule_to_rdkit_molecules(self._molecule))
    self.assertLen(mols, 6)  # 2 bond topologies * (1 opt geom + 2 init_geom)
    self.assertEqual([m.GetProp('_Name') for m in mols], [
        'SMU 618451001, RDKIT COC(=CF)OC, bt 618451(1/2), geom init(1/2)',
        'SMU 618451001, RDKIT COC(=CF)OC, bt 618451(1/2), geom init(2/2)',
        'SMU 618451001, RDKIT COC(=CF)OC, bt 618451(1/2), geom opt',
        'SMU 618451001, RDKIT COC(=CF)OC, bt 99999(2/2), geom init(1/2)',
        'SMU 618451001, RDKIT COC(=CF)OC, bt 99999(2/2), geom init(2/2)',
        'SMU 618451001, RDKIT COC(=CF)OC, bt 99999(2/2), geom opt'
    ])
    self.assertEqual(
        '[H]C(F)=C(OC([H])([H])[H])OC([H])([H])[H]',
        Chem.MolToSmiles(mols[0], kekuleSmiles=True, isomericSmiles=False))
    self.assertEqual(
        '[H]C(F)=C(OC([H])([H])[H])OC([H])([H])[H]',
        Chem.MolToSmiles(mols[4], kekuleSmiles=True, isomericSmiles=False))

  def test_initial_only(self):
    mols = list(
        smu_utils_lib.molecule_to_rdkit_molecules(
            self._molecule,
            include_initial_geometries=True,
            include_optimized_geometry=False,
            which_topologies=smu_utils_lib.WhichTopologies.STARTING))
    self.assertLen(mols, 2)
    self.assertEqual([m.GetProp('_Name') for m in mols], [
        'SMU 618451001, RDKIT COC(=CF)OC, bt 618451(1/2), geom init(1/2)',
        'SMU 618451001, RDKIT COC(=CF)OC, bt 618451(1/2), geom init(2/2)',
    ])
    # This is just one random atom I picked from the .dat file and converted to
    # angstroms instead of bohr.
    self.assertEqual('C', mols[0].GetAtomWithIdx(1).GetSymbol())
    np.testing.assert_allclose([0.6643, -3.470301, 3.4766],
                               list(mols[0].GetConformer().GetAtomPosition(1)),
                               atol=1e-6)

    self.assertEqual('C', mols[1].GetAtomWithIdx(1).GetSymbol())
    np.testing.assert_allclose([664.299998, -3470.300473, 3476.600215],
                               list(mols[1].GetConformer().GetAtomPosition(1)),
                               atol=1e-6)

  def test_optimized_only(self):
    mols = list(
        smu_utils_lib.molecule_to_rdkit_molecules(
            self._molecule,
            include_initial_geometries=False,
            include_optimized_geometry=True,
            which_topologies=smu_utils_lib.WhichTopologies.STARTING))
    self.assertLen(mols, 1)
    self.assertEqual(
        mols[0].GetProp('_Name'),
        'SMU 618451001, RDKIT COC(=CF)OC, bt 618451(1/2), geom opt',
    )
    self.assertEqual(
        '[H]C(F)=C(OC([H])([H])[H])OC([H])([H])[H]',
        Chem.MolToSmiles(mols[0], kekuleSmiles=True, isomericSmiles=False))
    # This is just two random atoms I picked from the .dat file and converted to
    # angstroms instead of bohr.
    self.assertEqual('C', mols[0].GetAtomWithIdx(1).GetSymbol())
    np.testing.assert_allclose([0.540254, -3.465543, 3.456982],
                               list(mols[0].GetConformer().GetAtomPosition(1)),
                               atol=1e-6)
    self.assertEqual('H', mols[0].GetAtomWithIdx(13).GetSymbol())
    np.testing.assert_allclose([2.135153, -1.817366, 0.226376],
                               list(mols[0].GetConformer().GetAtomPosition(13)),
                               atol=1e-6)


# Note that this class tests smiles to rdkit_molecule and rdkit_molecule_to
# bond_topology
class SmilesToBondTopologyTest(parameterized.TestCase):

  @parameterized.parameters([
      ['C', dataset_pb2.BondTopology.ATOM_C],
      ['N', dataset_pb2.BondTopology.ATOM_N],
      ['[N+]', dataset_pb2.BondTopology.ATOM_NPOS],
      ['O', dataset_pb2.BondTopology.ATOM_O],
      ['[O-]', dataset_pb2.BondTopology.ATOM_ONEG],
      ['F', dataset_pb2.BondTopology.ATOM_F],
  ])
  def test_atoms(self, smiles, expected):
    # mol = Chem.MolFromSmiles(smiles, sanitize=False)
    bt = smu_utils_lib.rdkit_molecule_to_bond_topology(
        smu_utils_lib.smiles_to_rdkit_molecule(smiles))
    got = None
    for atom in bt.atom:
      if atom != dataset_pb2.BondTopology.ATOM_H:
        got = atom
    self.assertEqual(got, expected)

  @parameterized.parameters([['CC', dataset_pb2.BondTopology.BOND_SINGLE],
                             ['C=C', dataset_pb2.BondTopology.BOND_DOUBLE],
                             ['C#C', dataset_pb2.BondTopology.BOND_TRIPLE]])
  def test_bonds(self, smiles, expected):
    bt = smu_utils_lib.rdkit_molecule_to_bond_topology(
        smu_utils_lib.smiles_to_rdkit_molecule(smiles))
    got = None
    for bond in bt.bond:
      if (bt.atom[bond.atom_a] == dataset_pb2.BondTopology.ATOM_C and
          bt.atom[bond.atom_b] == dataset_pb2.BondTopology.ATOM_C):
        got = bond.bond_type
    self.assertEqual(got, expected)


class SmilesCompareTest(absltest.TestCase):

  def test_string_format(self):
    # for some simplicity later on, we use shorter names
    self.assertEqual('MISSING', str(smu_utils_lib.SmilesCompareResult.MISSING))
    self.assertEqual('MISMATCH',
                     str(smu_utils_lib.SmilesCompareResult.MISMATCH))
    self.assertEqual('MATCH', str(smu_utils_lib.SmilesCompareResult.MATCH))

  def test_missing(self):
    bond_topology = str_to_bond_topology("""
atom: ATOM_O
atom: ATOM_O
bond {
  atom_b: 1
  bond_type: BOND_DOUBLE
}
""")
    result, with_h, without_h = smu_utils_lib.bond_topology_smiles_comparison(
        bond_topology)
    self.assertEqual(smu_utils_lib.SmilesCompareResult.MISSING, result)
    self.assertEqual('O=O', with_h)
    self.assertEqual('O=O', without_h)

    # Also directly test compute_smiles_for_bond_topology
    self.assertEqual(
        'O=O',
        smu_utils_lib.compute_smiles_for_bond_topology(
            bond_topology, include_hs=True))

  def test_mismatch(self):
    bond_topology = str_to_bond_topology("""
atom: ATOM_O
atom: ATOM_O
bond {
  atom_b: 1
  bond_type: BOND_DOUBLE
}
smiles: "BlahBlahBlah"
""")
    result, with_h, without_h = smu_utils_lib.bond_topology_smiles_comparison(
        bond_topology)
    self.assertEqual(smu_utils_lib.SmilesCompareResult.MISMATCH, result)
    self.assertEqual('O=O', with_h)
    self.assertEqual('O=O', without_h)

  def test_matched_and_h_stripping(self):
    bond_topology = str_to_bond_topology("""
atom: ATOM_O
atom: ATOM_H
atom: ATOM_H
bond {
  atom_b: 1
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 2
  bond_type: BOND_SINGLE
}
smiles: "O"
""")
    result, with_h, without_h = smu_utils_lib.bond_topology_smiles_comparison(
        bond_topology)
    self.assertEqual(smu_utils_lib.SmilesCompareResult.MATCH, result)
    self.assertEqual('[H]O[H]', with_h)
    self.assertEqual('O', without_h)

    # Also directly test compute_smiles_for_bond_topology
    self.assertEqual(
        '[H]O[H]',
        smu_utils_lib.compute_smiles_for_bond_topology(
            bond_topology, include_hs=True))
    self.assertEqual(
        'O',
        smu_utils_lib.compute_smiles_for_bond_topology(
            bond_topology, include_hs=False))

  def test_compute_smiles_from_rdkit_molecule_no_hs(self):
    mol = Chem.MolFromSmiles('FOC', sanitize=False)
    self.assertEqual(
        smu_utils_lib.compute_smiles_for_rdkit_molecule(mol, include_hs=False),
        'COF')
    # This is expected. Even with include_hs=True, if there were no Hs in the
    # molecule, they will not be in the smiles.
    self.assertEqual(
        smu_utils_lib.compute_smiles_for_rdkit_molecule(mol, include_hs=True),
        'COF')

  def test_compute_smiles_from_rdkit_molecule_with_hs(self):
    mol = Chem.MolFromSmiles('FOC', sanitize=False)
    Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS)
    mol = Chem.AddHs(mol)
    self.assertEqual(
        smu_utils_lib.compute_smiles_for_rdkit_molecule(mol, include_hs=False),
        'COF')
    self.assertEqual(
        smu_utils_lib.compute_smiles_for_rdkit_molecule(mol, include_hs=True),
        '[H]C([H])([H])OF')

  def test_compute_smiles_from_rdkit_molecule_special_case(self):
    mol = Chem.MolFromSmiles('C12=C3C4=C1C4=C23', sanitize=False)
    # Double check that this really is the special case -- we get back the
    # SMILES we put in even though it's not the one we want.
    self.assertEqual('C12=C3C4=C1C4=C23',
                     Chem.MolToSmiles(mol, kekuleSmiles=True))
    self.assertEqual(
        smu_utils_lib.compute_smiles_for_rdkit_molecule(mol, include_hs=False),
        'C12=C3C1=C1C2=C31')

  def test_compute_smiles_from_rdkit_molecule_labeled_with_h(self):
    mol = Chem.MolFromSmiles(
        '[O-][N+]([H])([H])N([H])OC([H])([H])F', sanitize=False)
    self.assertIsNotNone(mol)
    self.assertEqual(
        '[O-][N+:1]([H:2])([H:3])[N:4]([H:5])[O:6][C:7]([H:8])([H:9])[F:10]',
        smu_utils_lib.compute_smiles_for_rdkit_molecule(
            mol, include_hs=True, labeled_atoms=True))

  def test_compute_smiles_from_rdkit_molecule_labeled_no_h(self):
    mol = Chem.MolFromSmiles(
        '[O-][N+]([H])([H])N([H])OC([H])([H])F', sanitize=False)
    self.assertIsNotNone(mol)
    self.assertEqual(
        '[O-][NH2+:1][NH:2][O:3][CH2:4][F:5]',
        smu_utils_lib.compute_smiles_for_rdkit_molecule(
            mol, include_hs=False, labeled_atoms=True))


class MergeMoleculesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # We are relying on the fact that the first molecule in both x07_sample.dat
    # and x07_stage1.dat are the same.
    self.stage1_molecule = get_stage1_molecule()
    self.stage2_molecule = get_stage2_molecule()

    self.duplicate_molecule = dataset_pb2.Molecule()
    self.duplicate_molecule.mol_id = self.stage1_molecule.mol_id
    # A real duplicate molecule wouldn't have both of these fields filled in,
    # but it's fine for the test to make sure everything is copied.
    self.duplicate_molecule.duplicate_of = 123
    self.duplicate_molecule.duplicate_found.extend([111, 222])

  def test_two_stage2(self):
    with self.assertRaises(ValueError):
      smu_utils_lib.merge_molecule(self.stage2_molecule, self.stage2_molecule)

  def test_two_stage1(self):
    with self.assertRaises(ValueError):
      smu_utils_lib.merge_molecule(self.stage1_molecule, self.stage1_molecule)

  def test_two_duplicates(self):
    duplicate_molecule2 = copy.deepcopy(self.duplicate_molecule)
    duplicate_molecule2.duplicate_found[:] = [333, 444]

    got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.duplicate_molecule, duplicate_molecule2)
    self.assertIsNone(got_conflict)
    self.assertEqual(123, got_mol.duplicate_of)
    self.assertCountEqual([111, 222, 333, 444], got_mol.duplicate_found)

  def test_stage2_stage1(self):
    # Add a duplicate to stage1 to make sure it is copied
    self.stage1_molecule.duplicate_found.append(999)
    got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage2_molecule, self.stage1_molecule)
    self.assertIsNone(got_conflict)
    self.assertEqual(got_mol.duplicate_found, [999])
    # Just check a random field that is in stage2 but not stage1
    self.assertNotEmpty(got_mol.prop.vib_mode)

  def test_stage2_stage1_conflict_energy(self):
    self.stage2_molecule.ini_geo[0].energy.val = -1.23
    got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage2_molecule, self.stage1_molecule)
    self.assertEqual(got_conflict, [
        618451001, 1, 1, 1, 1, -406.51179, 0.052254, -406.522079, 2.5e-05, True,
        True, -1.23, 0.052254, -406.522079, 2.5e-05, True, True
    ])
    # Just check a random field that is in stage2 but not stage1
    self.assertNotEmpty(got_mol.prop.vib_mode)
    # This stage1 value should be returned
    self.assertEqual(got_mol.ini_geo[0].energy.val, -406.51179)

  def test_stage2_stage1_conflict_missing_iniital_geometry_field(self):
    del self.stage2_molecule.ini_geo[:]
    got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage2_molecule, self.stage1_molecule)
    self.assertEqual(got_conflict, [
        618451001, 1, 1, 1, 1, -406.51179, 0.052254, -406.522079, 2.5e-05, True,
        True, 0.0, 0.0, -406.522079, 2.5e-05, False, True
    ])
    # Just check a random field that is in stage2 but not stage1
    self.assertNotEmpty(got_mol.prop.vib_mode)

  def test_stage2_stage1_conflict_missing_iniital_geometry(self):
    del self.stage2_molecule.ini_geo[0].atompos[:]
    got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage2_molecule, self.stage1_molecule)
    self.assertEqual(got_conflict, [
        618451001, 1, 1, 1, 1, -406.51179, 0.052254, -406.522079, 2.5e-05, True,
        True, -406.51179, 0.052254, -406.522079, 2.5e-05, False, True
    ])
    # Just check a random field that is in stage2 but not stage1
    self.assertNotEmpty(got_mol.prop.vib_mode)

  def test_stage2_stage1_conflict_missing_optimized_geometry_field(self):
    self.stage2_molecule.ClearField('opt_geo')
    got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage2_molecule, self.stage1_molecule)
    self.assertEqual(got_conflict, [
        618451001, 1, 1, 1, 1, -406.51179, 0.052254, -406.522079, 2.5e-05, True,
        True, -406.51179, 0.052254, 0.0, 0.0, True, False
    ])
    # Just check a random field that is in stage2 but not stage1
    self.assertNotEmpty(got_mol.prop.vib_mode)

  def test_stage2_stage1_conflict_missing_optimized_geometry(self):
    del self.stage2_molecule.opt_geo.atompos[:]
    got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage2_molecule, self.stage1_molecule)
    self.assertEqual(got_conflict, [
        618451001, 1, 1, 1, 1, -406.51179, 0.052254, -406.522079, 2.5e-05, True,
        True, -406.51179, 0.052254, -406.522079, 2.5e-05, True, False
    ])
    # Just check a random field that is in stage2 but not stage1
    self.assertNotEmpty(got_mol.prop.vib_mode)

  def test_stage2_stage1_no_conflict_minus1(self):
    # If stage2 contains a -1, we keep that (stricter error checking later on)
    self.stage2_molecule.ini_geo[0].energy.val = -1.0
    got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage2_molecule, self.stage1_molecule)
    self.assertIsNone(got_conflict)
    # This stage1 value should be returned
    self.assertEqual(got_mol.ini_geo[0].energy.val, -406.51179)

  def test_stage2_stage1_no_conflict_approx_equal(self):
    self.stage2_molecule.ini_geo[0].energy.val += 1e-7
    got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage2_molecule, self.stage1_molecule)
    self.assertIsNone(got_conflict)
    # Just check a random field from stage2
    self.assertNotEmpty(got_mol.prop.vib_mode)

  def test_status_800(self):
    self.stage2_molecule.prop.calc.status = 800
    # Set a value so that we make sure we use the stage1 data
    self.stage2_molecule.ini_geo[0].energy.val += 12345
    expected_init_energy = (self.stage1_molecule.ini_geo[0].energy.val)
    got_mol, _ = smu_utils_lib.merge_molecule(self.stage2_molecule,
                                              self.stage1_molecule)
    self.assertEqual(got_mol.prop.calc.status, 580)
    self.assertEqual(got_mol.ini_geo[0].energy.val, expected_init_energy)
    self.assertEqual(got_mol.prop.calc.warn_vib_imag, 0)

  def test_status_700(self):
    self.stage2_molecule.prop.calc.status = 700
    # Set a value so that we make sure we use the stage1 data
    self.stage2_molecule.ini_geo[0].energy.val += 12345
    expected_init_energy = (self.stage1_molecule.ini_geo[0].energy.val)
    got_mol, _ = smu_utils_lib.merge_molecule(self.stage2_molecule,
                                              self.stage1_molecule)
    self.assertEqual(got_mol.prop.calc.status, 570)
    self.assertEqual(got_mol.ini_geo[0].energy.val, expected_init_energy)
    self.assertEqual(got_mol.prop.calc.warn_vib_imag, 0)

  def test_status_800_warn_vib_2(self):
    self.stage2_molecule.prop.calc.status = 800
    # We set two values because 1 is any negative and 2 is for a large negative
    self.stage1_molecule.prop.vib_freq.val[3] = -123
    self.stage1_molecule.prop.vib_freq.val[4] = -1
    got_mol, _ = smu_utils_lib.merge_molecule(self.stage2_molecule,
                                              self.stage1_molecule)
    self.assertEqual(got_mol.prop.calc.status, 580)
    self.assertEqual(got_mol.prop.calc.warn_vib_imag, 2)

  def test_status_800_warn_vib_1(self):
    self.stage2_molecule.prop.calc.status = 800
    self.stage1_molecule.prop.vib_freq.val[4] = -1
    got_mol, _ = smu_utils_lib.merge_molecule(self.stage2_molecule,
                                              self.stage1_molecule)
    self.assertEqual(got_mol.prop.calc.status, 580)
    self.assertEqual(got_mol.prop.calc.warn_vib_imag, 1)

  def test_error_frequencies_101(self):
    self.stage1_molecule.prop.calc.error_frequencies = 101
    unused_got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage1_molecule, self.stage2_molecule)
    self.assertIsNotNone(got_conflict)

  def test_error_frequencies_101_for_allowed_mol(self):
    self.stage1_molecule.mol_id = 795795001
    self.stage2_molecule.mol_id = 795795001
    self.stage1_molecule.prop.calc.error_frequencies = 101
    unused_got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage1_molecule, self.stage2_molecule)
    self.assertIsNone(got_conflict)

  def test_disallowed_error_flags(self):
    # each of these is allowed separately, but not together
    self.stage1_molecule.prop.calc.error_nstat1 = 3
    self.stage1_molecule.prop.calc.error_nstatc = 3
    self.stage1_molecule.prop.calc.error_frequencies = 3
    unused_got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage1_molecule, self.stage2_molecule)
    self.assertIsNotNone(got_conflict)

  def test_stage2_duplicate(self):
    got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage2_molecule, self.duplicate_molecule)
    self.assertIsNone(got_conflict)
    self.assertEqual(got_mol.duplicate_found, [111, 222])
    self.assertEqual(got_mol.duplicate_of, 123)
    # Just check a random field from stage2
    self.assertNotEmpty(got_mol.prop.vib_mode)

  def test_stage1_duplicate(self):
    got_mol, got_conflict = smu_utils_lib.merge_molecule(
        self.stage1_molecule, self.duplicate_molecule)
    self.assertIsNone(got_conflict)
    self.assertEqual(got_mol.duplicate_found, [111, 222])
    self.assertEqual(got_mol.duplicate_of, 123)
    # Just check a random field from stage1
    self.assertTrue(got_mol.prop.HasField('vib_freq'))

  def test_multiple_initial_geometries(self):
    bad_molecule = copy.deepcopy(self.stage1_molecule)
    bad_molecule.ini_geo.append(bad_molecule.ini_geo[0])
    with self.assertRaises(ValueError):
      smu_utils_lib.merge_molecule(bad_molecule, self.stage2_molecule)
    with self.assertRaises(ValueError):
      smu_utils_lib.merge_molecule(self.stage2_molecule, bad_molecule)

  def test_multiple_bond_topologies(self):
    bad_molecule = copy.deepcopy(self.stage1_molecule)
    bad_molecule.bond_topo.append(bad_molecule.bond_topo[0])
    with self.assertRaises(ValueError):
      smu_utils_lib.merge_molecule(bad_molecule, self.stage2_molecule)
    with self.assertRaises(ValueError):
      smu_utils_lib.merge_molecule(self.stage2_molecule, bad_molecule)

  def test_different_bond_topologies(self):
    self.stage1_molecule.bond_topo[0].atom[0] = (
        dataset_pb2.BondTopology.ATOM_H)
    with self.assertRaises(ValueError):
      smu_utils_lib.merge_molecule(self.stage1_molecule, self.stage2_molecule)
    with self.assertRaises(ValueError):
      smu_utils_lib.merge_molecule(self.stage2_molecule, self.stage1_molecule)


class MoleculeErrorTest(absltest.TestCase):

  def test_stage1_no_error(self):
    molecule = get_stage1_molecule()
    self.assertEqual(0,
                     smu_utils_lib.molecule_calculation_error_level(molecule))

  def test_stage1_error(self):
    molecule = get_stage1_molecule()
    molecule.prop.calc.error_frequencies = 123
    self.assertEqual(8,
                     smu_utils_lib.molecule_calculation_error_level(molecule))

  def test_stage2_no_error(self):
    molecule = get_stage2_molecule()
    self.assertEqual(0,
                     smu_utils_lib.molecule_calculation_error_level(molecule))

  def test_stage2_error_status_8(self):
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = 256
    self.assertEqual(8,
                     smu_utils_lib.molecule_calculation_error_level(molecule))

  def test_stage2_error_status_7(self):
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = 50
    self.assertEqual(7,
                     smu_utils_lib.molecule_calculation_error_level(molecule))

  def test_stage2_error_status_6(self):
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = 4
    self.assertEqual(6,
                     smu_utils_lib.molecule_calculation_error_level(molecule))

  def test_stage2_error_level_5(self):
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = 1
    molecule.prop.calc.warn_delta_t1 = 2
    self.assertEqual(5,
                     smu_utils_lib.molecule_calculation_error_level(molecule))

  def test_stage2_error_level_4(self):
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = 1
    molecule.prop.calc.warn_vib_linear = 1
    self.assertEqual(4,
                     smu_utils_lib.molecule_calculation_error_level(molecule))

  def test_stage2_error_level_3(self):
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = 2
    self.assertEqual(3,
                     smu_utils_lib.molecule_calculation_error_level(molecule))

  def test_stage2_error_level_2(self):
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = 0
    molecule.prop.calc.warn_delta_t1 = 2
    self.assertEqual(2,
                     smu_utils_lib.molecule_calculation_error_level(molecule))

  def test_stage2_error_level_1(self):
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = 0
    molecule.prop.calc.warn_vib_linear = 1
    self.assertEqual(1,
                     smu_utils_lib.molecule_calculation_error_level(molecule))


class FilterMoleculeByAvailabilityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._molecule = dataset_pb2.Molecule()
    properties = self._molecule.prop
    # A STANDARD field
    properties.spe_comp_b5.val = 1.23
    # A COMPLETE field
    properties.vib_zpe.val = 1.23
    # An INTERNAL_ONLY field
    properties.compute_cluster_info = 'not set'

  def test_standard(self):
    smu_utils_lib.filter_molecule_by_availability(self._molecule,
                                                  [dataset_pb2.STANDARD])
    self.assertTrue(self._molecule.prop.HasField('spe_comp_b5'))
    self.assertFalse(self._molecule.prop.HasField('vib_zpe'))
    self.assertFalse(self._molecule.prop.HasField('compute_cluster_info'))

  def test_complete_and_internal_only(self):
    smu_utils_lib.filter_molecule_by_availability(
        self._molecule, [dataset_pb2.COMPLETE, dataset_pb2.INTERNAL_ONLY])
    self.assertFalse(self._molecule.prop.HasField('spe_comp_b5'))
    self.assertTrue(self._molecule.prop.HasField('vib_zpe'))
    self.assertTrue(self._molecule.prop.HasField('compute_cluster_info'))


class MoleculeToStandardTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self._molecule = get_stage2_molecule()

  def test_field_filtering(self):
    # Check that all fields start out set
    self.assertTrue(self._molecule.ini_geo[0].HasField('energy'))
    self.assertTrue(self._molecule.opt_geo.HasField('energy'))
    self.assertTrue(self._molecule.opt_geo.HasField('enuc'))
    self.assertTrue(self._molecule.prop.HasField('vib_freq'))
    self.assertTrue(self._molecule.prop.HasField('vib_zpe'))

    _ = smu_utils_lib.molecule_to_standard(self._molecule)

    self.assertTrue(self._molecule.ini_geo[0].HasField('energy'))
    self.assertTrue(self._molecule.opt_geo.HasField('energy'))
    self.assertFalse(self._molecule.opt_geo.HasField('enuc'))
    self.assertTrue(self._molecule.prop.HasField('vib_freq'))
    self.assertFalse(self._molecule.prop.HasField('vib_zpe'))

  def test_remove_error_molecule(self):
    self._molecule.prop.calc.which_database = dataset_pb2.UNSPECIFIED
    self._molecule.prop.calc.status = 256

    self.assertIsNone(smu_utils_lib.molecule_to_standard(self._molecule))

  def test_remove_duplicate(self):
    self._molecule.prop.calc.which_database = dataset_pb2.UNSPECIFIED
    self._molecule.duplicate_of = 123

    self.assertIsNone(smu_utils_lib.molecule_to_standard(self._molecule))

  def test_remove_complete(self):
    self._molecule.prop.calc.which_database = dataset_pb2.COMPLETE

    self.assertIsNone(smu_utils_lib.molecule_to_standard(self._molecule))


class CleanUpErrorCodesTest(parameterized.TestCase):

  def test_stage2(self):
    molecule = get_stage2_molecule()
    molecule.prop.calc.error_nstat1 = 123
    smu_utils_lib.clean_up_error_codes(molecule)
    self.assertEqual(molecule.prop.calc.error_nstat1, 0)

  def test_stage1_dup(self):
    molecule = get_stage1_molecule()
    molecule.duplicate_of = 123
    smu_utils_lib.clean_up_error_codes(molecule)
    self.assertEqual(molecule.prop.calc.status, -1)
    self.assertEqual(molecule.prop.calc.error_nstat1, 0)

  def test_stage1_dup_with_no_record(self):
    molecule = get_stage1_molecule()
    smu_utils_lib.clean_up_error_codes(molecule)
    self.assertEqual(molecule.prop.calc.status, 0)
    self.assertEqual(molecule.prop.calc.error_nstat1, 0)

  def test_stage1_590(self):
    molecule = get_stage1_molecule()
    molecule.prop.calc.error_nstat1 = 5
    smu_utils_lib.clean_up_error_codes(molecule)
    self.assertEqual(molecule.prop.calc.status, 590)
    self.assertEqual(molecule.prop.calc.error_nstat1, 0)

  def test_stage1_600(self):
    molecule = get_stage1_molecule()
    molecule.prop.calc.error_nstat1 = 2
    smu_utils_lib.clean_up_error_codes(molecule)
    self.assertEqual(molecule.prop.calc.status, 600)
    self.assertEqual(molecule.prop.calc.error_nstat1, 0)
    self.assertFalse(molecule.ini_geo[0].HasField('energy'))
    self.assertFalse(molecule.ini_geo[0].HasField('gnorm'))
    self.assertFalse(molecule.HasField('opt_geo'))


class CleanUpSentinelValuestest(parameterized.TestCase):

  def test_no_change(self):
    molecule = get_stage2_molecule()
    smu_utils_lib.clean_up_sentinel_values(molecule)
    self.assertTrue(molecule.ini_geo[0].HasField('energy'))
    self.assertTrue(molecule.ini_geo[0].HasField('gnorm'))
    self.assertTrue(molecule.opt_geo.HasField('energy'))
    self.assertTrue(molecule.opt_geo.HasField('gnorm'))

  @parameterized.parameters('energy', 'gnorm')
  def test_one_field(self, field):
    molecule = get_stage2_molecule()
    getattr(molecule.ini_geo[0], field).val = -1.0
    getattr(molecule.opt_geo, field).val = -1.0
    smu_utils_lib.clean_up_sentinel_values(molecule)
    self.assertFalse(molecule.ini_geo[0].HasField(field))
    self.assertFalse(molecule.opt_geo.HasField(field))


class FindZeroValuesTest(parameterized.TestCase):

  def test_no_zeroes(self):
    molecule = get_stage2_molecule()
    got = list(smu_utils_lib.find_zero_values(molecule))
    self.assertEqual(got, [])

  def test_scalar(self):
    molecule = get_stage2_molecule()
    molecule.prop.orb_elumo_b3lyp_631ppgdp.val = 0.0
    got = list(smu_utils_lib.find_zero_values(molecule))
    self.assertEqual(got, ['orb_elumo_b3lyp_631ppgdp'])

  def test_excitation(self):
    molecule = get_stage2_molecule()
    molecule.prop.exc_ene_cc2_tzvp.val[2] = 0.0
    got = list(smu_utils_lib.find_zero_values(molecule))
    self.assertEqual(got, ['exc_ene_cc2_tzvp'])

  def test_atomic(self):
    molecule = get_stage2_molecule()
    molecule.prop.chg_esp_hf_631gd.val[3] = 0.0
    got = list(smu_utils_lib.find_zero_values(molecule))
    self.assertEqual(got, ['chg_esp_hf_631gd'])


class DetermineFateTest(parameterized.TestCase):

  def test_duplicate_same_topology(self):
    molecule = get_stage1_molecule()
    # bond topology is mol_id // 1000
    molecule.duplicate_of = molecule.mol_id + 1
    smu_utils_lib.clean_up_error_codes(molecule)
    self.assertEqual(dataset_pb2.Properties.FATE_DUPLICATE_SAME_TOPOLOGY,
                     smu_utils_lib.determine_fate(molecule))

  def test_duplicate_different_topology(self):
    molecule = get_stage1_molecule()
    # bond topology is mol_id // 1000
    molecule.duplicate_of = molecule.mol_id + 1000
    smu_utils_lib.clean_up_error_codes(molecule)
    self.assertEqual(dataset_pb2.Properties.FATE_DUPLICATE_DIFFERENT_TOPOLOGY,
                     smu_utils_lib.determine_fate(molecule))

  @parameterized.parameters(
      (2, dataset_pb2.Properties.FATE_FAILURE_GEO_OPT),
      (5, dataset_pb2.Properties.FATE_FAILURE_TOPOLOGY_CHECK),
      (6, dataset_pb2.Properties.FATE_FAILURE_NO_RESULTS))
  def test_geometry_failures(self, nstat1, expected_fate):
    molecule = get_stage1_molecule()
    molecule.prop.calc.error_nstat1 = nstat1
    smu_utils_lib.clean_up_error_codes(molecule)
    self.assertEqual(expected_fate, smu_utils_lib.determine_fate(molecule))

  def test_no_result(self):
    molecule = get_stage1_molecule()
    smu_utils_lib.clean_up_error_codes(molecule)
    self.assertEqual(dataset_pb2.Properties.FATE_FAILURE_NO_RESULTS,
                     smu_utils_lib.determine_fate(molecule))

  @parameterized.parameters(570, 580)
  def test_discarded_other(self, status):
    molecule = get_stage1_molecule()
    molecule.prop.calc.status = status
    smu_utils_lib.clean_up_error_codes(molecule)
    self.assertEqual(dataset_pb2.Properties.FATE_FAILURE_STAGE2,
                     smu_utils_lib.determine_fate(molecule))

  @parameterized.parameters((256, dataset_pb2.Properties.FATE_ERROR_SERIOUS),
                            (50, dataset_pb2.Properties.FATE_ERROR_MAJOR),
                            (4, dataset_pb2.Properties.FATE_ERROR_MODERATE))
  def test_calculation_errors(self, status, expected):
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = status
    self.assertEqual(expected, smu_utils_lib.determine_fate(molecule))

  @parameterized.parameters(
      (0, dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_SERIOUS),
      (1, dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_SERIOUS),
      (2, dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_SERIOUS),
      (3, dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_SERIOUS),
  )
  def test_calculation_warnings_serious(self, status, expected):
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = status
    molecule.prop.calc.warn_delta_t1 = 1234
    self.assertEqual(expected, smu_utils_lib.determine_fate(molecule))

  @parameterized.parameters(
      (0, dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_MEDIUM_VIB),
      (1, dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_MEDIUM_VIB),
      (2, dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_MEDIUM_VIB),
      (3, dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_MEDIUM_VIB),
  )
  def test_calculation_warnings_vibrational(self, status, expected):
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = status
    molecule.prop.calc.warn_vib_linear = 1234
    self.assertEqual(expected, smu_utils_lib.determine_fate(molecule))

  @parameterized.parameters(
      (0, dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW),
      (1, dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_LOW),
      (2, dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_LOW),
      (3, dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_LOW),
  )
  def test_success(self, status, expected):
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = status
    self.assertEqual(expected, smu_utils_lib.determine_fate(molecule))

  def test_multiple_warnings(self):
    # Checks that when we have both serious and medium warnings, serious wins
    molecule = get_stage2_molecule()
    molecule.prop.calc.status = 1
    molecule.prop.calc.warn_delta_t1 = 1234
    molecule.prop.calc.warn_vib_linear = 1234
    self.assertEqual(
        dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_SERIOUS,
        smu_utils_lib.determine_fate(molecule))

  def test_completed_molecule(self):
    molecule = get_stage2_molecule()
    smu_utils_lib.filter_molecule_by_availability(
        molecule, [dataset_pb2.COMPLETE, dataset_pb2.STANDARD])
    self.assertEqual(dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW,
                     smu_utils_lib.determine_fate(molecule))

  def test_completed_standard(self):
    molecule = get_stage2_molecule()
    smu_utils_lib.filter_molecule_by_availability(molecule,
                                                  [dataset_pb2.STANDARD])
    # This is a weird test. This function really just does not do anything
    # for a standard record because we've filtered out all the status/error
    # variables we woudl need to do it. But it's here to make sure nothing
    # explodes and we default to the best version.
    self.assertEqual(dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW,
                     smu_utils_lib.determine_fate(molecule))

  def test_completed_duplicate_same(self):
    molecule = get_stage2_molecule()
    smu_utils_lib.filter_molecule_by_availability(
        molecule, [dataset_pb2.COMPLETE, dataset_pb2.STANDARD])
    molecule.duplicate_of = 618451999
    molecule.prop.calc.status = -1
    self.assertEqual(dataset_pb2.Properties.FATE_DUPLICATE_SAME_TOPOLOGY,
                     smu_utils_lib.determine_fate(molecule))

  def test_completed_duplicate_different(self):
    molecule = get_stage2_molecule()
    smu_utils_lib.filter_molecule_by_availability(
        molecule, [dataset_pb2.COMPLETE, dataset_pb2.STANDARD])
    molecule.duplicate_of = 12345999
    molecule.prop.calc.status = -1
    self.assertEqual(dataset_pb2.Properties.FATE_DUPLICATE_DIFFERENT_TOPOLOGY,
                     smu_utils_lib.determine_fate(molecule))

  @parameterized.parameters(
      (570, dataset_pb2.Properties.FATE_FAILURE_STAGE2),
      (580, dataset_pb2.Properties.FATE_FAILURE_STAGE2),
      (590, dataset_pb2.Properties.FATE_FAILURE_TOPOLOGY_CHECK),
      (600, dataset_pb2.Properties.FATE_FAILURE_GEO_OPT),
  )
  def test_completed_stage1_error(self, status, expected):
    molecule = get_stage2_molecule()
    smu_utils_lib.filter_molecule_by_availability(
        molecule, [dataset_pb2.COMPLETE, dataset_pb2.STANDARD])
    molecule.prop.calc.status = status
    self.assertEqual(expected, smu_utils_lib.determine_fate(molecule))


class ToBondTopologySummaryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._molecule = get_stage2_molecule()

  def get_output_with(self, got, bt_id, field):
    out = None
    for summary in got:
      if (summary.bond_topology.topo_id == bt_id and
          getattr(summary, field) > 0):
        assert out is None
        out = summary
    return out

  def test_dup_same(self):
    self._molecule.prop.calc.fate = dataset_pb2.Properties.FATE_DUPLICATE_SAME_TOPOLOGY
    got = list(
        smu_utils_lib.molecule_to_bond_topology_summaries(self._molecule))
    self.assertLen(got, 1)
    self.assertEqual(got[0].bond_topology.topo_id,
                     self._molecule.bond_topo[0].topo_id)
    self.assertEqual(got[0].count_attempted_molecules, 1)
    self.assertEqual(got[0].count_duplicates_same_topology, 1)

  def test_dup_diff(self):
    self._molecule.prop.calc.fate = (
        dataset_pb2.Properties.FATE_DUPLICATE_DIFFERENT_TOPOLOGY)
    got = list(
        smu_utils_lib.molecule_to_bond_topology_summaries(self._molecule))
    self.assertLen(got, 1)
    self.assertEqual(got[0].count_attempted_molecules, 1)
    self.assertEqual(got[0].count_duplicates_different_topology, 1)

  def test_geometry_failed(self):
    self._molecule.prop.calc.fate = (dataset_pb2.Properties.FATE_FAILURE_STAGE2)
    got = list(
        smu_utils_lib.molecule_to_bond_topology_summaries(self._molecule))
    self.assertLen(got, 1)
    self.assertEqual(got[0].count_attempted_molecules, 1)
    self.assertEqual(got[0].count_failed_geometry_optimization, 1)

  def test_missing_calculation(self):
    self._molecule.prop.calc.fate = dataset_pb2.Properties.FATE_FAILURE_NO_RESULTS
    got = list(
        smu_utils_lib.molecule_to_bond_topology_summaries(self._molecule))
    self.assertLen(got, 1)
    self.assertEqual(got[0].count_attempted_molecules, 1)
    self.assertEqual(got[0].count_kept_geometry, 1)
    self.assertEqual(got[0].count_missing_calculation, 1)

  def _swap_bond_topologies(self):
    """Swaps the order of the first two topologies."""
    bt0 = self._molecule.bond_topo[0]
    bt1 = self._molecule.bond_topo[1]
    del self._molecule.bond_topo[:]
    self._molecule.bond_topo.extend([bt1, bt0])

  @parameterized.parameters(False, True)
  def test_calculation_with_error(self, swap_order):
    self._molecule.prop.calc.fate = (dataset_pb2.Properties.FATE_ERROR_SERIOUS)
    self._molecule.bond_topo.append(self._molecule.bond_topo[0])
    self._molecule.bond_topo[-1].topo_id = 123
    self._molecule.bond_topo[-1].info = (
        dataset_pb2.BondTopology.SOURCE_DDT
        | dataset_pb2.BondTopology.SOURCE_MLCR)
    self._molecule.bond_topo[0].info = (
        dataset_pb2.BondTopology.SOURCE_DDT
        | dataset_pb2.BondTopology.SOURCE_CSD
        | dataset_pb2.BondTopology.SOURCE_STARTING)

    if swap_order:
      self._swap_bond_topologies()

    got = list(
        smu_utils_lib.molecule_to_bond_topology_summaries(self._molecule))

    self.assertLen(got, 4)

    one_out = self.get_output_with(got, 123,
                                   'count_detected_match_itc_with_error')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_with_error, 0)

    one_out = self.get_output_with(got, 618451, 'count_attempted_molecules')
    self.assertEqual(one_out.count_kept_geometry, 1)
    self.assertEqual(one_out.count_calculation_with_error, 1)
    self.assertEqual(one_out.count_detected_match_itc_with_error, 0)

    one_out = self.get_output_with(got, 123,
                                   'count_detected_match_mlcr_with_error')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_with_error, 0)

    one_out = self.get_output_with(got, 618451,
                                   'count_detected_match_csd_with_error')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_with_error, 0)

  @parameterized.parameters(False, True)
  def test_calculation_with_warning(self, swap_order):
    self._molecule.prop.calc.fate = (
        dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_SERIOUS)
    self._molecule.bond_topo.append(self._molecule.bond_topo[0])
    self._molecule.bond_topo[-1].topo_id = 123
    self._molecule.bond_topo[-1].info = (
        dataset_pb2.BondTopology.SOURCE_DDT
        | dataset_pb2.BondTopology.SOURCE_MLCR)
    self._molecule.bond_topo[0].info = (
        dataset_pb2.BondTopology.SOURCE_DDT
        | dataset_pb2.BondTopology.SOURCE_CSD
        | dataset_pb2.BondTopology.SOURCE_STARTING)

    if swap_order:
      self._swap_bond_topologies()

    got = list(
        smu_utils_lib.molecule_to_bond_topology_summaries(self._molecule))

    self.assertLen(got, 4)

    one_out = self.get_output_with(got, 123,
                                   'count_detected_match_itc_with_warning')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_with_warning, 0)

    one_out = self.get_output_with(got, 618451, 'count_attempted_molecules')
    self.assertEqual(one_out.count_kept_geometry, 1)
    self.assertEqual(one_out.count_calculation_with_warning, 1)
    self.assertEqual(one_out.count_calculation_with_error, 0)
    self.assertEqual(one_out.count_detected_match_itc_with_warning, 0)
    self.assertEqual(one_out.count_detected_match_itc_with_error, 0)

    one_out = self.get_output_with(got, 123,
                                   'count_detected_match_mlcr_with_warning')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_with_warning, 0)
    self.assertEqual(one_out.count_calculation_with_error, 0)

    one_out = self.get_output_with(got, 618451,
                                   'count_detected_match_csd_with_warning')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_with_warning, 0)
    self.assertEqual(one_out.count_calculation_with_error, 0)

  @parameterized.parameters(False, True)
  def test_calculation_success_itc(self, swap_order):
    self._molecule.prop.calc.fate = dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW
    self._molecule.bond_topo.append(self._molecule.bond_topo[0])
    self._molecule.bond_topo[-1].topo_id = 123
    self._molecule.bond_topo[-1].info = (
        dataset_pb2.BondTopology.SOURCE_DDT
        | dataset_pb2.BondTopology.SOURCE_MLCR)
    self._molecule.bond_topo[0].info = (
        dataset_pb2.BondTopology.SOURCE_DDT
        | dataset_pb2.BondTopology.SOURCE_CSD
        | dataset_pb2.BondTopology.SOURCE_STARTING)
    if swap_order:
      self._swap_bond_topologies()

    got = list(
        smu_utils_lib.molecule_to_bond_topology_summaries(self._molecule))

    self.assertLen(got, 4)

    one_out = self.get_output_with(got, 123, 'count_detected_match_itc_success')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_success, 0)

    one_out = self.get_output_with(got, 618451, 'count_attempted_molecules')
    self.assertEqual(one_out.count_kept_geometry, 1)
    self.assertEqual(one_out.count_calculation_success, 1)
    self.assertEqual(one_out.count_calculation_with_error, 0)
    self.assertEqual(one_out.count_detected_match_itc_success, 0)
    self.assertEqual(one_out.count_detected_match_itc_with_error, 0)

    one_out = self.get_output_with(got, 123,
                                   'count_detected_match_mlcr_success')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_success, 0)
    self.assertEqual(one_out.count_calculation_with_error, 0)

    one_out = self.get_output_with(got, 618451,
                                   'count_detected_match_csd_success')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_success, 0)
    self.assertEqual(one_out.count_calculation_with_error, 0)

  def test_success_varied_sources(self):
    self._molecule.prop.calc.fate = dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW
    self._molecule.bond_topo.append(self._molecule.bond_topo[0])
    self._molecule.bond_topo.append(self._molecule.bond_topo[0])
    self._molecule.bond_topo[0].topo_id = 123
    self._molecule.bond_topo[0].info = (
        dataset_pb2.BondTopology.SOURCE_STARTING
        | dataset_pb2.BondTopology.SOURCE_DDT)
    self._molecule.bond_topo[1].topo_id = 456
    self._molecule.bond_topo[1].info = (
        dataset_pb2.BondTopology.SOURCE_CSD
        | dataset_pb2.BondTopology.SOURCE_MLCR)
    self._molecule.bond_topo[2].topo_id = 789
    self._molecule.bond_topo[2].info = (
        dataset_pb2.BondTopology.SOURCE_DDT
        | dataset_pb2.BondTopology.SOURCE_MLCR)

    got = list(
        smu_utils_lib.molecule_to_bond_topology_summaries(self._molecule))

    self.assertLen(got, 5)

    one_out = self.get_output_with(got, 123, 'count_calculation_success')
    self.assertEqual(one_out.count_detected_match_itc_success, 0)
    self.assertEqual(one_out.count_detected_match_mlcr_success, 0)
    self.assertEqual(one_out.count_detected_match_csd_success, 0)

    one_out = self.get_output_with(got, 789, 'count_detected_match_itc_success')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_success, 0)

    one_out = self.get_output_with(got, 456,
                                   'count_detected_match_mlcr_success')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_success, 0)

    one_out = self.get_output_with(got, 789,
                                   'count_detected_match_mlcr_success')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_success, 0)

    one_out = self.get_output_with(got, 456, 'count_detected_match_csd_success')
    self.assertEqual(one_out.count_attempted_molecules, 0)
    self.assertEqual(one_out.count_kept_geometry, 0)
    self.assertEqual(one_out.count_calculation_success, 0)

  def test_no_starting_topology(self):
    self._molecule.prop.calc.fate = dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW
    self._molecule.bond_topo.append(self._molecule.bond_topo[0])
    self._molecule.bond_topo[-1].topo_id = 123
    self._molecule.bond_topo[-1].info = (dataset_pb2.BondTopology.SOURCE_DDT)
    self._molecule.bond_topo[0].info = (dataset_pb2.BondTopology.SOURCE_DDT)

    got = list(
        smu_utils_lib.molecule_to_bond_topology_summaries(self._molecule))

    self.assertLen(got, 2)
    # We don't actually care about the order, but this is what comes out right
    # now.
    self.assertEqual(got[0].bond_topology.topo_id, 618451)
    self.assertEqual(got[0].count_detected_match_itc_success, 1)

    self.assertEqual(got[1].bond_topology.topo_id, 123)
    self.assertEqual(got[1].count_detected_match_itc_success, 1)

  @parameterized.parameters(0, 1, 2)
  def test_multiple_detection(self, starting_idx):
    self._molecule.prop.calc.fate = dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW
    # Even with 3 detections, we only want to output one multiple detection
    # record.
    self._molecule.bond_topo.append(self._molecule.bond_topo[0])
    self._molecule.bond_topo.append(self._molecule.bond_topo[0])
    for _ in range(3):
      self._molecule.bond_topo[starting_idx].info = (
          dataset_pb2.BondTopology.SOURCE_DDT)
    self._molecule.bond_topo[starting_idx].info |= (
        dataset_pb2.BondTopology.SOURCE_STARTING)

    got = list(
        smu_utils_lib.molecule_to_bond_topology_summaries(self._molecule))
    self.assertLen(got, 2)

    # We don't actually care about the order, but this is what comes out right
    # now.
    self.assertEqual(got[0].bond_topology.topo_id, 618451)
    self.assertEqual(got[0].count_calculation_success, 1)
    self.assertEqual(got[0].count_multiple_detections, 0)

    self.assertEqual(got[1].bond_topology.topo_id, 618451)
    self.assertEqual(got[1].count_calculation_success, 0)
    self.assertEqual(got[1].count_multiple_detections, 1)


class LabeledSmilesTester(absltest.TestCase):

  def test_atom_labels(self):
    mol = Chem.MolFromSmiles('FCON[NH2+][O-]', sanitize=False)
    self.assertIsNotNone(mol)
    smiles_before = Chem.MolToSmiles(mol)
    self.assertEqual(
        smu_utils_lib.labeled_smiles(mol), 'F[CH2:1][O:2][NH:3][NH2+:4][O-:5]')
    # Testing both the atom numbers and the smiles is redundant,
    # but guards against possible future changes.
    for atom in mol.GetAtoms():
      self.assertEqual(atom.GetAtomMapNum(), 0)
    self.assertEqual(Chem.MolToSmiles(mol), smiles_before)


class AtomSwapVariantTester(absltest.TestCase):

  def test_atom_swap_variant(self):
    """Test inspired by the following pair.

      C([C:3]1=[C:1]2[C:2]1=[NH+:5]2)=[N:4][O-:6]
      C([C:3]1=[C:2]2[C:1]1=[NH+:5]2)=[N:4][O-:6]
    """
    topo1 = """
atom: ATOM_C
atom: ATOM_C
atom: ATOM_C
atom: ATOM_C
atom: ATOM_N
atom: ATOM_N
atom: ATOM_O
bond {
  atom_b: 3
  bond_type: BOND_SINGLE
}
bond {
  atom_b: 4
  bond_type: BOND_DOUBLE
}
bond {
  atom_a: 1
  atom_b: 2
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 1
  atom_b: 3
  bond_type: BOND_DOUBLE
}
bond {
  atom_a: 1
  atom_b: 5
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 2
  atom_b: 3
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 2
  atom_b: 5
  bond_type: BOND_DOUBLE
}
bond {
  atom_a: 4
  atom_b: 6
  bond_type: BOND_SINGLE
}
"""
    bt1 = str_to_bond_topology(topo1)

    topo2 = """
atom: ATOM_C
atom: ATOM_C
atom: ATOM_C
atom: ATOM_C
atom: ATOM_N
atom: ATOM_N
atom: ATOM_O
bond {
  atom_a: 0
  atom_b: 3
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 0
  atom_b: 4
  bond_type: BOND_DOUBLE
}
bond {
  atom_a: 1
  atom_b: 2
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 1
  atom_b: 3
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 1
  atom_b: 5
  bond_type: BOND_DOUBLE
}
bond {
  atom_a: 2
  atom_b: 3
  bond_type: BOND_DOUBLE
}
bond {
  atom_a: 2
  atom_b: 5
  bond_type: BOND_SINGLE
}
bond {
  atom_a: 4
  atom_b: 6
  bond_type: BOND_SINGLE
}
"""
    bt2 = str_to_bond_topology(topo2)

    self.assertEqual(len(bt1.atom), len(bt2.atom))
    self.assertEqual(len(bt1.bond), len(bt2.bond))
    s1 = smu_utils_lib.compute_smiles_for_bond_topology(bt1, True)
    s2 = smu_utils_lib.compute_smiles_for_bond_topology(bt2, True)
    self.assertEqual(s1, s2)

    utilities.canonicalize_bond_topology(bt1)
    utilities.canonicalize_bond_topology(bt2)
    self.assertFalse(utilities.same_bond_topology(bt1, bt2))


if __name__ == '__main__':
  absltest.main()
