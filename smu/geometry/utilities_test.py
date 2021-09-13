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

# Tester for SMU utilities functions.

from absl.testing import absltest
from absl.testing import parameterized

from google.protobuf import text_format
from rdkit import Chem
import utilities

from smu import dataset_pb2
from smu.parser import smu_utils_lib


def zero2():
  """Return a Geometry with two points at the origin."""
  return text_format.Parse(
      """
        atom_positions: {
          x:0.0,
          y:0.0,
          z:0.0
        },
        atom_positions: {
          x:0.0,
          y:0.0,
          z:0.0
        }

""", dataset_pb2.Geometry())


class TestUtilities(parameterized.TestCase):

  def test_zero_distance(self):
    coords = zero2()
    self.assertEqual(utilities.distance_between_atoms(coords, 0, 1), 0.0)

  def test_unit_x(self):
    coords = zero2()
    coords.atom_positions[1].x = 1.0 / smu_utils_lib.BOHR_TO_ANGSTROMS
    self.assertAlmostEqual(utilities.distance_between_atoms(coords, 0, 1), 1.0)

  def test_unit_y(self):
    coords = zero2()
    coords.atom_positions[1].y = 1.0 / smu_utils_lib.BOHR_TO_ANGSTROMS
    self.assertAlmostEqual(utilities.distance_between_atoms(coords, 0, 1), 1.0)

  def test_unit_z(self):
    coords = zero2()
    coords.atom_positions[1].z = 1.0 / smu_utils_lib.BOHR_TO_ANGSTROMS
    self.assertAlmostEqual(utilities.distance_between_atoms(coords, 0, 1), 1.0)

  def test_connected(self):
    pass


#  @parameterized.expand(
#  [
#    ["[H]", 0, dataset_pb2.BondTopology.ATOM_H],
#    ["C", 0, dataset_pb2.BondTopology.ATOM_C],
#    ["N", 0, dataset_pb2.BondTopology.ATOM_N],
#    ["[N+]", 1, dataset_pb2.BondTopology.ATOM_NPOS],
#    ["O", 0, dataset_pb2.BondTopology.ATOM_O],
#    ["[O-]", -1, dataset_pb2.BondTopology.ATOM_ONEG],
#    ["F", 0, dataset_pb2.BondTopology.ATOM_F]
#  ]
#  )
#  def test_molecule_to_bond_topology_geom(self, smiles, charge, expected):
#    mol = Chem.MolFromSmiles(smiles, sanitize=False)
#    bt,geom = utilities.molecule_to_bond_topology_geom(mol)
#    self.assertEqual(len(bt.atoms), mol.GetNumAtoms())
#    self.assertEqual(bt.atoms[0], expected)
#
#  @parameterized.expand(
#  [
#    ["CC", dataset_pb2.BondTopology.BOND_SINGLE],
#    ["C=C", dataset_pb2.BondTopology.BOND_DOUBLE],
#    ["C#C", dataset_pb2.BondTopology.BOND_TRIPLE]
#  ]
#  )
#  def test_bonds(self, smiles, expected):
#    mol = Chem.MolFromSmiles(smiles, sanitize=False)
#    bt,geom = utilities.molecule_to_bond_topology_geom(mol)
#    self.assertEqual(len(bt.atoms), mol.GetNumAtoms())

  def test_canonical(self):
    bt = text_format.Parse(
        """
    atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
    bonds {
      atom_a: 2
      atom_b: 1
      bond_type: BOND_SINGLE
    },
    bonds {
      atom_a: 1
      atom_b: 0
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())

    expected = text_format.Parse(
        """ atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
    bonds {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    },
    bonds {
      atom_a: 1
      atom_b: 2
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())

    utilities.canonical_bond_topology(bt)
    self.assertEqual(
        text_format.MessageToString(bt), text_format.MessageToString(expected))

  def test_equality(self):
    bt1 = text_format.Parse(
        """
    atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
    bonds {
      atom_a: 2
      atom_b: 1
      bond_type: BOND_SINGLE
    },
    bonds {
      atom_a: 1
      atom_b: 0
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())

    bt2 = text_format.Parse(
        """ atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
    bonds {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    },
    bonds {
      atom_a: 1
      atom_b: 2
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())

    self.assertFalse(utilities.same_bond_topology(bt1, bt2))
    utilities.canonical_bond_topology(bt1)
    self.assertTrue(utilities.same_bond_topology(bt1, bt2))

  def test_single_fragment_single_atom(self):
    bt = text_format.Parse(""" atoms: ATOM_C
""", dataset_pb2.BondTopology())
    self.assertTrue(utilities.is_single_fragment(bt))

  def test_single_fragment_two_disconnected_atoms(self):
    bt = text_format.Parse(""" atoms: ATOM_C
    atoms: ATOM_C
""", dataset_pb2.BondTopology())
    self.assertFalse(utilities.is_single_fragment(bt))

  def test_single_fragment_two_connected_atoms(self):
    bt = text_format.Parse(
        """ atoms: ATOM_C
    atoms: ATOM_C
    bonds {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())
    self.assertTrue(utilities.is_single_fragment(bt))

  def test_single_fragment_3_atoms_0_bonds(self):
    bt = text_format.Parse(
        """ atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
""", dataset_pb2.BondTopology())
    self.assertFalse(utilities.is_single_fragment(bt))

  def test_single_fragment_3_atoms_1_bonds(self):
    bt = text_format.Parse(
        """ atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
    bonds {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())
    self.assertFalse(utilities.is_single_fragment(bt))

  def test_single_fragment_3_atoms_2_bonds(self):
    bt = text_format.Parse(
        """ atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
    bonds {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    },
    bonds {
      atom_a: 1
      atom_b: 2
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())
    self.assertTrue(utilities.is_single_fragment(bt))

  def test_single_fragment_4_atoms_0_bonds(self):
    bt = text_format.Parse(
        """ atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
""", dataset_pb2.BondTopology())
    self.assertFalse(utilities.is_single_fragment(bt))

  def test_single_fragment_4_atoms_3_bonds_ring(self):
    bt = text_format.Parse(
        """ atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
    bonds {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    }
    bonds {
      atom_a: 1
      atom_b: 2
      bond_type: BOND_SINGLE
    }
    bonds {
      atom_a: 0
      atom_b: 2
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())
    self.assertFalse(utilities.is_single_fragment(bt))

  def test_single_fragment_4_atoms_3_bonds_no_ring(self):
    bt = text_format.Parse(
        """ atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
    atoms: ATOM_C
    bonds {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    }
    bonds {
      atom_a: 1
      atom_b: 2
      bond_type: BOND_SINGLE
    }
    bonds {
      atom_a: 2
      atom_b: 3
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())
    self.assertTrue(utilities.is_single_fragment(bt))

  @parameterized.parameters([
      ["CC", True],
      ["C=C", True],
      ["C#C", True],
      ["C.C", False],
      ["CCCC", True],
      ["C1CCC1", True],
      ["CCC.C", False],
      ["CCC.CCC", False],
      ["C1CCCCC1.CCC", False],
      ["C.C1CCCCC1", False],
      ["C.C.C.C.F.N.O", False],
      ["C=N.O", False],
      ["CC1CC1.C", False],
      ["C12CC2C1.C", False],
  ])
  def test_with_smiles(self, smiles, expected):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    bt = utilities.molecule_to_bond_topology(mol)
    self.assertEqual(utilities.is_single_fragment(bt), expected)
    Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS)
    mol_h = Chem.AddHs(mol)
    bt_h = utilities.molecule_to_bond_topology(mol_h)
    self.assertEqual(utilities.is_single_fragment(bt_h), expected)

if __name__ == "__main__":
  absltest.main()
