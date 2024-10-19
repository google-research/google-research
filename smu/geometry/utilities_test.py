# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

# Tester for SMU utilities functions.

from absl.testing import absltest
from parameterized import parameterized
from rdkit import Chem

from google.protobuf import text_format
from smu import dataset_pb2
from smu.geometry import utilities
from smu.parser import smu_utils_lib


def zero2():
  """Return a Geometry with two points at the origin."""
  return text_format.Parse(
      """
        atompos: {
          x:0.0,
          y:0.0,
          z:0.0
        },
        atompos: {
          x:0.0,
          y:0.0,
          z:0.0
        }

""", dataset_pb2.Geometry())


class TestUtilities(absltest.TestCase):

  def test_zero_distance(self):
    coords = zero2()
    self.assertEqual(utilities.distance_between_atoms(coords, 0, 1), 0.0)

  def test_unit_x(self):
    coords = zero2()
    coords.atompos[1].x = 1.0 / smu_utils_lib.BOHR_TO_ANGSTROMS
    self.assertAlmostEqual(utilities.distance_between_atoms(coords, 0, 1), 1.0)

  def test_unit_y(self):
    coords = zero2()
    coords.atompos[1].y = 1.0 / smu_utils_lib.BOHR_TO_ANGSTROMS
    self.assertAlmostEqual(utilities.distance_between_atoms(coords, 0, 1), 1.0)

  def test_unit_z(self):
    coords = zero2()
    coords.atompos[1].z = 1.0 / smu_utils_lib.BOHR_TO_ANGSTROMS
    self.assertAlmostEqual(utilities.distance_between_atoms(coords, 0, 1), 1.0)

  def test_canonical(self):
    bt = text_format.Parse(
        """
    atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
    bond {
      atom_a: 2
      atom_b: 1
      bond_type: BOND_SINGLE
    },
    bond {
      atom_a: 1
      atom_b: 0
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())

    expected = text_format.Parse(
        """ atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
    bond {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    },
    bond {
      atom_a: 1
      atom_b: 2
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())

    utilities.canonicalize_bond_topology(bt)
    self.assertEqual(
        text_format.MessageToString(bt), text_format.MessageToString(expected))

  def test_equality(self):
    bt1 = text_format.Parse(
        """
    atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
    bond {
      atom_a: 2
      atom_b: 1
      bond_type: BOND_SINGLE
    },
    bond {
      atom_a: 1
      atom_b: 0
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())

    bt2 = text_format.Parse(
        """ atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
    bond {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    },
    bond {
      atom_a: 1
      atom_b: 2
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())

    self.assertFalse(utilities.same_bond_topology(bt1, bt2))
    utilities.canonicalize_bond_topology(bt1)
    self.assertTrue(utilities.same_bond_topology(bt1, bt2))

  def test_single_fragment_single_atom(self):
    bt = text_format.Parse(""" atom: ATOM_C
""", dataset_pb2.BondTopology())
    self.assertTrue(utilities.is_single_fragment(bt))

  def test_single_fragment_two_disconnected_atoms(self):
    bt = text_format.Parse(""" atom: ATOM_C
    atom: ATOM_C
""", dataset_pb2.BondTopology())
    self.assertFalse(utilities.is_single_fragment(bt))

  def test_single_fragment_two_connected_atoms(self):
    bt = text_format.Parse(
        """ atom: ATOM_C
    atom: ATOM_C
    bond {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())
    self.assertTrue(utilities.is_single_fragment(bt))

  def test_single_fragment_3_atoms_0_bonds(self):
    bt = text_format.Parse(
        """ atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
""", dataset_pb2.BondTopology())
    self.assertFalse(utilities.is_single_fragment(bt))

  def test_single_fragment_3_atoms_1_bonds(self):
    bt = text_format.Parse(
        """ atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
    bond {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())
    self.assertFalse(utilities.is_single_fragment(bt))

  def test_single_fragment_3_atoms_2_bonds(self):
    bt = text_format.Parse(
        """ atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
    bond {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    },
    bond {
      atom_a: 1
      atom_b: 2
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())
    self.assertTrue(utilities.is_single_fragment(bt))

  def test_single_fragment_4_atoms_0_bonds(self):
    bt = text_format.Parse(
        """ atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
""", dataset_pb2.BondTopology())
    self.assertFalse(utilities.is_single_fragment(bt))

  def test_single_fragment_4_atoms_3_bonds_ring(self):
    bt = text_format.Parse(
        """ atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
    bond {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
    }
    bond {
      atom_a: 1
      atom_b: 2
      bond_type: BOND_SINGLE
    }
    bond {
      atom_a: 0
      atom_b: 2
      bond_type: BOND_SINGLE
    }
""", dataset_pb2.BondTopology())
    self.assertFalse(utilities.is_single_fragment(bt))

  def test_single_fragment_4_atoms_3_bonds_no_ring(self):
    bt = text_format.Parse(
        """ atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
    atom: ATOM_C
    bond {
      atom_a: 0
      atom_b: 1
      bond_type: BOND_SINGLE
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
""", dataset_pb2.BondTopology())
    self.assertTrue(utilities.is_single_fragment(bt))

  @parameterized.expand([
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
    bt = smu_utils_lib.rdkit_molecule_to_bond_topology(mol)
    self.assertEqual(utilities.is_single_fragment(bt), expected)
    Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS)
    mol_h = Chem.AddHs(mol)
    bt_h = smu_utils_lib.rdkit_molecule_to_bond_topology(mol_h)
    self.assertEqual(utilities.is_single_fragment(bt_h), expected)

  @parameterized.expand([["C", 0], ["CC", 0], ["CCC", 0], ["C1CC1", 3],
                         ["CCCCC", 0], ["C1CCC1", 4], ["C1C(C)CC1", 4]])
  def test_ring_atom_count(self, smiles, expected):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    self.assertEqual(utilities.ring_atom_count_mol(mol), expected)


if __name__ == "__main__":
  absltest.main()
