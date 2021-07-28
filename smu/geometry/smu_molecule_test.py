import operator

import itertools

import numpy as np

from parameterized import parameterized, parameterized_class

import unittest
from google.protobuf import text_format

from smu import dataset_pb2
from smu.topology_from_geometry import smu_molecule

class TestSMuMolecule(unittest.TestCase):
  """Test the SmuMolecule class."""

  def test_ethane(self):
    """The simplest molecule, CC"""
    bond_topology = text_format.Parse("""
      atoms: ATOM_C
      atoms: ATOM_C
      bonds: {
        atom_a: 0,
        atom_b: 1,
        bond_type: BOND_SINGLE
      }
""", dataset_pb2.BondTopology())
    cc = text_format.Parse("""
      atoms: ATOM_C
      atoms: ATOM_C
""", dataset_pb2.BondTopology())
    scores = np.array([0.1, 1.1, 2.1, 3.1], dtype=np.float32)
    bonds_to_scores = {(0, 1): scores}
    mol = smu_molecule.SmuMolecule(cc, bonds_to_scores)
    state = mol.generate_search_state()
    self.assertEqual(len(state), 1)
    self.assertEqual(state, [[0, 1, 2, 3]])

    for i, s in enumerate(itertools.product(*state)):
      res = mol.place_bonds(s)
      self.assertIsNotNone(res)
      self.assertAlmostEqual(res.score, scores[i])

  @parameterized.expand(
  [
    [0, dataset_pb2.BondTopology.BOND_UNDEFINED],
    [1, dataset_pb2.BondTopology.BOND_SINGLE],
    [2, dataset_pb2.BondTopology.BOND_DOUBLE],
    [3, dataset_pb2.BondTopology.BOND_TRIPLE],
  ]
  )
  def test_ethane_all(self, btype, expected_bond):
    cc = text_format.Parse("""
      atoms: ATOM_C
      atoms: ATOM_C
""", dataset_pb2.BondTopology())
    bonds_to_scores = {(0, 1): np.zeros(4, dtype=np.float32)}
    bonds_to_scores[(0, 1)][btype] = 1.0
    mol = smu_molecule.SmuMolecule(cc, bonds_to_scores)
    state = mol.generate_search_state()
    for s in itertools.product(*state):
      res = mol.place_bonds(s)
      self.assertIsNotNone(res)
      if btype == 0:
        self.assertEqual(len(res.bonds), 0)
      else:
        self.assertEqual(len(res.bonds), 1)
        self.assertEqual(res.bonds[0].bond_type, expected_bond)

  @parameterized.expand(
  [
    [0, 0, 0, 2.0],
    [0, 1, 1, 2.0],
    [0, 2, 1, 2.0],
    [0, 3, 1, 2.0],
    [1, 1, 2, 2.0],
    [1, 2, 2, 2.0],
    [1, 3, 2, 2.0],
    [2, 2, 2, 2.0],
    [2, 3, 0, None],
    [3, 3, 0, None]
  ]
  )
  def test_propane_all(self, btype1, btype2,
                       expected_bonds, expected_score):
    cc = text_format.Parse("""
      atoms: ATOM_C
      atoms: ATOM_C
      atoms: ATOM_C
""", dataset_pb2.BondTopology())
#   print(f"Generating bonds {btype1} and {btype2}")
    bonds_to_scores = {(0, 1): np.zeros(4, dtype=np.float32),
                       (1, 2): np.zeros(4, dtype=np.float32)}
    bonds_to_scores[(0, 1)][btype1] = 1.0
    bonds_to_scores[(1, 2)][btype2] = 1.0
    mol = smu_molecule.SmuMolecule(cc, bonds_to_scores)
    state = mol.generate_search_state()
    for s in itertools.product(*state):
      res = mol.place_bonds(s)
      if expected_score is not None:
        self.assertIsNotNone(res)
        self.assertEqual(len(res.bonds), expected_bonds)
        self.assertAlmostEqual(res.score, expected_score)
        if btype1 == 0:
          if btype2 > 0:
            self.assertEqual(res.bonds[0].bond_type, btype2)
        else:
          self.assertEqual(res.bonds[0].bond_type, btype1)
          self.assertEqual(res.bonds[1].bond_type, btype2)
      else:
        self.assertIsNone(res)

  def test_operators(self):
    cc = text_format.Parse("""
      atoms: ATOM_C
      atoms: ATOM_C
      atoms: ATOM_C
""", dataset_pb2.BondTopology())
#   print(f"Generating bonds {btype1} and {btype2}")
    bonds_to_scores = {(0, 1): np.zeros(4, dtype=np.float32),
                       (1, 2): np.zeros(4, dtype=np.float32)}
    scores = np.array([1.0, 3.0], dtype=np.float32)
    bonds_to_scores[(0, 1)][1] = scores[0]
    bonds_to_scores[(1, 2)][1] = scores[1]
    mol = smu_molecule.SmuMolecule(cc, bonds_to_scores)
    mol.set_initial_score_and_incrementer(1.0, operator.mul)
    state = mol.generate_search_state()
    for s in itertools.product(*state):
      res = mol.place_bonds(s)
      self.assertAlmostEqual(res.score, np.product(scores))

if __name__ == "__main__":
  unittest.main()
