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

"""Tests for molecule_graph."""

import unittest

from absl import flags
import numpy as np
from rdkit import Chem
import six
from six.moves import range

from tensorflow.python.util.protobuf import compare
from gigamol.molecule_graph_proto import molecule_graph
from gigamol.molecule_graph_proto import molecule_graph_pb2 as mgpb


FLAGS = flags.FLAGS


class MoleculeGraphTest(unittest.TestCase):

  def testGetAllPairsFromProto(self):
    mol = Chem.MolFromSmiles('C(Cl)Cl')  # CID6344
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto()
    self.assertCountEqual([p.graph_distance for p in pb.atom_pairs], [1, 1, 2])

  def testGetNeighborsFromProto(self):
    mol = Chem.MolFromSmiles('C(Cl)Cl')  # CID6344
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto(max_pair_distance=1)
    self.assertCountEqual([p.graph_distance for p in pb.atom_pairs], [1, 1])

  def testGetRingSizes(self):
    mol = Chem.MolFromSmiles('C1=NC2=C(N1)C(=NC=N2)N')  # CID190
    mg = molecule_graph.MoleculeGraph(mol)
    self.assertCountEqual(
        six.iteritems(mg.get_ring_sizes()),
        [(0, [5]), (1, [5]), (2, [5, 6]), (3, [5, 6]), (4, [5]), (5, [6]),
         (6, [6]), (7, [6]), (8, [6])])

  def testGetRingSizesFromProto(self):
    mol = Chem.MolFromSmiles('C1=NC2=C(N1)C(=NC=N2)N')  # CID190
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto()
    self.assertCountEqual(
        [atom.ring_sizes for atom in pb.atoms],
        [[5], [5], [5], [5, 6], [5, 6], [6], [6], [6], [6], []])

  def testNoMismatches(self):
    """Test that using atom indices doesn't cause any mismatches.

    Smallest ring sizes and bond types use atom indices.
    """
    mol = Chem.MolFromSmiles('C1=CC=C(C=C1)C(=O)[C@@H](CCF)Cl')  # CID88643715
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto()
    count = 0
    for atom in pb.atoms:
      # Each aromatic carbon should be in a single ring of size 6
      if atom.type == mgpb.MoleculeGraph.Atom.ATOM_C and atom.aromatic:
        self.assertListEqual(list(atom.ring_sizes), [6])
        count += 1
    self.assertEqual(count, 6)  # should be six aromatic carbons
    count = 0
    for pair in pb.atom_pairs:
      # Each bonded aromatic pair should have aromatic bond type.
      a = pb.atoms[pair.a_idx]
      b = pb.atoms[pair.b_idx]
      if (a.type == b.type == mgpb.MoleculeGraph.Atom.ATOM_C and a.aromatic and
          b.aromatic and pair.graph_distance == 1):
        self.assertEqual(
            pair.bond_type, mgpb.MoleculeGraph.AtomPair.BOND_AROMATIC)
        count += 1
    self.assertEqual(count, 6)  # should be six aromatic bonds

  def testGetAtomType(self):
    mol = Chem.MolFromSmiles('C1=CC=C(C=C1)C(=O)[C@@H](CCF)Cl')  # CID88643715
    mg = molecule_graph.MoleculeGraph(mol)
    self.assertCountEqual(
        [mg.get_atom_type(atom) for atom in mol.GetAtoms()],
        [mgpb.MoleculeGraph.Atom.ATOM_C] * 10 + [
            mgpb.MoleculeGraph.Atom.ATOM_O, mgpb.MoleculeGraph.Atom.ATOM_F,
            mgpb.MoleculeGraph.Atom.ATOM_CL
        ])

    mol = Chem.MolFromSmiles('FB(F)F')
    mg = molecule_graph.MoleculeGraph(mol)
    self.assertCountEqual(
        [mg.get_atom_type(atom) for atom in mol.GetAtoms()],
        [mgpb.MoleculeGraph.Atom.ATOM_F] * 3 + [mgpb.MoleculeGraph.Atom.ATOM_B])

    mol = Chem.MolFromSmiles('Cl[Se][Se]Cl')
    mg = molecule_graph.MoleculeGraph(mol)
    self.assertCountEqual(
        [mg.get_atom_type(atom) for atom in mol.GetAtoms()],
        [mgpb.MoleculeGraph.Atom.ATOM_CL] * 2 + [
            mgpb.MoleculeGraph.Atom.ATOM_SE] * 2)

  def testGetAtomTypeFromProto(self):
    mol = Chem.MolFromSmiles('C1=CC=C(C=C1)C(=O)[C@@H](CCF)Cl')  # CID88643715
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto()
    self.assertCountEqual(
        [atom.type for atom in pb.atoms],
        [mgpb.MoleculeGraph.Atom.ATOM_C] * 10 + [
            mgpb.MoleculeGraph.Atom.ATOM_O, mgpb.MoleculeGraph.Atom.ATOM_F,
            mgpb.MoleculeGraph.Atom.ATOM_CL
        ])

    mol = Chem.MolFromSmiles('FB(F)F')
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto()
    self.assertCountEqual(
        [atom.type for atom in pb.atoms],
        [mgpb.MoleculeGraph.Atom.ATOM_F] * 3 + [mgpb.MoleculeGraph.Atom.ATOM_B])

    mol = Chem.MolFromSmiles('Cl[Se][Se]Cl')
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto()
    self.assertCountEqual(
        [atom.type for atom in pb.atoms],
        [mgpb.MoleculeGraph.Atom.ATOM_CL] * 2 + [
            mgpb.MoleculeGraph.Atom.ATOM_SE] * 2)

  def testGetChirality(self):
    mol = Chem.MolFromSmiles('C1=CC=C(C=C1)C(=O)[C@@H](CCF)Cl')  # CID88643715
    mg = molecule_graph.MoleculeGraph(mol)
    self.assertCountEqual(
        [mg.get_atom_chirality(atom) for atom in mol.GetAtoms()],
        [mgpb.MoleculeGraph.Atom.CHIRAL_R
        ] + [mgpb.MoleculeGraph.Atom.CHIRAL_NONE] * 12)

  def testGetChiralityFromProto(self):
    mol = Chem.MolFromSmiles('C1=CC=C(C=C1)C(=O)[C@@H](CCF)Cl')  # CID88643715
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto()
    self.assertCountEqual([atom.chirality for atom in pb.atoms],
                          [mgpb.MoleculeGraph.Atom.CHIRAL_R
                          ] + [mgpb.MoleculeGraph.Atom.CHIRAL_NONE] * 12)

  def testGetHybridization(self):
    # CID91796808: the amide nitrogen is sp2
    mol = Chem.MolFromSmiles('CC(C)(C#CC1=CC=C(C=C1)C(=O)N2CCC(C2)OC)O')
    mg = molecule_graph.MoleculeGraph(mol)
    self.assertCountEqual(
        [mg.get_hybridization(atom) for atom in mol.GetAtoms()],
        [mgpb.MoleculeGraph.Atom.HYBRIDIZATION_SP3] * 10 +
        [mgpb.MoleculeGraph.Atom.HYBRIDIZATION_SP2] * 9 +
        [mgpb.MoleculeGraph.Atom.HYBRIDIZATION_SP] * 2)

  def testGetHybridizationFromProto(self):
    # CID91796808: the amide nitrogen is sp2
    mol = Chem.MolFromSmiles('CC(C)(C#CC1=CC=C(C=C1)C(=O)N2CCC(C2)OC)O')
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto()
    self.assertCountEqual([atom.hybridization for atom in pb.atoms],
                          [mgpb.MoleculeGraph.Atom.HYBRIDIZATION_SP3] * 10 +
                          [mgpb.MoleculeGraph.Atom.HYBRIDIZATION_SP2] * 9 +
                          [mgpb.MoleculeGraph.Atom.HYBRIDIZATION_SP] * 2)

  def testGetHydrogenBonding(self):
    # CID91796808
    mol = Chem.MolFromSmiles('CC(C)(C#CC1=CC=C(C=C1)C(=O)N2CCC(C2)OC)O')
    mg = molecule_graph.MoleculeGraph(mol)
    self.assertCountEqual(
        six.itervalues(mg.get_hydrogen_bonding()),
        [molecule_graph.HydrogenBonding(acceptor=True, donor=False)] * 3 +
        [molecule_graph.HydrogenBonding(acceptor=True, donor=True)])

  def testGetHydrogenBondingFromProto(self):
    # CID91796808
    mol = Chem.MolFromSmiles('CC(C)(C#CC1=CC=C(C=C1)C(=O)N2CCC(C2)OC)O')
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto()
    self.assertCountEqual(
        [(atom.acceptor, atom.donor) for atom in pb.atoms],
        [(True, False)] * 3 + [(True, True)] + [(False, False)] * 17)

  def testGetSmiles(self):
    # CID68617
    mol = Chem.MolFromSmiles('CN[C@H]1CC[C@H](C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl')
    mg = molecule_graph.MoleculeGraph(mol)
    self.assertEqual(mg.smiles, 'CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21')

  def testGetSmilesFromProto(self):
    # CID68617
    mol = Chem.MolFromSmiles('CN[C@H]1CC[C@H](C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl')
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto()
    self.assertEqual(pb.smiles, 'CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21')

  def testGetBondTypes(self):
    mol = Chem.MolFromSmiles('C1=CC=C(C=C1)C(=O)[C@@H](CCF)Cl')  # CID88643715
    mg = molecule_graph.MoleculeGraph(mol)
    self.assertCountEqual(
        six.itervalues(mg.get_bond_types()),
        [mgpb.MoleculeGraph.AtomPair.BOND_SINGLE] * 6 + [
            mgpb.MoleculeGraph.AtomPair.BOND_DOUBLE
        ] + [mgpb.MoleculeGraph.AtomPair.BOND_AROMATIC] * 6)

  def testGetBondTypesFromProto(self):
    mol = Chem.MolFromSmiles('C(Cl)Cl')  # CID6344
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto()
    self.assertCountEqual([pair.bond_type for pair in pb.atom_pairs],
                          [mgpb.MoleculeGraph.AtomPair.BOND_SINGLE] * 2 +
                          [mgpb.MoleculeGraph.AtomPair.BOND_NONE])

  def testGetBondTypesNeighborsFromProto(self):
    mol = Chem.MolFromSmiles('C1=CC=C(C=C1)C(=O)[C@@H](CCF)Cl')  # CID88643715
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto(max_pair_distance=1)
    self.assertCountEqual([pair.bond_type for pair in pb.atom_pairs],
                          [mgpb.MoleculeGraph.AtomPair.BOND_SINGLE] * 6 + [
                              mgpb.MoleculeGraph.AtomPair.BOND_DOUBLE
                          ] + [mgpb.MoleculeGraph.AtomPair.BOND_AROMATIC] * 6)

  def testGetSameRingFromProto(self):
    mol = Chem.MolFromSmiles('C1=CC=C2C=CC=CC2=C1')  # CID931
    mg = molecule_graph.MoleculeGraph(mol)
    pb = mg.to_proto()
    self.assertCountEqual([pair.same_ring for pair in pb.atom_pairs],
                          [True] * 29 + [False] * 16)

  def testGetMMFF94PartialCharges(self):
    mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')  # CID2244
    mg = molecule_graph.MoleculeGraph(mol, partial_charges='mmff94')
    # Compares with PubChem partial charges for non-hydrogen atoms.
    self.assertCountEqual(
        np.round(list(mg.get_partial_charges().values()), 2), [
            -0.65, -0.57, -0.57, -0.23, -0.15, -0.15, -0.15, -0.15, 0.06, 0.08,
            0.09, 0.63, 0.66
        ])

  def testGetGasteigerPartialCharges(self):
    mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')  # CID2244
    mg = molecule_graph.MoleculeGraph(mol, partial_charges='gasteiger')
    self.assertCountEqual(
        np.round(list(mg.get_partial_charges().values()), 2), [
            -0.48, -0.43, -0.25, -0.25, -0.06, -0.06, -0.04, -0.02, 0.03, 0.1,
            0.14, 0.31, 0.34
        ])

  def testGetPartialChargesFromProto(self):
    mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')  # CID2244
    mg = molecule_graph.MoleculeGraph(mol, partial_charges='mmff94')
    pb = mg.to_proto()
    self.assertCountEqual(
        np.round([atom.partial_charge for atom in pb.atoms], 2), [
            -0.65, -0.57, -0.57, -0.23, -0.15, -0.15, -0.15, -0.15, 0.06, 0.08,
            0.09, 0.63, 0.66
        ])

  def testGetCircularFingerprintFromProto(self):
    mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')  # CID2244
    mg = molecule_graph.MoleculeGraph(mol, partial_charges='mmff94')
    pb = mg.to_proto()
    self.assertEqual(len(pb.binary_features), 1024)
    self.assertGreater(sum(pb.binary_features), 0)

  def testCircularFingerprintUsesChirality(self):
    # These molecules differ only in the chirality of their stereocenters.
    mol1 = Chem.MolFromSmiles(
        'CN[C@H]1CC[C@H](C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl')
    mol2 = Chem.MolFromSmiles(
        'CN[C@@H]1CC[C@@H](C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl')
    mg1 = molecule_graph.MoleculeGraph(mol1)
    mg2 = molecule_graph.MoleculeGraph(mol2)
    pb1 = mg1.to_proto()
    pb2 = mg2.to_proto()
    self.assertNotEqual(pb1.binary_features, pb2.binary_features)

  def testHasNonZeroSpatialDistance(self):
    mol = Chem.MolFromSmiles('CCCC')
    mol_pb = molecule_graph.MoleculeGraph(
        mol, partial_charges=None, compute_conformer=True).to_proto(
            calc_pair_spatial_distances=True)
    self.assertTrue(mol_pb.atom_pairs[0].HasField('spatial_distance'))
    self.assertGreater(mol_pb.atom_pairs[0].spatial_distance, 0)

  def testHasNoSpatialDistance(self):
    mol = Chem.MolFromSmiles('CCCC')
    mol_pb = molecule_graph.MoleculeGraph(
        mol, partial_charges=None).to_proto(
            calc_pair_spatial_distances=False)
    self.assertFalse(mol_pb.atom_pairs[0].HasField('spatial_distance'))

  def testPermutation(self):
    # Cyanic acid is handy because all 4 atoms are different.
    mol = Chem.AddHs(Chem.MolFromSmiles('C(#N)O'))
    mg = molecule_graph.MoleculeGraph(mol, partial_charges=None)
    random_state = np.random.RandomState(seed=123)
    num_times_carbon_first = 0
    for _ in range(50):
      mol_pb = mg.to_proto(max_pair_distance=-1,
                           calc_pair_spatial_distances=False,
                           random_permute=random_state)
      self.assertEqual(len(mol_pb.atoms), 4)
      self.assertEqual(len(mol_pb.atom_pairs), 6)
      pos_to_atom = [mol_pb.atoms[i].element for i in range(4)]
      if pos_to_atom[0] == 'C':
        num_times_carbon_first += 1
      # Checks that the carbon is triple bonded to nitrogen and single bonded to
      # oxygen.
      num_pairs_checked = 0
      for pair in mol_pb.atom_pairs:
        atom_types = set([pos_to_atom[pair.a_idx], pos_to_atom[pair.b_idx]])
        if atom_types == set(['C', 'N']):
          self.assertEqual(pair.bond_type,
                           mgpb.MoleculeGraph.AtomPair.BOND_TRIPLE)
          num_pairs_checked += 1
        if atom_types == set(['C', 'O']):
          self.assertEqual(pair.bond_type,
                           mgpb.MoleculeGraph.AtomPair.BOND_SINGLE)
          num_pairs_checked += 1
      self.assertEqual(num_pairs_checked, 2)
    self.assertLess(num_times_carbon_first, 50)
    self.assertGreater(num_times_carbon_first, 0)


class SimpleFeaturesTest(compare.ProtoAssertions, unittest.TestCase):

  def smiles_to_features(self, smiles):
    mol = Chem.MolFromSmiles(smiles)
    mg = molecule_graph.MoleculeGraph(
        mol, partial_charges='gasteiger', compute_conformer=True)
    mol_pb = mg.to_proto(max_pair_distance=-1,
                         calc_pair_spatial_distances=True)
    return molecule_graph.proto_to_simple_features(mol_pb)

  def testMethane(self):
    out_features = self.smiles_to_features('C')
    self.assertProtoEqual(
        out_features,
        mgpb.SimpleMoleculeFeatures(
            element_type_counts={'C': 1},
            num_atoms=1,  # hydrogens are implicit
            num_heavy_atoms=1,
            is_chiral=False,
            num_hbond_acceptor=0,
            num_hbond_donor=0,
            is_aromatic=False,
            bond_type_counts={},
            ring_sizes=[],
            partial_charges_distribution=
            mgpb.SimpleMoleculeFeatures.DistributionSummary(
                min=-0.0775578911931,
                max=-0.0775578911931,
                mean=-0.0775578911931,
                median=-0.0775578911931,
                count=1.0,
                std=np.nan),
            formal_charges_distribution=
            mgpb.SimpleMoleculeFeatures.DistributionSummary(
                min=0.0,
                max=0.0,
                mean=0.0,
                median=0.0,
                count=1.0,
                std=np.nan),
            graph_distances_distribution=
            mgpb.SimpleMoleculeFeatures.DistributionSummary(
                min=np.nan,
                max=np.nan,
                mean=np.nan,
                median=np.nan,
                count=0.0,
                std=np.nan),
            spatial_distances_distribution=
            mgpb.SimpleMoleculeFeatures.DistributionSummary(
                min=np.nan,
                max=np.nan,
                mean=np.nan,
                median=np.nan,
                count=0.0,
                std=np.nan),
        ))

  def testMethaneWithHs(self):
    mol = Chem.rdmolops.AddHs(Chem.MolFromSmiles('C'))
    mg = molecule_graph.MoleculeGraph(mol)
    mol_pb = mg.to_proto(max_pair_distance=-1,
                         calc_pair_spatial_distances=False)
    out_features = molecule_graph.proto_to_simple_features(mol_pb)
    self.assertEqual(out_features.num_atoms, 5)
    self.assertEqual(out_features.num_heavy_atoms, 1)
    self.assertEqual(out_features.element_type_counts, {'C': 1, 'H': 4})

  def testBenzofuran(self):
    # Benzofuran is a simple aromatic molecule with 2 fused rings.
    out_features = self.smiles_to_features('o2c1ccccc1cc2')
    self.assertProtoEqual(
        out_features,
        mgpb.SimpleMoleculeFeatures(
            element_type_counts={
                'C': 8,
                'O': 1
            },
            num_atoms=9,  # hydrogens are implicit
            num_heavy_atoms=9,
            is_chiral=False,
            num_hbond_acceptor=1,
            num_hbond_donor=0,
            is_aromatic=True,
            bond_type_counts={4: 10},
            ring_sizes=[5, 6],
            partial_charges_distribution=mgpb.SimpleMoleculeFeatures
            .DistributionSummary(
                min=-0.464352920848,
                max=0.133344576552,
                mean=-0.047186106,
                median=-0.018712824,
                count=9.0,
                std=0.17051615),
            formal_charges_distribution=mgpb.SimpleMoleculeFeatures
            .DistributionSummary(
                min=0.0, max=0.0, mean=0.0, median=0.0, count=9.0, std=0.0),
            graph_distances_distribution=mgpb.SimpleMoleculeFeatures
            .DistributionSummary(
                min=1.0,
                max=4.0,
                mean=2.19444444444,
                median=2.0,
                count=36.0,
                std=0.980362744657),
            spatial_distances_distribution=mgpb.SimpleMoleculeFeatures
            .DistributionSummary(
                min=1.5,
                max=4.96333164827,
                mean=2.76933290278,
                median=2.59807621135,
                count=36.0,
                std=1.0782908346),
        ))

  def testChiral(self):
    # This is alanine.
    out_features = self.smiles_to_features('O=C(O)[C@@H](N)C')
    self.assertEqual(out_features.num_heavy_atoms, 6)
    self.assertTrue(out_features.is_chiral)


if __name__ == '__main__':
  unittest.main()
