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

"""Functional tests for molecule parsing ops."""
import os
from typing import List, Tuple
import unittest

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf  # tf

from google.protobuf import text_format
from gigamol.molecule_graph_parsing_ops.py import molecule_graph_parsing_ops as mgpo
from gigamol.molecule_graph_proto import molecule_graph_pb2 as mgpb

TESTDIR = 'gigamol/molecule_graph_parsing_ops/py/testdata'


def OneHot(size, index):
  x = [0.0] * size
  x[index] = 1.0
  return x


def ReadTestProtosFromCSV(test_datafile):
  """Reads test protos from csv and returns list of keys and values in string format."""
  df = pd.read_csv(test_datafile)
  test_protos = {}
  for _, row in df.iterrows():
    mol_pb = mgpb.MoleculeGraph()
    text_format.Parse(row.proto_string, mol_pb)
    test_protos[row.example_id] = mol_pb.SerializeToString()
  return list(test_protos.keys()), list(test_protos.values())


class MoleculeParserOpTest(tf.test.TestCase):

  def setUp(self):
    super(MoleculeParserOpTest, self).setUp()
    self.batch_size = 7
    self.max_atoms = 15
    test_datafile = os.path.join(TESTDIR, 'molecule_graph_test.csv')
    self.test_dataset = ReadTestProtosFromCSV(test_datafile)

    with self.test_session() as sess:
      keys, values = self.test_dataset
      self.mols = {
          key: idx for idx, key in enumerate(keys)
      }
      (self.example_ids, self.atoms_t, self.pairs_t,
       self.atom_mask_t, self.pair_mask_t) = mgpo.MoleculeGraphParser(
           keys,
           values,
           self.max_atoms)

      (self.atoms, self.pairs, self.atom_mask, self.pair_mask) = sess.run(
          [self.atoms_t, self.pairs_t, self.atom_mask_t, self.pair_mask_t])
    self.graph_distance_pb = mgpb.MoleculeGraph(
        atoms=[
            mgpb.MoleculeGraph.Atom(type=mgpb.MoleculeGraph.Atom.ATOM_C),
            mgpb.MoleculeGraph.Atom(type=mgpb.MoleculeGraph.Atom.ATOM_C),
            mgpb.MoleculeGraph.Atom(type=mgpb.MoleculeGraph.Atom.ATOM_C),
            mgpb.MoleculeGraph.Atom(type=mgpb.MoleculeGraph.Atom.ATOM_C),
            mgpb.MoleculeGraph.Atom(type=mgpb.MoleculeGraph.Atom.ATOM_C),
        ],
        atom_pairs=[
            mgpb.MoleculeGraph.AtomPair(
                a_idx=0,
                b_idx=1,
                bond_type=mgpb.MoleculeGraph.AtomPair.BOND_NONE,
                graph_distance=0),
            mgpb.MoleculeGraph.AtomPair(
                a_idx=0,
                b_idx=2,
                bond_type=mgpb.MoleculeGraph.AtomPair.BOND_SINGLE,
                graph_distance=1),
            mgpb.MoleculeGraph.AtomPair(
                a_idx=0,
                b_idx=3,
                bond_type=mgpb.MoleculeGraph.AtomPair.BOND_NONE,
                graph_distance=7),
            mgpb.MoleculeGraph.AtomPair(
                a_idx=0,
                b_idx=4,
                bond_type=mgpb.MoleculeGraph.AtomPair.BOND_NONE,
                graph_distance=8),
        ])

  def testShapes(self):
    atoms_shape = (self.batch_size, self.max_atoms, 27)
    pairs_shape = (self.batch_size, self.max_atoms, self.max_atoms, 12)

    self.assertEqual(self.atoms.shape, atoms_shape)
    self.assertEqual(self.pairs.shape, pairs_shape)
    self.assertEqual(self.atom_mask.shape, atoms_shape[:-1])
    self.assertEqual(self.pair_mask.shape, pairs_shape[:-1])

  def testMasks(self):
    self.assertAllEqual(self.atom_mask[self.mols['CID1647']],
                        [1] * 5 + [0] * 10)
    expected_pair_mask = np.zeros((15, 15))
    expected_pair_mask[:5, :5] = [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1],
                                  [1, 1, 0, 1, 1], [1, 1, 1, 0, 1],
                                  [1, 1, 1, 1, 0]]
    self.assertAllEqual(self.pair_mask[self.mols['CID1647']],
                        expected_pair_mask)

    self.assertAllEqual(self.atom_mask[self.mols['CID6344']],
                        [1] * 3 + [0] * 12)
    expected_pair_mask = np.zeros((15, 15))
    expected_pair_mask[:3, :3] = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    self.assertAllEqual(self.pair_mask[self.mols['CID6344']],
                        expected_pair_mask)

  def testAtomType(self):
    self.assertAllEqual(
        self.atoms[self.mols['CID6344'], :, :11],  # Cl, Cl, C
        [OneHot(11, 7),
         OneHot(11, 7),
         OneHot(11, 1)] +
        [np.zeros(11)] * 12)

  def testOtherAtomType(self):
    self.assertAllEqual(
        self.atoms[self.mols['CID24575'], :, :11],  # I, I, I, As
        [OneHot(11, 9),
         OneHot(11, 9),
         OneHot(11, 9)] +
        [np.zeros(11)] * 12)

  def testChirality(self):
    expected = np.zeros((self.max_atoms, 2))
    expected[3] = OneHot(2, 0)
    self.assertAllEqual(
        self.atoms[self.mols['CID88643715'], :, 11:13], expected)

  def testFormalCharge(self):
    expected = np.zeros(15)
    expected[1] = -1
    self.assertAllEqual(
        self.atoms[self.mols['CID16760658'], :, 13],
        expected)

  def testPartialCharge(self):
    self.assertAllEqual(self.atoms[self.mols['CID1647'], :, 14], np.zeros(15))

  def testRingSizes(self):
    # multiple ring sizes
    self.assertAllEqual(
        self.atoms[self.mols['CID190'], :, 15:21],
        [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # multiple ring counts
    self.assertAllEqual(
        self.atoms[self.mols['CID6812'], :, 15:21],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

  def testHybridization(self):
    self.assertAllEqual(
        self.atoms[self.mols['CID1647'], :, 21:24],
        [OneHot(3, 2),
         OneHot(3, 0),
         OneHot(3, 2),
         OneHot(3, 2),
         OneHot(3, 0)] +
        [np.zeros(3)] * 10)

  def testHydrogenBonding(self):
    self.assertAllEqual(
        self.atoms[self.mols['CID16760658'], :, 24:26],
        [[1.0, 1.0],
         [1.0, 0.0],
         [1.0, 0.0]] +
        [np.zeros(2)] * 12)

  def testAromaticity(self):
    self.assertAllEqual(
        self.atoms[self.mols['CID190'], :, 26],
        [1.0, 1.0, 1.0, 1.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0,
         0.0, 0.0, 0.0, 0.0, 0.0])

  def testBondType(self):
    self.assertAllEqual(
        self.pairs[self.mols['CID6344'], :3, :3, :4],
        [[np.zeros(4),
          np.zeros(4),
          OneHot(4, 0)],
         [np.zeros(4),
          np.zeros(4),
          OneHot(4, 0)],
         [OneHot(4, 0),
          OneHot(4, 0),
          np.zeros(4)]])

  def testGraphDistance(self):
    expected = np.zeros((self.max_atoms, self.max_atoms, 7))
    expected[:3, :3] = (
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
         [[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
         [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    self.assertAllEqual(self.pairs[self.mols['CID6344'], :, :, 4:11], expected)

  def testSameRing(self):
    expected = np.zeros((self.max_atoms, self.max_atoms))
    expected[:10, :10] = (
        [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
         [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
         [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
         [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    self.assertAllEqual(self.pairs[self.mols['CID190'], :, :, 11], expected)

  def testMaxPairDistance(self):
    # Constrains to adjacent atoms
    with self.test_session() as sess:
      keys, values = self.test_dataset
      mg_input = mgpo.MoleculeGraphParser(
          keys, values,
          self.max_atoms,
          max_pair_distance=1)
      pair_mask = sess.run(mg_input.pair_mask)
      self.assertAllEqual(
          pair_mask[self.mols['CID6344']][:3, :3],
          [[0, 0, 1],
           [0, 0, 1],
           [1, 1, 0]])

  def testMaxPairDistanceDefault(self):
    # Verifies that passing None is the same as giving the default value (-1)
    with self.test_session() as sess:
      keys, values = self.test_dataset
      mg_input = mgpo.MoleculeGraphParser(
          keys, values,
          self.max_atoms,
          max_pair_distance=None)
      mg_input2 = mgpo.MoleculeGraphParser(
          keys, values, self.max_atoms)
      pairs1, pairs2, pair_mask1, pair_mask2 = sess.run(
          [mg_input.pairs, mg_input2.pairs, mg_input.pair_mask,
           mg_input2.pair_mask])
      self.assertAllEqual(pairs1, pairs2)
      self.assertAllEqual(pair_mask1, pair_mask2)

  def testBadParse(self):
    # This is a very trimmed down value of something we actually encountered in
    # practice (where a text format proto was passed in instead of a wire format
    # one. Arbitrary strings don't work because when they misparse, they usually
    # leave the number of atoms as 0, which is caught as invalid.
    bad_input_value = '\n{           # \n}'
    with self.test_session() as sess:
      keys, values = self.test_dataset
      mod_values = values
      mod_values[-1] = bad_input_value
      mg_input = mgpo.MoleculeGraphParser(
          keys, mod_values,
          max_atoms=1)
      with self.assertRaisesOpError('Failed to parse MoleculeGraph'):
        sess.run(mg_input.atoms)

  def testOversizedMol(self):
    # Verifies that reading in molecules with more than max_atoms atoms doesn't
    # raise an error
    with self.test_session() as sess:
      keys, values = self.test_dataset
      mg_input = mgpo.MoleculeGraphParser(
          keys, values,
          max_atoms=1)
      atoms, pairs = sess.run([mg_input.atoms, mg_input.pairs])
      self.assertEqual(atoms.shape[:2], (self.batch_size, 1))
      self.assertEqual(pairs.shape[:3], (self.batch_size, 1, 1))

    # Verifies that reading in molecules with more than max_atoms and
    # allow_overflow=False raises an error
    with self.test_session() as sess:
      keys, values = self.test_dataset
      mg_input = mgpo.MoleculeGraphParser(
          keys, values,
          max_atoms=1,
          allow_overflow=False)
      with self.assertRaisesOpError(
          'CID1647 has 5 atoms, which is more than the max allowed'):
        sess.run(mg_input.atoms)


if __name__ == '__main__':
  tf.disable_eager_execution()
  unittest.main()
