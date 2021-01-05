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

# Lint as: python2, python3
"""Tests for research.biology.chemgraph.py.molecules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from rdkit import Chem

import tensorflow.compat.v1 as tf
from mol_dqn.chemgraph.dqn.py import molecules


class MoleculesTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(MoleculesTest, self).setUp()
    self.atom_types = ['H', 'C', 'N', 'O', 'F']

  def test_atom_valences(self):
    self.assertEqual([1, 4, 3, 2, 1], molecules.atom_valences(self.atom_types))

  def test_get_scaffold(self):
    self.assertEqual('', molecules.get_scaffold(Chem.MolFromSmiles('CCO')))
    self.assertEqual('c1ccccc1',
                     molecules.get_scaffold(Chem.MolFromSmiles('CC1=CC=CC=C1')))
    # Note that the carbonyl is included in the scaffold...
    self.assertEqual(
        'O=C(c1ccccc1)c1ccccc1',
        molecules.get_scaffold(
            Chem.MolFromSmiles('C1=CC=C(C=C1C(C2=CC(=CC=C2)C(C)C)=O)C(C)C')))
    # ...but the hydroxyl is not.
    self.assertEqual(
        'c1ccc(Cc2ccccc2)cc1',
        molecules.get_scaffold(
            Chem.MolFromSmiles('C1=CC=C(C=C1C(C2=CC(=CC=C2)C(C)C)O[H])C(C)C')))

  @parameterized.parameters(
      ('c1ccccc1', 'c1ccccc1', True),
      ('c1ccccc1CC', 'c1ccccc1', True),
      ('C1=CC=C2C=CC=CC2=C1', 'c1ccccc1', True),
      ('C1=CC=CC=C1CCC2=CC=CC=C2', 'c1ccccc1', True),
      ('C1CCCCC1', 'c1ccccc1', False),
      ('C1CCCC1', 'c1ccccc1', False),
  )
  def test_contains_scaffold(self, smiles, scaffold, expected):
    self.assertEqual(
        expected,
        molecules.contains_scaffold(Chem.MolFromSmiles(smiles), scaffold))

  @parameterized.parameters(('C1CCC1', 4), ('C1CCCCC1', 6), ('C1CCCCCC1', 7),
                            ('c2ccc1ccccc1c2', 6), ('C2CCC1CCCC1CC2', 7),
                            ('CC1CC2CCCCCC3CC(C1)C23', 8))
  def test_get_ring_size(self, mol, size):
    mol = Chem.MolFromSmiles(mol)
    self.assertEqual(molecules.get_largest_ring_size(mol), size)

  @parameterized.parameters(
      ('C1CCC1', 0.5604),
      ('C1CCCCC1', 1.3406),
      ('C1CCCCCC1', 0.7307),
      # Make sure they are consistent with reported values.
      # https://github.com/wengong-jin/icml18-jtnn/blob/master/data/opt.test.logP-SA
      ('COc1cc2c(cc1OC)CC([NH3+])C2', -2.50504567445),
      ('OC[C@@H](Br)C(F)(F)Br', -2.45357941743),
      ('NC(=O)C1(N2CCCC2)CC[NH2+]CC1', -5.375513278))
  def test_penalized_logp(self, mol, score):
    mol = Chem.MolFromSmiles(mol)
    self.assertAlmostEqual(molecules.penalized_logp(mol), score)


if __name__ == '__main__':
  tf.test.main()
