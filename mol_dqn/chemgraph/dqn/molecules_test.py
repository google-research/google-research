# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Tests for research.biology.chemgraph.mcts.molecules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow.compat.v1 as tf

from mol_dqn.chemgraph.dqn import molecules


class MoleculesTest(tf.test.TestCase):

  def test_empty_init(self):
    mol = molecules.Molecule({'C', 'O'})
    mol.initialize()
    self.assertSetEqual(mol.get_valid_actions(), {'C', 'O'})

  def test_empty_action(self):
    mol = molecules.Molecule({'C', 'O'})
    mol.initialize()
    result = mol.step('C')
    self.assertEqual(result.state, 'C')
    self.assertEqual(result.reward, 0)
    self.assertEqual(result.terminated, False)

  def test_benzene_init(self):
    mol = molecules.Molecule({'C', 'O'}, 'c1ccccc1')
    mol.initialize()
    self.assertSetEqual(
        mol.get_valid_actions(),
        {'Oc1ccccc1', 'c1ccccc1', 'Cc1ccccc1', 'c1cc2cc-2c1', 'c1cc2ccc1-2'})

  def test_benzene_action(self):
    mol = molecules.Molecule({'C', 'O'}, 'c1ccccc1')
    mol.initialize()
    result = mol.step('Cc1ccccc1')
    self.assertEqual(result.state, 'Cc1ccccc1')
    self.assertEqual(result.reward, 0)
    self.assertEqual(result.terminated, False)

  def test_ethane_init(self):
    mol = molecules.Molecule({'C', 'O'}, 'CC')
    mol.initialize()
    self.assertSetEqual(
        mol.get_valid_actions(),
        {'CC', 'C=C', 'CCC', 'C#CC', 'CCO', 'CC=O', 'C', 'C=CC', 'C#C'})

  def test_cyclobutane_init(self):
    # We want to know that it is possible to form another
    # ring when there is one ring present.
    mol = molecules.Molecule({'C', 'O'}, 'C1CCC1')
    mol.initialize()
    self.assertSetEqual(
        mol.get_valid_actions(), {
            'C1CCC1', 'C=C1CCC1', 'C1C2CC12', 'C1=CCC1', 'CCCC', 'O=C1CCC1',
            'CC1CCC1', 'OC1CCC1', 'C1#CCC1', 'C1C2=C1C2'
        })

  def test_do_not_allow_removal(self):
    mol = molecules.Molecule({'C', 'O'}, 'CC', allow_removal=False)
    mol.initialize()
    self.assertSetEqual(
        mol.get_valid_actions(),
        {'CC', 'CCC', 'C#CC', 'CCO', 'CC=O', 'C=CC', 'C=C', 'C#C'})

  def test_do_not_allow_no_modification(self):
    mol = molecules.Molecule({'C', 'O'}, 'C#C', allow_no_modification=False)
    mol.initialize()
    actions_noallow_no_modification = mol.get_valid_actions()
    mol = molecules.Molecule({'C', 'O'}, 'C#C', allow_no_modification=True)
    mol.initialize()
    actions_allow_no_modification = mol.get_valid_actions()
    self.assertSetEqual(
        {'C#C'},
        actions_allow_no_modification - actions_noallow_no_modification)

  def test_do_not_allow_bonding_between_rings(self):
    atom_types = {'C'}
    start_smiles = 'CC12CC1C2'
    mol = molecules.Molecule(
        atom_types, start_smiles, allow_bonds_between_rings=True)
    mol.initialize()
    actions_true = mol.get_valid_actions()
    mol = molecules.Molecule(
        atom_types, start_smiles, allow_bonds_between_rings=False)
    mol.initialize()
    actions_false = mol.get_valid_actions()

    self.assertSetEqual({'CC12C3C1C32', 'CC12C3=C1C32'},
                        actions_true - actions_false)

  def test_limited_ring_formation(self):
    atom_types = {'C'}
    start_smiles = 'CCCCC'
    mol = molecules.Molecule(
        atom_types, start_smiles, allowed_ring_sizes={3, 4, 5})
    mol.initialize()
    actions_allow_5_member_ring = mol.get_valid_actions()
    mol = molecules.Molecule(
        atom_types, start_smiles, allowed_ring_sizes={3, 4})
    mol.initialize()
    actions_do_not_allow_5_member_ring = mol.get_valid_actions()

    self.assertSetEqual(
        {'C1CCCC1', 'C1#CCCC1', 'C1=CCCC1'},
        actions_allow_5_member_ring - actions_do_not_allow_5_member_ring)

  def test_initialize(self):
    mol = molecules.Molecule({'C', 'O'}, 'c1ccccc1', record_path=True)
    mol.initialize()
    # Test if the molecule is correctly initialized.
    self.assertEqual(mol.state, 'c1ccccc1')
    self.assertEqual(mol.num_steps_taken, 0)
    self.assertListEqual(mol.get_path(), ['c1ccccc1'])
    # Take a step
    result = mol.step('Cc1ccccc1')
    self.assertEqual(result.state, 'Cc1ccccc1')
    self.assertEqual(result.reward, 0)
    self.assertListEqual(mol.get_path(), ['c1ccccc1', 'Cc1ccccc1'])
    # Test if the molecule is reset to its initial state.
    mol.initialize()
    self.assertEqual(mol.state, 'c1ccccc1')
    self.assertEqual(mol.num_steps_taken, 0)
    self.assertListEqual(mol.get_path(), ['c1ccccc1'])

  def test_state_transition(self):
    mol = molecules.Molecule({'C', 'O'}, 'c1ccccc1')
    mol.initialize()
    result = mol.step('Cc1ccccc1')
    self.assertEqual(result.state, 'Cc1ccccc1')
    self.assertEqual(result.reward, 0)
    self.assertEqual(result.terminated, False)
    self.assertEqual(mol.state, 'Cc1ccccc1')
    self.assertEqual(mol.num_steps_taken, 1)

  def test_invalid_actions(self):
    mol = molecules.Molecule({'C', 'O'}, 'c1ccccc1')
    mol.initialize()
    with self.assertRaisesRegexp(ValueError, 'Invalid action.'):
      mol.step('C')

  def test_episode_not_started(self):
    mol = molecules.Molecule({'C', 'O'}, 'c1ccccc1')
    with self.assertRaisesRegexp(ValueError, 'This episode is terminated.'):
      mol.step('Cc1ccccc1')

  def test_end_episode(self):
    mol = molecules.Molecule({'C', 'O'}, 'c1ccccc1', max_steps=3)
    mol.initialize()
    for _ in range(3):
      action = mol.get_valid_actions().pop()
      result = mol.step(action)
    self.assertEqual(result.terminated, True)
    with self.assertRaisesRegexp(ValueError, 'This episode is terminated.'):
      mol.step(mol.get_valid_actions().pop())

  def test_goal_settings(self):
    mol = molecules.Molecule(
        {'C', 'O'}, 'c1ccccc1', target_fn=lambda x: x == 'Cc1ccccc1')
    mol.initialize()
    result = mol.step('Cc1ccccc1')
    self.assertEqual(result.state, 'Cc1ccccc1')
    self.assertEqual(result.reward, 0)
    self.assertEqual(result.terminated, True)
    with self.assertRaisesRegexp(ValueError, 'This episode is terminated.'):
      mol.step(mol.get_valid_actions().pop())

  def test_reward_settings(self):

    class TargetedMolecule(molecules.Molecule):

      def _reward(self):
        return int(self._state == 'Cc1ccccc1')

    mol = TargetedMolecule({'C', 'O'}, 'c1ccccc1')
    mol.initialize()
    result = mol.step('Cc1ccccc1')
    self.assertEqual(result.state, 'Cc1ccccc1')
    self.assertEqual(result.reward, 1)
    self.assertEqual(result.terminated, False)

  def test_image_generation(self):
    mol = molecules.Molecule({'C', 'O'}, 'c1ccccc1', max_steps=3)
    mol.initialize()
    image = mol.visualize_state()
    del image

  def test_record(self):
    mol = molecules.Molecule({'C', 'O'}, 'c1ccccc1', record_path=True)
    mol.initialize()
    mol.step('Cc1ccccc1')
    mol.step('CCc1ccccc1')
    mol.step('Cc1ccccc1')
    mol.step('c1ccccc1')
    self.assertListEqual(
        mol.get_path(),
        ['c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1', 'Cc1ccccc1', 'c1ccccc1'])

  def test_more_than_three_possible_bonds(self):
    mol = molecules.Molecule({'C', 'S'})
    mol.initialize()
    mol.step('C')


if __name__ == '__main__':
  tf.test.main()
