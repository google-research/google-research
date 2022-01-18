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

# Lint as: python2, python3
"""Optimizes QED of a molecule with DQN.

This experiment tries to find the molecule with the highest QED
starting from a given molecule.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

from absl import app
from absl import flags

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from mol_dqn.chemgraph.dqn import deep_q_networks
from mol_dqn.chemgraph.dqn import molecules as molecules_mdp
from mol_dqn.chemgraph.dqn import run_dqn
from mol_dqn.chemgraph.dqn.tensorflow_core import core

FLAGS = flags.FLAGS


class MultiObjectiveRewardMolecule(molecules_mdp.Molecule):
  """Defines the subclass of generating a molecule with a specific reward.

  The reward is defined as a scalar
    reward = weight * similarity_score + (1 - weight) *  qed_score
  """

  def __init__(self, target_molecule, similarity_weight, discount_factor,
               **kwargs):
    """Initializes the class.

    Args:
      target_molecule: SMILES string. The target molecule against which we
        calculate the similarity.
      similarity_weight: Float. The weight applied similarity_score.
      discount_factor: Float. The discount factor applied on reward.
      **kwargs: The keyword arguments passed to the parent class.
    """
    super(MultiObjectiveRewardMolecule, self).__init__(**kwargs)
    target_molecule = Chem.MolFromSmiles(target_molecule)
    self._target_mol_fingerprint = self.get_fingerprint(target_molecule)
    self._sim_weight = similarity_weight
    self._discount_factor = discount_factor

  def get_fingerprint(self, molecule):
    """Gets the morgan fingerprint of the target molecule.

    Args:
      molecule: Chem.Mol. The current molecule.

    Returns:
      rdkit.ExplicitBitVect. The fingerprint of the target.
    """
    return AllChem.GetMorganFingerprint(molecule, radius=2)

  def get_similarity(self, smiles):
    """Gets the similarity between the current molecule and the target molecule.

    Args:
      smiles: String. The SMILES string for the current molecule.

    Returns:
      Float. The Tanimoto similarity.
    """

    structure = Chem.MolFromSmiles(smiles)
    if structure is None:
      return 0.0
    fingerprint_structure = self.get_fingerprint(structure)

    return DataStructs.TanimotoSimilarity(self._target_mol_fingerprint,
                                          fingerprint_structure)

  def _reward(self):
    """Calculates the reward of the current state.

    The reward is defined as a tuple of the similarity and QED value.

    Returns:
      A tuple of the similarity and qed value
    """
    # calculate similarity.
    # if the current molecule does not contain the scaffold of the target,
    # similarity is zero.
    if self._state is None:
      return 0.0
    mol = Chem.MolFromSmiles(self._state)
    if mol is None:
      return 0.0
    similarity_score = self.get_similarity(self._state)
    # calculate QED
    qed_value = QED.qed(mol)
    reward = (
        similarity_score * self._sim_weight +
        qed_value * (1 - self._sim_weight))
    discount = self._discount_factor**(self.max_steps - self._counter)
    return reward * discount


def main(argv):
  del argv  # unused.
  if FLAGS.hparams is not None:
    with open(FLAGS.hparams, 'r') as f:
      hparams = deep_q_networks.get_hparams(**json.load(f))
  else:
    hparams = deep_q_networks.get_hparams()

  environment = MultiObjectiveRewardMolecule(
      target_molecule=FLAGS.target_molecule,
      similarity_weight=FLAGS.similarity_weight,
      discount_factor=hparams.discount_factor,
      atom_types=set(hparams.atom_types),
      init_mol=FLAGS.start_molecule,
      allow_removal=hparams.allow_removal,
      allow_no_modification=hparams.allow_no_modification,
      allow_bonds_between_rings=hparams.allow_bonds_between_rings,
      allowed_ring_sizes=set(hparams.allowed_ring_sizes),
      max_steps=hparams.max_steps_per_episode)

  dqn = deep_q_networks.DeepQNetwork(
      input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
      q_fn=functools.partial(
          deep_q_networks.multi_layer_model, hparams=hparams),
      optimizer=hparams.optimizer,
      grad_clipping=hparams.grad_clipping,
      num_bootstrap_heads=hparams.num_bootstrap_heads,
      gamma=hparams.gamma,
      epsilon=1.0)

  run_dqn.run_training(hparams=hparams, environment=environment, dqn=dqn)

  core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


if __name__ == '__main__':
  app.run(main)
