# coding=utf-8
# Copyright 2018 The Google Research Authors.
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
import random

from absl import app
from absl import flags
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tensorflow import gfile

from mol_dqn.chemgraph.mcts import deep_q_networks
from mol_dqn.chemgraph.mcts import molecules as molecules_mdp
from mol_dqn.chemgraph.mcts import run_dqn
from mol_dqn.chemgraph.py import molecules
from mol_dqn.chemgraph.tensorflow import core


flags.DEFINE_float(
    'similarity_constraint', 0.0, 'The constraint of similarity.'
    'The similarity of the generated molecule must'
    'greater than this constraint')
FLAGS = flags.FLAGS


class LogPRewardWithSimilarityConstraintMolecule(molecules_mdp.Molecule):
  """The molecule whose reward is the penalized logP with similarity constraint.

  Each time the environment is initialized, we uniformly choose
    a molecule from all molecules as target.
  """

  def __init__(self, all_molecules, discount_factor, similarity_constraint,
               **kwargs):
    """Initializes the class.

    Args:
      all_molecules: List of SMILES string. the molecules to select
      discount_factor: Float. The discount factor.
      similarity_constraint: Float. The lower bound of similarity of the
        molecule must satisfy.
      **kwargs: The keyword arguments passed to the parent class.
    """
    super(LogPRewardWithSimilarityConstraintMolecule, self).__init__(**kwargs)
    self._all_molecules = all_molecules
    self._discount_factor = discount_factor
    self._similarity_constraint = similarity_constraint
    self._target_mol_fingerprint = None

  def initialize(self):
    """Resets the MDP to its initial state.

    Each time the environment is initialized, we uniformly choose
    a molecule from all molecules as target.
    """
    self._state = random.choice(self._all_molecules)
    self._target_mol_fingerprint = self.get_fingerprint(
        Chem.MolFromSmiles(self._state))
    if self.record_path:
      self._path = [self._state]
    self._valid_actions = self.get_valid_actions(force_rebuild=True)
    self._counter = 0

  def get_fingerprint(self, molecule):
    """Gets the morgan fingerprint of the target molecule.

    Args:
      molecule: Chem.Mol. The current molecule.

    Returns:
      rdkit.ExplicitBitVect. The fingerprint of the target.
    """
    return AllChem.GetMorganFingerprint(molecule, radius=2)

  def get_similarity(self, molecule):
    """Gets the similarity between the current molecule and the target molecule.

    Args:
      molecule: String. The SMILES string for the current molecule.

    Returns:
      Float. The Tanimoto similarity.
    """

    fingerprint_structure = self.get_fingerprint(molecule)
    return DataStructs.TanimotoSimilarity(self._target_mol_fingerprint,
                                          fingerprint_structure)

  def _reward(self):
    """Reward of a state.

    If the similarity constraint is not satisfied,
    the reward is decreased by the difference times a large constant
    If the similarity constrain is satisfied,
    the reward is the penalized logP of the molecule.

    Returns:
      Float. The reward.

    Raises:
      ValueError: if the current state is not a valid molecule.
    """
    molecule = Chem.MolFromSmiles(self._state)
    if molecule is None:
      raise ValueError('Current state %s is not a valid molecule' % self._state)
    similarity = self.get_similarity(molecule)
    if similarity <= self._similarity_constraint:
      # 40 is an arbitrary number. Suppose we have a molecule that is not
      # similar to the target at all, but has a high logP. The logP improvement
      # can be 20, and the similarity difference can be 0.5. To discourage that
      # molecule, similarity difference is timed by 20 / 0.5 = 40.
      reward = molecules.penalized_logp(molecule) + 40 * (
          similarity - self._similarity_constraint)
    else:
      reward = molecules.penalized_logp(molecule)
    return reward * self._discount_factor**(self.max_steps - self._counter)


def main(argv):
  del argv  # unused.
  if FLAGS.hparams is not None:
    with gfile.Open(FLAGS.hparams, 'r') as f:
      hparams = deep_q_networks.get_hparams(**json.load(f))
  else:
    hparams = deep_q_networks.get_hparams()

  filename = 'all_800_mols.json'
  with gfile.Open(filename) as fp:
    all_molecules = json.load(fp)

  environment = LogPRewardWithSimilarityConstraintMolecule(
      similarity_constraint=FLAGS.similarity_constraint,
      discount_factor=hparams.discount_factor,
      all_molecules=all_molecules,
      atom_types=set(hparams.atom_types),
      init_mol=None,
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

  run_dqn.run_training(
      hparams=hparams,
      environment=environment,
      dqn=dqn,)

  core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


if __name__ == '__main__':
  app.run(main)
