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

"""Maximizes the QED of the molecule while keep similarity.

Multi-Objective optimization using single Q function.
  Obj1: QED;
  Obj2: similarity.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
from absl import app
from absl import flags
import networkx as nx
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Contrib import SA_Score
from tensorflow.compat.v1 import gfile
from mol_dqn.chemgraph.mcts import deep_q_networks
from mol_dqn.chemgraph.mcts import molecules as molecules_mdp
from mol_dqn.chemgraph.mcts import run_dqn
from mol_dqn.chemgraph.tensorflow import core


flags.DEFINE_float('target_logp', 3.0, 'The target logP value')
flags.DEFINE_float('gamma', 0.999, 'discount')
FLAGS = flags.FLAGS


def num_long_cycles(mol):
  """Calculate the number of long cycles.

  Args:
    mol: Molecule. A molecule.

  Returns:
    negative cycle length.
  """
  cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
  if not cycle_list:
    cycle_length = 0
  else:
    cycle_length = max([len(j) for j in cycle_list])
  if cycle_length <= 6:
    cycle_length = 0
  else:
    cycle_length = cycle_length - 6
  return -cycle_length


def penalized_logp(molecule):
  log_p = Descriptors.MolLogP(molecule)
  sas_score = SA_Score.sascorer.calculateScore(molecule)
  cycle_score = num_long_cycles(molecule)
  return log_p - sas_score + cycle_score


class Molecule(molecules_mdp.Molecule):
  """Penalized logP molecule."""

  def __init__(self, target_molecule, **kwargs):
    """Initializes the class.

    Args:
      target_molecule: SMILES string. the target molecule against which we
        calculate the similarity.
      **kwargs: The keyword arguments passed to the parent class.
    """
    super(Molecule, self).__init__(**kwargs)
    target_molecule = Chem.MolFromSmiles(target_molecule)
    self._target_mol_fingerprint = self.get_fingerprint(target_molecule)

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
    molecule = Chem.MolFromSmiles(self._state)
    if molecule is None:
      return 0.0
    log_p = Descriptors.MolLogP(molecule)
    sas_score = SA_Score.sascorer.calculateScore(molecule)
    cycle_score = num_long_cycles(molecule)
    logp_score = -abs(log_p - sas_score + cycle_score - FLAGS.target_logp)
    similarity_score = self.get_similarity(molecule)
    return (FLAGS.similarity_weight * similarity_score +
            (1 - FLAGS.similarity_weight) * logp_score) * FLAGS.gamma**(
                self.max_steps - self._counter)


def main(argv):
  del argv
  if FLAGS.hparams is not None:
    with gfile.Open(FLAGS.hparams, 'r') as f:
      hparams = deep_q_networks.get_hparams(**json.load(f))
  else:
    hparams = deep_q_networks.get_hparams()

  environment = Molecule(
      target_molecule=FLAGS.start_molecule,
      atom_types=set(hparams.atom_types),
      init_mol=FLAGS.start_molecule,
      allow_removal=hparams.allow_removal,
      allow_no_modification=hparams.allow_no_modification,
      allow_bonds_between_rings=hparams.allow_bonds_between_rings,
      allowed_ring_sizes=set(hparams.allowed_ring_sizes),
      max_steps=hparams.max_steps_per_episode)

  dqn = deep_q_networks.DeepQNetwork(
      input_shape=(hparams.batch_size, hparams.fingerprint_length),
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
      dqn=dqn,
  )

  hparams.add_hparam('start_molecule', FLAGS.start_molecule)
  hparams.add_hparam('target_molecule', FLAGS.start_molecule)
  core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


if __name__ == '__main__':
  app.run(main)
