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

"""Generates molecules that satisfy two targets.

Target1: SAS
Target2: QED
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
from rdkit.Chem import QED
from rdkit.Contrib import SA_Score
from tensorflow.compat.v1 import gfile
from mol_dqn.chemgraph.mcts import deep_q_networks
from mol_dqn.chemgraph.mcts import molecules as molecules_mdp
from mol_dqn.chemgraph.mcts import run_dqn
from mol_dqn.chemgraph.tensorflow import core


flags.DEFINE_float('target_sas', 1, 'The target SAS of the molecule.')
flags.DEFINE_float('target_qed', 0.5, 'The target QED of the molecule.')
flags.DEFINE_float('gamma', 0.999, 'discount')
FLAGS = flags.FLAGS


class MultiObjectiveRewardMolecule(molecules_mdp.Molecule):
  """Defines the subclass of generating a molecule with a specific reward.

  The reward is defined as a 1-D vector with 2 entries: similarity and QED
    reward = (similarity_score, qed_score)
  """

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
      return 0.0, 0.0
    mol = Chem.MolFromSmiles(self._state)
    if mol is None:
      return 0.0, 0.0

    qed_value = QED.qed(mol)
    sas = SA_Score.sascorer.calculateScore(mol)
    return -abs(sas - FLAGS.target_sas), -abs(qed_value - FLAGS.target_qed)


def soft_cst(v, l, r):
  if l <= v <= r:
    return 1
  return -min(abs(l - v), abs(r - v))


class Molecule(molecules_mdp.Molecule):
  """SAS and QED reward molecule."""

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
      return 0.0, 0.0
    mol = Chem.MolFromSmiles(self._state)
    if mol is None:
      return 0.0, 0.0

    qed_value = QED.qed(mol)
    sas = SA_Score.sascorer.calculateScore(mol)

    # c1 = soft_cst(sas, FLAGS.target_sas - 0.2, FLAGS.target_sas + 0.2)
    # c2 = soft_cst(qed_value, FLAGS.target_qed - 0.1, FLAGS.target_qed + 0.1)
    # # if c1 < 0 and c2 < 0:
    # #   return - c1 * c2
    # # else:
    # #   return c1 * c2
    return (soft_cst(sas, FLAGS.target_sas - 0.2, FLAGS.target_sas + 0.2) +
            soft_cst(qed_value, FLAGS.target_qed - 0.1,
                     FLAGS.target_qed + 0.1)) * FLAGS.gamma**(
                         self.max_steps - self._counter)


def main(argv):
  del argv
  if FLAGS.hparams is not None:
    with gfile.Open(FLAGS.hparams, 'r') as f:
      hparams = deep_q_networks.get_hparams(**json.load(f))
  else:
    hparams = deep_q_networks.get_hparams()

  hparams.add_hparam('target_qed', FLAGS.target_qed)
  hparams.add_hparam('target_sas', FLAGS.target_sas)

  environment = Molecule(
      atom_types=set(hparams.atom_types),
      init_mol='CCc1c(C)[nH]c2CCC(CN3CCOCC3)C(=O)c12',
      allow_removal=hparams.allow_removal,
      allow_no_modification=hparams.allow_no_modification,
      allow_bonds_between_rings=False,
      allowed_ring_sizes={3, 4, 5, 6},
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
      dqn=dqn,
  )

  core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


if __name__ == '__main__':
  app.run(main)
