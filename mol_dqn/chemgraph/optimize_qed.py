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

from absl import app
from absl import flags

from rdkit import Chem

from rdkit.Chem import QED
from tensorflow import gfile
from mol_dqn.chemgraph.dqn import deep_q_networks
from mol_dqn.chemgraph.dqn import molecules as molecules_mdp
from mol_dqn.chemgraph.dqn import run_dqn
from mol_dqn.chemgraph.dqn.tensorflow_core import core

FLAGS = flags.FLAGS


class QEDRewardMolecule(molecules_mdp.Molecule):
  """The molecule whose reward is the QED."""

  def __init__(self, discount_factor, **kwargs):
    """Initializes the class.

    Args:
      discount_factor: Float. The discount factor. We only
        care about the molecule at the end of modification.
        In order to prevent a myopic decision, we discount
        the reward at each step by a factor of
        discount_factor ** num_steps_left,
        this encourages exploration with emphasis on long term rewards.
      **kwargs: The keyword arguments passed to the base class.
    """
    super(QEDRewardMolecule, self).__init__(**kwargs)
    self.discount_factor = discount_factor

  def _reward(self):
    """Reward of a state.

    Returns:
      Float. QED of the current state.
    """
    molecule = Chem.MolFromSmiles(self._state)
    if molecule is None:
      return 0.0
    qed = QED.qed(molecule)
    return qed * self.discount_factor ** (self.max_steps - self.num_steps_taken)


def main(argv):
  del argv  # unused.
  if FLAGS.hparams is not None:
    with gfile.Open(FLAGS.hparams, 'r') as f:
      hparams = deep_q_networks.get_hparams(**json.load(f))
  else:
    hparams = deep_q_networks.get_hparams()

  environment = QEDRewardMolecule(
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

  run_dqn.run_training(
      hparams=hparams,
      environment=environment,
      dqn=dqn)

  core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


if __name__ == '__main__':
  app.run(main)
