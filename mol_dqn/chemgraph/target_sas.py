# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Generates molecules whose SA score stays with in a range."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
from absl import app
from absl import flags
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

from mol_dqn.chemgraph.dqn import deep_q_networks
from mol_dqn.chemgraph.dqn import molecules as molecules_mdp
from mol_dqn.chemgraph.dqn import run_dqn
from mol_dqn.chemgraph.dqn.tensorflow_core import core

flags.DEFINE_float('target_sas', 2.5,
                   'The target synthetic accessibility value')
flags.DEFINE_string('loss_type', 'l2', 'The loss type')
FLAGS = flags.FLAGS


class TargetSASMolecule(molecules_mdp.Molecule):
  """Target SAS reward Molecule."""

  def __init__(self, discount_factor, target_sas, loss_type, **kwargs):
    """Initializes the class.

    Args:
      discount_factor: Float. The discount factor. We only care about the
        molecule at the end of modification. In order to prevent a myopic
        decision, we discount the reward at each step by a factor of
        discount_factor ** num_steps_left, this encourages exploration with
        emphasis on long term rewards.
      target_sas: Float. Target synthetic accessibility value.
      loss_type: String. 'l2' for l2 loss, 'l1' for l1 loss.
      **kwargs: The keyword arguments passed to the base class.
    """
    super(TargetSASMolecule, self).__init__(**kwargs)
    self.discount_factor = discount_factor
    self.target_sas = target_sas
    if loss_type == 'l1':
      self.loss_fn = abs
    elif loss_type == 'l2':
      self.loss_fn = lambda x: x**2
    else:
      raise ValueError('loss_type must by "l1" or "l2"')

  def _reward(self):
    molecule = Chem.MolFromSmiles(self._state)
    if molecule is None:
      return -self.loss_fn(self.target_sas)
    sas = sascorer.calculateScore(molecule)
    return -self.loss_fn(sas - self.target_sas) * (
        self.discount_factor**(self.max_steps - self.num_steps_taken))


def main(argv):
  del argv
  if FLAGS.hparams is not None:
    with open(FLAGS.hparams, 'r') as f:
      hparams = deep_q_networks.get_hparams(**json.load(f))
  else:
    hparams = deep_q_networks.get_hparams()

  environment = TargetSASMolecule(
      discount_factor=hparams.discount_factor,
      target_sas=FLAGS.target_sas,
      loss_type=FLAGS.loss_type,
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
      dqn=dqn,
  )

  core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


if __name__ == '__main__':
  app.run(main)
