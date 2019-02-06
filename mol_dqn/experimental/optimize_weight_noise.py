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

"""Optimize weight with noise."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

from absl import app
from absl import flags
import numpy as np
from rdkit import Chem

from rdkit.Chem import Descriptors
from tensorflow import gfile
from mol_dqn.chemgraph.mcts import deep_q_networks_noise
from mol_dqn.chemgraph.mcts import molecules as molecules_mdp
from mol_dqn.chemgraph.mcts import run_dqn
from mol_dqn.chemgraph.tensorflow import core

flags.DEFINE_string('error_type', 'robust', 'error_type.')
flags.DEFINE_float('noise_std', 0.1, 'std dev of noise.')
flags.DEFINE_float('gamma', 0.999, 'discount')
FLAGS = flags.FLAGS


class Molecule(molecules_mdp.Molecule):
  """Defines the subclass of a molecule MDP with a target molecular weight."""

  def __init__(self, target_weight, **kwargs):
    """Initializes the class.

    Args:
      target_weight: Float. the target molecular weight.
      **kwargs: The keyword arguments passed to the parent class.
    """
    super(Molecule, self).__init__(**kwargs)
    self.target_weight = target_weight

  def _reward(self):
    """Calculates the reward of the current state.

    The reward is defined as the negative l2 distance between the current
    molecular weight and target molecular weight.

    Returns:
      Float. The negative distance.
    """
    factor = 1.0 + np.random.normal(0, FLAGS.noise_std)
    if self._counter == self.max_steps:
      factor = 1.0
    factor *= FLAGS.gamma**(self.max_steps - self._counter)
    molecule = Chem.MolFromSmiles(self._state)
    if molecule is None:
      return -self.target_weight**2 * factor
    return -(Descriptors.MolWt(molecule) - self.target_weight)**2 * factor


def main(argv):
  del argv
  if FLAGS.hparams is not None:
    with gfile.Open(FLAGS.hparams, 'r') as f:
      hparams = run_dqn.get_hparams(**json.load(f))
  else:
    hparams = run_dqn.get_hparams()

  environment = Molecule(
      target_weight=FLAGS.target_weight,
      atom_types=set(hparams.atom_types),
      init_mol=FLAGS.start_molecule,
      allow_removal=hparams.allow_removal,
      allow_no_modification=hparams.allow_no_modification,
      max_steps=hparams.max_steps_per_episode)

  if FLAGS.error_type.lower() == 'l2':
    klass = deep_q_networks_noise.DeepQNetworkL2
  else:
    klass = deep_q_networks_noise.DeepQNetwork
  dqn = klass(
      input_shape=(hparams.batch_size, hparams.fingerprint_length),
      q_fn=functools.partial(
          deep_q_networks_noise.multi_layer_model, hparams=hparams),
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

  hparams.add_hparam('noise_std', FLAGS.noise_std)
  hparams.add_hparam('error_type', FLAGS.error_type)
  core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


if __name__ == '__main__':
  app.run(main)
