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

"""Solves a non-interacting system."""

from absl import app
from absl import flags
from absl import logging

import jax
from jax.config import config
from jax_dft import scf
from jax_dft import utils
import numpy as np

# Set the default dtype as float64
config.update('jax_enable_x64', True)

flags.DEFINE_integer(
    'num_electrons', 1, 'Number of electrons in the system.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('JAX devices: %s', jax.devices())

  grids = np.arange(-256, 257) * 0.08
  external_potential = utils.get_atomic_chain_potential(
      grids=grids,
      locations=np.array([-0.8, 0.8]),
      nuclear_charges=np.array([1., 1.]),
      interaction_fn=utils.exponential_coulomb)

  density, total_eigen_energies, _ = scf.solve_noninteracting_system(
      external_potential, num_electrons=FLAGS.num_electrons, grids=grids)
  logging.info('density: %s', density)
  logging.info('total energy: %f', total_eigen_energies)


if __name__ == '__main__':
  app.run(main)
