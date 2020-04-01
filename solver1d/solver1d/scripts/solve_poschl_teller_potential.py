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

r"""Solves Poschl-Teller potential.

Poschl-Teller potential is a special class of potentials for which the
one-dimensional Schrodinger equation can be solved in terms of Special
functions.

https://en.wikipedia.org/wiki/P%C3%B6schl%E2%80%93Teller_potential

The general form of the potential is

v(x) = -\frac{\lambda(\lambda + 1)}{2} a^2 \frac{1}{\cosh^2(a x)}

It holds M=ceil(\lambda) levels, where \lambda is a positive float.

For example, \lambda=0.5 holds 1 electron. The exact eigen energy is -0.125.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import app
from absl import flags
from absl import logging

import numpy as np

from solver1d import single_electron


flags.DEFINE_enum('solver', 'EigenSolver',
                  ['EigenSolver', 'SparseEigenSolver'],
                  'Solver classes in single_electron module.')
flags.DEFINE_float(
    'lam', 0.5, 'Lambda in the Poschl-Teller potential function.')
flags.DEFINE_float(
    'scaling', 1.,
    'Scaling coefficient in the Poschl-Teller potential function.')
flags.DEFINE_float('grid_lower', -20.,
                   'Lower boundary of the range of the grid.')
flags.DEFINE_float(
    'grid_upper', 20., 'Upper boundary of the range of the grid.')
flags.DEFINE_integer('num_grids', 1001, 'Number of grid points.')
flags.DEFINE_integer(
    'num_electrons', 1, 'Number of electrons in the system.')


FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info(
      'Solve Poschl-Teller potential with lambda=%f, scaling=%f',
      FLAGS.lam, FLAGS.scaling)
  logging.info(
      'Number of electrons: %d', FLAGS.num_electrons)
  logging.info(
      'Grids: linspace(%f, %f, %d)',
      FLAGS.grid_lower, FLAGS.grid_upper, FLAGS.num_grids)

  exact_energy = single_electron.poschl_teller_eigen_energy(
      level=FLAGS.num_electrons, lam=FLAGS.lam)
  logging.info('Exact energy: %f', exact_energy)

  logging.info('Solve with solver: %s', FLAGS.solver)
  solver = getattr(single_electron, FLAGS.solver)(
      grids=np.linspace(FLAGS.grid_lower, FLAGS.grid_upper, FLAGS.num_grids),
      potential_fn=functools.partial(
          single_electron.poschl_teller, lam=FLAGS.lam),
      num_electrons=FLAGS.num_electrons)
  solver.solve_ground_state()
  logging.info('Numerical solution: %f', solver.total_energy)
  logging.info('Difference: %e', solver.total_energy - exact_energy)


if __name__ == '__main__':
  app.run(main)
