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

"""Run experiment."""

import os

from absl import app
from absl import flags
import gin
import tensorflow.compat.v1 as tf

from amortized_bo import controller
from amortized_bo import deep_evolution_solver  # pylint: disable=unused-import
from amortized_bo import simple_ising_model  # pylint: disable=unused-import
from amortized_bo import utils

flags.DEFINE_string('work_dir', '/tmp/amortized_bo/experiment',
                    'Root directory for writing logs/summaries/checkpoints.')

flags.DEFINE_multi_string('gin_files', [], 'List of paths to the config files.')

flags.DEFINE_multi_string('gin_bindings', [],
                          'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS


@gin.configurable
def experiment(work_dir,
               problem_cls=None,
               solver_cls=None,
               num_rounds=None,
               batch_size=None,
               seed=None):
  """Run experiment."""
  # Must be specified via Gin.
  assert problem_cls
  assert solver_cls
  assert num_rounds
  assert batch_size

  tf.gfile.MakeDirs(work_dir)
  if seed is not None:
    utils.set_seed(seed)
  print('Running experiment with %s on %s' % (problem_cls, solver_cls))
  problem = problem_cls()
  solver = solver_cls(problem.domain)
  population = controller.run(
      problem, solver, num_rounds=num_rounds, batch_size=batch_size)
  print('Writing output to %s/population.csv' % work_dir)
  population.to_csv(os.path.join(work_dir, 'population.csv'))
  return population


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)
  experiment(work_dir=FLAGS.work_dir)


if __name__ == '__main__':
  app.run(main)
