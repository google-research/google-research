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

# Lint as: python3
r"""Launch group testing simulator.

For example: (use verbosity 1 for debug, 0 for info, -1 for warning, etc.)

python3 -m run_experiment --gin_config=configs/toy.gin --verbosity=1
"""
from absl import app
from absl import flags
import gin

from grouptesting import simulator


flags.DEFINE_string('base_dir', None, 'Directory to save data to.')
flags.DEFINE_integer('seed', None, 'Random seed to be used in this run.')
flags.DEFINE_multi_string('gin_config', [],
                          'List of paths to the config files.')
flags.DEFINE_multi_string('gin_bindings', [],
                          'Newline separated list of Gin parameter bindings.')
flags.DEFINE_bool('interactive_mode',
                  True,
                  'Run in interactive mode to let the user enters the positive '
                  'groups at each test cycle. Use --nointeractive_mode to '
                  'disable it.')
FLAGS = flags.FLAGS


def main(unused_argv):
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  sim = simulator.Simulator(workdir=FLAGS.base_dir)
  run_fn = sim.run if not FLAGS.interactive_mode else sim.interactive_loop
  run_fn(rngkey=FLAGS.seed)


if __name__ == '__main__':
  app.run(main)
