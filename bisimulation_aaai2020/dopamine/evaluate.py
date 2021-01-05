# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

r"""The entry point for making a trained Rainbow agent play an Atari game.

Sample run:
  ```
  GAME=SpaceInvaders
  python -m bisimulation_aaai2020/dopamine/evaluate \
      --base_dir=/tmp/dopamine/evals/${GAME} \
      --metric_checkpoint=/tmp/dopamine/trained_metrics/${GAME}
  ```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
from absl import flags

from dopamine.discrete_domains import run_experiment as dopamine_run_experiment

import tensorflow.compat.v1 as tf
from bisimulation_aaai2020.dopamine import run_experiment


flags.DEFINE_integer('num_global_steps', 2000, 'Number of episodes to run.')
flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_string('metric_checkpoint', None,
                    'Path to agent checkpoint with metric.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')

FLAGS = flags.FLAGS




def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  gin_file = 'bisimulation_aaai2020/dopamine/configs/rainbow.gin'
  dopamine_run_experiment.load_gin_configs([gin_file], FLAGS.gin_bindings)
  gin_binding = 'BisimulationRainbowAgent.evaluate_metric_only=True'
  FLAGS.gin_bindings.append(gin_binding)
  runner = run_experiment.create_runner(FLAGS.base_dir,
                                        FLAGS.metric_checkpoint)
  runner.visualize(num_global_steps=FLAGS.num_global_steps)


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
