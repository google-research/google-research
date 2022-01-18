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

r"""Launcher for training original agent.

This file trains the initial (DQN / Rainbow) agent and keeps track
of the best states visited so far as potential candidates for resetting.


"""

from absl import app
from absl import flags
from dopamine.discrete_domains import run_experiment

import tensorflow as tf
from memento import run_experiment_from_checkpoint

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"third_party/py/dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_atari_environment.game_name="Pong"").')

FLAGS = flags.FLAGS


def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = run_experiment_from_checkpoint.create_runner_checkpoint(
      FLAGS.base_dir, run_experiment.create_agent, schedule='save_best')
  runner.run_experiment()

if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
