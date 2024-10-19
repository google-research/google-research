# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

r"""Launcher for running Memento agent.

This file should be run *after* the original agent has been trained
using train_original_agent.py.

The following is a sample command to run the Memento agent after the
original agent has been trained using the sample command given in
train_original_agent.py


"""

from absl import app
from absl import flags
from dopamine.discrete_domains import run_experiment

import tensorflow as tf
from memento import run_experiment_from_checkpoint

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_string('original_base_dir', None,
                    'Base directory for original agent (for Memento).')
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
  del unused_argv
  gin_bindings = FLAGS.gin_bindings + \
      ['LoadFromRunner.original_base_dir="{}"'.format(FLAGS.original_base_dir)]

  tf.logging.set_verbosity(tf.logging.INFO)
  run_experiment.load_gin_configs(FLAGS.gin_files, gin_bindings)
  runner = run_experiment_from_checkpoint.create_runner_checkpoint(
      FLAGS.base_dir, run_experiment.create_agent, schedule='load_from_best')
  runner.run_experiment()

if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  flags.mark_flag_as_required('original_base_dir')
  app.run(main)
