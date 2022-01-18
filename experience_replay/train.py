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

r"""Train an agent.

"""
import os


from absl import app
from absl import flags

from dopamine.discrete_domains import run_experiment
import tensorflow.compat.v1 as tf

from experience_replay import run_experience_replay_experiment


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
flags.DEFINE_string(
    'schedule', 'continuous_train_and_eval',
    'The schedule with which to run the experiment and choose an appropriate '
    'Runner. Supported choices are '
    '{continuous_train, eval, continuous_train_and_eval}.')


FLAGS = flags.FLAGS



def create_runner(base_dir, create_agent_fn,
                  schedule='continuous_train_and_eval'):
  """Creates an experiment Runner.

  TODO(b/): Figure out the right idiom to create a Runner. The current mechanism
  of using a number of flags will not scale and is not elegant.

  Args:
    base_dir: Base directory for hosting all subdirectories.
    create_agent_fn: A function that takes as args a Tensorflow session and a
     Gym Atari 2600 environment, and returns an agent.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `run_experiment.Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None

  # Continuously runs training and eval till max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return run_experience_replay_experiment.ElephantRunner(
        base_dir, create_agent_fn)

  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


def launch_experiment(create_runner_fn, create_agent_fn):
  """Launches the experiment.

  Args:
    create_runner_fn: A function that takes as args a base directory and a
      function for creating an agent and returns a `Runner` like object.
    create_agent_fn: A function that takes as args a Tensorflow session and a
      Gym environment, and returns an agent.
  """
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = create_runner_fn(FLAGS.base_dir, create_agent_fn,
                            schedule=FLAGS.schedule)
  runner.run_experiment()


def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  launch_experiment(create_runner,
                    run_experience_replay_experiment.create_agent)

if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
