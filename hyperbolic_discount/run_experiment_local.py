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

from absl import flags
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import run_experiment

import gin


from hyperbolic_discount import hyperbolic_dqn_agent
from hyperbolic_discount import hyperbolic_rainbow_agent

FLAGS = flags.FLAGS


def create_agent(sess, environment, summary_writer=None):
  """Creates a DQN agent.

  Args:
    sess: A `tf.Session`object  for running associated ops.
    environment: An Atari 2600 environment.
    summary_writer: A Tensorflow summary writer to pass to the agent for
      in-agent training statistics in Tensorboard.

  Returns:
    A DQN or SARSA agent.
  """
  if FLAGS.agent_name == 'hyperbolic_dqn':
    return hyperbolic_dqn_agent.HyperDQNAgent(
        sess,
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif FLAGS.agent_name == 'hyperbolic_rainbow':
    return hyperbolic_rainbow_agent.HyperRainbowAgent(
        sess,
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)


@gin.configurable()
def create_runner(base_dir,
                  create_agent_fn,
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
    return run_experiment.Runner(base_dir, create_agent_fn,
                                 atari_lib.create_atari_environment)
  # Continuously runs training till maximum num_iterations is hit.
  elif schedule == 'continuous_train':
    return run_experiment.TrainRunner(base_dir, create_agent_fn,
                                      atari_lib.create_atari_environment)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))
