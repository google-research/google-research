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

"""A training routine for the `sail_rl` agents.

The script is adapted from:
https://github.com/google-research/google-research/tree/master/munchausen_rl.
"""

import datetime
import os
import random
from absl import flags

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment

import gin.tf
import numpy as np
import tensorflow.compat.v1 as tf

from sail_rl.agents import al_dqn
from sail_rl.agents import al_iqn
from sail_rl.agents import sail_dqn
from sail_rl.agents import sail_iqn

tf.disable_v2_behavior()

flags.DEFINE_string('workdir', None, """Working directory.""")
flags.DEFINE_string('prefix', '', 'workdir prefix.')
flags.DEFINE_string(
    'agent_type', 'sail_dqn', """Type of agent to train.
     Valid values: dqn, iqn, sail_dqn, sail_iqn, al_dqn, al_iqn.""")
flags.DEFINE_string('env', 'atari',
                    'Training environment. Valid values are: `gym`, `atari`.')
flags.DEFINE_string(
    'game', 'Pong',
    """Which game to train on. Valid values: any Atari games for the Atari env,
    or `CartPole` for the gym env.""")
flags.DEFINE_string(
    'gin_file', '',
    'Paths to gin configuration file (e.g.`sail_rl/configs/atari_dqn.gin`).')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files')
flags.DEFINE_string('gym_version', 'v0', 'Gym version')
flags.DEFINE_integer('seed', None,
                     'Random generator seed. If None, the seed is random.')
flags.DEFINE_bool(
    'set_seed', False,
    'If False, do not set the seed (seed is used as a run number).')

FLAGS = flags.FLAGS


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in the
      config files.
  """
  gin.parse_config_files_and_bindings(
      gin_files, bindings=gin_bindings, skip_unknown=False)


def parse_flags():
  """Parse the flags."""
  agent_type = FLAGS.agent_type
  date = datetime.datetime.now()
  run_name = str(date).replace(' ', '_')
  run_name = '%s_' % agent_type + run_name


  log_dir = os.path.join(FLAGS.workdir, run_name)
  gin_file = FLAGS.gin_file

  if FLAGS.env == 'atari':
    def create_atari_env_fn():
      """Creates the appropriate atari environement."""
      return atari_lib.create_atari_environment(FLAGS.game)
    create_env_fn = create_atari_env_fn
  elif FLAGS.env == 'gym':
    def create_gym_env_fn():
      return gym_lib.create_gym_environment(FLAGS.game, version='v0')
    create_env_fn = create_gym_env_fn
  else:
    raise ValueError('Wrong env: %s' % FLAGS.env)
  return (agent_type, log_dir, run_name, create_env_fn, gin_file)


def main(argv):
  del argv
  if FLAGS.seed is not None and FLAGS.set_seed:
    print('Seed set to %i.' % FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

  (agent_type, log_dir, run_name, create_environment_fn,
   gin_file) = parse_flags()
  print('Flags parsed.')
  print('log_dir = {}'.format(log_dir))

  print(run_name)

  def create_agent_fn(sess, environment, summary_writer):
    """Creates the appropriate agent."""
    if agent_type == 'dqn':
      return dqn_agent.DQNAgent(
          sess=sess,
          num_actions=environment.action_space.n,
          summary_writer=summary_writer)
    elif agent_type == 'iqn':
      return implicit_quantile_agent.ImplicitQuantileAgent(
          sess=sess,
          num_actions=environment.action_space.n,
          summary_writer=summary_writer)
    elif agent_type == 'al_dqn':
      return al_dqn.ALDQNAgent(
          sess=sess,
          num_actions=environment.action_space.n,
          summary_writer=summary_writer)
    elif agent_type == 'al_iqn':
      return al_iqn.ALImplicitQuantileAgent(
          sess=sess,
          num_actions=environment.action_space.n,
          summary_writer=summary_writer)
    elif agent_type == 'sail_dqn':
      return sail_dqn.SAILDQNAgent(
          sess=sess,
          num_actions=environment.action_space.n,
          summary_writer=summary_writer)
    elif agent_type == 'sail_iqn':
      return sail_iqn.SAILImplicitQuantileAgent(
          sess=sess,
          num_actions=environment.action_space.n,
          summary_writer=summary_writer)
    else:
      raise ValueError('Wrong agent %s' % agent_type)

  if gin_file:
    load_gin_configs([gin_file], FLAGS.gin_bindings)

  print('lets run!')
  runner = run_experiment.TrainRunner(log_dir, create_agent_fn,
                                      create_environment_fn)

  print('Agent of type %s created.' % agent_type)
  # pylint: disable=protected-access
  for k in sorted(runner._agent.__dict__):
    if not k.startswith('_'):
      print(k, runner._agent.__dict__[k])
  print()

  # pylint: enable=protected-access
  runner.run_experiment()

if __name__ == '__main__':
  tf.app.run(main)
