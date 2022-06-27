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

r"""The entry point for running experiments.
"""

import functools
import json
import os

from absl import app
from absl import flags
from absl import logging
from dopamine.discrete_domains import run_experiment as base_run_experiment
from dopamine.discrete_domains import train as base_train

from generalization_representations_rl_aistats22.offline_rl import offline_incoherent_dqn_agent
from generalization_representations_rl_aistats22.offline_rl import offline_incoherent_iqn_agent
from generalization_representations_rl_aistats22.offline_rl import run_experiment


# flags are defined when importing run_xm_preprocessing
flags.DEFINE_string('agent_name', 'incoherent_dqn', 'Name of the agent.')
flags.DEFINE_string('replay_dir', None, 'Directory from which to load the '
                    'replay data')

FLAGS = flags.FLAGS


def create_offline_agent(sess,  # pylint: disable=unused-argument
                         environment,
                         agent_name,
                         replay_data_dir,
                         summary_writer=None):
  """Creates an online agent.

  Args:
    sess: A `tf.Session` object for running associated ops. This argument is
      ignored for JAX agents.
    environment: An Atari 2600 environment.
    agent_name: Name of the agent to be created.
    replay_data_dir: Directory from which to load the fixed replay buffers.
    summary_writer: A Tensorflow summary writer to pass to the agent for
      in-agent training statistics in Tensorboard.

  Returns:
    An agent with metrics.
  """
  if agent_name == 'incoherent_dqn':
    agent = offline_incoherent_dqn_agent.OfflineIncoherentDQNAgent
  elif agent_name == 'incoherent_iqn':
    agent = offline_incoherent_iqn_agent.OfflineIncoherentImplicitQuantileAgent
  else:
    raise ValueError('{} is not a valid agent name'.format(FLAGS.agent_name))

  return agent(
      num_actions=environment.action_space.n,
      replay_data_dir=replay_data_dir,
      summary_writer=summary_writer)


def create_replay_dir(xm_parameters):
  """Creates the replay data directory from xm_parameters."""
  replay_dir = FLAGS.replay_dir
  if xm_parameters:
    xm_params = json.loads(xm_parameters)
    problem_name, run_number = '', ''
    for param, value in xm_params.items():
      if param.endswith('game_name'):
        problem_name = value
      elif param.endswith('run_number'):
        run_number = str(value)
    replay_dir = os.path.join(replay_dir, problem_name, run_number)
  return os.path.join(replay_dir, 'replay_logs')


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  logging.set_verbosity(logging.INFO)

  xm_xid = None if 'xm_xid' not in FLAGS else FLAGS.xm_xid
  xm_wid = None if 'xm_wid' not in FLAGS else FLAGS.xm_wid
  xm_parameters = (None
                   if 'xm_parameters' not in FLAGS else FLAGS.xm_parameters)
  base_dir, gin_files, gin_bindings = base_train.run_xm_preprocessing(
      xm_xid, xm_wid, xm_parameters, FLAGS.base_dir,
      FLAGS.custom_base_dir_from_hparams, FLAGS.gin_files, FLAGS.gin_bindings)
  create_agent = functools.partial(
      create_offline_agent,
      agent_name=FLAGS.agent_name
      )
  base_run_experiment.load_gin_configs(gin_files, gin_bindings)
  runner = run_experiment.FixedReplayRunner(base_dir, create_agent)
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
