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

"""Train SAC with DBC or MICo."""


from typing import Optional

from absl import app
from absl import flags
from absl import logging

from dopamine.continuous_domains import run_experiment
from dopamine.discrete_domains import gym_lib
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.sac import sac_agent
from flax.metrics import tensorboard
from gym import spaces
from mico.dm_control import dbc_agent
from mico.dm_control import metric_sac_agent

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_string('agent_name', 'mico', 'Agent name.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

FLAGS = flags.FLAGS


def create_continuous_bisim_agent(
    environment,
    summary_writer = None
):
  """Creates an agent."""
  assert FLAGS.agent_name is not None
  if FLAGS.agent_name.startswith('sac'):
    assert isinstance(environment.action_space, spaces.Box)
    assert isinstance(environment.observation_space, spaces.Box)
    return sac_agent.SACAgent(
        action_shape=environment.action_space.shape,
        action_limits=(environment.action_space.low,
                       environment.action_space.high),
        observation_shape=environment.observation_space.shape,
        action_dtype=environment.action_space.dtype,
        observation_dtype=environment.observation_space.dtype,
        summary_writer=summary_writer)
  elif FLAGS.agent_name.startswith('dbc'):
    assert isinstance(environment.action_space, spaces.Box)
    assert isinstance(environment.observation_space, spaces.Box)
    return dbc_agent.DBCAgent(
        action_shape=environment.action_space.shape,
        action_limits=(environment.action_space.low,
                       environment.action_space.high),
        observation_shape=environment.observation_space.shape,
        action_dtype=environment.action_space.dtype,
        observation_dtype=environment.observation_space.dtype,
        summary_writer=summary_writer)
  elif FLAGS.agent_name.startswith('mico'):
    assert isinstance(environment.action_space, spaces.Box)
    assert isinstance(environment.observation_space, spaces.Box)
    return metric_sac_agent.MetricSACAgent(
        action_shape=environment.action_space.shape,
        action_limits=(environment.action_space.low,
                       environment.action_space.high),
        observation_shape=environment.observation_space.shape,
        action_dtype=environment.action_space.dtype,
        observation_dtype=environment.observation_space.dtype,
        summary_writer=summary_writer)
  else:
    raise ValueError(f'Unknown agent: {FLAGS.agent_name}')


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  logging.set_verbosity(logging.INFO)

  base_dir = FLAGS.base_dir
  gin_files = FLAGS.gin_files
  gin_bindings = FLAGS.gin_bindings
  xm_xid = None if 'xm_xid' not in FLAGS else FLAGS.xm_xid
  xm_wid = None if 'xm_wid' not in FLAGS else FLAGS.xm_wid
  xm_parameters = (
      None if 'xm_parameters' not in FLAGS else FLAGS.xm_parameters)
  base_dir, gin_files, gin_bindings = xm_utils.run_xm_preprocessing(
      xm_xid, xm_wid, xm_parameters, base_dir,
      FLAGS.custom_base_dir_from_hparams, gin_files, gin_bindings)
  run_experiment.load_gin_configs(gin_files, gin_bindings)
  runner = run_experiment.ContinuousTrainRunner(base_dir,
                                                create_continuous_bisim_agent)
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
