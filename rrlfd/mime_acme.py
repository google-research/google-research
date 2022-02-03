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

# Lint as: python3
"""Train an Acme agent on mime."""

import datetime
import os
from typing import Dict, Sequence

from absl import app
from absl import flags
from acme import specs
from acme import types
from acme.agents.tf import dmpo
from acme.agents.tf import mpo
from acme.tf import networks
from acme.tf import utils as tf_utils
from acme.utils import counting
from acme.utils.loggers.google import cns
import numpy as np
import sonnet as snt
import tensorflow as tf

from rrlfd import environment_loop
from rrlfd.env_wrapper import DmMimeWrapper
from rrlfd.env_wrapper import KwargWrapper

flags.DEFINE_string('task', 'Pick', 'Mime task.')
flags.DEFINE_enum('input_type', 'position',
                  ['depth', 'rgb', 'rgbd', 'position'],
                  'Input modality.')
flags.DEFINE_boolean('dense_reward', True, 'If True, use dense reward signal.')
flags.DEFINE_float('dense_reward_multiplier', 1.0,
                   'Multiplier for dense rewards.')

flags.DEFINE_string('agent', 'DMPO', 'Acme agent to train.')
flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to run for.')
flags.DEFINE_integer('max_episode_steps', None,
                     'If set, override environment default for max episode '
                     'length.')
flags.DEFINE_integer('seed', 0, 'Experiment seed.')

flags.DEFINE_string('logdir', None, 'Location to log results to.')
flags.DEFINE_boolean('log_learner', False, 'If True, save learner logs.')
flags.DEFINE_boolean('render', False, 'If True, render environment.')
flags.DEFINE_boolean('verbose', False, 'If True, log actions at each step.')

FLAGS = flags.FLAGS


def make_mpo_networks(
    action_spec,
    policy_layer_sizes = (300, 200),
    critic_layer_sizes = (400, 300),
):
  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)
  critic_layer_sizes = list(critic_layer_sizes) + [1]

  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes),
      networks.MultivariateNormalDiagHead(num_dimensions)
  ])
  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = networks.CriticMultiplexer(
      critic_network=networks.LayerNormMLP(critic_layer_sizes),
      action_network=networks.ClipToSpec(action_spec))

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': tf_utils.batch_concat,
  }


def make_dmpo_networks(
    action_spec,
    policy_layer_sizes = (300, 200),
    critic_layer_sizes = (400, 300),
    vmin = -150.,
    vmax = 150.,
    num_atoms = 51,
):
  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)

  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes),
      networks.MultivariateNormalDiagHead(num_dimensions)
  ])
  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = networks.CriticMultiplexer(
      critic_network=networks.LayerNormMLP(critic_layer_sizes),
      action_network=networks.ClipToSpec(action_spec))
  critic_network = snt.Sequential(
      [critic_network,
       networks.DiscreteValuedHead(vmin, vmax, num_atoms)])

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': tf_utils.batch_concat,
  }


def main(_):
  tf.random.set_seed(FLAGS.seed)

  if FLAGS.logdir is not None:
    logdir = FLAGS.logdir
  else:
    logdir = os.path.join(
        FLAGS.logdir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

  # Create an environment and grab the spec.
  env = DmMimeWrapper(
      task=FLAGS.task,
      seed=FLAGS.seed,
      input_type=FLAGS.input_type,
      dense_reward=FLAGS.dense_reward,
      dense_reward_multiplier=FLAGS.dense_reward_multiplier,
      max_episode_steps=FLAGS.max_episode_steps,
      logdir=logdir,
      render=FLAGS.render,
      verbose=FLAGS.verbose)
  environment = KwargWrapper(env)
  environment_spec = specs.make_environment_spec(environment)
  print(environment_spec)

  counter = counting.Counter()
  agent_logger = (
      cns.CNSLogger(logdir, 'learner')
      if logdir is not None and FLAGS.log_learner else None)

  if FLAGS.agent == 'MPO':
    agent_networks = make_mpo_networks(environment_spec.actions)

    agent = mpo.MPO(
        environment_spec=environment_spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        observation_network=agent_networks['observation'],
        checkpoint=True,
        logger=agent_logger,
        counter=counter,
    )
  elif FLAGS.agent == 'DMPO':
    agent_networks = make_dmpo_networks(environment_spec.actions)

    agent = dmpo.DistributionalMPO(
        environment_spec=environment_spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        observation_network=agent_networks['observation'],
        checkpoint=True,
        logger=agent_logger,
        counter=counter,
    )
  else:
    raise NotImplementedError('Supported agents: MPO, DMPO.')
  env_logger = (
      cns.CNSLogger(logdir, 'env_loop') if logdir is not None else None)

  # Run the environment loop.
  loop = environment_loop.EnvironmentLoop(
      environment, agent, logger=env_logger, counter=counter)
  loop.run(num_episodes=FLAGS.num_episodes)


if __name__ == '__main__':
  app.run(main)
