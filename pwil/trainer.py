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

"""PWIL training script."""

from absl import app
from absl import flags
from acme import specs
from acme.agents.tf import d4pg
from acme.agents.tf.actors import FeedForwardActor
from acme.utils.loggers import csv as csv_logger
import sonnet as snt

from pwil import imitation_loop
from pwil import rewarder
from pwil import utils


flags.DEFINE_string('workdir', None, 'Logging directory')
flags.DEFINE_string('env_name', None, 'Environment name.')
flags.DEFINE_string('demo_dir', None, 'Directory of expert demonstrations.')
flags.DEFINE_boolean('state_only', False,
                     'Use only state for reward computation')
flags.DEFINE_float('sigma', 0.2, 'Exploration noise.')
flags.DEFINE_integer('num_transitions_rb', 50000,
                     'Number of transitions to fill the rb with.')
flags.DEFINE_integer('num_demonstrations', 1, 'Number of expert episodes.')
flags.DEFINE_integer('subsampling', 20, 'Subsampling factor of demonstrations.')
flags.DEFINE_integer('random_seed', 1, 'Experiment random seed.')
flags.DEFINE_integer('num_steps_per_iteration', 10000,
                     'Number of training steps per iteration.')
flags.DEFINE_integer('num_iterations', 100, 'Number of iterations.')
flags.DEFINE_integer('num_eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('samples_per_insert', 256, 'Controls update frequency.')
flags.DEFINE_float('policy_learning_rate', 1e-4,
                   'Larning rate for policy updates')
flags.DEFINE_float('critic_learning_rate', 1e-4,
                   'Larning rate for critic updates')

FLAGS = flags.FLAGS


def main(_):
  # Load environment.
  environment = utils.load_environment(FLAGS.env_name)
  environment_spec = specs.make_environment_spec(environment)

  # Create Rewarder.
  demonstrations = utils.load_demonstrations(
      demo_dir=FLAGS.demo_dir, env_name=FLAGS.env_name)
  pwil_rewarder = rewarder.PWILRewarder(
      demonstrations,
      subsampling=FLAGS.subsampling,
      env_specs=environment_spec,
      num_demonstrations=FLAGS.num_demonstrations,
      observation_only=FLAGS.state_only)

  # Define optimizers
  policy_optimizer = snt.optimizers.Adam(
      learning_rate=FLAGS.policy_learning_rate)
  critic_optimizer = snt.optimizers.Adam(
      learning_rate=FLAGS.critic_learning_rate)

  # Define D4PG agent.
  agent_networks = utils.make_d4pg_networks(environment_spec.actions)
  agent = d4pg.D4PG(
      environment_spec=environment_spec,
      policy_network=agent_networks['policy'],
      critic_network=agent_networks['critic'],
      observation_network=agent_networks['observation'],
      policy_optimizer=policy_optimizer,
      critic_optimizer=critic_optimizer,
      samples_per_insert=FLAGS.samples_per_insert,
      sigma=FLAGS.sigma,
  )

  # Prefill the agent's Replay Buffer.
  utils.prefill_rb_with_demonstrations(
      agent=agent,
      demonstrations=pwil_rewarder.demonstrations,
      num_transitions_rb=FLAGS.num_transitions_rb,
      reward=pwil_rewarder.reward_scale)

  # Create the eval policy (without exploration noise).
  eval_policy = snt.Sequential([
      agent_networks['observation'],
      agent_networks['policy'],
  ])
  eval_agent = FeedForwardActor(policy_network=eval_policy)

  # Define train/eval loops.

  train_logger = csv_logger.CSVLogger(
      directory=FLAGS.workdir, label='train_logs')
  eval_logger = csv_logger.CSVLogger(
      directory=FLAGS.workdir, label='eval_logs')


  train_loop = imitation_loop.TrainEnvironmentLoop(
      environment, agent, pwil_rewarder, logger=train_logger)

  eval_loop = imitation_loop.EvalEnvironmentLoop(
      environment, eval_agent, pwil_rewarder, logger=eval_logger)

  for _ in range(FLAGS.num_iterations):
    train_loop.run(num_steps=FLAGS.num_steps_per_iteration)
    eval_loop.run(num_episodes=FLAGS.num_eval_episodes)

if __name__ == '__main__':
  app.run(main)
