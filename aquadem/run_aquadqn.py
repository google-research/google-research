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

"""Runner of the AQuaDQN agent."""

from absl import app
from absl import flags
import acme
from acme import specs
from acme.agents.jax import dqn
from acme.jax.layouts import local_layout
from acme.utils import loggers
import jax

from aquadem import builder as aquadem_builder
from aquadem import config
from aquadem import networks as aquadem_networks
from aquadem import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', '/tmp/aquadqn', 'Log directory')
flags.DEFINE_string('env_name', 'door-human-v1', 'What environment to run')
flags.DEFINE_integer('num_demonstrations', None,
                     'Number of expert demonstrations to use.')
flags.DEFINE_integer('num_steps', 1000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('eval_every', 10000, 'Evaluation frequency.')
flags.DEFINE_integer('seed', 0, 'Seed of the RL agent.')


def main(_):
  # Create an environment, grab the spec.
  environment = utils.make_environment(task=FLAGS.env_name)
  aqua_config = config.AquademConfig()
  spec = specs.make_environment_spec(environment)
  discretized_spec = aquadem_builder.discretize_spec(spec,
                                                     aqua_config.num_actions)

  # Create AQuaDem builder.
  loss_fn = dqn.losses.MunchausenQLearning(max_abs_reward=100.)
  dqn_config = dqn.DQNConfig(
      samples_per_insert_tolerance_rate=float('inf'),
      min_replay_size=1,
      n_step=3,
      num_sgd_steps_per_step=8,
      learning_rate=1e-4,
      samples_per_insert=256)
  rl_agent = dqn.DQNBuilder(config=dqn_config, loss_fn=loss_fn)
  make_demonstrations = utils.get_make_demonstrations_fn(
      FLAGS.env_name, FLAGS.num_demonstrations, FLAGS.seed)
  builder = aquadem_builder.AquademBuilder(
      rl_agent=rl_agent,
      config=aqua_config,
      make_demonstrations=make_demonstrations)

  # Create networks.
  q_network = aquadem_networks.make_q_network(
      spec=discretized_spec,)
  networks = aquadem_networks.make_action_candidates_network(
      spec=spec,
      num_actions=aqua_config.num_actions,
      discrete_rl_networks=q_network)
  exploration_epsilon = 0.01
  discrete_policy = dqn.default_behavior_policy(q_network, exploration_epsilon)
  behavior_policy = aquadem_builder.get_aquadem_policy(discrete_policy,
                                                       networks)

  # Create the environment loop used for training.
  agent = local_layout.LocalLayout(
      seed=FLAGS.seed,
      environment_spec=spec,
      builder=builder,
      networks=networks,
      policy_network=behavior_policy,
      batch_size=dqn_config.batch_size * dqn_config.num_sgd_steps_per_step,
      samples_per_insert=dqn_config.samples_per_insert)

  train_logger = loggers.CSVLogger(FLAGS.workdir, label='train')
  train_loop = acme.EnvironmentLoop(environment, agent, logger=train_logger)

  # Create the evaluation actor and loop.
  eval_policy = dqn.default_behavior_policy(q_network, 0.)
  eval_policy = aquadem_builder.get_aquadem_policy(eval_policy, networks)
  eval_actor = builder.make_actor(
      random_key=jax.random.PRNGKey(FLAGS.seed),
      policy_network=eval_policy,
      variable_source=agent)
  eval_env = utils.make_environment(task=FLAGS.env_name, evaluation=True)

  eval_logger = loggers.CSVLogger(FLAGS.workdir, label='eval')
  eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, logger=eval_logger)

  assert FLAGS.num_steps % FLAGS.eval_every == 0
  for _ in range(FLAGS.num_steps // FLAGS.eval_every):
    eval_loop.run(num_episodes=10)
    train_loop.run(num_steps=FLAGS.eval_every)
  eval_loop.run(num_episodes=10)


if __name__ == '__main__':
  app.run(main)
