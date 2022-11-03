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

Example command:

python -m aux_tasks.auxiliary_mc.train \
  --agent_name=cumulant_jax_dqn \
  --gin_files='aux_tasks/auxiliary_mc/dqn.gin' \
  --base_dir=/tmp/online_rl/dqn \
  --gin_bindings="Runner.evaluation_steps=10" \
  --gin_bindings="JaxDQNAgent.min_replay_history = 40" \
  --gin_bindings="Runner.max_steps_per_episode = 10" \
  --gin_bindings="OutOfGraphReplayBufferWithMC.replay_capacity = 10000" \
  --gin_bindings="OutOfGraphReplayBufferWithMC.batch_size = 5" \
  --gin_bindings="Runner.training_steps = 100"

"""

from absl import app
from absl import flags
from absl import logging
from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import train as base_train  # pylint: disable=unused-import
from jax.config import config

from aux_tasks.auxiliary_mc import discounted_dqn_agent
from aux_tasks.auxiliary_mc import dqn_agent



flags.DEFINE_string('agent_name', 'jax_dqn', 'Name of the agent.')
flags.DEFINE_boolean('disable_jit', False, 'Name of the agent.')
FLAGS = flags.FLAGS


def create_agent(
    sess,  # pylint:disable=unused-argument
    environment,
    summary_writer=None):
  """Creates a DQN agent with RC auxiliary tasks and MC replay buffer.

  Args:
    sess: A `tf.Session`object  for running associated ops.
    environment: An Atari 2600 environment.
    summary_writer: A Tensorflow summary writer to pass to the agent for
      in-agent training statistics in Tensorboard.

  Returns:
    A DQN agent with metrics.
  """
  if FLAGS.agent_name == 'cumulant_jax_dqn':
    agent = dqn_agent.CumulantJaxDQNAgentWithAuxiliaryMC
  elif FLAGS.agent_name == 'discounted_jax_dqn':
    agent = discounted_dqn_agent.DiscountedJaxDQNAgentWithAuxiliaryMC
  else:
    raise ValueError('{} is not a valid agent name'.format(FLAGS.agent_name))

  return agent(num_actions=environment.action_space.n,
               summary_writer=summary_writer)


def main(unused_argv):
  config.update('jax_disable_jit', FLAGS.disable_jit)
  logging.set_verbosity(logging.INFO)
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = run_experiment.Runner(FLAGS.base_dir, create_agent)
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
