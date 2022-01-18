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

"""Binary entry-point for Metric RL experiments."""

from absl import app
from absl import flags
from absl import logging

from dopamine.discrete_domains import run_experiment
from dopamine.jax.agents.dqn import dqn_agent as jax_dqn_agent
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent as jax_implicit_quantile_agent
from dopamine.jax.agents.quantile import quantile_agent as jax_quantile_agent
from dopamine.jax.agents.rainbow import rainbow_agent as jax_rainbow_agent
import gin
import jax
import jax.numpy as jnp

from mico.atari import metric_dqn_agent
from mico.atari import metric_implicit_quantile_agent
from mico.atari import metric_quantile_agent
from mico.atari import metric_rainbow_agent


flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

FLAGS = flags.FLAGS


def sample_gaussian(rng, mu, var):
  rng1, rng2 = jax.random.split(rng)
  return rng1, mu + jnp.sqrt(var) * jax.random.normal(rng2)


@gin.configurable
def create_metric_agent(sess, environment, agent_name='metric_dqn',
                        summary_writer=None, debug_mode=False):
  """Creates a metric agent.

  Args:
    sess: TF session, unused since we are in JAX.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, unused.

  Returns:
    An active and passive agent.
  """
  assert agent_name is not None
  del sess
  del debug_mode
  if agent_name == 'dqn':
    return jax_dqn_agent.JaxDQNAgent(num_actions=environment.action_space.n,
                                     summary_writer=summary_writer)
  elif agent_name == 'quantile':
    return jax_quantile_agent.JaxQuantileAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return jax_rainbow_agent.JaxRainbowAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    return jax_implicit_quantile_agent.JaxImplicitQuantileAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'metric_dqn':
    return metric_dqn_agent.MetricDQNAgent(
        num_actions=environment.action_space.n, summary_writer=summary_writer)
  elif agent_name == 'metric_c51' or agent_name == 'metric_rainbow':
    return metric_rainbow_agent.MetricRainbowAgent(
        num_actions=environment.action_space.n, summary_writer=summary_writer)
  elif agent_name == 'metric_quantile':
    return metric_quantile_agent.MetricQuantileAgent(
        num_actions=environment.action_space.n, summary_writer=summary_writer)
  elif agent_name == 'metric_implicit_quantile':
    return metric_implicit_quantile_agent.MetricImplicitQuantileAgent(
        num_actions=environment.action_space.n, summary_writer=summary_writer)
  elif agent_name == 'mimplicit_quantile':
    return metric_implicit_quantile_agent.MetricImplicitQuantileAgent(
        num_actions=environment.action_space.n, mico_weight=0.0,
        tau=0.03, summary_writer=summary_writer)
  elif agent_name == 'metric_mimplicit_quantile':
    return metric_implicit_quantile_agent.MetricImplicitQuantileAgent(
        num_actions=environment.action_space.n, tau=0.03,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  logging.set_verbosity(logging.INFO)
  base_dir = FLAGS.base_dir
  gin_files = FLAGS.gin_files
  gin_bindings = FLAGS.gin_bindings
  run_experiment.load_gin_configs(gin_files, gin_bindings)
  runner = run_experiment.TrainRunner(base_dir, create_metric_agent)
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
