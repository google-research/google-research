# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Script to render videos of the Proximal Policy Gradient algorithm.

Command line:

  python3 -m agents.scripts.visualize \
      --logdir=/path/to/logdir/<time>-<config> --outdir=/path/to/outdir/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import gym
import tensorflow as tf

from pybullet_envs.minitaur.agents import tools
from pybullet_envs.minitaur.agents.scripts import utility


def _create_environment(config, outdir):
  """Constructor for an instance of the environment.

  Args:
    config: Object providing configurations via attributes.
    outdir: Directory to store videos in.

  Returns:
    Wrapped OpenAI Gym environment.
  """
  if isinstance(config.env, str):
    env = gym.make(config.env)
  else:
    env = config.env()
  # Ensure that the environment has the specification attribute set as expected
  # by the monitor wrapper.
  if not hasattr(env, 'spec'):
    setattr(env, 'spec', getattr(env, 'spec', None))
  if config.max_length:
    env = tools.wrappers.LimitDuration(env, config.max_length)
#  env = gym.wrappers.Monitor(
#      env, outdir, lambda unused_episode_number: True)
  env = tools.wrappers.RangeNormalize(env)
  env = tools.wrappers.ClipAction(env)
  env = tools.wrappers.ConvertTo32Bit(env)
  return env


def _define_loop(graph, eval_steps):
  """Create and configure an evaluation loop.

  Args:
    graph: Object providing graph elements via attributes.
    eval_steps: Number of evaluation steps per epoch.

  Returns:
    Loop object.
  """
  loop = tools.Loop(
      None, graph.step, graph.should_log, graph.do_report, graph.force_reset)
  loop.add_phase(
      'eval', graph.done, graph.score, graph.summary, eval_steps,
      report_every=eval_steps,
      log_every=None,
      checkpoint_every=None,
      feed={graph.is_training: False})
  return loop


def visualize(
    logdir, outdir, num_agents, num_episodes, checkpoint=None,
    env_processes=True):
  """Recover checkpoint and render videos from it.

  Args:
    logdir: Logging directory of the trained algorithm.
    outdir: Directory to store rendered videos in.
    num_agents: Number of environments to simulate in parallel.
    num_episodes: Total number of episodes to simulate.
    checkpoint: Checkpoint name to load; defaults to most recent.
    env_processes: Whether to step environments in separate processes.
  """
  config = utility.load_config(logdir)
  with config.unlocked:
    config.network = functools.partial(
        utility.define_network, config.network, config)
    config.policy_optimizer = getattr(tf.train, config.policy_optimizer)
    config.value_optimizer = getattr(tf.train, config.value_optimizer)
  with tf.device('/cpu:0'):
    batch_env = utility.define_batch_env(
        lambda: _create_environment(config, outdir),
        num_agents, env_processes)
    graph = utility.define_simulation_graph(
        batch_env, config.algorithm, config)
    total_steps = num_episodes * config.max_length
    loop = _define_loop(graph, total_steps)
  saver = utility.define_saver(
      exclude=(r'.*_temporary/.*', r'global_step'))
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    utility.initialize_variables(
        sess, saver, config.logdir, checkpoint, resume=True)
    for unused_score in loop.run(sess, saver, total_steps):
      pass
  batch_env.close()


def main(_):
  """Load a trained algorithm and render videos."""
  utility.set_up_logging()
  if not FLAGS.logdir or not FLAGS.outdir:
    raise KeyError('You must specify logging and outdirs directories.')
  FLAGS.logdir = os.path.expanduser(FLAGS.logdir)
  FLAGS.outdir = os.path.expanduser(FLAGS.outdir)
  visualize(
      FLAGS.logdir, FLAGS.outdir, FLAGS.num_agents, FLAGS.num_episodes,
      FLAGS.checkpoint, FLAGS.env_processes)


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string(
      'logdir', None,
      'Directory to the checkpoint of a training run.')
  tf.app.flags.DEFINE_string(
      'outdir', None,
      'Local directory for storing the monitoring outdir.')
  tf.app.flags.DEFINE_string(
      'checkpoint', None,
      'Checkpoint name to load; defaults to most recent.')
  tf.app.flags.DEFINE_integer(
      'num_agents', 1,
      'How many environments to step in parallel.')
  tf.app.flags.DEFINE_integer(
      'num_episodes', 5,
      'Minimum number of episodes to render.')
  tf.app.flags.DEFINE_boolean(
      'env_processes', True,
      'Step environments in separate processes to circumvent the GIL.')
  tf.app.run()
