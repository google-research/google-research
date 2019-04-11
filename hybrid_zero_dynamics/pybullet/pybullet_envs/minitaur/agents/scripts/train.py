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

r"""Script to train a batch reinforcement learning algorithm.

Command line:

  python3 -m agents.scripts.train --logdir=/path/to/logdir --config=pendulum
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import functools
import os

import gym
import tensorflow as tf

from pybullet_envs.minitaur.agents import tools
from pybullet_envs.minitaur.agents.scripts import configs
from pybullet_envs.minitaur.agents.scripts import utility


def _create_environment(config):
  """Constructor for an instance of the environment.

  Args:
    config: Object providing configurations via attributes.

  Returns:
    Wrapped OpenAI Gym environment.
  """
  if isinstance(config.env, str):
    env = gym.make(config.env)
  else:
    env = config.env()
  if config.max_length:
    env = tools.wrappers.LimitDuration(env, config.max_length)
  env = tools.wrappers.RangeNormalize(env)
  env = tools.wrappers.ClipAction(env)
  env = tools.wrappers.ConvertTo32Bit(env)
  return env


def _define_loop(graph, logdir, train_steps, eval_steps):
  """Create and configure a training loop with training and evaluation phases.

  Args:
    graph: Object providing graph elements via attributes.
    logdir: Log directory for storing checkpoints and summaries.
    train_steps: Number of training steps per epoch.
    eval_steps: Number of evaluation steps per epoch.

  Returns:
    Loop object.
  """
  loop = tools.Loop(
      logdir, graph.step, graph.should_log, graph.do_report,
      graph.force_reset)
  loop.add_phase(
      'train', graph.done, graph.score, graph.summary, train_steps,
      report_every=None,
      log_every=train_steps // 2,
      checkpoint_every=None,
      feed={graph.is_training: True})
  loop.add_phase(
      'eval', graph.done, graph.score, graph.summary, eval_steps,
      report_every=eval_steps,
      log_every=eval_steps // 2,
      checkpoint_every=10 * eval_steps,
      feed={graph.is_training: False})
  return loop


def train(config, env_processes):
  """Training and evaluation entry point yielding scores.

  Resolves some configuration attributes, creates environments, graph, and
  training loop. By default, assigns all operations to the CPU.

  Args:
    config: Object providing configurations via attributes.
    env_processes: Whether to step environments in separate processes.

  Yields:
    Evaluation scores.
  """
  tf.reset_default_graph()
  with config.unlocked:
    config.network = functools.partial(
        utility.define_network, config.network, config)
    config.policy_optimizer = getattr(tf.train, config.policy_optimizer)
    config.value_optimizer = getattr(tf.train, config.value_optimizer)
  if config.update_every % config.num_agents:
    tf.logging.warn('Number of agents should divide episodes per update.')
  with tf.device('/cpu:0'):
    batch_env = utility.define_batch_env(
        lambda: _create_environment(config),
        config.num_agents, env_processes)
    graph = utility.define_simulation_graph(
        batch_env, config.algorithm, config)
    loop = _define_loop(
        graph, config.logdir,
        config.update_every * config.max_length,
        config.eval_episodes * config.max_length)
    total_steps = int(
        config.steps / config.update_every *
        (config.update_every + config.eval_episodes))
  # Exclude episode related variables since the Python state of environments is
  # not checkpointed and thus new episodes start after resuming.
  saver = utility.define_saver(exclude=(r'.*_temporary/.*',))
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    utility.initialize_variables(sess, saver, config.logdir)
    for score in loop.run(sess, saver, total_steps):
      yield score
  batch_env.close()


def main(_):
  """Create or load configuration and launch the trainer."""
  utility.set_up_logging()
  if not FLAGS.config:
    raise KeyError('You must specify a configuration.')
  logdir = FLAGS.logdir and os.path.expanduser(os.path.join(
      FLAGS.logdir, '{}-{}'.format(FLAGS.timestamp, FLAGS.config)))
  try:
    config = utility.load_config(logdir)
  except IOError:
    config = tools.AttrDict(getattr(configs, FLAGS.config)())
    config = utility.save_config(config, logdir)
  for score in train(config, FLAGS.env_processes):
    tf.logging.info('Score {}.'.format(score))


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string(
      'logdir', None,
      'Base directory to store logs.')
  tf.app.flags.DEFINE_string(
      'timestamp', datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
      'Sub directory to store logs.')
  tf.app.flags.DEFINE_string(
      'config', None,
      'Configuration to execute.')
  tf.app.flags.DEFINE_boolean(
      'env_processes', True,
      'Step environments in separate processes to circumvent the GIL.')
  tf.app.run()
