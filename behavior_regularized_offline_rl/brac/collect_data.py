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

"""Main run file for data collection."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import importlib
import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
import tensorflow.compat.v1 as tf

from behavior_regularized_offline_rl.brac import dataset
from behavior_regularized_offline_rl.brac import policy_loader
from behavior_regularized_offline_rl.brac import train_eval_utils
from behavior_regularized_offline_rl.brac import utils

tf.compat.v1.enable_v2_behavior()

flags.DEFINE_string('root_dir',
                    os.path.join(os.getenv('HOME', '/'), 'tmp/offlinerl/data'),
                    'Root directory for saving data.')
flags.DEFINE_string('sub_dir', '0', 'sub directory for saving data.')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'env name.')
flags.DEFINE_string('data_name', 'random', 'data name.')
flags.DEFINE_string('env_loader', 'mujoco', 'env loader, suite/gym.')
flags.DEFINE_string('config_dir',
                    'behavior_regularized_offline_rl.brac.configs',
                    'config file dir.')
flags.DEFINE_string('config_file', 'dcfg_pure', 'config file name.')
flags.DEFINE_string('policy_root_dir', None,
                    'Directory in which to find the behavior policy.')
flags.DEFINE_integer('n_samples', int(1e6), 'number of transitions to collect.')
flags.DEFINE_integer('n_eval_episodes', 20,
                     'number episodes to eval each policy.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


def get_sample_counts(n, distr):
  """Provides size of each sub-dataset based on desired distribution."""
  distr = np.array(distr)
  distr = distr / np.sum(distr)
  counts = []
  remainder = n
  for i in range(distr.shape[0] - 1):
    count = int(n * distr[i])
    remainder -= count
    counts.append(count)
  counts.append(remainder)
  return counts


def collect_n_transitions(tf_env, policy, data, n, log_freq=10000):
  """Adds desired number of transitions to dataset."""
  collector = train_eval_utils.DataCollector(tf_env, policy, data)
  time_st = time.time()
  timed_at_step = 0
  steps_collected = 0
  while steps_collected < n:
    count = collector.collect_transition()
    steps_collected += count
    if (steps_collected % log_freq == 0
        or steps_collected == n) and count > 0:
      steps_per_sec = ((steps_collected - timed_at_step)
                       / (time.time() - time_st))
      timed_at_step = steps_collected
      time_st = time.time()
      logging.info('(%d/%d) steps collected at %.4g steps/s.', steps_collected,
                   n, steps_per_sec)


def collect_data(
    log_dir,
    data_config,
    n_samples=int(1e6),
    env_name='HalfCheetah-v2',
    log_freq=int(1e4),
    n_eval_episodes=20,
    ):
  """Creates dataset of transitions based on desired config."""
  tf_env = train_eval_utils.env_factory(env_name)
  observation_spec = tf_env.observation_spec()
  action_spec = tf_env.action_spec()

  # Initialize dataset.
  sample_sizes = list([cfg[-1] for cfg in data_config])
  sample_sizes = get_sample_counts(n_samples, sample_sizes)
  with tf.device('/cpu:0'):
    data = dataset.Dataset(
        observation_spec,
        action_spec,
        n_samples,
        circular=False)
  data_ckpt = tf.train.Checkpoint(data=data)
  data_ckpt_name = os.path.join(log_dir, 'data')

  # Collect data for each policy in data_config.
  time_st = time.time()
  test_results = collections.OrderedDict()
  for (policy_name, policy_cfg, _), n_transitions in zip(
      data_config, sample_sizes):
    policy_cfg = policy_loader.parse_policy_cfg(policy_cfg)
    policy = policy_loader.load_policy(policy_cfg, action_spec)
    logging.info('Testing policy %s...', policy_name)
    eval_mean, eval_std = train_eval_utils.eval_policy_episodes(
        tf_env, policy, n_eval_episodes)
    test_results[policy_name] = [eval_mean, eval_std]
    logging.info('Return mean %.4g, std %.4g.', eval_mean, eval_std)
    logging.info('Collecting data from policy %s...', policy_name)
    collect_n_transitions(tf_env, policy, data, n_transitions, log_freq)

  # Save final dataset.
  assert data.size == data.capacity
  data_ckpt.write(data_ckpt_name)
  time_cost = time.time() - time_st
  logging.info('Finished: %d transitions collected, '
               'saved at %s, '
               'time cost %.4gs.', n_samples, data_ckpt_name, time_cost)


def main(_):
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)
  sub_dir = FLAGS.sub_dir
  log_dir = os.path.join(
      FLAGS.root_dir,
      FLAGS.env_name,
      FLAGS.data_name,
      sub_dir,
      )
  utils.maybe_makedirs(log_dir)
  config_module = importlib.import_module(
      '{}.{}'.format(FLAGS.config_dir, FLAGS.config_file))
  collect_data(
      log_dir=log_dir,
      data_config=config_module.get_data_config(FLAGS.env_name,
                                                FLAGS.policy_root_dir),
      n_samples=FLAGS.n_samples,
      env_name=FLAGS.env_name,
      n_eval_episodes=FLAGS.n_eval_episodes
      )


if __name__ == '__main__':
  app.run(main)
