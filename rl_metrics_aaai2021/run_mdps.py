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

r"""Compute metrics on random MDPs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow.compat.v1 as tf

from rl_metrics_aaai2021 import random_mdp
from rl_metrics_aaai2021 import utils

flags.DEFINE_string('base_dir', None, 'Base directory to store stats.')
flags.DEFINE_multi_string('metrics', None, 'List of metrics to compute.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')
flags.DEFINE_bool('verbose', False, 'Whether to print verbose messages.')
flags.DEFINE_float('tolerance', 0.001, 'Error tolerance for value iteration.')
flags.DEFINE_float('gamma', 0.9, 'Discount factor to use for value iteration.')
flags.DEFINE_string(
    'custom_base_dir_from_hparams', None,
    'If not None, will set the base_directory prefixed with '
    'the value of this flag, and a subdirectory using game '
    'name and hparam settings. For example, if your game is '
    'Breakout with hparams epsilon=0.01 and horizon=6, '
    'the resulting directory would be:'
    'FLAGS.base_dir/Breakout/0.01_6/')
flags.DEFINE_integer('num_mdps', 10, 'Number of random MDPs to generate.')
flags.DEFINE_integer('num_states', 10, 'Number of states in MDP.')
flags.DEFINE_integer('num_actions', 4, 'Number of actions in MDP.')
flags.DEFINE_integer('max_iterations', 1000,
                     'Maximum number of iterations to use for sammpling.')

FLAGS = flags.FLAGS


def main(_):
  flags.mark_flags_as_required(['base_dir'])
  if FLAGS.custom_base_dir_from_hparams is not None:
    FLAGS.base_dir = os.path.join(FLAGS.base_dir,
                                  FLAGS.custom_base_dir_from_hparams)
  else:
    # Add Work unit to base directory path, if it exists.
    if 'xm_wid' in FLAGS and FLAGS.xm_wid > 0:
      FLAGS.base_dir = os.path.join(FLAGS.base_dir, str(FLAGS.xm_wid))
  base_dir = os.path.join(
      FLAGS.base_dir, '{}_{}'.format(FLAGS.num_states, FLAGS.num_actions))
  if not tf.io.gfile.exists(base_dir):
    tf.io.gfile.makedirs(base_dir)
  gin.parse_config(bindings=FLAGS.gin_bindings, skip_unknown=False)
  metrics = utils.METRICS if FLAGS.metrics is None else FLAGS.metrics
  mdp_stats = {}
  for i in range(FLAGS.num_mdps):
    tf.logging.info('Starting run %d', i)
    env = random_mdp.RandomMDP(FLAGS.num_states, FLAGS.num_actions)
    # We add the discount factor to the environment.
    env.gamma = FLAGS.gamma
    # We will store the values in our environment, as it may be used by the
    # metrics.
    start_time = time.time()
    env.values, env.q_values = utils.value_iteration(env, FLAGS.tolerance,
                                                     verbose=FLAGS.verbose)
    value_iteration_time = time.time() - start_time
    for metric_name in metrics:
      if metric_name not in utils.METRICS:
        if FLAGS.verbose:
          logging.info('Unknown metric %s, skipping', metric_name)
        continue
      logging.info('Will compute %s', metric_name)
      metric = utils.METRICS[metric_name].constructor(
          metric_name, utils.METRICS[metric_name].label, env, base_dir,
          gamma=FLAGS.gamma)
      metric.compute(save_and_reload=False, verbose=FLAGS.verbose)
      min_gap, avg_gap, max_gap = metric.maybe_pretty_print_and_compute_gaps()
      if metric.name not in mdp_stats:
        mdp_stats[metric.name] = []
      runtime = (value_iteration_time if metric_name == 'd_delta_star'
                 else metric.statistics.time)
      mdp_stats[metric.name].append(
          utils.MDPStats(time=runtime,
                         num_iterations=metric.statistics.num_iterations,
                         min_gap=min_gap, avg_gap=avg_gap, max_gap=max_gap))
  with tf.io.gfile.GFile(os.path.join(base_dir, 'mdp_stats.pkl'), 'w') as f:
    pickle.dump(mdp_stats, f)

if __name__ == '__main__':
  app.run(main)

