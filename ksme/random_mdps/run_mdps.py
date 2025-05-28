# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

import os
import pickle

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf

from ksme.random_mdps import metric_registry
from ksme.random_mdps import random_mdp
from ksme.random_mdps import utils

flags.DEFINE_string('base_dir', None, 'Base directory to store stats.')
flags.DEFINE_multi_string('metrics', None, 'List of metrics to compute.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')
flags.DEFINE_bool('verbose', False, 'Whether to print verbose messages.')
flags.DEFINE_float('tolerance', 0.001, 'Error tolerance for value iteration.')
flags.DEFINE_float('gamma', 0.9, 'Discount factor to use for value iteration.')
flags.DEFINE_integer('num_runs', 10, 'The number of random MDPs to generate.')
flags.DEFINE_integer('num_states', 10, 'Number of states in MDP.')
flags.DEFINE_integer('num_actions', 4, 'Number of actions in MDP.')
flags.DEFINE_float('reward_variance', 1.0,
                   'Reward variance for sampled rewards.')
flags.DEFINE_integer('max_iterations', 100000,
                     'Maximum number of iterations to use for sammpling.')
flags.DEFINE_integer('seed', 0, 'PRNG seed to use in experiment')

FLAGS = flags.FLAGS


def compute_metrics(base_dir, env, metrics, mdp_stats, run_number):
  """Compute the requested metrics on the environment and update mdp_stats.

  Args:
    base_dir: str, base directory where to store statistics.
    env: Environment, a Random MDP.
    metrics: list of the metrics to compute.
    mdp_stats: dict, accumulated statistics.
    run_number: int, the mdp number.

  Returns:
    Updated mdp_stats with statistics from current run.
  """
  # We add the discount factor to the environment.
  env.gamma = FLAGS.gamma
  # We will store the values in our environment, as it may be used by the
  # metrics.
  env.values = utils.compute_value(env.policy_transition_probs,
                                   env.policy_rewards,
                                   env.gamma)
  for metric_name in metrics:
    if metric_name not in metric_registry.METRICS:
      logging.info('Unknown metric %s, skipping', metric_name)
      continue
    logging.info('Will compute %s', metric_name)
    metric = metric_registry.METRICS[metric_name].constructor(
        metric_name, metric_registry.METRICS[metric_name].label, env, base_dir,
        run_number=run_number)
    metric.compute(save_and_reload=False)
    avg_gap, min_gap, max_gap = metric.compute_gap()
    if metric.name not in mdp_stats:
      mdp_stats[metric.name] = []
    mdp_stats[metric.name].append(
        utils.MDPStats(
            metric.time,
            metric.num_iterations,
            avg_gap, min_gap, max_gap))
  return mdp_stats


def random_mdp_experiment(metrics):
  """Compute the metrics on a set of randomly generated MDPs."""
  gin.parse_config(bindings=FLAGS.gin_bindings, skip_unknown=False)
  base_dir = os.path.join(
      FLAGS.base_dir,
      f'{FLAGS.num_states}_{FLAGS.num_actions}_{FLAGS.reward_variance}')
  if not tf.io.gfile.exists(base_dir):
    tf.io.gfile.makedirs(base_dir)
  mdp_stats = {}
  for i in range(FLAGS.num_runs):
    logging.info('Starting run %d', i)
    env = random_mdp.RandomMDP(FLAGS.num_states, FLAGS.num_actions,
                               reward_variance=FLAGS.reward_variance)
    mdp_stats = compute_metrics(base_dir, env, metrics, mdp_stats, i)
  return base_dir, mdp_stats


def main(_):
  flags.mark_flags_as_required(['base_dir'])
  metrics = utils.METRICS if FLAGS.metrics is None else FLAGS.metrics
  base_dir, mdp_stats = random_mdp_experiment(metrics)
  with tf.io.gfile.GFile(os.path.join(base_dir, 'mdp_stats.pkl'), 'w') as f:
    pickle.dump(mdp_stats, f)

if __name__ == '__main__':
  app.run(main)
