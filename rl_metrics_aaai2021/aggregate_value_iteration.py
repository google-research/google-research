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

"""This file runs value iteration on an aggregated state space.

It aggregates states using the supplied metric.

This module will run a number of trials on a set of possible metrics and compile
the results in a plot.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import logging
import gin
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from six.moves import range
import tensorflow.compat.v1 as tf


def greedy(metric, num_states, num_states_target, max_iterations,
           verbose=False):
  """Greedily aggregate states until a desired number of aggregate states.

  Args:
    metric: matrix of distances.
    num_states: int, number of total states.
    num_states_target: int, desired number of states.
    max_iterations: int, maximum number of iterations to run algorithm.
    verbose: bool, whether to print verbose messages.

  Returns:
    list of aggregated states and list mapping state to its cluster.
  """
  curr_metric = np.copy(metric)
  # First we ensure that we won't aggregate states with themselves.
  np.fill_diagonal(curr_metric, np.inf)
  aggregate_states = [[x] for x in range(num_states)]
  state_to_aggregate_states = list(range(num_states))
  num_iterations = 1
  while len(aggregate_states) > num_states_target:
    # Pick a pair of the closest states randomly.
    min_distance = np.min(curr_metric)
    # We add a little epsilon here to avoid floating point precision issues.
    x, y = np.where(curr_metric <= min_distance + 1e-8)
    i = np.random.randint(len(x))
    s, t = x[i], y[i]
    # So we no longer try to aggregate these states.
    curr_metric[s, t] = np.inf
    curr_metric[t, s] = np.inf
    # For simplicity we'll put the new aggregation at the front.
    c1 = state_to_aggregate_states[s]
    c2 = state_to_aggregate_states[t]
    new_aggregate_states = [[]]
    for c in [c1, c2]:
      for s in aggregate_states[c]:
        if s in new_aggregate_states[0]:
          # If c1 == c2, this would cause duplicates which causes never-ending
          # loops.
          continue
        new_aggregate_states[0].append(s)
        state_to_aggregate_states[s] = 0
    # Re-index all the other aggregations.
    for i, c in enumerate(aggregate_states):
      if i == c1 or i == c2:
        continue
      for s in c:
        state_to_aggregate_states[s] = len(new_aggregate_states)
      new_aggregate_states.append(c)
    aggregate_states = new_aggregate_states
    if num_iterations % 1000 == 0 and verbose:
      logging.info('Iteration %d', num_iterations)
    num_iterations += 1
    if num_iterations > max_iterations:
      break
  return aggregate_states, state_to_aggregate_states


def k_medians(metric, num_states, num_states_target, max_iterations,
              verbose=False):
  """Aggregate states using the k-medians algorithm.

  Args:
    metric: matrix of distances.
    num_states: int, number of total states.
    num_states_target: int, desired number of states.
    max_iterations: int, maximum number of iterations to run algorithm.
    verbose: bool, whether to print verbose messages.

  Returns:
    list of aggregated states and dict mapping state to its cluster.
  """
  # Pick an initial set of centroids.
  centroids = np.random.choice(num_states, size=num_states_target,
                               replace=False)
  state_to_centroid = [0 for _ in range(num_states)]
  for k, s in enumerate(centroids):
    state_to_centroid[s] = k
  # We first put each state in a random cluster.
  for s in range(num_states):
    if s in centroids:
      continue
    k = s % num_states_target
    state_to_centroid[s] = k
  clusters_changing = True
  num_iterations = 1
  while clusters_changing:
    clusters_changing = False
    clusters = [[x] for x in centroids]
    for s in range(num_states):
      if s in centroids:
        continue
      nearest_centroid = 0
      smallest_distance = np.inf
      for k, t in enumerate(centroids):
        if metric[s, t] < smallest_distance:
          smallest_distance = metric[s, t]
          nearest_centroid = k
      if nearest_centroid != state_to_centroid[s]:
        clusters_changing = True
      state_to_centroid[s] = nearest_centroid
      clusters[nearest_centroid].append(s)
    # Re-calculate centroids.
    for k, c in enumerate(clusters):
      min_avg_distance = np.inf
      new_centroid = 0
      for s in c:
        avg_distance = 0.
        for t in c:
          avg_distance += metric[s, t]
        avg_distance /= len(c)
        if avg_distance < min_avg_distance:
          min_avg_distance = avg_distance
          new_centroid = s
      centroids[k] = new_centroid
    if num_iterations % 1000 == 0 and verbose:
      logging.info('Iteration %d', num_iterations)
    num_iterations += 1
    if num_iterations > max_iterations:
      break
  return clusters, state_to_centroid


@gin.configurable
def value_iteration(env, aggregate_states, tolerance=0.001, verbose=False):
  r"""Run value iteration on the aggregate MDP.

  This constructs a new MDP using the aggregate states as follows:
  ```
  R(c, a) = 1/|c| * \sum_{s \in c} R(s, a)
  P(c, a)(c') = 1/|c| * \sum_{s \in c}\sum_{s' \in c'} P(s, a)(s')
  ```

  Args:
    env: the original environment.
    aggregate_states: list of aggregate states.
    tolerance: float, maximum difference in value between successive
      iterations. Once this threshold is past, computation stops.
    verbose: bool, whether to print verbose messages.

  Returns:
    list of floats representing cluster values.
  """
  num_clusters = len(aggregate_states)
  transition_probs = np.zeros((num_clusters, env.num_actions, num_clusters))
  rewards = np.zeros((num_clusters, env.num_actions))
  for c1 in range(num_clusters):
    for a in range(env.num_actions):
      for s1 in aggregate_states[c1]:
        rewards[c1, a] += env.rewards[s1, a]
        for c2 in range(num_clusters):
          for s2 in aggregate_states[c2]:
            transition_probs[c1, a, c2] += env.transition_probs[s1, a, s2]
      rewards[c1, a] /= len(aggregate_states[c1])
      transition_probs[c1, a, :] /= len(aggregate_states[c1])
  q_values = np.zeros((num_clusters, env.num_actions))
  error = tolerance * 2.
  num_iterations = 1
  while error > tolerance:
    for c in range(num_clusters):
      for a in range(env.num_actions):
        old_q_values = np.copy(q_values[c, a])
        q_values[c, a] = rewards[c, a] + env.gamma * np.matmul(
            transition_probs[c, a, :], np.max(q_values, axis=1))
        error = np.max(abs(q_values[c, a] - old_q_values))
    if num_iterations % 1000 == 0 and verbose:
      logging.info('Iteration %d: %f', num_iterations, error)
    num_iterations += 1
  return q_values


@gin.configurable
def experiment(base_dir,
               env,
               metrics,
               max_iterations=100,
               run=0,
               random_mdp=False,
               verbose=False,
               aggregation_method='greedy'):
  """Module to run the experiment.

  Args:
    base_dir: str, base directory where to save the files.
    env: an environment specifying the true underlying MDP.
    metrics: list of metrics which will be used for the nearest-neighbour
      approximants.
    max_iterations: int, maximum number of iterations for each of the
      aggregation methods.
    run: int, run id.
    random_mdp: bool, whether the environment is a random MDP or not.
    verbose: bool, whether to print verbose messages.
    aggregation_method: string, greedy or k_median method

  Returns:
    Dict containing statistics.
  """
  if env.values is None:
    logging.info('Values must have already been computed.')
    return
  cmap = cm.get_cmap('plasma', 256)
  data = {
      'Metric': [],
      'num_states_target': [],
      'run': [],
      'qg': [],
      'exact_qvalues': [],
      'error': []
  }
  num_states_targets = np.linspace(1, env.num_states, 10).astype(int)
  for num_states_target in num_states_targets:
    # -(-x//1) is the same as ceil(x).
    # num_states_target = max(int(-(-state_fraction * env.num_states // 1)), 1)
    for metric in metrics:
      if metric.metric is None:
        continue
      if verbose:
        logging.info('***Run %d, %s, %d',
                     num_states_target, metric.name, run)
      if aggregation_method == 'k_median':
        aggregate_states, state_to_aggregate_states = (
            k_medians(
                metric.metric,
                env.num_states,
                num_states_target,
                max_iterations,
                verbose=verbose))
      if aggregation_method == 'greedy':
        aggregate_states, state_to_aggregate_states = (
            greedy(
                metric.metric,
                env.num_states,
                num_states_target,
                max_iterations,
                verbose=verbose))
      if not random_mdp:
        # Generate plot of neighborhoods.
        neighbourhood_path = os.path.join(
            base_dir, metric.name,
            'neighborhood_{}_{}.pdf'.format(num_states_target, run))
        obs_image = env.render_custom_observation(
            env.reset(), state_to_aggregate_states, cmap,
            boundary_values=[-1, num_states_target])
        plt.imshow(obs_image)
        with tf.gfile.GFile(neighbourhood_path, 'w') as f:
          plt.savefig(f, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()
      # Perform value iteration on aggregate states.
      q_aggregate = value_iteration(env, aggregate_states)
      # Now project the values of the aggregate states to the ground states.
      q_projected = [
          q_aggregate[state_to_aggregate_states[s]]
          for s in range(env.num_states)]
      data['Metric'].append(metric.label)
      data['num_states_target'].append(num_states_target)
      data['run'].append(run)
      data['qg'].append(q_projected)
      data['exact_qvalues'].append(env.q_val_it_q_values)
      data['error'].append(
          np.mean(
              np.max((np.abs(q_projected - env.q_val_it_q_values)), axis=1)))
  return data


def plot_data(base_dir, data):
  """Plot the data collected from all experiment runs."""
  del data['qg']
  del data['exact_qvalues']
  df = pd.DataFrame(data=data)
  plt.subplots(1, 1, figsize=(8, 6))
  sns.lineplot(x='num_states_target', y='error', hue='Metric', data=df,
               ci=99, lw=3)
  plt.xlabel('Number of aggregate states', fontsize=24)
  plt.ylabel('Avg. Error', fontsize=24)
  plt.legend(fontsize=18)
  pdf_file = os.path.join(base_dir, 'aggregate_value_iteration.pdf')
  with tf.io.gfile.GFile(pdf_file, 'w') as f:
    plt.savefig(f, format='pdf', dpi=300, bbox_inches='tight')
  plt.clf()
  plt.close('all')
