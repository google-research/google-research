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

"""This file runs subsampled value iteration using metrics.

It estimates the value function for a set of points using a pre-defined state
metric to obtain the approximant from the nearest known neighbour.
This module will run a number of trials on a set of possible metrics and compile
the results in a plot.
"""

import os
import gin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.compat.v1 as tf


DEFAULT_SUBSAMPLING_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                 0.8, 0.9, 1.0)


@gin.configurable
def experiment(base_dir, env, metrics,
               subsampling_fractions=DEFAULT_SUBSAMPLING_FRACTIONS, run=0,
               random_mdp=True, verbose=False, aggregation_method=None):
  """Module to run the experiment.

  Args:
    base_dir: str, base directory where to save the files.
    env: an environment specifying the true underlying MDP.
    metrics: list of metrics which will be used for the nearest-neighbour
      approximants.
    subsampling_fractions: list of floats specifying the number of states to
      subsample for the approximant.
    run: int, run id.
    random_mdp: bool, whether the environment is a random MDP or not.
    verbose: bool, whether to print verbose messages.
    aggregation_method: str, dummy variable

  Returns:
    Dict containing statistics.
  """
  del base_dir
  del aggregation_method
  del random_mdp
  if env.values is None:
    tf.logging.info('Values must have already been computed.')
    return
  data = {
      'Metric': [],
      'num_known_states': [],
      'run': [],
      'avg_error': [],
      'max_error': [],
      'avg_error_q': [],
      'max_error_q': []
  }
  for subsample_fraction in subsampling_fractions:
    for metric in metrics:
      if metric.metric is None:
        continue
      if verbose:
        tf.logging.info('***Run {}, {}, {}'.format(
            subsample_fraction, metric.name, run))
      num_subsampled_states = int(env.num_states * subsample_fraction)
      num_known_states = env.num_states - num_subsampled_states
      subsamples = np.random.choice(env.num_states,
                                    size=num_subsampled_states, replace=False)
      subsampled_metric = np.copy(metric.metric)
      # We first set all distances between subsampled states to np.inf.
      for s1 in subsamples:
        for s2 in subsamples:
          subsampled_metric[s1, s2] = np.inf
      max_error = 0.
      avg_error = 0.
      max_error_q = 0.
      avg_error_q = 0.
      for subsample in subsamples:
        nearest_neighbor = np.argmin(subsampled_metric[subsample])
        value_estimate_error = abs(
            env.values[nearest_neighbor] - env.values[subsample])
        q_value_estimate_error = np.max(abs(
            env.q_values[nearest_neighbor] - env.q_values[subsample]))
        if value_estimate_error > max_error:
          max_error = value_estimate_error
        if q_value_estimate_error > max_error_q:
          max_error_q = q_value_estimate_error
        avg_error += value_estimate_error
        avg_error_q += q_value_estimate_error
      avg_error /= env.num_states
      avg_error_q /= env.num_states
      data['Metric'].append(metric.label)
      data['num_known_states'].append(num_known_states)
      data['run'].append(run)
      data['max_error'].append(max_error)
      data['avg_error'].append(avg_error)
      data['max_error_q'].append(max_error_q)
      data['avg_error_q'].append(avg_error_q)
  return data


def plot_data(base_dir, data):
  """Plot the data collected from all experiment runs."""
  df = pd.DataFrame(data=data)
  for error in ['max_error_q', 'avg_error_q']:
    plt.subplots(1, 1, figsize=(8, 6))
    sns.lineplot(x='num_known_states', y=error, hue='Metric', data=df,
                 ci=99, lw=3)
    plt.xlabel('Number of known states', fontsize=24)
    ylabel = r'$L_{\infty}$ Error' if error == 'max_error' else 'Avg. Error'
    plt.ylabel(ylabel, fontsize=24)
    plt.legend(fontsize=18)
    pdf_file = os.path.join(base_dir,
                            'subsampled_value_iteration_{}.pdf'.format(error))
    with tf.gfile.GFile(pdf_file, 'w') as f:
      plt.savefig(f, format='pdf', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close('all')
