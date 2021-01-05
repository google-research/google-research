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

"""Utils for experiments on clustering linear dynamical systems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
import timeit
import warnings

# pylint: disable=g-bad-import-order
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab  # pylint: disable=g-import-not-at-top
import numpy as np
import pandas as pd
import seaborn as sns
import six
from sklearn import cluster
from sklearn import metrics
from statsmodels.tools import sm_exceptions
import tslearn
from tslearn import clustering as tslearn_clustering

import lds

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def pad_seqs_to_matrix(seqs, max_seq_len=None):
  if max_seq_len is None:
    max_seq_len = np.max([s.seq_len for s in seqs])
  output_dim = seqs[0].output_dim
  padded = np.zeros((len(seqs), max_seq_len * output_dim))
  for i, s in enumerate(seqs):
    for j in xrange(output_dim):
      padded[i, max_seq_len * j:max_seq_len * j + s.seq_len] = s.outputs[:, j]
  return padded


# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
def generate_cluster_centers(num_clusters,
                             hidden_state_dim,
                             input_dim,
                             cluster_center_dist_lower_bound,
                             diagonalizable=True):
  """Generates cluster center eigenvalues with distance requirement.

  The generated eigenvalues are drawn uniformly from (-1, 1) until the
  pairwise distance between cluster center eigenvalues >= lower bound.
  """
  min_cluster_center_dist = -1.
  while min_cluster_center_dist < cluster_center_dist_lower_bound:
    cluster_centers = []
    for _ in xrange(num_clusters):
      c = lds.generate_linear_dynamical_system(
          hidden_state_dim, input_dim, diagonalizable=diagonalizable)
      cluster_centers.append(c)
    min_cluster_center_dist = np.inf
    for s1 in xrange(num_clusters):
      for s2 in xrange(s1 + 1, num_clusters):
        d = lds.eig_dist(cluster_centers[s1], cluster_centers[s2])
        if d < min_cluster_center_dist:
          min_cluster_center_dist = d
  logging.info('generated min_cluster_center_dist %.2f',
               min_cluster_center_dist)
  return cluster_centers


def generate_lds_clusters(cluster_centers,
                          num_systems,
                          cluster_radius,
                          diagonalizable=True):
  """Generates clusters of linear dynamical systems.

  Args:
    cluster_centers: A list of LinearDynamicalSystem instances.
    num_systems: Total number of systems in all clusters.
    cluster_radius: Desired mean distance from the centers.

  Returns:
    - A list of LinearDynamicalSystem of size num_systems.
    - A list of true cluster ids.
  """
  num_clusters = len(cluster_centers)
  cluster_id = np.random.randint(0, num_clusters, num_systems)
  hidden_state_dim = cluster_centers[0].hidden_state_dim
  for c in cluster_centers:
    if c.hidden_state_dim != hidden_state_dim:
      raise ValueError('Hidden state dimension mismatch.')
  generated_systems = []
  dist_to_center = np.zeros(num_systems)
  for i in xrange(num_systems):
    c = cluster_centers[cluster_id[i]]
    if diagonalizable:
      eigvalues_new = c.get_spectrum() + cluster_radius / np.sqrt(
          hidden_state_dim) * np.random.randn(hidden_state_dim)
      generated_systems.append(
          lds.generate_linear_dynamical_system(
              hidden_state_dim, eigvalues=eigvalues_new))
    else:
      transition_matrix = c.transition_matrix + cluster_radius / np.sqrt(
          hidden_state_dim) * np.random.randn(hidden_state_dim,
                                              hidden_state_dim)
      generated_systems.append(
          lds.LinearDynamicalSystem(
              transition_matrix,
              np.random.randn(c.input_matrix.shape[0], c.input_matrix.shape[1]),
              np.random.randn(c.output_matrix.shape[0],
                              c.output_matrix.shape[1])))
    dist_to_center[i] = lds.eig_dist(c, generated_systems[-1])

  # For logging purpose.
  dist_bw_centers = np.zeros((num_clusters, num_clusters))
  for i in xrange(num_clusters):
    for j in xrange(num_clusters):
      dist_bw_centers[i, j] = lds.eig_dist(cluster_centers[i],
                                           cluster_centers[j])
  logging.info('Distances between cluster centers:\n%s', str(dist_bw_centers))
  logging.info('Average distance from cluster centers: %.3f',
               np.average(dist_to_center))
  for i in xrange(num_clusters):
    logging.info('Eigenvalues of cluster center %d: %s', i,
                 str(cluster_centers[i].get_spectrum()))

  return generated_systems, cluster_id


def get_kmeans_clusters(sequences, num_clusters, transform_fn):
  """Computes clusters on transformed sequences using KMeans.

  Args:
    sequences: A list of LinearDynamicalSystemSequence objects.
    num_clusters: The desired number of clusters.
    transform_fn: Transformation to be applied on sequences before clustering.

  Returns:
    An array of shape [num_sequences], the cluster ids.
  """
  transformed = [transform_fn(s).flatten() for s in sequences]
  transformed = np.stack(transformed, axis=0)
  if num_clusters > 0:
    return cluster.KMeans(
        n_clusters=num_clusters).fit_predict(transformed), num_clusters
  max_score = -np.inf
  best_n = 0
  results_n = {}
  for n in xrange(2, 10):
    model = cluster.KMeans(n_clusters=n)
    results_n[n] = model.fit_predict(transformed)
    score_n = metrics.silhouette_score(transformed, results_n[n])
    print(score_n)
    if score_n > max_score:
      max_score = score_n
      best_n = n
  return results_n[best_n], best_n


def get_results(sequences,
                num_clusters,
                true_cluster_ids,
                transform_fns,
                include_tslearn,
                include_slow_methods=False):
  """Compares KMeans clustering results with different transform_fns.

  See Section 2.3.9 in https://scikit-learn.org/stable/modules/clustering.html
  for more details on metrics.

  Args:
    sequences: A list of LinearDynamicalSystemSequence objects.
    num_clusters: Desired number of clusters, may differ from ground truth.
    true_cluster_ids: Ground truth from generated data.
    true_systems: Ground truth LDS.
    transform_fns: A dict of transformation fns to be tested with clustering.
      For each fn, we call the fn on the sequence output first and then cluster
      based on the function return value.
    include_tslearn: Whether to include tslearn methods for comparison.
    save_visualization_path: Filepath to save plots for visualizing clusters.

  Returns:
    A pandas DataFrame with columns `method`, `t_secs`, `failed_ratio`, and
    columns for clustering metrics such as `adj_mutual_info` and `v_measure`.
  """
  cluster_ids = collections.OrderedDict({'true': true_cluster_ids})
  pred_n_clusters = collections.OrderedDict(
      {'true': np.max(true_cluster_ids) + 1})
  transform_fns = collections.OrderedDict(transform_fns)
  t_record = collections.OrderedDict({'true': 0.0})
  failure_cnt = collections.OrderedDict({'true': 0})
  for k, fn in transform_fns.iteritems():
    logging.info('Running clustering method %s.', k)
    start_t = timeit.default_timer()
    with warnings.catch_warnings(record=True) as caught:
      warnings.filterwarnings(
          'always', category=sm_exceptions.ConvergenceWarning)
      cluster_ids[k], pred_n_clusters[k] = get_kmeans_clusters(
          sequences, num_clusters, fn)
    t_elapsed = timeit.default_timer() - start_t
    t_record[k] = t_elapsed
    failure_cnt[k] = 0
    logging.info('Caught %d warnings from %s.', len(caught), k)
    for w in caught:
      if w.category in [
          RuntimeWarning, sm_exceptions.ConvergenceWarning,
          sm_exceptions.HessianInversionWarning
      ]:
        failure_cnt[k] += 1
      else:
        warnings.warn(w.message, w.category)
  if include_tslearn:
    # kShape is supposed to work on monomodal data.
    max_seq_len = np.max([s.seq_len for s in sequences])
    x_train = pad_seqs_to_matrix(sequences)[:, :max_seq_len]
    if include_slow_methods:
      tslearn_methods = {
          'gak_km':
              tslearn_clustering.GlobalAlignmentKernelKMeans(
                  n_clusters=num_clusters,
                  n_init=2,
                  sigma=tslearn.metrics.sigma_gak(x_train),
                  verbose=True,
                  random_state=0),
          'dtw_km':
              tslearn_clustering.TimeSeriesKMeans(
                  n_clusters=num_clusters, verbose=False, metric='softdtw'),
          'kshape':
              tslearn_clustering.KShape(n_clusters=num_clusters, verbose=False),
      }
    else:
      tslearn_methods = {
          'kshape':
              tslearn_clustering.KShape(n_clusters=num_clusters, verbose=False),
      }
    ts = tslearn.utils.to_time_series_dataset(x_train)
    for k, m in tslearn_methods.iteritems():
      logging.info('Running clustering method %s.', k)
      start_t = timeit.default_timer()
      cluster_ids[k] = m.fit_predict(ts)
      t_elapsed = timeit.default_timer() - start_t
      t_record[k] = t_elapsed
      failure_cnt[k] = 0
      if cluster_ids[k] is None:
        cluster_ids[k] = np.random.randint(
            low=0, high=num_clusters, size=len(sequences))
        failure_cnt[k] = len(sequences)

  metric_fns_with_truth = {
      'adj_rand_score':
          metrics.adjusted_rand_score,
      'adj_mutual_info':
          metrics.adjusted_mutual_info_score,
      'fowlkes_mallows':
          metrics.fowlkes_mallows_score,
      'homogeneity':
          metrics.homogeneity_score,
      'completeness':
          metrics.completeness_score,
      # pylint: disable=g-long-lambda
      'v_measure': (lambda t, pred: metrics.homogeneity_completeness_v_measure(
          t, pred)[2]),
  }
  metric_dict = {k: [] for k in metric_fns_with_truth.keys()}
  metric_dict['t_secs'] = t_record.values()
  metric_dict['failed_ratio'] = np.array(failure_cnt.values()).astype(
      np.float32) / len(true_cluster_ids)
  metric_dict['method'] = []
  metric_dict['pred_n_clusters'] = []
  for k, pred_ids in cluster_ids.iteritems():
    for metric_key, metric_fn in metric_fns_with_truth.iteritems():
      try:
        metric_dict[metric_key].append(metric_fn(true_cluster_ids, pred_ids))
      except Exception as e:  # pylint: disable=broad-except
        metric_dict[metric_key].append(0.0)
        logging.info('Error computing %s: %s', metric_key, e.message)
    metric_dict['method'].append(k)
    metric_dict['pred_n_clusters'].append(pred_n_clusters.get(k, np.nan))
  return pd.DataFrame(data=metric_dict).set_index('method').reset_index()


def plot_sample_trajectories(sequences,
                             cluster_ids_dict,
                             plot_filepath,
                             num_selected_examples=10):
  """Visualization of the trajectory in the first dim of outputs.

  Args:
    sequences: A list of DynamicalSystemSequence objects.
    cluster_ids_dict: A dict mapping clustering method name as keys to resulting
      cluster ids.
    plot_filepath: Path to save the plot.
    num_selected_examples: Number of examples to plot.
  """
  color_list = ['red', 'blue', 'green', 'orange', 'brown']
  selected_examples = np.random.randint(0, len(sequences),
                                        num_selected_examples)
  seq_len = sequences[0].seq_len
  pylab.figure(figsize=(12, 16))
  for m, k in enumerate(cluster_ids_dict):
    pylab.subplot(len(cluster_ids_dict), 1, m + 1)
    for i in selected_examples:
      pylab.plot(
          range(seq_len),
          sequences[i].outputs,
          color=color_list[cluster_ids_dict[k][i]])
    pylab.title(k)
  output = six.StringIO()
  pylab.savefig(output, format='png')
  image = output.getvalue()
  with open(plot_filepath, 'w+') as f:
    f.write(image)


def visualize_clusters(seqs, true_systems, true_cluster_ids, transform_fns,
                       plot_filepath):
  """Visualizes learned params with colors of ground truth clusters.

  Only plots the first two dimensions of learned params.

  Args:
    seqs: A list of numpy arrays.
    true_systems: A list of LinearDynamicalSystem objects.
    true_cluster_ids: A numpy array of ground truth cluster ids.
    transform_fns: The transfrom fns for clustering.
    filepath: Path to save the plot.
  """
  num_subplots = len(transform_fns)
  subplot_id = 1
  subplot_cols = (num_subplots + 1) / 2
  pylab.figure(figsize=(4 * 2, 4 * subplot_cols))
  if true_systems is not None:
    num_subplots += 1
    subplot_cols = (num_subplots + 1) / 2
    pylab.figure(figsize=(4 * 2, 4 * subplot_cols))
    pylab.subplot(subplot_cols, 2, subplot_id)
    subplot_id += 1
    ax = sns.scatterplot(
        x=[a.get_spectrum()[0] for a in true_systems],
        y=[a.get_spectrum()[1] for a in true_systems],
        palette=sns.color_palette('Set2',
                                  np.max(true_cluster_ids) + 1),
        hue=true_cluster_ids)
    ax.set(xlabel='true eig value 1', ylabel='true eig value 2')
    ax.set_title('Ground Truth')

  plot_data_collection = None
  for i, k in enumerate(transform_fns.keys()):
    pylab.subplot(subplot_cols, 2, subplot_id)
    subplot_id += 1
    learned_params = [transform_fns[k](s) for s in seqs]

    # pylint: disable=g-complex-comprehension
    plot_data = {str(i): [p[i] for p in learned_params] for i \
        in xrange(len(learned_params[0]))}
    plot_data['method'] = [k] * len(learned_params)
    plot_data['true_cluster_id'] = list(true_cluster_ids)
    if plot_data_collection is None:
      plot_data_collection = plot_data
    else:
      for data_key in plot_data:
        plot_data_collection[data_key].extend(plot_data[data_key])
    ax = sns.scatterplot(
        x=[p[0] for p in learned_params],
        y=[p[1] for p in learned_params],
        palette=sns.color_palette('Set2', len(np.unique(true_cluster_ids))),
        hue=true_cluster_ids)
    ax.set(xlabel='Learned param 1', ylabel='Learned param 2')
    ax.set_title(k)
  pd.DataFrame(plot_data_collection).to_csv(plot_filepath + '_data.csv')
  pylab.subplots_adjust(hspace=0.3, wspace=0.4)
  output = six.StringIO()
  pylab.savefig(output, format='png')
  image = output.getvalue()
  with open(plot_filepath, 'w+') as f:
    f.write(image)
