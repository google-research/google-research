# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

r"""Script for running experiments.

Example to run locally:
python experiments.py --output_dir=may19_3d --hidden_state_dim=3 \
        --min_seq_len=100 --max_seq_len=2000 --num_sampled_seq_len=20 \
        --num_systems=100 --num_repeat=100 \
        --cluster_center_dist_lower_bound=0.1 --hide_inputs=true
The outputs will show up in output_dir may19_3d.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
import os

# pylint: disable=g-bad-import-order
from absl import app
from absl import flags
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab  # pylint: disable=g-import-not-at-top
import numpy as np
import pandas as pd
import seaborn as sns
import six
import sklearn
import tqdm

import arma
import clustering
import lds

sns.set(style='whitegrid')

FLAGS = flags.FLAGS

# Flags for IO and plotting.
flags.DEFINE_string('output_dir', None, 'Output filepath.')
flags.DEFINE_boolean(
    'load_results', False, 'Whether to skip experiments '
    'and only plot existing results from output_dir.')
flags.DEFINE_boolean(
    'plot_clusters', False, 'Whether to visualize each '
    'experiment run and plot clustering results.')

# Flags for generating simulated clusters of LDSs.
flags.DEFINE_boolean('generate_diagonalizable_only', False, 'Whether to only '
                     'generate diagonalizable LDSs.')
flags.DEFINE_integer('num_clusters', 2, 'Number of clusters in experiments.')
flags.DEFINE_integer('num_systems', 100,
                     'Number of dynamical systems to cluster.')
flags.DEFINE_integer('hidden_state_dim', 2, 'Hidden state dim in experiments.')
flags.DEFINE_integer('input_dim', 1, 'Input dim in experiments.')
flags.DEFINE_boolean(
    'hide_inputs', True, 'Whether the inputs are observable '
    'to the clustering algorithm.')
flags.DEFINE_spaceseplist(
    'cluster_center_eigvalues', None, 'Optional List of lists of eigenvalues '
    'for each cluster. The outer list is space separated, and the inner list '
    'is comma separated. E.g. `0.9,0.1 0.5,0.1`. When null, generate random '
    'clusters centers by drawing eigenvalues uniformly from [-1, 1].')
flags.DEFINE_float(
    'cluster_center_dist_lower_bound', 0.2, 'Desired distance lower bound '
    'between cluster centers. Only applicable when cluster_center_eigvalues '
    'is None. Generate cluster centers until distance >= this val.')
flags.DEFINE_float('cluster_radius', 0.05,
                   'Radius of each dynamical system cluster.')
flags.DEFINE_integer('random_seed', 0, 'Random seed.')
flags.DEFINE_integer('num_repeat', 1,
                     'Number of repeated runs for each fixed seq len.')

# Flags for output sequences from LDSs.
flags.DEFINE_integer('min_seq_len', 10, 'Min seq len in experiments.')
flags.DEFINE_integer('max_seq_len', 1000, 'Max seq len in experiments.')
flags.DEFINE_integer(
    'num_sampled_seq_len', 10, 'Number of sampled seq len '
    'values in between min and max seq len.')
flags.DEFINE_float('input_mean', 0.0, 'Input mean.')
flags.DEFINE_float('input_stddev', 1.0, 'Input stddev.')
flags.DEFINE_float('output_noise_stddev', 0.01, 'Output noise stddev.')
flags.DEFINE_float('init_state_mean', 0.0, 'Init state mean.')
flags.DEFINE_float('init_state_stddev', 0.0, 'Init state stddev.')

# Flags for hparams in clustering algorithms.
flags.DEFINE_integer('guessed_hidden_dim', 0,
                     'Assumed hidden dim. If 0, use true hidden dim.')
flags.DEFINE_integer(
    'guessed_num_clusters', 0,
    'Desired number of clusters. If 0, find best number '
    'adaptively from maximizing kmeans objective score.')
flags.DEFINE_integer(
    'LDS_GIBBS_num_update_samples', 100, 'Number of update '
    'samples used for fitting LDS in pylds package.')
flags.DEFINE_integer('spectral_filtering_num_filters', 25, 'Number of filters '
                     'used in spectral filtering method.')
flags.DEFINE_float('spectral_filtering_learning_rate', 0.0001, 'Learning rate '
                   'in spectral filtering method.')

# Flags for whether to include certain baselines.
flags.DEFINE_boolean(
    'include_LDS_MLE', False, 'Whether to include MLE '
    'estimation for LDS in the experiments. Could be slow.')
flags.DEFINE_boolean(
    'include_tslearn', True, 'Whether to include time series '
    'clustering methods from the tslearn package in the '
    'experiments.')
flags.DEFINE_boolean(
    'include_tslearn_slow', False, 'Whether to include time '
    'series clustering methods from the tslearn package '
    'that are slow: DTW and GAK.')
flags.DEFINE_boolean('include_LDS_GIBBS', True, 'Whether to include the '
                     'Gibbs sampling method for LDS.')
flags.DEFINE_boolean('include_ARMA_MLE', False, 'Whether to include the '
                     'MLE method for ARMA.')


def create_model_fns(hdim):
  """Util function to create model fns to fit model params to sequences.

  Args:
    hdim: Guessed hidden dimension for model fitting.

  Returns:
    A dictionary mapping method names to model_fns. Each model_fn
    takes output seq and input seq, and returns fitted model params.
  """
  model_fns = collections.OrderedDict()
  # Using raw outputs.
  # model_fns['raw_output'] = lambda o, i: o
  # pylint: disable=g-long-lambda
  # Pure AR.
  model_fns['AR'] = lambda o, i: arma.fit_ar(o, i, hdim)
  # Iterated regression without regularization and with regularization.
  model_fns['ARMA_OLS'] = lambda o, i: arma.fit_arma_iter(o, i, hdim)
  model_fns['ARMA_RLS'] = lambda o, i: arma.fit_arma_iter(
      o, i, hdim, l2_reg=0.01)
  # Fit AR model and cluster based on AR param roots.
  # model_fns['AR_roots'] = lambda o, i: arma.get_eig_from_arparams(
  #     arma.fit_ar(o, i, hdim))
  # Fit ARMA model and cluster based on AR param roots.
  # model_fns['ARMA_OLS_roots'] = lambda o, i: arma.get_eig_from_arparams(
  #     arma.fit_arma_iter(o, i, hdim))
  # model_fns['ARMA_RLS_roots_0.01'] = lambda o, i: arma.get_eig_from_arparams(
  #     arma.fit_arma_iter(o, i, hdim, l2_reg=0.01))
  if FLAGS.include_LDS_GIBBS:
    model_fns['LDS_GIBBS'] = lambda o, i: lds.fit_lds_gibbs(
        o, i, hdim, num_update_samples=FLAGS.LDS_GIBBS_num_update_samples)
  if FLAGS.include_ARMA_MLE:
    model_fns['ARMA_MLE'] = lambda o, i: arma.fit_arma_mle(o, i, hdim)
  if FLAGS.include_LDS_MLE:
    model_fns['LDS_MLE'] = lambda o, i: lds.fit_lds_mle(o, i, hdim)
  return model_fns


def _compose_model_fn(model_fn):
  if FLAGS.hide_inputs:
    return lambda seq: model_fn(seq.outputs, None)
  return lambda seq: model_fn(seq.outputs, seq.inputs)


def _create_pca_model_fn(pca_model):
  return lambda o, _: pca_model.transform(o.flatten()).flatten()


# pylint: disable=g-doc-args
def get_results(cluster_center_eigvalues,
                cluster_center_dist_lower_bound,
                hidden_state_dim,
                input_dim,
                guessed_hidden_dim,
                num_clusters,
                guessed_num_clusters,
                min_seq_len,
                max_seq_len,
                num_sampled_seq_len,
                num_repeat,
                num_systems,
                cluster_radius,
                input_mean,
                input_stddev,
                output_noise_stddev,
                init_state_mean=0.0,
                init_state_stddev=0.0,
                generate_diagonalizable_only=False,
                random_seed=0,
                results_path=None):
  """Get results for varying sequence lengths.

  Args:
    cluster_center_eigvalues: List of lists of eigenvalues for each cluster.
      E.g. [[0.9,0.1], [0.5,0.1], [0.2,0.2], or None. If None, eigenvalues will
      be generated from uniform(-1,1) with respect to
      cluster_center_dist_lower_bound.
    cluster_center_dist_lower_bound: Desired distance lower bound between
      clusters. When generating cluster centers, try repeatedly until distance
      is greater than cluster_center_dist_lower_bound.
    hidden_state_dim: True hidden state dim.
    input_dim: The input dim.
    guessed_hidden_dim: Assumed hidden dim. If 0, use true hidden dim.
    num_clusters: True number of clusters.
    guessed_num_clusters: Desired number of clusters. If 0, use true number.
    min_seq_len: Min seq len in experiments.
    max_seq_len: Max seq len in experiments.
    num_sampled_seq_len: Number of sampled seq len values in between min and max
      seq len.
    num_repeat: Number of repeated experiments for each seq_len.
    num_systems: Number of dynamical system in each clustering experiments.
    cluster_radius: Expected distance of generated systems from cluster centers.
    input_mean: Scalar or 1D array of length hidden state dim.
    input_stddev: Scalar of 1D array of length hidden state dim.
    output_noise_stddev: Scalar.
    init_state_mean: Scalar or 1D array of length hidden state dim.
    init_state_stddev: Scalar of 1D array of length hidden state dim.
    random_seed: Random seed, integer.

  Returns:
    A pandas DataFrame with columns `method`, `seq_len`, `t_secs`,
    `failed_ratio`, and columns for clustering metrics such as `adj_mutual_info`
    and `v_measure`. The same method and seq_len will appear in num_repeat many
    rows.
  """
  if cluster_center_eigvalues is not None:
    if len(cluster_center_eigvalues) <= 1:
      raise ValueError('Need at least two cluster centers.')
    cluster_center_eigvalues = np.array(cluster_center_eigvalues)
    if cluster_center_eigvalues.shape != (num_clusters, hidden_state_dim):
      raise ValueError(
          'Cluter center eig has shape %s, expected (%d, %d).' %
          (str(cluster_center_eigvalues.shape), num_clusters, hidden_state_dim))
  np.random.seed(random_seed)
  progress_bar = tqdm.tqdm(total=num_repeat * num_sampled_seq_len)
  # Generator for output sequences.
  gen = lds.SequenceGenerator(
      input_mean=input_mean,
      input_stddev=input_stddev,
      output_noise_stddev=output_noise_stddev,
      init_state_mean=init_state_mean,
      init_state_stddev=init_state_stddev)
  seq_len_vals = np.linspace(min_seq_len, max_seq_len, num_sampled_seq_len)
  seq_len_vals = [int(round(x)) for x in seq_len_vals]
  if guessed_hidden_dim == 0:
    guessed_hidden_dim = hidden_state_dim
  if guessed_num_clusters == 0:
    guessed_num_clusters = num_clusters
  results_dfs = []
  for i in xrange(num_repeat):
    logging.info('---Starting experiments in repeat run #%d---', i)
    if cluster_center_eigvalues is not None:
      cluster_centers = []
      for eig_val in cluster_center_eigvalues:
        c = lds.generate_linear_dynamical_system(
            hidden_state_dim, input_dim, eigvalues=eig_val)
        cluster_centers.append(c)
    else:
      cluster_centers = clustering.generate_cluster_centers(
          num_clusters,
          hidden_state_dim,
          input_dim,
          cluster_center_dist_lower_bound,
          diagonalizable=generate_diagonalizable_only)
    true_systems, true_cluster_ids = clustering.generate_lds_clusters(
        cluster_centers,
        num_systems,
        cluster_radius,
        diagonalizable=generate_diagonalizable_only)
    for seq_len in seq_len_vals:
      logging.info('Running experiment with seq_len = %d.', seq_len)
      seqs = [gen.generate_seq(s, seq_len=seq_len) for s in true_systems]
      # Create transform_fns.
      model_fns = create_model_fns(guessed_hidden_dim)
      pca = sklearn.decomposition.PCA(n_components=guessed_hidden_dim).fit(
          np.stack([s.outputs.flatten() for s in seqs], axis=0))
      model_fns['PCA'] = _create_pca_model_fn(pca)
      transform_fns = collections.OrderedDict()
      for k in model_fns:
        transform_fns[k] = _compose_model_fn(model_fns[k])
      # Get clustering results.
      results_df = clustering.get_results(
          seqs,
          guessed_num_clusters,
          true_cluster_ids,
          true_systems,
          transform_fns,
          FLAGS.include_tslearn,
          include_slow_methods=FLAGS.include_tslearn_slow)
      results_df['seq_len'] = seq_len
      results_df['n_guessed_clusters'] = guessed_num_clusters
      results_df['n_true_clusters'] = num_clusters
      results_df['true_hidden_dim'] = hidden_state_dim
      results_df['guessed_hidden_dim'] = guessed_hidden_dim
      results_dfs.append(results_df)
      logging.info('Results:\n%s', str(results_df))
      plot_filepath = os.path.join(
          FLAGS.output_dir,
          'cluster_visualization_run_%d_seq_len_%d.png' % (i, seq_len))
      if FLAGS.plot_clusters:
        clustering.visualize_clusters(seqs, true_systems, true_cluster_ids,
                                      transform_fns, plot_filepath)
      progress_bar.update(1)
    if results_path:
      with open(results_path, 'w+') as f:
        pd.concat(results_dfs).to_csv(f, index=False)
  progress_bar.close()
  return pd.concat(results_dfs)


def plot_results(results_df, output_dir):
  """Plots metrics and saves plots as png files."""
  for metric_name in results_df.columns:
    if metric_name == 'seq_len' or metric_name == 'method':
      continue
    # Other than the silhouette metric, the metric value for ground truth is
    # always 1 for adj_mutual_info, adj_rand_score etc., so skip ground truth
    # in plotting.
    if metric_name != 'silhouette':
      results_df = results_df[results_df.method != 'true']
    pylab.figure()
    sns.pointplot(
        x='seq_len',
        y=metric_name,
        data=results_df,
        hue='method',
        scale=0.5,
        estimator=np.mean,
        err_style='bars',
        capsize=.1)
    output = six.StringIO()
    pylab.savefig(output, format='png')
    image = output.getvalue()
    with open(os.path.join(output_dir, metric_name + '.png'), 'w+') as f:
      f.write(image)


def main(unused_argv):
  if FLAGS.load_results:
    with open(os.path.join(FLAGS.output_dir, 'results.csv'), 'r') as f:
      combined_result_df = pd.read_csv(f, index_col=False)
    plot_results(combined_result_df, FLAGS.output_dir)
    return

  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
  cluster_center_eigvalues = None
  if FLAGS.cluster_center_eigvalues is not None:
    try:
      cluster_center_eigvalues = []
      for e in FLAGS.cluster_center_eigvalues:
        cluster_center_eigvalues.append([float(x) for x in e.split(',')])
    except:
      raise ValueError('Expected cluster_center_eigvalues to be list of comma '
                       'separated strings, e.g. [`0.9,0.1`, `0.5,0.1`].')
  with open(os.path.join(FLAGS.output_dir, 'flags.txt'), 'w+') as f:
    f.write(str(FLAGS.flag_values_dict()))
  results_path = os.path.join(FLAGS.output_dir, 'results_inc.csv')
  df = get_results(
      cluster_center_eigvalues,
      cluster_center_dist_lower_bound=FLAGS.cluster_center_dist_lower_bound,
      hidden_state_dim=FLAGS.hidden_state_dim,
      input_dim=FLAGS.input_dim,
      guessed_hidden_dim=FLAGS.guessed_hidden_dim,
      num_clusters=FLAGS.num_clusters,
      guessed_num_clusters=FLAGS.guessed_num_clusters,
      min_seq_len=FLAGS.min_seq_len,
      max_seq_len=FLAGS.max_seq_len,
      num_sampled_seq_len=FLAGS.num_sampled_seq_len,
      num_repeat=FLAGS.num_repeat,
      num_systems=FLAGS.num_systems,
      cluster_radius=FLAGS.cluster_radius,
      input_mean=FLAGS.input_mean,
      input_stddev=FLAGS.input_stddev,
      output_noise_stddev=FLAGS.output_noise_stddev,
      init_state_mean=FLAGS.init_state_mean,
      init_state_stddev=FLAGS.init_state_stddev,
      random_seed=FLAGS.random_seed,
      generate_diagonalizable_only=FLAGS.generate_diagonalizable_only,
      results_path=results_path)
  with open(os.path.join(FLAGS.output_dir, 'results.csv'), 'w+') as f:
    df.to_csv(f, index=False)
  plot_results(df, FLAGS.output_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('output_dir')
  app.run(main)
