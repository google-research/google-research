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

r"""Script for running experiments.

Example to run locally:
python bih.py --output_dir=bih_may21 --channel=both\
    --hdim=3 --num_clusters=2
The outputs will show up in output_dir ucr_may19.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
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

import arma
import clustering
import lds

FLAGS = flags.FLAGS

# Flags for IO and plotting.
flags.DEFINE_string('output_dir', None, 'Output filepath.')
flags.DEFINE_boolean('use_fixed_len', True, 'Whether to use fixed len 500.')
flags.DEFINE_boolean(
    'load_results', False, 'Whether to skip experiments '
    'and only plot existing results from output_dir.')
flags.DEFINE_boolean(
    'plot_clusters', False, 'Whether to visualize each '
    'experiment run and plot clustering results.')

flags.DEFINE_integer('num_repeat', 1,
                     'Number of repeated runs for bootstrapping neg examples.')
flags.DEFINE_integer('subsample_step_size', 1, '1 for not subsampling')
flags.DEFINE_string('channel', 'both', 'Which channel to use.')

# Flags for hparams in clustering algorithms.
flags.DEFINE_integer('hdim', 0, 'Hidden state dimension.')
flags.DEFINE_integer('num_clusters', 0, 'Desired number of clusters.')
flags.DEFINE_integer(
    'LDS_GIBBS_num_update_samples', 100, 'Number of update '
    'samples used for fitting LDS in pylds package.')
flags.DEFINE_integer('random_seed', 0, 'Random seed.')

# Flags for whether to include certain baselines.
flags.DEFINE_boolean(
    'include_LDS_MLE', False, 'Whether to include MLE '
    'estimation for LDS in the experiments. Could be slow.')
flags.DEFINE_boolean(
    'include_tslearn', True, 'Whether to include time series '
    'clustering methods from the tslearn package in the '
    'experiments.')
flags.DEFINE_boolean(
    'include_slow_methods', False, 'Whether to include the '
    'significantly slower methods, LDS_GIBBS, ARMA_MLE, and '
    'gak_km.')


def _replace_nan_with_0(arr):
  return np.where(np.isnan(arr), 0.0, arr)


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
  # model_fns['raw_output'] = lambda s: _replace_nan_with_0(s.outputs)
  # pylint: disable=g-long-lambda
  # Pure AR.
  model_fns['AR'] = lambda s: arma.fit_ar(s.outputs, None, hdim)
  # Iterated regression without regularization and with regularization.
  model_fns['ARMA_OLS'] = lambda s: arma.fit_arma_iter(s.outputs, None, hdim)
  model_fns['ARMA'] = lambda s: arma.fit_arma_iter(
      s.outputs, None, hdim, l2_reg=0.01)
  if FLAGS.include_slow_methods:
    model_fns['LDS'] = lambda s: lds.fit_lds_gibbs(
        s.outputs,
        None,
        hdim,
        num_update_samples=FLAGS.LDS_GIBBS_num_update_samples)
    model_fns['ARMA_MLE'] = lambda s: arma.fit_arma_mle(s.outputs, None, hdim)
  if FLAGS.include_LDS_MLE:
    model_fns['LDS_MLE'] = lambda s: lds.fit_lds_mle(s.outputs, None, hdim)
  return model_fns


def parse_csv(filename, hdim):
  """Parses ECG data from csv file."""
  df = pd.read_csv(filename)
  labels = df[df['input channel '] == 0].label.values
  seqs = []
  for i, row in df.iterrows():
    if i % 2:
      assert row[1] == 1
      continue
    assert row[1] == 0
    ch1_vals = row[3:].values.astype(np.float32).reshape(-1, 1)
    ch2_vals = df.iloc[i + 1][3:].values.astype(np.float32).reshape(-1, 1)
    if FLAGS.channel == 'both':
      values = np.concatenate([ch1_vals, ch2_vals], axis=1)
    elif FLAGS.channel == '1':
      values = ch1_vals
    elif FLAGS.channel == '2':
      values = ch2_vals
    else:
      raise ValueError('Channel flag expected to be both, 1, or 2.')
    seq_len = values.shape[0]
    if np.isnan(values).any():
      seq_len = np.min(np.where(np.isnan(values).any(axis=1)))
      assert np.isnan(values[seq_len:]).all()
      values = values[:seq_len]
    seqs.append(
        lds.LinearDynamicalSystemSequence(
            np.zeros((seq_len, 1)), np.zeros((seq_len, hdim)), values))
  return seqs, labels


def parse_examples_to_seq(hdim, pos_samples=25, neg_samples=25):
  """Parses ECG data."""
  if FLAGS.use_fixed_len:
    pos_seqs, pos_labels = parse_csv('mit-bih/pos-len500.csv', hdim)
    neg_seqs, neg_labels = parse_csv('mit-bih/neg-shuffled-len500.csv', hdim)
  else:
    pos_seqs, pos_labels = parse_csv('mit-bih/pos-sample.csv', hdim)
    neg_seqs, neg_labels = parse_csv('mit-bih/neg-shuffled-sample.csv', hdim)
  assert np.unique(pos_labels) == [1.]
  assert np.unique(neg_labels) == [0.]
  pos_seqs, pos_labels = sklearn.utils.resample(
      pos_seqs, pos_labels, n_samples=pos_samples)
  neg_seqs, neg_labels = sklearn.utils.resample(
      neg_seqs, neg_labels, n_samples=neg_samples)
  seqs = list(pos_seqs) + list(neg_seqs)
  labels = list(pos_labels) + list(neg_labels)
  if FLAGS.subsample_step_size > 1:
    seqs = subsample(seqs, FLAGS.subsample_step_size)
  return sklearn.utils.shuffle(seqs, labels)


def _subsample_rows(arr, step_size):
  return np.concatenate(
      [arr[j:j + 1, :] for j in xrange(0, arr.shape[0], step_size)], axis=0)


def subsample(sequences, step_size=5):
  subsampled = []
  for s in sequences:
    subsampled.append(
        lds.LinearDynamicalSystemSequence(
            _subsample_rows(s.inputs, step_size),
            _subsample_rows(s.hidden_states, step_size),
            _subsample_rows(s.outputs, step_size)))
  return subsampled


def get_results_bih_dataset(hdim, num_clusters, pos_samples=25, neg_samples=25):
  """Returns ECG clustering results in a pandas DataFrame."""
  seqs, labels = parse_examples_to_seq(hdim, pos_samples, neg_samples)
  model_fns = create_model_fns(hdim)
  padded = clustering.pad_seqs_to_matrix(seqs)
  max_seq_len = np.max([s.seq_len for s in seqs])
  pca = sklearn.decomposition.PCA(n_components=hdim).fit(padded)
  # pylint: disable=g-long-lambda
  model_fns['PCA'] = lambda s: pca.transform(
      clustering.pad_seqs_to_matrix([s], max_seq_len)).flatten()
  # Get clustering results.
  results_df = clustering.get_results(
      seqs,
      num_clusters,
      labels,
      None,
      model_fns,
      include_tslearn=FLAGS.include_tslearn)
  print(results_df)
  clustering.visualize_clusters(
      seqs, None, labels, model_fns,
      os.path.join(FLAGS.output_dir, 'visualization.png'))
  return results_df


def get_agg_stats(df):
  """Returns aggregated stats."""
  for metric in df.columns.values:
    if metric == 'method':
      continue
    stats = df.groupby(['method'])[metric].agg(['mean', 'count', 'std'])
    ci95_hi = []
    ci95_lo = []
    mean_w_ci = []
    for i in stats.index:
      m, c, s = stats.loc[i]
      ci95_hi.append(m + 1.96 * s / np.sqrt(c))
      ci95_lo.append(m - 1.96 * s / np.sqrt(c))
      mean_w_ci.append(
          '%.2f (%.2f-%.2f)' %
          (m, m - 1.96 * s / np.sqrt(c), m + 1.96 * s / np.sqrt(c)))
    stats['ci95_hi'] = ci95_hi
    stats['ci95_lo'] = ci95_lo
    stats['mean_w_ci'] = mean_w_ci
    print(metric)
    print(stats[['mean_w_ci']])
    stats.to_csv(os.path.join(FLAGS.output_dir, metric + '_agg.csv'))


def plot_results(results_df, output_dir):
  """Plots metrics and saves plots as png files."""
  for metric_name in results_df.columns:
    if metric_name == 'seq_len' or metric_name == 'method':
      continue
    pylab.figure()
    sns.lineplot(
        x='seq_len',
        y=metric_name,
        data=results_df,
        hue='method',
        estimator=np.mean,
        err_style='bars')
    output = six.StringIO()
    pylab.savefig(output, format='png')
    image = output.getvalue()
    with open(os.path.join(output_dir, metric_name + '.png'), 'w+') as f:
      f.write(image)


def main(unused_argv):
  np.random.seed(0)
  if FLAGS.load_results:
    with open(os.path.join(FLAGS.output_dir, 'results.csv'), 'r') as f:
      results_df = pd.read_csv(f, index_col=False)
    plot_results(results_df, FLAGS.output_dir)
    return
  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
  combined_results_list = []
  for _ in xrange(FLAGS.num_repeat):
    results_df = get_results_bih_dataset(FLAGS.hdim, FLAGS.num_clusters)
    combined_results_list.append(results_df)
  results_df = pd.concat(combined_results_list)
  with open(os.path.join(FLAGS.output_dir, 'flags.txt'), 'w+') as f:
    f.write(str(FLAGS.flag_values_dict()))
  with open(os.path.join(FLAGS.output_dir, 'results.csv'), 'w+') as f:
    results_df.to_csv(f, index=False)
  get_agg_stats(results_df)
  # plot_results(results_df, FLAGS.output_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('output_dir')
  app.run(main)
