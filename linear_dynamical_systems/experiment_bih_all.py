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
python bih.py --output_dir=bih_may21 --channel=both\
    --hdim=3 --num_clusters=2
The outputs will show up in output_dir ucr_may19.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import logging
import os

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

# pylint: disable=g-bad-import-order
import arma
import clustering
import lds


FLAGS = flags.FLAGS

# Flags for IO and plotting.
flags.DEFINE_string('output_dir', None, 'Output filepath.')
flags.DEFINE_boolean(
    'load_results', False, 'Whether to skip experiments '
    'and only plot existing results from output_dir.')
flags.DEFINE_boolean(
    'plot_clusters', False, 'Whether to visualize each '
    'experiment run and plot clustering results.')

flags.DEFINE_integer('sample_size', None, 'Sample size of signals for each '
                     'clustering run.')
flags.DEFINE_boolean(
    'filter_type', False, 'Whether to select only certain '
    'types of labels according to prior work.')
flags.DEFINE_integer(
    'label_count_threshold', 0, 'Threshold for label counts, '
    'label as `other` if below the threshold.')
flags.DEFINE_integer('num_repeat', 1,
                     'Number of repeated runs for bootstrapping neg examples.')
flags.DEFINE_integer('subsample_step_size', 1, '1 for not subsampling')
flags.DEFINE_string('channel', 'both', 'Which channel to use, both or 0 or 1.')

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
    'include_tslearn_slow', False, 'Whether to include time '
    'series clustering methods from the tslearn package '
    'that are slow: DTW and GAK.')
flags.DEFINE_boolean('include_LDS_GIBBS', True, 'Whether to include the '
                     'Gibbs sampling method for LDS.')
flags.DEFINE_boolean('include_ARMA_MLE', False, 'Whether to include the '
                     'MLE method for ARMA.')


def _drop_nan_rows(arr):
  return arr[~np.isnan(arr).any(axis=1)]


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
  model_fns['AR'] = lambda s: arma.fit_ar(
      _replace_nan_with_0(s.outputs), None, hdim)
  # Iterated regression without regularization and with regularization.
  model_fns['ARMA_OLS'] = lambda s: arma.fit_arma_iter(s.outputs, None, hdim)
  model_fns['ARMA'] = lambda s: arma.fit_arma_iter(
      s.outputs, None, hdim, l2_reg=0.01)
  model_fns['ARMA_roots'] = lambda s: arma.get_eig_from_arparams(
      arma.fit_arma_iter(s.outputs, None, hdim, l2_reg=0.01))
  if FLAGS.include_LDS_GIBBS:
    model_fns['LDS'] = lambda s: lds.fit_lds_gibbs(
        _replace_nan_with_0(s.outputs),
        None,
        hdim,
        num_update_samples=FLAGS.LDS_GIBBS_num_update_samples)
  if FLAGS.include_ARMA_MLE:
    model_fns['ARMA_MLE'] = lambda s: arma.fit_arma_mle(
        _replace_nan_with_0(s.outputs), None, hdim)
  if FLAGS.include_LDS_MLE:
    model_fns['LDS_MLE'] = lambda s: lds.fit_lds_mle(
        _replace_nan_with_0(s.outputs), None, hdim)
  return model_fns


def parse_csv(filename, hdim):
  """Reads ECG data from csv file."""
  labels = []
  seqs = []
  unprocessed_key = None
  unprocessed_label = None
  unprocessed_ch0 = None
  not_full_length = 0
  with open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      key = row[0]
      channel = row[1]
      label = row[2]
      channel_signal = np.array(row[3:]).reshape(-1, 1)
      try:
        channel_signal = channel_signal.astype(np.float32)
      except ValueError:
        channel_signal = np.array([float(x) if x else np.nan for x in row[3:]
                                  ]).reshape(-1, 1)
        # logging.info('Partial signal of len %d with key %s',
        #     sum(~np.isnan(channel_signal)), key)
        not_full_length += 1
      if channel == '0':
        assert unprocessed_ch0 is None
        unprocessed_ch0 = channel_signal
        unprocessed_key = key
        unprocessed_label = label
      if channel == '1':
        assert unprocessed_ch0 is not None
        seq_len = len(channel_signal)
        assert len(unprocessed_ch0) == seq_len
        if FLAGS.channel == 'both':
          vals = np.concatenate([unprocessed_ch0, channel_signal], axis=1)
        elif FLAGS.channel == '0':
          vals = unprocessed_ch0
        elif FLAGS.channel == '1':
          vals = channel_signal
        else:
          raise ValueError('Unexpected FLAGS.channel value: %s' % FLAGS.channel)
        seqs.append(
            lds.LinearDynamicalSystemSequence(
                np.zeros((seq_len, 1)), np.zeros((seq_len, hdim)), vals))
        assert label == unprocessed_label
        assert key.split(':')[:2] == unprocessed_key.split(':')[:2]
        labels.append(label)
        unprocessed_label = None
        unprocessed_key = None
        unprocessed_ch0 = None
  logging.info('Total seqs: %d, partial length seqs: %d.', len(seqs),
               not_full_length)
  if FLAGS.filter_type:
    seqs, labels = filter_type(seqs, labels)
  seqs, labels = drop_infreq_labels(seqs, labels)
  return seqs, labels


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


def print_label_info(labels):
  label_vocab, label_counts = np.unique(labels, return_counts=True)
  df = pd.DataFrame(index=label_vocab, data={'count': label_counts})
  print(df.sort_values('count', ascending=False).to_latex())


def filter_type(seqs, labels):
  types = ['N', 'AFIB', 'VT', 'P', 'AFL']
  seqs = [seqs[i] for i in xrange(len(seqs)) if labels[i] in types]
  labels = [l for l in labels if l in types]
  return seqs, labels


def drop_infreq_labels(seqs, labels):
  """Filter out infrequent labels."""
  label_vocab, label_counts = np.unique(labels, return_counts=True)
  is_dropped = {}
  for i in xrange(len(label_vocab)):
    logging.info('Found label %s, with count %d.', label_vocab[i],
                 label_counts[i])
    if label_counts[i] < FLAGS.label_count_threshold:
      logging.info('Dropped label %s.', label_vocab[i])
      is_dropped[label_vocab[i]] = True
    else:
      is_dropped[label_vocab[i]] = False
  seqs = [seqs[i] for i in xrange(len(seqs)) if not is_dropped[labels[i]]]
  labels = [l for l in labels if not is_dropped[l]]
  return seqs, labels


def sample_rebalance(seqs, labels):
  """Resample the data to have equal prevalence for each label."""
  label_vocab = np.unique(labels)
  n_samples_per_class = int(FLAGS.sample_size / len(label_vocab))
  sampled_seqs = []
  sampled_labels = []
  for l in label_vocab:
    l_seqs = [seqs[i] for i in xrange(len(seqs)) if labels[i] == l]
    l_labels = [labels[i] for i in xrange(len(seqs)) if labels[i] == l]
    sampled_l_seqs, sampled_l_labels = sklearn.utils.resample(
        l_seqs, l_labels, n_samples=n_samples_per_class)
    sampled_seqs.extend(sampled_l_seqs)
    sampled_labels.extend(sampled_l_labels)
  return sklearn.utils.shuffle(sampled_seqs, sampled_labels)


def get_results_bih_dataset(seqs, labels, hdim, num_clusters):
  """Returns a dataframe of clustering results on the ECG dataset."""
  label_vocab, label_counts = np.unique(labels, return_counts=True)
  logging.info('Counts of labels in current run: %s',
               str(label_vocab) + ' ' + str(label_counts))
  label_lookup = {l: i for i, l in enumerate(label_vocab)}
  cluster_ids = [label_lookup[l] for l in labels]
  model_fns = create_model_fns(hdim)
  padded = clustering.pad_seqs_to_matrix(seqs)
  max_seq_len = np.max([s.seq_len for s in seqs])
  pca = sklearn.decomposition.PCA(n_components=hdim).fit(_drop_nan_rows(padded))
  # pylint: disable=g-long-lambda
  model_fns['PCA'] = lambda s: pca.transform(
      _replace_nan_with_0(clustering.pad_seqs_to_matrix([s], max_seq_len))
  ).flatten()
  # Get clustering results.
  results_df = clustering.get_results(
      seqs,
      num_clusters,
      cluster_ids,
      None,
      model_fns,
      include_tslearn=FLAGS.include_tslearn,
      include_slow_methods=FLAGS.include_tslearn_slow)
  logging.info(results_df)
  if FLAGS.plot_clusters:
    clustering.visualize_clusters(
        seqs, None, labels, model_fns,
        os.path.join(FLAGS.output_dir, 'visualization.png'))
  return results_df


def get_agg_stats(df):
  """Writes a csv file with aggregated stats."""
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
    logging.info(metric)
    logging.info(stats[['mean_w_ci']])
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
  with open(os.path.join(FLAGS.output_dir, 'flags.txt'), 'w+') as f:
    f.write(str(FLAGS.flag_values_dict()))
  seqs, labels = parse_csv('mit-bih/all_classes.csv', FLAGS.hdim)
  for _ in xrange(FLAGS.num_repeat):
    seqs, labels = sample_rebalance(seqs, labels)
    results_df = get_results_bih_dataset(seqs, labels, FLAGS.hdim,
                                         FLAGS.num_clusters)
    combined_results_list.append(results_df)
    results_df = pd.concat(combined_results_list)
    with open(os.path.join(FLAGS.output_dir, 'results.csv'), 'w+') as f:
      results_df.to_csv(f, index=False)
  get_agg_stats(results_df)
  # plot_results(results_df, FLAGS.output_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('output_dir')
  app.run(main)
