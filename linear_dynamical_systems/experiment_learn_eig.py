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

r"""Script for running experiments about learning eigenvalues.

Example to run locally:
 python experiment_learn_eig.py --output_dir=may19_b_eig_2d --min_seq_len=20 \
         --max_seq_len=2000 --num_sampled_seq_len=20 --num_repeat=100 \
         --hidden_dim=2 --output_noise_stddev=0.1 --hide_inputs=true \
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import timeit
import warnings

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
from statsmodels.tools import sm_exceptions
import tqdm

import arma
import lds

FLAGS = flags.FLAGS

# Flags for IO and plotting.
flags.DEFINE_string('output_dir', None, 'Output filepath.')
flags.DEFINE_boolean(
    'load_results', False, 'Whether to skip experiments '
    'and only plot existing results from output_dir.')

# Flags for generating random ground truth LDSs.
flags.DEFINE_boolean('generate_diagonalizable_only', True, 'Whether to only '
                     'generate diagonalizable LDSs.')
flags.DEFINE_list(
    'true_eig', None, 'List of ground truth eigenvalues. If '
    'None, random eigenvalues are generated in each repeat of '
    'experiment.')
flags.DEFINE_boolean(
    'hide_inputs', True, 'Whether the inputs are observable '
    'to the clustering algorithm.')
flags.DEFINE_integer('hidden_dim', 0, 'Hidden state dimension.')
flags.DEFINE_integer('input_dim', 1, 'Input dim in experiments.')
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
flags.DEFINE_float('output_noise_stddev', 0.1, 'Output noise stddev.')
flags.DEFINE_float('init_state_mean', 0.0, 'Init state mean.')
flags.DEFINE_float('init_state_stddev', 0.0, 'Init state stddev.')

# Flags for hparams in clustering algorithms.
flags.DEFINE_boolean(
    'include_slow_methods', False, 'Whether to include the '
    'significantly slower methods, LDS_GIBBS, ARMA_MLE, and '
    'gak_km.')
flags.DEFINE_integer(
    'LDS_GIBBS_num_update_samples', 100, 'Number of update '
    'samples used for fitting LDS in pylds package.')

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def get_eig_from_arparams(arparams):
  eigs = np.roots(np.r_[1, -arparams])
  return eigs[np.argsort(eigs.real)[::-1]]


def create_learning_fns(hidden_dim):
  """Util function to create learning fns to get learned eigenvalues."""
  learning_fns = collections.OrderedDict()
  # pylint: disable=g-long-lambda
  learning_fns['AR'] = lambda o, i: get_eig_from_arparams(
      arma.fit_ar(o, i, hidden_dim))
  # The ARMA_OLS method gives very big errors, hence not included.
  # learning_fns['ARMA_OLS'] = lambda o, i: get_eig_from_arparams(
  #     arma.fit_arma_iter(o, i, hidden_dim))
  # learning_fns['ARMA_RLS_0.1'] = lambda o, i: get_eig_from_arparams(
  #     arma.fit_arma_iter(o, i, hidden_dim, l2_reg=0.1))
  learning_fns['ARMA_RLS'] = lambda o, i: get_eig_from_arparams(
      arma.fit_arma_iter(o, i, hidden_dim, l2_reg=0.01))
  # learning_fns['ARMA_RLS_0.005'] = lambda o, i: get_eig_from_arparams(
  #     arma.fit_arma_iter(o, i, hidden_dim, l2_reg=0.005))
  # learning_fns['ARMA_RLS_0.001'] = lambda o, i: get_eig_from_arparams(
  #     arma.fit_arma_iter(o, i, hidden_dim, l2_reg=0.001))
  if FLAGS.include_slow_methods:
    learning_fns['ARMA_MLE'] = lambda o, i: get_eig_from_arparams(
        arma.fit_arma_mle(o, i, hidden_dim))
    # The LDS_MLE  method fails convergence too often, hence not included.
    # learning_fns['LDS_MLE'] = lambda o, i: lds.fit_lds_mle(
    #     o, i, hidden_dim)
    learning_fns['LDS_GIBBS'] = lambda o, i: lds.fit_lds_gibbs(
        o, i, hidden_dim, num_update_samples=FLAGS.LDS_GIBBS_num_update_samples)
  return learning_fns


def get_results_seq_len(given_true_eig,
                        hidden_dim,
                        input_dim,
                        min_seq_len,
                        max_seq_len,
                        num_sampled_seq_len,
                        num_repeat,
                        input_mean,
                        input_stddev,
                        output_noise_stddev,
                        init_state_mean=0.0,
                        init_state_stddev=0.0,
                        generate_diagonalizable_only=False,
                        random_seed=0):
  """Get results for varying sequence lengths.

  Args:
    given_true_eig: Ground truth of eigenvalues. If None, generate random
      eigenvalues from uniform [-1,1] in each repeat of experiment.
    hidden_dim: Assumed hidden dim. If 0, use true hidden dim.
    input_dim: The input dim.
    min_seq_len: Min seq len in experiments.
    max_seq_len: Max seq len in experiments.
    num_sampled_seq_len: Number of sampled seq len values in between min and max
      seq len.
    num_repeat: Number of repeated experiments for each seq_len.
    input_mean: Scalar or 1D array of length hidden state dim.
    input_stddev: Scalar of 1D array of length hidden state dim.
    output_noise_stddev: Scalar.
    init_state_mean: Scalar or 1D array of length hidden state dim.
    init_state_stddev: Scalar of 1D array of length hidden state dim.
    generate_diagonalizable_only: Whether to only use diagonalizable LDSs in
      simulations.
    random_seed: Random seed, integer.

  Returns:
    A pandas DataFrame with columns `method`, `seq_len`, `t_secs`,
    `failed_ratio`, and `l2_r_error`.
    The same method and seq_len will appear in num_repeat many rows.
  """
  np.random.seed(random_seed)
  progress_bar = tqdm.tqdm(total=num_repeat * num_sampled_seq_len)
  gen = lds.SequenceGenerator(
      input_mean=input_mean,
      input_stddev=input_stddev,
      output_noise_stddev=output_noise_stddev,
      init_state_mean=init_state_mean,
      init_state_stddev=init_state_stddev)
  # seq_len_vals = np.linspace(min_seq_len, max_seq_len, num_sampled_seq_len)
  # seq_len_vals = [int(round(x)) for x in seq_len_vals]
  min_inv_sqrt_seq_len = 1. / np.sqrt(max_seq_len)
  max_inv_sqrt_seq_len = 1. / np.sqrt(min_seq_len)
  inv_sqrt_seq_len_vals = np.linspace(min_inv_sqrt_seq_len,
                                      max_inv_sqrt_seq_len, num_sampled_seq_len)
  seq_len_vals = [int(round(1. / (x * x))) for x in inv_sqrt_seq_len_vals]
  learning_fns = create_learning_fns(hidden_dim)
  metric_dict = {
      k: [] for k in [
          'method', 'seq_len', 't_secs', 'l2_a_error', 'l2_r_error',
          'failed_convg'
      ]
  }
  for _ in xrange(num_repeat):
    if given_true_eig is not None:
      ground_truth = lds.generate_linear_dynamical_system(
          hidden_dim, input_dim, eigvalues=given_true_eig)
    else:
      ground_truth = lds.generate_linear_dynamical_system(
          hidden_dim, input_dim, diagonalizable=generate_diagonalizable_only)
    true_eig = ground_truth.get_spectrum()
    for seq_len in seq_len_vals:
      seq = gen.generate_seq(ground_truth, seq_len=seq_len)
      for k, fn in learning_fns.iteritems():
        start_t = timeit.default_timer()
        with warnings.catch_warnings(record=True) as caught:
          warnings.filterwarnings(
              'always', category=sm_exceptions.ConvergenceWarning)
          if FLAGS.hide_inputs:
            eig_pred = fn(seq.outputs, None)
          else:
            eig_pred = fn(seq.outputs, seq.inputs)
        t_elapsed = timeit.default_timer() - start_t
        metric_dict['seq_len'].append(seq_len)
        metric_dict['method'].append(k)
        metric_dict['t_secs'].append(t_elapsed)
        metric_dict['l2_a_error'].append(np.linalg.norm(true_eig - eig_pred))
        metric_dict['l2_r_error'].append(
            np.linalg.norm(true_eig - eig_pred) / np.linalg.norm(true_eig))
        metric_dict['failed_convg'].append(False)
        for w in caught:
          if w.category in [
              RuntimeWarning, sm_exceptions.ConvergenceWarning,
              sm_exceptions.HessianInversionWarning
          ]:
            metric_dict['failed_convg'][-1] = True
          else:
            warnings.warn(w.message, w.category)
      progress_bar.update(1)
  progress_bar.close()
  return pd.DataFrame(data=metric_dict)


def plot_results(results_df, output_dir):
  """Plots metrics and saves plots as png files."""
  pylab.figure()
  method_name_mapping = {
      'AR': 'pure_AR',
      'ARMA_RLS': 'ARMA_iter',
      'LDS_GIBBS': 'LDS_MLE',
      'kshape': 'k-Shape',
      'PCA': 'PCA',
      # 'dtw_km': 'DTW',
      'ARMA_MLE': 'ARMA_MLE',
      # 'raw_output': 'raw_outputs'
  }
  results_df['inv_sqrt_seq_len'] = results_df.seq_len.apply(
      lambda x: round(1.0 / np.sqrt(x), 3))
  results_df['method'] = results_df.method.map(method_name_mapping)
  mean_df = results_df.groupby(['seq_len', 'method']).mean().reset_index()
  sns.lineplot(x='seq_len', y='failed_convg', data=mean_df, hue='method')
  output = six.StringIO()
  pylab.savefig(output, format='png')
  image = output.getvalue()
  with open(output_dir + '/' + 'convergence_failure_ratio.png', 'w+') as f:
    f.write(image)

  for metric_name in ['l2_a_error', 'l2_r_error', 't_secs']:
    pylab.figure()
    g = sns.pointplot(
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
    with open(output_dir + '/' + metric_name + '.png', 'w+') as f:
      f.write(image)

  for metric_name in ['l2_a_error', 'l2_r_error', 't_secs']:
    pylab.figure()
    g = sns.pointplot(
        x='inv_sqrt_seq_len',
        y=metric_name,
        data=results_df,
        hue='method',
        scale=0.5,
        estimator=np.mean,
        err_style='bars',
        capsize=.1)
    g.set_xlim(g.get_xlim()[::-1])
    # g.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    output = six.StringIO()
    pylab.savefig(output, format='png')
    image = output.getvalue()
    with open(output_dir + '/' + metric_name + '_sqrt.png', 'w+') as f:
      f.write(image)


def main(unused_argv):
  if FLAGS.load_results:
    with open(FLAGS.output_dir + '/eig_results.csv', 'r') as f:
      result_df = pd.read_csv(f, index_col=False)
    plot_results(result_df, FLAGS.output_dir)
    return

  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
  with open(FLAGS.output_dir + '/flags.txt', 'w+') as f:
    f.write(str(FLAGS.flag_values_dict()))
  true_eig = None
  if FLAGS.true_eig is not None:
    try:
      true_eig = [float(x) for x in FLAGS.true_eig]
    except:
      raise ValueError('Error parsing true_eig %s' % str(true_eig))
  result_df = get_results_seq_len(
      given_true_eig=true_eig,
      hidden_dim=FLAGS.hidden_dim,
      input_dim=FLAGS.input_dim,
      min_seq_len=FLAGS.min_seq_len,
      max_seq_len=FLAGS.max_seq_len,
      num_sampled_seq_len=FLAGS.num_sampled_seq_len,
      num_repeat=FLAGS.num_repeat,
      input_mean=FLAGS.input_mean,
      input_stddev=FLAGS.input_stddev,
      output_noise_stddev=FLAGS.output_noise_stddev,
      init_state_mean=FLAGS.init_state_mean,
      init_state_stddev=FLAGS.init_state_stddev,
      random_seed=FLAGS.random_seed,
      generate_diagonalizable_only=FLAGS.generate_diagonalizable_only,
  )
  with open(FLAGS.output_dir + '/eig_results.csv', 'w+') as f:
    result_df.to_csv(f, index=False)
  plot_results(result_df, FLAGS.output_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('output_dir')
  app.run(main)
