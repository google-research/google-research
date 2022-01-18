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

"""Plot estimated calibration error versus true calibration error."""
import json
import os

from absl import app
from absl import flags
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import tensorflow as tf

from caltrain import ce_type_paper_name_map
from caltrain.get_ece_bias import get_ece_bias
from caltrain.simulation.polynomial import TruePolynomial

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './caltrain/data',
                    'location of the source data')
flags.DEFINE_string('plot_dir', './caltrain/plots', 'location to write plots')

matplotlib.use('Agg')
font = {'size': 26}
matplotlib.rc('font', **font)


def generate_params(num_datasets, dataset, a, b):
  """Generate params for simulated datasets."""
  np.random.seed(2379587)
  keys = ['a', 'b', 'd', 'alpha', 'beta', 'dataset', 'true_ce']
  params = {key: [] for key in keys}
  d_params = np.linspace(1, 10, num_datasets)
  for i in range(num_datasets):
    d = d_params[i]
    alpha = a
    beta = b
    params['a'].append(a)
    params['b'].append(b)
    params['d'].append(d)
    params['alpha'].append(alpha)
    params['beta'].append(beta)
    params['dataset'].append(dataset)
    true_dataset = TruePolynomial(alpha=a, beta=b, d=d, n_samples=0)
    params['true_ce'].append(100 * true_dataset.true_calib_error())

  params['a'].append(a)
  params['b'].append(b)
  params['d'].append(1)
  params['alpha'].append(alpha)
  params['beta'].append(beta)
  params['dataset'].append(dataset)
  true_dataset = TruePolynomial(alpha=a, beta=b, d=1, n_samples=0)
  params['true_ce'].append(100 * true_dataset.true_calib_error())
  return params


def get_params_and_bias(cached_result, param_name, a, b):
  """Get parameter for x axis and bias."""
  n_samples = [200, 5000]
  num_datasets = 100
  ce_types = ['ew_ece_bin', 'em_ece_sweep']

  config = {}
  config['num_reps'] = 1000
  config['num_bins'] = 15
  config['split'] = ''
  config['norm'] = 2
  config['calibration_method'] = 'no_calibration'
  config['bin_method'] = ''

  params = generate_params(num_datasets, 'polynomial', a=a, b=b)
  ece_bias = get_ece_bias(config, n_samples, ce_types, params, cached_result)
  return params[param_name], ece_bias


def plot_bias_vs_params(raw_data, param_name='true_ce'):
  """Plot bias versus param."""
  clrs = [['b', 'navy'], ['r', 'darkred']]

  a = [1.1, 1]
  b = [0.1, 1]

  prior_labels = ['f(x)~Beta({}, {})'.format(a[0], b[0]), 'f(x)~Uniform(0, 1)']
  linestyles = ['-', '--']

  ce_types = ['ew_ece_bin', 'em_ece_sweep']
  paper_ce_type = [ce_type_paper_name_map[ce_type] for ce_type in ce_types]

  for ce_idx in range(len(ce_types)):
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(a)):
      true_ce, ece_bias = get_params_and_bias(
          raw_data, param_name, a=a[i], b=b[i])
      true_ce, bias_1, bias_2 = zip(
          *sorted(zip(true_ce, ece_bias[0, ce_idx, :], ece_bias[1, ce_idx, :])))
      ece_1 = np.squeeze(np.array(bias_1)) + np.squeeze(np.array(true_ce))
      ax.plot(
          true_ce,
          ece_1,
          linestyle=linestyles[0],
          linewidth=3,
          color=clrs[ce_idx][i],
          label='{}'.format('n=200, {}'.format(prior_labels[i])))
      ece_2 = np.squeeze(np.array(bias_2)) + np.squeeze(np.array(true_ce))
      ax.plot(
          true_ce,
          ece_2,
          linestyle=linestyles[1],
          linewidth=3,
          color=clrs[ce_idx][i],
          label='{}'.format('n=5,000, {}'.format(prior_labels[i])))
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', label='y=x', linewidth=3)
    ax.axhline(
        y=12,
        xmin=0,
        xmax=15,
        color='k',
        linewidth=3,
        linestyle='--',
        label='12% error')

    major_locator = MultipleLocator(2)
    minor_locator = MultipleLocator(2)
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=1.5)
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_major_locator(major_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.set_xlabel('True Calibration Error (%)')
    ax.set_ylabel('Estimated Calibration Error (%)')
    plt.title(paper_ce_type[ce_idx])
    plt.legend(loc='lower right', fontsize=20)
    save_dir = os.path.join(FLAGS.plot_dir, 'bias_vs_param')
    os.makedirs(save_dir, exist_ok=True)
    fig_filename = os.path.join(
        save_dir, '{}_{}_{}_{}.pdf'.format(paper_ce_type[ce_idx], param_name,
                                           a[0], b[0]))
    fig.savefig(fig_filename, dpi='figure', bbox_inches='tight')


def main(_):
  calibration_results_cache_file = os.path.join(FLAGS.data_dir,
                                                'calibration_results.json')
  with tf.io.gfile.GFile(calibration_results_cache_file, 'r') as f:
    raw_data = json.load(f)
  plot_bias_vs_params(raw_data)


if __name__ == '__main__':
  app.run(main)
