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

"""Heat map plots."""
import os

from absl import app
from absl import flags
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from caltrain.run_calibration import estimate_ece

matplotlib.use('Agg')
font = {'size': 26}
matplotlib.rc('font', **font)

FLAGS = flags.FLAGS
flags.DEFINE_string('plot_dir', './caltrain/plots', 'location to write plots')
flags.DEFINE_string('data_dir', './caltrain/data',
                    'location of the source data')


def plot_imshow(ce_type):
  """Plot bias and variance heat map for given ce type."""
  n_samples = [200, 400, 800, 1600, 3200, 6400]
  num_bins = [2, 4, 8, 16, 32, 64]

  config = {}
  config['num_reps'] = 1000
  config['split'] = ''
  config['norm'] = 2
  config['calibration_method'] = 'no_calibration'
  config['bin_method'] = ''
  config['ce_type'] = ce_type
  config['dataset'] = 'polynomial'
  config['a'] = 1.1
  config['b'] = 0.1
  config['d'] = 2
  config['alpha'] = 1.1
  config['beta'] = 0.1

  ece_bias = np.zeros((len(num_bins), len(n_samples)))
  ece_var = np.zeros((len(num_bins), len(n_samples)))

  np.random.seed(2379587)

  for i, num_samples in enumerate(n_samples):
    for j, num_bin in enumerate(num_bins):
      config['num_samples'] = num_samples
      config['num_bins'] = num_bin
      mean, var, _ = estimate_ece(config, FLAGS.data_dir)
      ece_bias[j, i] = mean
      ece_var[j, i] = var

  for p in [0, 1]:
    fig, ax = plt.subplots(figsize=(10, 10))
    cur_data = ece_bias if p == 0 else ece_var
    ax.imshow(np.abs(cur_data), cmap='Reds', vmin=0, vmax=9.0)

    ax.set_xticks(np.arange(len(n_samples)))
    ax.set_yticks(np.arange(len(num_bins)))

    ax.set_xticklabels(n_samples)
    ax.set_ylabel('# Bins')
    ax.set_yticklabels(num_bins)
    ax.set_xlabel('# Samples')
    cur_data_type = 'Bias' if p == 0 else 'Sqrt(Variance)'
    cur_title_data_type = 'Bias' if p == 0 else r'$\sqrt{\mathrm{Variance}}$'
    cur_title_ce_type = r'Equal Width $\mathrm{ECE}_\mathrm{BIN}$' if ce_type == 'ew_ece_bin' else r'Equal Mass $\mathrm{ECE}_\mathrm{BIN}$'
    ax.set_title('{} in {}'.format(cur_title_data_type, cur_title_ce_type))

    for i in range(cur_data.shape[0]):
      for j in range(cur_data.shape[1]):
        ax.text(
            j,
            i,
            '%.2f' % (cur_data[i, j]),
            ha='center',
            va='center',
            color='#000000')
    plt.tight_layout()
    save_dir = os.path.join(FLAGS.plot_dir, 'heat')
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        os.path.join(
            save_dir,
            '{}_{}_alpha_{}_beta_{}.pdf'.format(ce_type, cur_data_type,
                                                config['alpha'],
                                                config['beta'])),
        dpi='figure',
        bbox_inches='tight')


def main(_):
  plot_imshow('em_ece_bin')
  plot_imshow('ew_ece_bin')


if __name__ == '__main__':
  app.run(main)
