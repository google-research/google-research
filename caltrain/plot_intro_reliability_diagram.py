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

"""Reliability diagram for intro."""
import os

from absl import app
from absl import flags
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from caltrain.run_calibration import calibrate
from caltrain.run_calibration import get_true_dataset

matplotlib.use('Agg')
font = {'size': 40}
matplotlib.rc('font', **font)

FLAGS = flags.FLAGS
flags.DEFINE_string('plot_dir', './plots', 'location to write plots')


def bin_means(fx, y, bins, n_bins):
  """Compute mean confidence score and mean label in each bin."""
  bin_fx = []
  bin_y = []
  for i in range(n_bins):
    cur = bins == i
    if any(cur):
      fxm = np.mean(fx[cur])
      bin_fx.append(fxm)
      ym = np.mean(y[cur])
      bin_y.append(ym)
    else:
      bin_y.append(0.0)
  return np.array(bin_fx), np.array(bin_y)


def plot_reliability_diagram(alpha=1.0, beta=1.0):
  """Plot reliability diagram."""
  n_bins = 15
  config = {}
  config['num_reps'] = 1000
  config['num_bins'] = n_bins
  config['split'] = ''
  config['norm'] = 2
  config['calibration_method'] = 'no_calibration'
  config['bin_method'] = ''
  config['d'] = 1
  config['alpha'] = alpha
  config['beta'] = beta
  config['a'] = alpha
  config['b'] = beta
  config['dataset'] = 'polynomial'
  config['ce_type'] = 'ew_ece_bin'
  config['num_samples'] = 200

  true_dataset = get_true_dataset(config)
  fx, ytrue, y = true_dataset.dataset()
  y = np.array(y)
  ytrue = np.array(ytrue)

  # Determine bins
  bins = np.minimum(n_bins - 1, np.floor(fx * n_bins)).astype(int)

  # Sort the data
  sort_ix = np.argsort(fx)
  fx = fx[sort_ix]
  y = y[sort_ix]
  ytrue = ytrue[sort_ix]
  bins = bins[sort_ix]

  _, bin_y = bin_means(fx, 1 * y, bins, n_bins)

  ece = calibrate(config, true_dataset)
  print('ECE is {}'.format(ece))

  fig, ax = plt.subplots(figsize=(10, 10))
  ax.plot(fx, y, 'x', label='Raw data', markersize=20)

  bin_centers = np.linspace(0, 1, n_bins, endpoint=False) + (1. / (n_bins * 2))
  ax.bar(
      bin_centers,
      bin_y,
      width=1.0 / 15,
      color='r',
      alpha=0.3,
      linewidth=1,
      edgecolor='r',
      label='ECE Bins')
  ax.plot([0.0, 1.0], [0.0, 1.0], 'k-', linewidth=2, label='True calibration')
  ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

  plt.title('Sample A')
  plt.xlabel('Predicted confidence')
  plt.ylabel('Empirical accuracy')
  ax.set_ylim([-0.05, 1.05])
  ax.set_xlim([-0.05, 1.05])
  ax.legend(loc=(0, 0.6), framealpha=0.4)
  save_dir = os.path.join(FLAGS.plot_dir, 'intro')
  os.makedirs(save_dir, exist_ok=True)
  fig.savefig(
      os.path.join(
          save_dir,
          'reliability_diagram_Beta_alpha={}_beta={}.pdf'.format(alpha, beta)),
      dpi='figure',
      bbox_inches='tight')


def main(_):
  plot_reliability_diagram(alpha=2.8, beta=0.05)


if __name__ == '__main__':
  app.run(main)
