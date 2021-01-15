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

"""Plot simple assumptions needed to compute TCE."""
import os

from absl import app
from absl import flags
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from caltrain.simulation.true_dataset import TrueDatasetBetaPrior

font = {'size': 40}
matplotlib.rc('font', **font)

FLAGS = flags.FLAGS
flags.DEFINE_string('plot_dir', './plots', 'location to write plots')


def plot_beta_distribution(alpha=1, beta=1):
  """Plot beta distribution."""
  beta_pdf = TrueDatasetBetaPrior(alpha=alpha, beta=beta).pdf_fx

  x = np.linspace(0.001, 0.999, 1000)
  y = beta_pdf(x)

  fig, ax = plt.subplots(figsize=(10, 10))

  ax.plot(
      x, y, linewidth=3, label=r'f(x)$\sim$Beta({}, {})'.format(alpha, beta))
  ax.set_xlim([0, 1])
  ax.set_ylim([0, 5])

  ax.set_xlabel('f(X)=c')
  ax.set_ylabel('PDF')
  plt.xticks([0, 1], [0, 1])
  ax.legend()
  save_dir = os.path.join(FLAGS.plot_dir, 'intro')
  os.makedirs(save_dir, exist_ok=True)
  fig.savefig(
      os.path.join(save_dir,
                   '{}.pdf'.format('Beta_alpha={}_beta={}').format(alpha,
                                                                   beta)),
      dpi='figure',
      bbox_inches='tight')


def plot_polynomial(d=2):
  """Plot true calibration curve that is polynomial in nature."""
  x = np.linspace(0, 1, 1000)
  y = x * x

  fig, ax = plt.subplots(figsize=(10, 10))

  ax.plot(x, y, linewidth=3, label=r'E$[Y \mid f(X)=c] = c^2$')
  ax.plot(x, x, 'k--', linewidth=3, label='Perfect calibration')
  ax.set_xlim([0, 1])
  ax.set_ylim([0, 1])
  ax.legend()

  ax.set_xlabel('f(X)=c')
  ax.set_ylabel(r'E$[Y \mid f(X)=c]$')
  plt.xticks([0, 1], [0, 1])
  plt.yticks([0, 1], [0, 1])

  save_dir = os.path.join(FLAGS.plot_dir, 'intro')
  os.makedirs(save_dir, exist_ok=True)
  fig.savefig(
      os.path.join(save_dir, '{}.pdf'.format('polynomial_d={}').format(d)),
      dpi='figure',
      bbox_inches='tight')


def main(_):
  plot_beta_distribution(alpha=2.8, beta=0.05)
  plot_polynomial()


if __name__ == '__main__':
  app.run(main)
