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

"""Plot bias vs. num_samples for CIFAR-10, CIFAR-100, ImageNet."""
import json
import os
import subprocess

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import caltrain as caltrain
from caltrain import ce_type_paper_name_map
from caltrain import cetype_color_map
from caltrain import ml_data_name_map
from caltrain import ml_model_name_map
from caltrain import SAVEFIG_FORMAT
from caltrain.get_ece_bias import get_ece_bias
from caltrain.glm_modeling import get_beta_fit_data
from caltrain.glm_modeling import get_datasets
from caltrain.glm_modeling import get_glm_fit_data
from caltrain.glm_modeling.glmmodel import get_glm_model_container

FLAGS = flags.FLAGS
flags.DEFINE_string('plot_dir', './caltrain/plots', 'location to write plots')
flags.DEFINE_string('data_dir', './caltrain/data',
                    'location of the source data')
flags.DEFINE_boolean('include_legend', True, 'Include figure legend')


def generate_params_from_fits(fits):
  """Convert fits to params."""
  params = {}
  params['a'] = []
  params['b'] = []
  params['d'] = []
  params['alpha'] = []
  params['beta'] = []
  params['dataset'] = []
  params['name'] = []
  for fit in fits:
    params['a'].append(fit['beta_0'])
    params['b'].append(fit['beta_1'])
    params['alpha'].append(fit['alpha'])
    params['beta'].append(fit['beta'])
    params['d'].append(None)

    if fit['glm_name'] == 'logflip_logflip_b0_b1':
      dataset = 'two_param_flip_polynomial'
    elif fit['glm_name'] == 'logit_logit_b0_b1':
      dataset = 'logistic_log_odds'
    elif fit['glm_name'] == 'logit_logit_b1':
      dataset = 'logistic_log_odds'
      params['a'][-1] = 0
    elif fit['glm_name'] == 'logflip_logflip_b1':
      dataset = 'two_param_flip_polynomial'
      params['a'][-1] = 0
    elif fit['glm_name'] == 'log_log_b0_b1':
      dataset = 'two_param_polynomial'
    elif fit['glm_name'] in ['logit_logflip_b1']:
      dataset = 'logistic_two_param_flip_polynomial'
      params['a'][-1] = 0
    elif fit['glm_name'] in ['logit_logflip_b0_b1']:
      dataset = 'logistic_two_param_flip_polynomial'
    else:
      raise NotImplementedError
    params['dataset'].append(dataset)
    params['name'].append(fit['name'])

  return params


def plot_bias_vs_num_samples(result,
                             params=None,
                             dataset_family='polynomial',
                             include_legend=True):
  """Plot bias vs num samples."""
  ce_types = ['em_ece_bin', 'ew_ece_bin', 'em_ece_sweep', 'ew_ece_sweep']
  n_samples = [200, 400, 800, 1600, 3200, 6400, 12800]

  config = {}
  config['num_reps'] = 1000
  config['num_bins'] = 15
  config['split'] = ''
  config['norm'] = 2
  config['calibration_method'] = 'no_calibration'
  config['bin_method'] = ''
  np.random.seed(2379587)
  num_datasets = len(params['a'])

  ece_bias = get_ece_bias(config, n_samples, ce_types, params, result)

  fig, ax = plt.subplots(figsize=(10, 10))
  markersize = 22
  x = np.linspace(180, 16000, 100)
  ax.plot(x, 0 * x, color='lime', linewidth=3)
  for ce_idx in range(len(ce_types)):
    color = cetype_color_map[ce_types[ce_idx]]
    linestyles = ['-', '--', '-.', ':']
    markers = ['*', '^', 'o', 'd']
    for j in range(num_datasets):
      ax.plot(
          n_samples,
          ece_bias[:, ce_idx, j],
          marker=markers[j],
          linestyle=linestyles[j],
          color=color,
          label='{}'.format(ce_types[ce_idx]),
          linewidth=3,
          markeredgecolor='k',
          markersize=markersize)

  if include_legend:

    def f(m, c, l):
      return plt.plot([], [],
                      marker=m,
                      color=c,
                      ls=l,
                      linewidth=2,
                      markeredgecolor='k',
                      markersize=markersize - 4)[0]

    handles = [
        f('s', cetype_color_map[ce_types[ce_idx]], 'None')
        for ce_idx in range(len(ce_types))
    ]
    handles += [f(markers[j], 'k', linestyles[j]) for j in range(num_datasets)]
    handles += [f(None, 'lime', '-')]
    labels = [ce_type_paper_name_map[ce_type] for ce_type in ce_types]
    labels += [
        ml_model_name_map[params['name'][j]] for j in range(num_datasets)
    ]
    labels += ['Bias=0']
    legend = plt.legend(
        handles, labels, loc='upper right', framealpha=1, ncol=5)
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_alpha(1)
    frame.set_linewidth(10)

  ax.tick_params(axis='both', which='major', labelsize=32)
  ax.grid(which='both', color='lightgray', linestyle='-', linewidth=1.5)

  dataset_family_paper_name = dataset_family
  if dataset_family in ml_data_name_map:
    dataset_family_paper_name = ml_data_name_map[dataset_family]
  plt.title('{}'.format(dataset_family_paper_name), fontsize=40)

  ax.set_ylabel('Bias', fontsize=40)
  ax.set_xlabel('Num samples (log scale)', fontsize=40)
  plt.xscale('log', basex=2)
  ax.set_xlim([180, 16000])
  ax.set_xticks([200, 400, 800, 1600, 3200, 6400, 12800])
  ax.set_xticklabels(['200', '', '800', '', '3,200', '', '12,800'])

  fig.tight_layout()
  save_dir = os.path.join(FLAGS.plot_dir, '{}'.format(dataset_family))
  os.makedirs(save_dir, exist_ok=True)
  fig_filename = os.path.join(
      save_dir,
      'bias_vs_num_samples_include_legend={}.{}'.format(include_legend,
                                                        SAVEFIG_FORMAT))
  fig.savefig(fig_filename, dpi='figure', bbox_inches='tight')
  if include_legend:
    legend_filename = os.path.join(save_dir, 'legend.{}'.format(SAVEFIG_FORMAT))
    caltrain.utils.export_legend(
        legend, filename=legend_filename, expand=[-350, -10, 5, 7])
    try:
      subprocess.Popen(['pdfcrop', legend_filename], stdout=subprocess.PIPE)
    except FileNotFoundError:
      raise RuntimeError('pdfcrop is needed for constructing legend')


def get_fits_from_caltrain_cache(data_dir):
  """Get fits from caltrain cache."""
  fits = []
  glm_models = get_glm_model_container(data_dir)
  glm_fit_data = get_glm_fit_data(data_dir)
  beta_fit_data = get_beta_fit_data(data_dir)
  dataset_list = get_datasets(data_dir=data_dir)  # All datasets
  for dataset in dataset_list:
    gm_name_aic_dict = {
        gm_name:
        glm_fit_data['data'][dataset.name][gm_name]['AIC']['mean']['value']
        for gm_name, gm in glm_models.items()
    }
    gm_best_name = min(gm_name_aic_dict, key=gm_name_aic_dict.get)
    curr_fit_dict = {}
    curr_fit_dict['name'] = dataset.name
    curr_fit_dict['glm_name'] = gm_best_name
    curr_fit_dict['beta_0'] = glm_fit_data['data'][
        dataset.name][gm_best_name]['b0']['mean']['value']
    curr_fit_dict['beta_1'] = glm_fit_data['data'][
        dataset.name][gm_best_name]['b1']['mean']['value']
    curr_fit_dict['alpha'] = beta_fit_data['data'][dataset.name]['a']
    curr_fit_dict['beta'] = beta_fit_data['data'][dataset.name]['b']
    fits.append(curr_fit_dict)

  return fits


def main(_):
  calibration_results_cache_file = os.path.join(FLAGS.data_dir,
                                                'calibration_results.json')
  with tf.io.gfile.GFile(calibration_results_cache_file, 'r') as f:
    result = json.load(f)

  fits = get_fits_from_caltrain_cache(data_dir=FLAGS.data_dir)
  # Generate plots using ml model/dataset fits
  for ml_dataset in ['imgnet', 'c10', 'c100']:
    filtered_fits = [
        fit for fit in fits if fit['name'].endswith(ml_dataset) and
        not fit['name'].startswith('lenet')
    ]

    params = generate_params_from_fits(filtered_fits)

    # Called twice intentionally:
    plot_bias_vs_num_samples(result, params, ml_dataset, include_legend=False)
    if FLAGS.include_legend:
      plot_bias_vs_num_samples(result, params, ml_dataset, include_legend=True)


if __name__ == '__main__':
  app.run(main)
