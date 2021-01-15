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

"""Plot distribution fits empirical datasets."""
import collections
import os

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps

from caltrain import dataset_mlmodel_imageset_map
from caltrain import imageset_color_map
from caltrain import mlmodel_linestyle_map
from caltrain import mlmodel_marker_map
from caltrain.glm_modeling import Folds
from caltrain.glm_modeling import get_beta_fit_data
from caltrain.glm_modeling import get_datasets
from caltrain.glm_modeling import get_glm_fit_data
from caltrain.glm_modeling.glmmodel import get_glm_model_container
from caltrain.run_calibration import calibrate

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', None, 'location of the source data')
flags.DEFINE_string('plot_dir', './plots', 'location to write plots')


def main(_):

  data_dir = FLAGS.data_dir
  glm_models = get_glm_model_container(data_dir)
  glm_fit_data = get_glm_fit_data(data_dir)
  beta_fit_data = get_beta_fit_data(data_dir)
  datasets = get_datasets(data_dir=FLAGS.data_dir)

  excluded_names = ['lenet5_c10', 'lenet5_c100', 'resnet50_birds']
  dataset_dict = {
      key: val for key, val in datasets.items() if key not in excluded_names
  }

  fontsize = 11
  legend_fontsize = 8
  figsize = (8, 2.5)
  save_dir = os.path.join(FLAGS.plot_dir, 'glm_modeling')
  os.makedirs(save_dir, exist_ok=True)
  save_file_name = os.path.join(save_dir, 'glm_beta_summary.pdf')
  fig, ax_list = plt.subplots(1, 3, figsize=figsize)
  ax_beta = ax_list[0]
  ax_glm = ax_list[1]
  ax_compare = ax_list[2]

  # (A) Beta fit plot:
  x = np.linspace(0, 1, 100000)
  for ii, dataset_name in enumerate(dataset_dict):
    mlmodel, imageset = dataset_mlmodel_imageset_map[dataset_name]
    curr_marker = mlmodel_marker_map[mlmodel]
    curr_color = imageset_color_map[imageset]
    curr_ls = mlmodel_linestyle_map[mlmodel]
    dataset = dataset_dict[dataset_name]
    ds = dataset[Folds.test]
    alpha = beta_fit_data['data'][ds.model]['a']
    beta = beta_fit_data['data'][ds.model]['b']
    dist = sps.beta(a=alpha, b=beta)
    ax_beta.plot(
        x,
        dist.pdf(x),
        '-',
        color=curr_color,
        label=ds.model,
        ls=curr_ls,
        linewidth=1)

  ax_beta.set_xlabel('$f(X)=c$')
  ax_beta.set_ylabel('PDF')
  ax_beta.set_xlim([0, 1])
  ax_beta.set_ylim([0, 2])
  ax_beta.set_xticks([0, 1])
  ax_beta.set_yticks([0, 2])
  ax_beta.text(0.04, 2 * .9, '(a)', fontsize=fontsize)
  ax_beta.xaxis.set_label_coords(.5, -.055)
  ax_beta.yaxis.set_label_coords(-.055, .5)
  ax_beta.set_xticklabels(['0', '1'])
  ax_beta.set_yticklabels(['0', '2'])

  # (B) Write calibration curve plot:
  for ii, (ds_name, ds_dict) in enumerate(dataset_dict.items()):
    mlmodel, imageset = dataset_mlmodel_imageset_map[ds_name]
    curr_marker = mlmodel_marker_map[mlmodel]
    curr_color = imageset_color_map[imageset]
    curr_ls = mlmodel_linestyle_map[mlmodel]
    ds = ds_dict[Folds.test]
    gm_name_AIC_dict = {  # pylint: disable=invalid-name
        gm_name: glm_fit_data['data'][ds_name][gm_name]['AIC']['mean']['value']
        for gm_name, gm in glm_models.items()
    }
    gm_best_name = min(gm_name_AIC_dict, key=gm_name_AIC_dict.get)
    gm_best = {key: val for key, val in glm_models.items()}[gm_best_name]
    gm_best.plot_calibration(
        ax_glm,
        ds,
        plot_yx=ii == 0,
        color=curr_color,
        linestyle=curr_ls)
  ax_glm.text(0.04, .9, '(b)', fontsize=fontsize)
  ax_glm.set_xlabel('$f(X)=c$')
  ax_glm.set_ylabel('$E[Y|f(x)=c]$')
  ax_glm.yaxis.set_label_coords(-.055, .5)
  ax_glm.xaxis.set_label_coords(.5, -.055)
  ax_glm.set_xlim([0, 1])
  ax_glm.set_ylim([0, 1])
  ax_glm.set_xticks([0, 1])
  ax_glm.set_yticks([0, 1])
  ax_glm.set_xticklabels(['0', '1'])
  ax_glm.set_yticklabels(['0', '1'])

  # (C) EECE vs SECE:
  N_repeats, n_samples = 1000, 1000  # pylint: disable=invalid-name
  data = collections.defaultdict(list)
  for dataset_name in dataset_dict:
    print(dataset_name)
    dataset = dataset_dict[dataset_name]
    ds = dataset[Folds.val]
    eece = ds.compute_error(ce_type='ew_ece_bin', norm=2) * 100
    gm_name_AIC_dict = {  # pylint: disable=invalid-name
        gm_name:
        glm_fit_data['data'][dataset_name][gm_name]['AIC']['mean']['value']
        for gm_name, gm in glm_models.items()
    }
    gm_best_name = min(gm_name_AIC_dict, key=gm_name_AIC_dict.get)
    gm_best = {key: val for key, val in glm_models.items()}[gm_best_name]
    config = {
        'dataset': dataset_name,
        'split': Folds.test,
        'calibration_method': 'no_calibration',
        'ce_type': 'ew_ece_bin',
        'num_bins': 15,
        'bin_method': 'equal_width',
        'norm': 2,
        'num_samples': n_samples
    }
    beta_hat_poly, _, _ = ds.fit_glm(gm_best)
    alpha = beta_fit_data['data'][ds.model]['a']
    beta = beta_fit_data['data'][ds.model]['b']
    p1 = beta_fit_data['data'][ds.model]['p1']
    a = beta_hat_poly[0]
    b = beta_hat_poly[1]
    true_dataset = gm_best.get_true_dist(
        n_samples=n_samples, alpha=alpha, beta=beta, a=a, b=b, p1=p1)
    sece = np.mean([
        calibrate(config, true_dataset=true_dataset) for _ in range(N_repeats)
    ])
    data['dataset'].append(dataset_name)
    data['eece_L2'].append(eece)
    data['sece_L2'].append(sece)
  df = pd.DataFrame(data)
  xlabel, ylabel = 'eece_L2', 'sece_L2'
  ax_compare.plot([0, 25], [0, 25], 'k--')
  for ii, (xi, yi, dataset_name) in enumerate(
      zip(df[xlabel], df[ylabel], df['dataset'])):
    mlmodel, imageset = dataset_mlmodel_imageset_map[dataset_name]
    curr_marker = mlmodel_marker_map[mlmodel]
    curr_color = imageset_color_map[imageset]
    curr_ls = mlmodel_linestyle_map[mlmodel]
    ax_compare.plot([xi], [yi],
                    marker=curr_marker,
                    color=curr_color,
                    ls=curr_ls,
                    alpha=.75,
                    markeredgecolor='k',
                    markersize=5,
                    markeredgewidth=.1)
  ax_compare.set_xlabel(r'ECE$_\mathrm{bin}$ (%)')
  ax_compare.set_ylabel(r'$\langle$ECE$_\mathrm{bin}\rangle$ (%, simulated)')
  ax_compare.text(0.04 * 25, .9 * 25, '(c)', fontsize=fontsize)
  ax_compare.grid(which='both', color='lightgray', linestyle='-')
  f = lambda m, c, l: plt.plot(  # pylint: disable=g-long-lambda
      [], [], marker=m, color=c, ls=l, linewidth=1, markersize=3)[0]
  handles = []
  labels = []
  for dataset_name, dataset in dataset_dict.items():
    mlmodel, imageset = dataset_mlmodel_imageset_map[dataset_name]
    curr_marker = mlmodel_marker_map[mlmodel]
    curr_color = imageset_color_map[imageset]
    curr_ls = mlmodel_linestyle_map[mlmodel]
    curr_handle = f(curr_marker, curr_color, curr_ls)
    handles.append(curr_handle)
    labels.append(dataset_name)
  plt.legend(
      handles,
      labels,
      loc='center left',
      bbox_to_anchor=(1, 0.5),
      prop={'size': legend_fontsize},
      frameon=False)
  axis_range = np.linspace(0, 25, 6)
  # ticklabels = ['']*len(axis_range)
  # ticklabels[0]='0'
  # ticklabels[-1]='25'
  ticklabels = range(0, 25 + 5, 5)
  ax_compare.set_xlim([axis_range[0], axis_range[-1]])
  ax_compare.set_ylim([axis_range[0], axis_range[-1]])
  ax_compare.set_xticks(axis_range)
  ax_compare.set_yticks(axis_range)
  ax_compare.set_xticklabels(ticklabels)
  ax_compare.set_yticklabels(ticklabels)

  for ax in ax_list:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
      item.set_fontsize(fontsize)
  fig.tight_layout(pad=.2, rect=[0, 0.03, 1, 0.95], w_pad=.5)
  fig.savefig(save_file_name, dpi='figure', bbox_inches='tight')


if __name__ == '__main__':
  app.run(main)
