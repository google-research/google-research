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

"""GLM modeling main analysis file."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import itertools
import json
import os
import pickle

from caltrain.run_calibration import calibrate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from six.moves import range
from six.moves import zip

from caltrain.glm_modeling import beta_fit_data
from caltrain.glm_modeling import Datasets
from caltrain.glm_modeling import ece_comparison_dir
from caltrain.glm_modeling import Folds
from caltrain.glm_modeling import glm_fit_data
from caltrain.glm_modeling import GLMModels
from caltrain.glm_modeling import Guo_et_al_data


def get_glm_fit_data(n=50):
  """Compute GLM fit across empirical datasets."""

  col_dict = collections.defaultdict(list)
  for dataset_name, dataset_dict in list(Datasets.items()):
    dataset = dataset_dict[Folds.test]
    for glm_name, glm_model in list(GLMModels.items()):
      print((dataset_name, glm_name))
      row_idx = pd.MultiIndex.from_tuples([(dataset_name, glm_name)],
                                          names=('dataset_name', 'glm_name'))
      curr_glm_fit_data = dataset.fit_glm_bootstrap(glm_model, n=n)
      for metric_name, metric_dict in curr_glm_fit_data.items():
        for stat_name, stat_dict in metric_dict.items():
          col_idx = (metric_name, stat_name, 'value')
          col_dict[col_idx].append(
              pd.Series([stat_dict['statistic']], index=row_idx))
          col_idx = (metric_name, stat_name, 'lower')
          col_dict[col_idx].append(
              pd.Series([stat_dict['minmax'][0]], index=row_idx))
          col_idx = (metric_name, stat_name, 'upper')
          col_dict[col_idx].append(
              pd.Series([stat_dict['minmax'][1]], index=row_idx))

  for key, val in col_dict.items():
    col_dict[key] = functools.reduce(lambda x, y: x.append(y), val)
  df_glm_fit = pd.DataFrame(col_dict)
  df_glm_fit.columns.names = ['parameter', 'statistic', 'estimate']

  def f():
    return collections.defaultdict(f)

  glm_fit_data_dict = collections.defaultdict(f)
  for curr_ds, glm, parameter, statistic, estimate in itertools.product(
      Datasets, GLMModels, ['AIC', 'nll', 'b0', 'b1'], ['mean', 'std'],
      ['value', 'lower', 'upper']):
    try:
      datum = df_glm_fit.loc[curr_ds.name, glm.name].loc[parameter, statistic,
                                                         estimate]
    except KeyError:
      continue
    glm_fit_data_dict[curr_ds.name][
        glm.name][parameter][statistic][estimate] = datum
  glm_fit_data_dict = json.loads(json.dumps(glm_fit_data_dict))

  return {
      'data': glm_fit_data_dict,
      'dataframe': df_glm_fit,
      'metadata': {
          'N': n
      }
  }


def recursive_beta_shift_fit(dataset,
                             arange,
                             brange,
                             n_s,
                             result_dict=None,
                             cf=.1,
                             tol=.00001,
                             shift=1e-16):
  """Compute MLE estimate of beta distribution with recursive brute-force."""

  if max(arange[1] - arange[0], brange[1] - brange[0]) < tol:
    return result_dict

  result_dict, success = dataset.beta_shift_fit(
      arange=arange, brange=brange, n_s=n_s, shift=shift)

  if not success:
    arange_l = max(arange[0] - ((arange[0] + arange[1]) / 2 - result_dict['a']),
                   0)
    arange_r = arange[1] - ((arange[0] + arange[1]) / 2 - result_dict['a'])
    brange_l = max(brange[0] - ((brange[0] + brange[1]) / 2 - result_dict['b']),
                   0)
    brange_r = brange[1] - ((brange[0] + brange[1]) / 2 - result_dict['b'])
  else:
    arange_l = (1 - cf) * arange[0] + cf * result_dict['a']
    arange_r = (1 - cf) * arange[1] + cf * result_dict['a']
    brange_l = (1 - cf) * brange[0] + cf * result_dict['b']
    brange_r = (1 - cf) * brange[1] + cf * result_dict['b']
  return recursive_beta_shift_fit(
      dataset, [arange_l, arange_r], [brange_l, brange_r],
      n_s,
      result_dict=result_dict,
      cf=cf,
      tol=tol,
      shift=shift)


def get_beta_fit_data():
  """Perform MLE of Beta distribution of best fit."""

  data_dict_beta_fit = collections.defaultdict(list)
  for dataset_name, dataset_dict in list(Datasets.items()):
    dataset = dataset_dict[Folds.test]
    beta_fit_best_param_dict = {'nll': float('inf')}
    for shift in [1e-16]:
      print((dataset_name, shift))
      beta_fit_p1_dict = recursive_beta_shift_fit(
          dataset,
          arange=(0, 200),
          brange=(0, 50),
          n_s=11,
          tol=1e-5,
          cf=.5,
          shift=shift)
      if beta_fit_p1_dict['nll'] < beta_fit_best_param_dict['nll']:
        beta_fit_best_param_dict = beta_fit_p1_dict
    data_dict_beta_fit['dataset_name'].append(dataset_name)
    for key, val in beta_fit_best_param_dict.items():
      data_dict_beta_fit[key].append(val)
  df_beta_fit = pd.DataFrame(data_dict_beta_fit).set_index(['dataset_name'])
  print(df_beta_fit)

  def f():
    return collections.defaultdict(f)

  beta_fit_data_dict = collections.defaultdict(f)
  for curr_ds, parameter in itertools.product(Datasets,
                                              ['a', 'b', 'loc', 'scale', 'p1']):
    datum = df_beta_fit.loc[curr_ds.name, parameter]
    if isinstance(datum, np.int64):
      datum = int(datum)
    beta_fit_data_dict[curr_ds.name][parameter] = datum
  beta_fit_data_dict = json.loads(json.dumps(beta_fit_data_dict))
  return {'data': beta_fit_data_dict, 'dataframe': df_beta_fit}


if __name__ == '__main__':

  # Write glm_fit analyis:
  data = get_glm_fit_data(n=10)
  pickle.dump(data, open('glm_fit_data.p', 'wb'))

  # Write beta_fit analyis:
  data = get_beta_fit_data()
  pickle.dump(data, open('beta_fit_data.p', 'wb'))

  # Write glm_fit summary plot:
  for curr_Dataset, curr_GLMModel in itertools.product(Datasets, GLMModels):
    gm = curr_GLMModel.value
    ds = curr_Dataset.value[Folds.test]
    save_file_path = os.path.join('glm_fit_figs', curr_Dataset.name)
    if not os.path.exists(save_file_path):
      os.mkdir(save_file_path)
    save_file_name = os.path.join(save_file_path,
                                  '{}.png'.format(curr_GLMModel.name))
    fig = gm.plot_fit_sequence(ds, figsize_single=3, fontsize=10)
    fig.savefig(save_file_name)

  # Write calibration curve plot:
  fig, ax = plt.subplots(figsize=(5.1, 3.1))
  fontsize = 8
  clrs = sns.color_palette('husl', n_colors=len(list(Datasets.items())))
  LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
  NUM_STYLES = len(LINE_STYLES)
  for ii, (ds_name, ds_dict) in enumerate(Datasets.items()):
    ds = ds_dict[Folds.test]
    gm_name_AIC_dict = {
        gm_name: glm_fit_data['data'][ds_name][gm_name]['AIC']['mean']['value']
        for gm_name, gm in GLMModels.items()
    }
    gm_best_name = min(gm_name_AIC_dict, key=gm_name_AIC_dict.get)
    gm_best = {key: val for key, val in GLMModels.items()}[gm_best_name]
    gm_best.plot_calibration(
        ax,
        ds,
        plot_yx=ii == 0,
        color=clrs[ii],
        linestyle=LINE_STYLES[ii % NUM_STYLES],
        fontsize=fontsize)
  ax.set_title('E[Y | f(x)]')
  ax.set_xlabel('Predicted confidence')
  ax.set_ylabel('Empirical accuracy')
  ax.set_xlim([0, 1])
  ax.set_ylim([0, 1])
  for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
               ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fontsize)
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': fontsize})
  fig.tight_layout(pad=.2, rect=[0, 0.03, 1, 0.95], w_pad=.5)
  fig.savefig('calibration_curve.png')

  # Write score density plots:
  for ds_name, ds_dict in Datasets.items():
    ds = ds_dict[Folds.test]
    fig, ax = plt.subplots(1, 1, figsize=[3] * 2)
    ds.plot_top_score_histogram(ax)
    fig.tight_layout(pad=.2, rect=[0, 0.03, 1, 0.95], w_pad=.5)
    save_file_path = os.path.join('density_figs', ds_name)
    fig.savefig(save_file_path)

  # Write ECE comparison plots:
  dataset_list = Datasets
  N_repeats, n_samples = 100, 1000
  fontsize = 8
  data = collections.defaultdict(list)
  for curr_dataset in dataset_list:
    print((curr_dataset.name, 'L1'))
    ds = curr_dataset.value[Folds.val]
    eece = ds.compute_error(ce_type='ew_ece_bin', norm=1)
    guo_ece = Guo_et_al_data['ECE'][curr_dataset]
    gm_name_AIC_dict = {
        gm_name:
        glm_fit_data['data'][curr_dataset.name][gm_name]['AIC']['mean']['value']
        for gm_name, gm in GLMModels.items()
    }
    gm_best_name = min(gm_name_AIC_dict, key=gm_name_AIC_dict.get)
    gm_best = {key: val for key, val in GLMModels.items()}[gm_best_name]
    tece_best = gm_best.get_calibration_error_beta_dist(
        ds, n_samples=n_samples, norm='L1')
    config = {
        'dataset': curr_dataset.name,
        'split': Folds.test,
        'calibration_method': 'no_calibration',
        'ce_type': 'ew_ece_bin',
        'num_bins': 15,
        'bin_method': 'equal_width',
        'norm': 1,
        'num_samples': n_samples
    }
    beta_hat_poly, nll, AIC = ds.fit_glm(gm_best)
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
    data['dataset'].append(curr_dataset.name)
    data['eece_L1'].append(eece)
    data['guo_ece'].append(guo_ece)
    data['tece_best_L1'].append(tece_best)
    data['sece_L1'].append(sece / 100)
  for curr_dataset in dataset_list:
    print((curr_dataset.name, 'L2'))
    ds = curr_dataset.value[Folds.val]
    eece = ds.compute_error(ce_type='ew_ece_bin', norm=2)
    guo_ece = Guo_et_al_data['ECE'][curr_dataset]
    gm_name_AIC_dict = {
        gm_name:
        glm_fit_data['data'][curr_dataset.name][gm_name]['AIC']['mean']['value']
        for gm_name, gm in GLMModels.items()
    }
    gm_best_name = min(gm_name_AIC_dict, key=gm_name_AIC_dict.get)
    gm_best = {key: val for key, val in GLMModels.items()}[gm_best_name]
    tece_best = gm_best.get_calibration_error_beta_dist(
        ds, n_samples=n_samples, norm='L2')
    config = {
        'dataset': curr_dataset.name,
        'split': Folds.test,
        'calibration_method': 'no_calibration',
        'ce_type': 'ew_ece_bin',
        'num_bins': 15,
        'bin_method': 'equal_width',
        'norm': 2,
        'num_samples': n_samples
    }
    beta_hat_poly, nll, AIC = ds.fit_glm(gm_best)
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
    data['eece_L2'].append(eece)
    data['tece_best_L2'].append(tece_best)
    data['sece_L2'].append(sece / 100)
  df = pd.DataFrame(data)
  clrs = sns.color_palette('husl', n_colors=len(df))
  data_cols = [c for c in df.columns if c != 'dataset']
  for xlabel, ylabel in itertools.combinations(data_cols, 2):
    print((xlabel, ylabel))
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    xmax = max(np.max(df[xlabel]), np.max(df[ylabel]))
    ax.plot([0, .3], [0, .3], 'r--')
    for ii, (xi, yi) in enumerate(zip(df[xlabel], df[ylabel])):
      ax.plot([xi], [yi], '*', color=clrs[ii], label=df.loc[ii]['dataset'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
      item.set_fontsize(fontsize)
    ax.grid(which='both', color='lightgray', linestyle='-')
    ax.legend(
        loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': fontsize})
    ax.set_xlim([0, .3])
    ax.set_ylim([0, .3])
    fig.tight_layout(pad=.2, rect=[0, 0.03, 1, 0.95], w_pad=.5)
    save_file_name = os.path.join(ece_comparison_dir,
                                  '{}_{}.png'.format(xlabel, ylabel))
    fig.savefig(save_file_name)
