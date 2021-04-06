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

"""Runs eval metrics for the shilling attack experiment in Section 4."""

# pylint: disable=use-symbolic-message-instead
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=dangerous-default-value
# pylint: disable=invalid-name
# pylint: disable=C6204

import collections
import copy
import json
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')


# User-defined hyperparameters for the experiment. These should match the first
# three user parameters in polblogs_experiment.py
SAVE_DIR = 'experiment_data/shilling'
NUMBER_OF_EXPERIMENTS = 10
# Copy line 739 and 742 from shilling_experiment.py
methods = ['deepwalk', 'glove', 'monet0', 'monet', 'random', 'nlp']
DB_LEVELS = [v / 100.0 for v in list(range(75, 100, 5)) + [50, 25]]
################################################################################

# Register results saving directory
EVAL_SAVE_DIR = os.path.join(SAVE_DIR, 'exp_results')
if not os.path.isdir(EVAL_SAVE_DIR):
  os.mkdir(EVAL_SAVE_DIR)

# Helper function to get method name from debias (DB) level
monet_alpha_encoder = lambda x: 'monet%0.2f' % x

# Register names of methods and display names
methods.extend([monet_alpha_encoder(db_level) for db_level in DB_LEVELS])
replace_dict = {
    'deepwalk': 'DeepWalk',
    'monet0': 'GloVe_meta',
    'monet': 'MONET_G',
    'random': 'Random',
    'glove': 'GloVe',
    'nlp': 'NLP'
}


def movielens_result_2d(  # pylint: disable=dangerous-default-value, missing-docstring
    df,
    cpalette,
    ppalette,
    figsize=(13, 10),
    title='Attacked Vids in Top-20 vs MRR-Lift, k=20',
    xtitle=None,
    ytitle=None,
    ignore_methods=['Random', 'Adversary', 'MONET_G-0.75', 'MONET_G-0.25'],
    x_col='MRR@k / random-MRR@k',
    x_subtitle='(higher better)',
    y_col='Attacked Vids in Top-20',
    y_subtitle='(lower better)',
    method_col='Method',
    annotate_size=26.0,
    title_size=40.0,
    ax_label_size=28.0,
    ax_tick_size=26.0,
    legend_text_size=26.0,
    xlim=(3.0, 8.0),
    ylim=(-0.5, 11.0),
    markersize=300,
    legend_markersize=18,
    text_loff1=0.7,
    text_uoff1=0.1,
    text_loff2=0.35,
    text_uoff2=0.25,
    legpos='lower right',
    filename=None):

  if xtitle is None:
    xtitle = x_col
  if ytitle is None:
    ytitle = y_col

  method_names = colors_palette.keys()

  # General figure specs
  _ = plt.figure(figsize=figsize)
  plt.rc('axes', titlesize=title_size)  # fontsize of the axes title
  plt.rc('axes', labelsize=ax_label_size)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=ax_tick_size)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=ax_tick_size)  # fontsize of the tick labels
  plt.rc('legend', fontsize=legend_text_size)  # legend fontsize
  plt.suptitle(title, fontsize=title_size)
  plt.title('')
  plt.xlim(xlim)
  plt.ylim(ylim)
  plt.xlabel(xtitle)

  custom_points = []
  # Plotting individual results
  for m in method_names:
    if m not in ignore_methods:
      x_mean = numpy.mean(df[df[method_col] == m][x_col])
      y_mean = numpy.mean(df[df[method_col] == m][y_col])
      plt.scatter(
          x=x_mean,
          y=y_mean,
          marker=ppalette[m],
          color=cpalette[m],
          s=markersize)
      plt.xlabel('%s\n%s' % (xtitle, x_subtitle))
      plt.ylabel('%s\n%s' % (ytitle, y_subtitle))
      if 'MONET' in m:
        if m == 'MONET_G':
          text = r'$\lambda$=1.00'
          custom_points.append(
              Line2D([0], [0],
                     color='w',
                     marker=ppalette[m],
                     markerfacecolor=cpalette[m],
                     label=m,
                     markersize=legend_markersize))
        else:
          text = r'$\lambda$=%s' % m[-4:]
        if m[-2:] == '50':
          plt.annotate(
              text, (x_mean - text_loff2, y_mean + text_uoff2),
              size=annotate_size)
        else:
          plt.annotate(
              text, (x_mean - text_loff1, y_mean + text_uoff1),
              size=annotate_size)
      else:
        custom_points.append(
            Line2D([0], [0],
                   color='w',
                   marker=ppalette[m],
                   markerfacecolor=cpalette[m],
                   label=m,
                   markersize=legend_markersize))
  # Plot GloVe_meta again
  m = 'GloVe_meta'
  x_mean = numpy.mean(df[df[method_col] == m][x_col])
  y_mean = numpy.mean(df[df[method_col] == m][y_col])
  plt.scatter(
      x=x_mean,
      y=y_mean,
      marker=ppalette[m],
      color=cpalette[m],
      s=markersize)
  plt.legend(
      handles=custom_points,
      loc=legpos,
      numpoints=1,
      shadow=True,
      fancybox=False)
  if filename is not None:
    plt.savefig(filename, bbox_inches='tight')


# Load results and create master list
exp_result_list = []
for experiment_number in range(NUMBER_OF_EXPERIMENTS):
  exp_save_dir = os.path.join(SAVE_DIR, 'experiment%d' % experiment_number)
  with open(os.path.join(exp_save_dir, '%d.txt' % experiment_number)) as f:
    exp_result = json.loads(f.read())
    exp_result_list.append(exp_result)
result_df = pd.DataFrame(exp_result_list)

# Create timing and embedding distance CIs
distcorr_dict = collections.defaultdict(list)
time_dict = collections.defaultdict(list)
for exp_result in exp_result_list:
  for method in methods:
    if '.' not in method:
      distcorr_dict[method].append(exp_result['%s_vs_glove_distcorr' % method])
      if method not in ['nlp', 'random']:
        time_dict[method].append(exp_result['%s_time' % method])

# Change dict names to display names
for method in methods:
  if method in time_dict:
    time_dict[replace_dict[method]] = time_dict[method]
    del time_dict[method]
  if method in distcorr_dict:
    distcorr_dict[replace_dict[method]] = distcorr_dict[method]
    del distcorr_dict[method]


def m_pm_s3(m, ss):
  return '%0.3f $\pm$ %0.3f' % (m, ss)  # pylint: disable=anomalous-backslash-in-string


def m_pm_sint(m, ss):
  return '%d $\pm$ %d' % (m, ss)  # pylint: disable=anomalous-backslash-in-string


def two_col_float_with_std(name, mm1, ss1, mm2, ss2):
  if numpy.isnan(mm2):
    string2 = 'N/A'
  else:
    string2 = m_pm_sint(round(mm2), round(ss2))
  return '%s & %s & %s \\\\' % (name, m_pm_s3(mm1, ss1), string2)


flines = []
for method in methods:
  if '.' not in method:
    m1 = s1 = m2 = s2 = numpy.nan
    if replace_dict[method] in distcorr_dict:
      m1 = numpy.mean(distcorr_dict[replace_dict[method]])
      s1 = numpy.std(distcorr_dict[replace_dict[method]])
    if replace_dict[method] in time_dict:
      m2 = numpy.mean(time_dict[replace_dict[method]])
      s2 = numpy.std(time_dict[replace_dict[method]])
    flines.append(two_col_float_with_std(replace_dict[method], m1, s1, m2, s2))
with open(os.path.join(EVAL_SAVE_DIR, 'time_and_distcorr.txt'), 'w') as f:
  f.writelines([s + '\n' for s in flines])

# Prep MRR df
mrr_df_list = []
for idx, row in result_df.iterrows():
  for method in methods:
    mrr_col = '%s_mrr_curve_full' % method
    mrr_col_rand = 'random_mrr_curve_full'
    if method not in ['random']:
      for k in [1, 5, 10, 20]:
        mrr_ratio = row[mrr_col][0][k - 1] / row[mrr_col_rand][0][k - 1]
        mrr_df_list.append({
            'k': k,
            'Method': method,
            'MRR@k / random-MRR@k': mrr_ratio,
            'Attacked Vids in Top-20': row[method]
        })
mrr_df = pd.DataFrame(mrr_df_list)


def replace_method_column(df, input_replace_dict):
  new_df = copy.deepcopy(df)
  for m in input_replace_dict:
    new_df = new_df.replace(m, input_replace_dict[m])
  return new_df


acceptable_points = ['^', 'o', 'v', 's', 'p', 'x', 'D', '*']
acceptable_colors = sns.color_palette()
sorted_names = sorted(replace_dict.values())
sorted_names = [n for n in sorted_names if '.' not in n]
colors_palette = {n: acceptable_colors[i] for i, n in enumerate(sorted_names)}
points_palette = {n: acceptable_points[i] for i, n in enumerate(sorted_names)}
# Add the backed-off monets to the list
for method in methods:
  if 'monet0.' in method:
    meth_name = 'MONET_G-%s' % method[-4:]
    colors_palette.update({meth_name: colors_palette['MONET_G']})
    points_palette.update({meth_name: points_palette['MONET_G']})
    replace_dict.update({method: meth_name})

mrr_df = replace_method_column(mrr_df, replace_dict)
movielens_result_2d(
    df=mrr_df[mrr_df['k'] == 1],
    xlim=[7.0, 15.0],
    title='Attacked Items in Top-20 vs MRR',
    xtitle='MRR / MRR-random',
    ytitle='Attacked Items in Top-20',
    legpos='lower right',
    cpalette=colors_palette,
    ppalette=points_palette,
    text_loff1=1.0,
    text_uoff1=0.25,
    text_loff2=1.0,
    filename=os.path.join(EVAL_SAVE_DIR, 'mrr_curve.png'))
