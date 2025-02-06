# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Utilities used for approxNN project."""
import time
import gc
import itertools
import os
import pickle
import sys

from absl import flags
from absl import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy.stats
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_kernels
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile
import tensorflow_datasets as tfds
from tqdm import tqdm

import config
import explanation_utils
import other
import utils

logging.set_verbosity(logging.INFO)

from debug import ipsh

FLAGS = flags.FLAGS

# pylint:disable=line-too-long

def plot_and_save_samples_and_explans(
    samples, explans, count, filename_prefix=''):
  """A debugging method used to show samples and generated explans.

  Args:
    samples: the samples for which explanations are generated.
    explans: the explans corresponding to the samples above.
    count: the number of (sample, explan) pairs to plot.
  """
  num_rows = 2
  num_cols = count
  plotting.update_matplotlib_defaults()
  fig, axes = plt.subplots(
      num_rows,
      num_cols,
      figsize=(num_cols*6, num_rows*6),
      sharex='col',
      sharey='col',
  )
  assert samples.shape == explans.shape
  channel_1_and_2_dim = (
      other.get_dataset_info(config.cfg.DATASET)['data_shape'][:2]
  )
  for col_idx in range(count):
    axes[0, col_idx].imshow(samples[col_idx].reshape(channel_1_and_2_dim))
    axes[1, col_idx].imshow(explans[col_idx].reshape(channel_1_and_2_dim))

  fig.savefig(
      gfile.GFile(
          os.path.join(
              config.cfg.PLOTS_DIR_PATH,
              f'{filename_prefix}_samples_and_explans',
          ),
          'wb',
      ),
      dpi=150,
  )


def latexify(
    fig_width=None,
    fig_height=None,
    columns=1,
    large_fonts=False,
    font_scale=1,
):
  """Set up matplotlib's RC params for LaTeX plotting.
  Call this before plotting a figure.

  Parameters
  ----------
  fig_width : float, optional, inches
  fig_height : float,  optional, inches
  columns : {1, 2}
  """

  # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

  # Width and max height in inches for IEEE journals taken from
  # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

  assert(columns in [1, 2])

  if fig_width is None:
      fig_width = 3.39 if columns == 1 else 6.9  # width in inches

  if fig_height is None:
      golden_mean = (np.sqrt(5) - 1.0) / 2.0    # Aesthetic ratio
      fig_height = fig_width * golden_mean  # height in inches

  # MAX_HEIGHT_INCHES = 8.0
  # if fig_height > MAX_HEIGHT_INCHES:
  #     print("WARNING: fig_height too large:" + fig_height +
  #           "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
  #     fig_height = MAX_HEIGHT_INCHES

  new_font_size = font_scale * 10 if large_fonts else font_scale * 7
  params = {'backend': 'ps',
            'text.latex.preamble': ['\\usepackage{gensymb}'],
            # fontsize for x and y labels (was 10)
            'axes.labelsize': new_font_size,
            'axes.titlesize': new_font_size,
            'font.size': new_font_size,
            'legend.fontsize': new_font_size,
            'xtick.labelsize': new_font_size,
            'ytick.labelsize': new_font_size,
            'text.usetex': True,
            'figure.figsize': [fig_width, fig_height],
            'font.family': 'serif',
            'xtick.minor.size': 0.5,
            'xtick.major.pad': 1.5,
            'xtick.major.size': 1,
            'ytick.minor.size': 0.5,
            'ytick.major.pad': 1.5,
            'ytick.major.size': 1,
            'lines.linewidth': 1.5,
            'lines.markersize': 0.5,
            'hatch.linewidth': 0.5
            }

  matplotlib.rcParams.update(params)
  plt.rcParams.update(params)


def update_matplotlib_defaults():
  plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
  plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def save_paper_figures(file_name, dpi=200):
  # plt.subplots_adjust(hspace=0.4, wspace=0.4)
  plt.savefig(
      gfile.GFile(os.path.join(config.cfg.PLOTS_DIR_PATH, file_name), 'wb'),
      dpi=dpi,
      bbox_inches="tight",
  )


def plot_vanilla_ite_values():
  """Show sample ITE/ATE value by showing distribution over Y and E."""

  if not config.cfg.USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS:
    raise ValueError('Expected use of identical samples for base models.')

  all_loaded_data_files = utils.load_processed_data_files({
      'samples': False,
      'y_preds': True,
      'y_trues': True,
      'w_chkpt': False,
      'w_final': False,
      'explans': False,
      'hparams': True,
      'metrics': True,
  }, explainer=config.cfg.EXPLANATION_TYPE)
  y_preds = all_loaded_data_files['y_preds']
  y_trues = all_loaded_data_files['y_trues']
  hparams = all_loaded_data_files['hparams']

  # Reorder columns for easier readability when debugging.
  hparams = hparams[[*config.CAT_HPARAMS, *config.NUM_HPARAMS]]
  hparams = utils.process_hparams(hparams, round_num=True, cat_to_code=False)

  num_base_models_times_samples = y_preds.shape[0]

  # For each of the desired hparams
  for col in config.ALL_HPARAMS:

    logging.info('Plotting ITE and ATE for hparam `%s`...', col)

    ite_tracker = pd.DataFrame({
        'sample_str': [],
        'x_y_trues': [],
        'x_y_preds': [],
        'hparam_col': [],
        'hparam_val': [],
    })

    # UPDATED CODE: plot over a class-balanced subsamples of the dataset. The
    # y_preds is ~960,000 dimensional (32 x 30,000), and so we have to find the
    # correct x_offset_indices. To do so, we first bundle tuples of size 30,000
    # and assert they belong to the same instance (through a *noisy* proxy by
    # checking the y_pred is identical for all). Then we have an ordered list
    # of y_pred for 32 instances from which we select the index of the first
    # occurance of the unique values as x_offset_idx

    tmp_y_trues = []  # Keep track for plotting; easier than indexing later.

    for x_offset_idx in range(config.cfg.NUM_SAMPLES_PER_BASE_MODEL):

      # x_* prefix is used for variables that correspond to instance x.
      x_indices = range(
          x_offset_idx,
          num_base_models_times_samples,
          config.cfg.NUM_SAMPLES_PER_BASE_MODEL,
      )  # TODO(amirhkarimi): does this still work when filtering to
         # subset of models based on performance?
      x_y_preds = np.argmax(y_preds[x_indices, :], axis=1)
      x_y_trues = np.argmax(y_trues[x_indices, :], axis=1)
      x_hparams = hparams.iloc[x_indices]

      # Sanity check: irrespective of the base model,
      # X_i is shared and so should share y_true value.
      assert np.all(x_y_trues == x_y_trues[0])
      tmp_y_trues.append(x_y_trues[0])

    unique_y_trues, unique_indices = np.unique(tmp_y_trues, return_index=True)

    for idx, x_offset_idx in enumerate(unique_indices):

      sample_str = f'x{idx}'

      # x_* prefix is used for variables that correspond to instance x.
      x_indices = range(
          x_offset_idx,
          num_base_models_times_samples,
          config.cfg.NUM_SAMPLES_PER_BASE_MODEL,
      )
      x_y_preds = np.argmax(y_preds[x_indices, :], axis=1)
      x_y_trues = np.argmax(y_trues[x_indices, :], axis=1)
      x_hparams = hparams.iloc[x_indices]

      # For each of the unique values of this hparam
      for val in x_hparams[col].unique():

        # Filter to those samples that were predicted
        # on models trained using this unique hparam.
        condition = x_hparams[col] == val
        matching_count = condition.sum()

        # Add to ite_tracker.
        ite_tracker = pd.concat([ite_tracker, pd.DataFrame({
                'sample_str': [sample_str] * matching_count,
                'x_y_trues': list(x_y_trues[condition]),
                'x_y_preds': list(x_y_preds[condition]),
                'hparam_col': [col] * matching_count,
                'hparam_val': [val] * matching_count,
            })], ignore_index=True)

    # For some unknown reason, although process_hparams saves hparams as float32
    # the column values in ite_tracker are being saved as float64 which is then
    # shown all garbled when converted to string for the legend. Corrrect below.
    if ite_tracker['hparam_val'].dtype == np.float64:
      ite_tracker['hparam_val'] = ite_tracker['hparam_val'].astype(np.float32)

    latexify(10, 4, font_scale=1.2, large_fonts=True)
    g = sns.catplot(
        x='sample_str',
        y='x_y_preds',
        hue='hparam_val',
        data=ite_tracker,
        kind='violin',
        alpha=0.3,
        legend=False,
    )
    fig = g.fig
    fig.set_size_inches(18, 6)
    g.set_axis_labels(r"$X_i$", r"$Y_h(X_i)$")

    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    # Inspired by: https://stackoverflow.com/a/47392973
    black_star = matplotlib.lines.Line2D([], [], color='black', marker='*',
                                         linestyle='None', markersize=12,
                                         label='True Label')
    blue = matplotlib.lines.Line2D([], [], color='blue', marker='D',
                                         linestyle='None', markersize=6,
                                         label='Average Prediction')
    handles.append(black_star)
    labels.append('True Label')
    handles.append(blue)
    labels.append('Average Prediction')
    plt.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(x_hparams[col].unique()) + 2,
        title=config.HPARAM_NAME_CONVERTER[col],
    )

    # For every X_i:
    # Put a star on the plot close to where the true label is.
    # Inspired by https://stackoverflow.com/a/37518947
    for loc_x in unique_y_trues:
      loc_y = loc_x
      plt.plot(loc_x, loc_y, color='black', marker='*', markersize=14)

    # For every X_i:
    # Put a diamond on the plot close to where the average of that column is.
    for loc_x in unique_y_trues:
      unique_hparam_types = x_hparams[col].unique()
      tmp = list(range(len(unique_hparam_types)))
      # The width of each subgroup of barplots is 0.8 of each column; set below.
      x_shifts = 0.8 * (tmp - np.mean(tmp)) / len(tmp)
      for tmp_unique_hparam_idxs, x_shift in enumerate(x_shifts):
        filtered = ite_tracker.where(
            (ite_tracker['sample_str'] == f'x{loc_x}') &
            (ite_tracker['hparam_val'] == unique_hparam_types[
                tmp_unique_hparam_idxs])
        )
        loc_y = filtered['x_y_preds'].mean()
        plt.plot(loc_x + x_shift, loc_y, color='blue', marker='D', markersize=8)

    fig.suptitle(
        r'$Y(X_i)$ averaged over models with test accuracy in '
        f'[\%{100 * config.cfg.MIN_BASE_MODEL_ACCURACY:.2f}, '
        f'\%{100 * config.cfg.MAX_BASE_MODEL_ACCURACY:.2f}]'
    )
    plt.tight_layout()
    save_paper_figures(f'vanilla_ite_{config.cfg.DATASET}_{col}.png')

    latexify(4, 4, font_scale=1.2, large_fonts=True)
    g = sns.catplot(
        x='hparam_val',
        y='x_y_preds',
        data=ite_tracker,
        kind='violin',
    )
    fig = g.fig
    fig.set_size_inches(6, 6)
    g.set_axis_labels('', '')
    for ax in g.axes[0]:
      ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.tight_layout()
    save_paper_figures(f'vanilla_ate_{config.cfg.DATASET}_{col}.png')


def plot_kernel_ite_comparison(treatment_effect_tracker, prefix=None):
  """Show that kernel choice does not matter. --> then continue w/ RBF."""
  range_accuracy = '0.0_1.0'
  # unique_hparam_types = treatment_effect_tracker['hparam_type'].unique()
  # for hparam_type in unique_hparam_types:
  for hparam_type in ['config.optimizer']:
    filtered = treatment_effect_tracker.where(
        (treatment_effect_tracker['hparam_type'] == hparam_type) &
        (treatment_effect_tracker['range_accuracy'] == range_accuracy)
    ).dropna()
    for target_type in ['explan']:
      g = sns.catplot(
          x='explan_type',
          y='%s_ite' % target_type,
          hue='h1_h2_str',
          col='kernel_type',
          data=filtered,
          kind='violin',
          sharey=False,
          legend=False,
          # height=3,
          # gridspec_kws={"wspace":0.4},
      )
      g.fig.set_size_inches(14, 4)
      for ax_idx, ax in enumerate(g.axes[0]):
        if ax_idx == 0 and target_type == 'explan':
          ax.legend(
              loc='upper left',
              ncol=1,
              title='Hyperparameter',  # cfg.HPARAM_NAME_CONVERTER[hparam_type],
          )

      for ax_idx, ax in enumerate(g.axes.flatten()):
        # ax.set_xlabel('Explanation Type')
        ax.set_xlabel('')
        explan_type = ax.get_title().split('=')[1][1:]
        explan_type = config.KERNEL_NAME_CONVERTER[explan_type]
        ax.set_title(explan_type)
        if ax_idx == 0:
          ax.set_ylabel(r'$ITE_{E}(x)$', fontsize=14)
        else:
          ax.set_ylabel('')

      for ax_idx, ax in enumerate(g.axes.flatten()):
        labels = ax.get_xticklabels()
        for label in labels:
          label.set_text(config.EXPLAN_NAME_CONVERTER[label.get_text()])
        ax.set_xticklabels(labels)

      plt.subplots_adjust(hspace=0.1, wspace=0.1)
      prefix = 'kernel' if not prefix else prefix
      save_paper_figures(
          '%s_ite_comparison_%s_%s_%s_%s.png' %
          (prefix, config.cfg.DATASET, target_type, range_accuracy, hparam_type)
      )

  # range_accuracy = '0.0_1.0'
  # filtered = treatment_effect_tracker.where(
  #     (treatment_effect_tracker['range_accuracy'] == range_accuracy)
  # ).dropna()
  # # TODO(amirhkarimi): support both without having expl type for y_pred plot
  # # for target_type in ['y_pred', 'explan']:
  # for target_type in ['explan']:
  #   g = sns.catplot(
  #       x='explan_type',
  #       y='%s_ite' % target_type,
  #       hue='h1_h2_str',
  #       col='kernel_type',
  #       row='hparam_type',
  #       data=filtered,
  #       kind='violin',
  #       sharey=False,
  #       legend=False,
  #       height=3,
  #   )
  #   g.fig.set_size_inches(16, 36)
  #   for ax in g.axes[0]:
  #     ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
  #   # plt.legend(
  #   #     # loc='upper center',
  #   #     # bbox_to_anchor=(0.5, 1.0),
  #   #     # ncol=len(filtered['h1_h2_str'].unique()),
  #   #     title=hparam_type,
  #   # )
  #   save_paper_figures(
  #       'kernel_ite_comparison_%s_%s_all.png' %
  #       (config.cfg.DATASET, hparam_type)
  #   )


def plot_hparam_ite_comparison(treatment_effect_tracker):
  """Show that hparams affect some explanation methods more."""
  range_accuracy = '0.0_1.0'
  kernel_type = 'rbf'
  filtered = treatment_effect_tracker.where(
      (treatment_effect_tracker['range_accuracy'] == range_accuracy) &
      (treatment_effect_tracker['kernel_type'] == kernel_type)
  ).dropna()
  for target_type in ['y_pred', 'explan']:
    g = sns.catplot(
        x='hparam_type',
        y='%s_ite' % target_type,
        col=None if target_type == 'y_pred' else 'explan_type',
        data=filtered,
        kind='violin',
        sharey=True,
        legend=False,
    )
    if target_type == 'y_pred':
      g.fig.set_size_inches(4, 4)
    else:
      g.fig.set_size_inches(16, 4)

    legend_handles = []
    legend_labels = []
    for idx, hparam in enumerate(
        treatment_effect_tracker['hparam_type'].unique()):
      color = sns.color_palette()[idx]
      legend_handles.append(matplotlib.patches.Patch(color=color))
      legend_labels.append(config.HPARAM_NAME_CONVERTER[hparam])

    for ax_idx, ax in enumerate(g.axes[0]):
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        if ax_idx == 1 and target_type == 'explan':
          ax.legend(
              legend_handles,
              legend_labels,
              loc='upper left',
              ncol=1,
              title='Hyperparameter Type',
          )

    for ax_idx, ax in enumerate(g.axes.flatten()):
      ax.set_xlabel('Hyperparameters')
      if target_type == 'y_pred':
        ax.set_title('Prediction')
        if ax_idx == 0: ax.set_ylabel(r'$ITE_{Y}(x)$', fontsize=14)
      else:
        explan_type = ax.get_title().split('=')[1][1:]
        explan_type = config.EXPLAN_NAME_CONVERTER[explan_type]
        ax.set_title(explan_type)
        if ax_idx == 0: ax.set_ylabel(r'$ITE_{E}(x)$', fontsize=14)

    for ax_idx, ax in enumerate(g.axes.flatten()):
      labels = ax.get_xticklabels()
      for label in labels:
        # label.set_text(config.HPARAM_NAME_CONVERTER[label.get_text()])
        label.set_text('')
      ax.set_xticklabels(labels, rotation=30)

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    save_paper_figures(
        'hparam_ite_comparison_%s_%s_%s_%s.png' %
        (config.cfg.DATASET, target_type, range_accuracy, kernel_type)
    )


def plot_bucket_ite_comparison(treatment_effect_tracker):
  """Compare ITE values for diff-performance models."""


  def tmp_joint_y_e_plotter(filtered, config):
    unique_hparam_types = filtered['hparam_type'].unique()
    unique_explan_types = filtered['explan_type'].unique()

    num_rows = len(unique_hparam_types)
    num_cols = len(unique_explan_types) + 1

    fig, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey='row')
    if axes.ndim == 1:  # if only 1 row (i.e., 1 hparam)
      axes = np.expand_dims(axes, 0)  # to support row, col indexing below.

    for row_idx in range(num_rows):
      for col_idx in range(num_cols):
        ax = axes[row_idx, col_idx]
        hparam_type = unique_hparam_types[row_idx]

        if col_idx == 0:
          y = 'y_pred_ite'
          data = filtered.where(
              (filtered['hparam_type'] == hparam_type)
          ).dropna()
          col_title = 'Prediction'
        else:
          y = 'explan_ite'
          explan_type = unique_explan_types[col_idx - 1]
          data = filtered.where(
              (filtered['hparam_type'] == hparam_type) &
              (treatment_effect_tracker['explan_type'] == explan_type)
          ).dropna()
          explan_type = config.EXPLAN_NAME_CONVERTER[explan_type]
          col_title = explan_type

        sns.violinplot(
            x='range_accuracy',
            y=y,
            hue='h1_h2_str',
            data=data,
            ax=ax,
            linewidth=0.4,
            legend=False
        )

        if col_idx == 0:
          ax.set_ylabel(r'$ITE_{Y}(x)$', fontsize=14)
        elif col_idx == 1:
          ax.set_ylabel(r'$ITE_{E}(x)$', fontsize=14)
        else:
          ax.set_ylabel('')

        if row_idx == 0:
          ax.set_title(col_title)
        if row_idx == len(unique_hparam_types) - 1:
          ax.set_xlabel('Test Accuracy', fontsize=14)
          labels = ax.get_xticklabels()
          for label in labels:
            range_accuracy = label.get_text()
            range_percentile = config.RANGE_ACCURACY_CONVERTER[range_accuracy]
            label.set_text(range_percentile)
          ax.set_xticklabels(labels, rotation=30)
        else:
          ax.set_xlabel('')
          ax.set_xticklabels('')

        if col_idx == 2:
          if num_rows > 1:
            ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 1.18),
                ncol=len(data['h1_h2_str'].unique()),
                title='',
            )
          else:
            ax.legend(
                loc='upper left',
                ncol=1,
                title=hparam_type,
            )
        else:
          ax.legend().set_visible(False)
    return fig, axes


  # unique_hparam_types = treatment_effect_tracker['hparam_type'].unique()
  # for hparam_type in unique_hparam_types:
  for hparam_type in ['config.optimizer']:
    not_range_accuracy = '0.0_1.0'
    kernel_type = 'rbf'
    filtered = treatment_effect_tracker.where(
        (treatment_effect_tracker['hparam_type'] == hparam_type) &
        (treatment_effect_tracker['kernel_type'] == kernel_type) &
        (treatment_effect_tracker['range_accuracy'] != not_range_accuracy)
    ).dropna()

    # ############################################################################
    # # plot ITE_Y and ITE_E separately
    # ############################################################################
    # for target_type in ['y_pred', 'explan']:
    #   g = sns.catplot(
    #       x='range_accuracy',
    #       y='%s_ite' % target_type,
    #       hue='h1_h2_str',
    #       col=None if target_type == 'y_pred' else 'explan_type',
    #       data=filtered,
    #       kind='violin',
    #       sharey='row',
    #       legend=False,
    #       # height=3,
    #       linewidth=0.6,
    #   )
    #   if target_type == 'y_pred':
    #     g.fig.set_size_inches(4, 4)
    #   else:
    #     g.fig.set_size_inches(16, 4)
    #   for ax_idx, ax in enumerate(g.axes[0]):
    #     ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    #     if ax_idx == 0 and target_type == 'explan':
    #       ax.legend(
    #           loc='upper left',
    #           ncol=1,
    #           title=hparam_type,
    #       )

    #   for ax_idx, ax in enumerate(g.axes.flatten()):
    #     ax.set_xlabel('Test Accuracy', fontsize=14)
    #     if target_type == 'y_pred':
    #       ax.set_title('Prediction')
    #       if ax_idx == 0: ax.set_ylabel(r'$ITE_{Y}(x)$', fontsize=14)
    #     else:
    #       explan_type = ax.get_title().split('=')[1][1:]
    #       explan_type = config.EXPLAN_NAME_CONVERTER[explan_type]
    #       ax.set_title(explan_type)
    #       if ax_idx == 0: ax.set_ylabel(r'$ITE_{E}(x)$', fontsize=14)

    #   for ax_idx, ax in enumerate(g.axes.flatten()):
    #     labels = ax.get_xticklabels()
    #     for label in labels:
    #       range_accuracy = label.get_text()
    #       range_percentile = config.RANGE_ACCURACY_CONVERTER[range_accuracy]
    #       label.set_text(range_percentile)
    #     ax.set_xticklabels(labels)

    #   plt.subplots_adjust(hspace=0.1, wspace=0.1)
    #   save_paper_figures(
    #       'bucket_ite_comparison_%s_%s_%s_%s.png' %
    #       (config.cfg.DATASET, kernel_type, hparam_type, target_type)
    #   )

    ############################################################################
    # repeat for one hparams in 1 fig
    ############################################################################
    fig, axes = tmp_joint_y_e_plotter(filtered, config)
    plt.plot([0.27, 0.27], [0.10, 0.88], color='lightgray', lw=2,
      transform=plt.gcf().transFigure, clip_on=False)
    fig.set_size_inches(24, 4)
    save_paper_figures(
        'bucket_ite_comparison_%s_%s_%s_all.png' %
        (config.cfg.DATASET, kernel_type, hparam_type)
    )

  ##############################################################################
  # repeat for all hparams in 1 fig
  ##############################################################################

  not_range_accuracy = '0.0_1.0'
  kernel_type = 'rbf'
  filtered = treatment_effect_tracker.where(
      (treatment_effect_tracker['kernel_type'] == kernel_type) &
      (treatment_effect_tracker['range_accuracy'] != not_range_accuracy)
  ).dropna()

  fig, axes = tmp_joint_y_e_plotter(filtered, config)
  plt.plot([0.27, 0.27], [0.10, 0.88], color='lightgray', lw=2,
      transform=plt.gcf().transFigure, clip_on=False)
  fig.set_size_inches(24, 32)
  save_paper_figures(
      'bucket_ite_comparison_%s_%s_all.png' %
      (config.cfg.DATASET, kernel_type)
  )


def plot_baseline_ite_comparison(treatment_effect_tracker):
  """Show that baseline choice does not matter. --> then go w/ h=n vs h!=n."""
  # pylint:disable=line-too-long
  treatment_effect_tracker['h1_h2_str'] = treatment_effect_tracker['h1_h2_str'].replace(
    ['optimizer: rmsprop vs adam'], 'optimizer: adam vs rmsprop')
  treatment_effect_tracker['h1_h2_str'] = treatment_effect_tracker['h1_h2_str'].replace(
    ['optimizer: rmsprop vs sgd'], 'optimizer: sgd vs rmsprop')
  treatment_effect_tracker['h1_h2_str'] = treatment_effect_tracker['h1_h2_str'].replace(
    ['optimizer: sgd vs adam'], 'optimizer: adam vs sgd')
  plot_kernel_ite_comparison(treatment_effect_tracker, 'baseline')


def plot_y_vs_e_ite_scattering(treatment_effect_tracker, treatment_effect_tracker_all):
  """Show scatter plots for ITE of Y vs E OR for d_yy' dee'."""
  kernel_type = 'rbf'
  treatment_effect_tracker = treatment_effect_tracker.where(
      (treatment_effect_tracker['kernel_type'] == kernel_type)
  ).dropna()
  treatment_effect_tracker_all = treatment_effect_tracker_all.where(
      (treatment_effect_tracker_all['kernel_type'] == kernel_type)
  ).dropna()
  unique_explan_types = treatment_effect_tracker['explan_type'].unique()
  corrs_pearson_all = {}
  corrs_spearman_all = {}
  for explan_type in unique_explan_types:
    corrs_pearson_all[explan_type] = []
    corrs_spearman_all[explan_type] = []

  latexify(10 * 2, 6 * 2, font_scale=1.4, large_fonts=True)
  def plot_and_save_fig_7(treatment_effect_tracker, hue, hue_order,
                          style, style_order, x_key, y_key,
                          x_label, y_label, suffix, plot_averages, append_corr):
    ''' TODO(amirhkarimi): add description.'''

    # make rows show model perf and cols show hparam type
    unique_range_accuracies = treatment_effect_tracker['range_accuracy'].unique()
    unique_range_accuracies = [e for e in unique_range_accuracies if e != '0.0_1.0']
    unique_explan_types = treatment_effect_tracker['explan_type'].unique()

    fig, axes = plt.subplots(
        len(unique_explan_types),
        len(unique_range_accuracies),
        # sharex=True, # 'col',
        # sharey=True, # 'row',
    )

    for j, explan_type in enumerate(unique_explan_types):

      for i, range_accuracy in enumerate(unique_range_accuracies):

        ax = axes[j, i]

        filtered = treatment_effect_tracker.where(
            (treatment_effect_tracker['range_accuracy'] == range_accuracy) &
            (treatment_effect_tracker['explan_type'] == explan_type)
        ).dropna()

        g = sns.scatterplot(
            data=filtered,
            x=x_key,
            y=y_key,
            # hue=hue,
            # hue_order=hue_order,
            # style=style,
            # style_order=style_order,
            ax=ax,
            alpha=0.45,
            s=15,
            # gridspec_kws={'hspace':0.1, 'wspace':0.1},
        )
        # plt.grid()

        # fig.set_size_inches(24, 10)

        if plot_averages:
          # Add averages to scatter plot (this is the ITE value).
          # marker_color = ['r', 'k', 'g']
          for (h1_h2_str, marker_color, marker_shape) in zip(
              filtered['h1_h2_str'].unique(),
              config.MARKER_COLORS,
              config.MARKER_SHAPES,
          ):
            filtered2 = filtered.where(filtered['h1_h2_str'] == h1_h2_str).dropna()
            y_pred_ite = np.mean(filtered2[x_key])
            explan_ite = np.mean(filtered2[y_key])
            y_pred_ite_var = np.var(filtered2[x_key])
            explan_ite_var = np.var(filtered2[y_key])
            ax.plot(
                y_pred_ite,
                explan_ite,
                color=marker_color,
                marker=marker_shape,
                markersize=10,
                mfc='none',
            )

        ax.legend().set_visible(False)
        # ax.get_legend().set_title(col)
        if j == len(unique_explan_types) - 1:
          ax.set_xlabel(x_label, fontsize=14)
        else:
          ax.set_xlabel('', fontsize=4)
          ax.set_xticklabels('', fontsize=14)
        if i == 0:
          ax.set_ylabel(y_label, fontsize=14)
        else:
          ax.set_ylabel('', fontsize=4)
          ax.set_yticklabels('', fontsize=4)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)


    # handles, labels = ax.get_legend_handles_labels()  # TODO(amirhkarimi): Only gets last axis; fix.
    # fig.legend(  # TODO(amirhkarimi): why are some legend elems missing?
    #     handles,
    #     labels,
    #     bbox_to_anchor=(1.05, 0.5),
    #     loc="center left",
    #     # borderaxespad=0,
    #     # ncol=1 if len(handles) <= 5 else 2,
    #     ncol=1,
    #     markerscale=8,
    # )

    # Add x=y line and correct aspect ratio.
    ref_ax = axes[-1, -1]  # OK to get limits on last ax, since they are shared.
    lims = [
        np.min([ref_ax.get_xlim(), ref_ax.get_ylim()]),  # min of both axes
        np.max([ref_ax.get_xlim(), ref_ax.get_ylim()]),  # max of both axes
    ]
    for col_idx in range(len(axes[0])):
      for row_idx in range(len(axes)):
        ax = axes[row_idx, col_idx]
        ax.plot(lims, lims, 'k--', alpha=0.3) # 0.3
        ax.set_aspect('equal')

    # Inspired by: https://stackoverflow.com/a/25814386
    rows = [config.EXPLAN_NAME_CONVERTER[col] for col in unique_explan_types]
    cols = []
    for range_accuracy in unique_range_accuracies:
      # range_old = range_accuracy
      # range_min, range_max = range_old.split('_')
      # range_new = f'{100 * float(range_min):.2f} - {100 * float(range_max):.2f}%' # TODO(amirhkarimi): fix percent printing
      # cols.append(range_new)
      range_percentile = config.RANGE_ACCURACY_CONVERTER[range_accuracy]
      cols.append(range_percentile)

    # fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 8))
    # plt.setp(axes.flat, xlabel='X-label', ylabel='Y-label')

    pad = 3 # in points

    for ax, col in zip(axes[0], cols):
      ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                  xycoords='axes fraction', textcoords='offset points',
                  size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:,0], rows):
      ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  size='large', ha='right', va='center')

    # # https://stackoverflow.com/a/45161551
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # # tight_layout doesn't take these labels into account. We'll need
    # # to make some room. These numbers are are manually tweaked.
    # # You could automatically calculate them, but it's a pain.
    # # fig.subplots_adjust(left=0.15, top=0.95)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    # plt.suptitle(suffix)
    save_paper_figures('y_vs_e_ite_scattering_%s_%s.png' % (config.cfg.DATASET, suffix))


    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    # for j, explan_type in enumerate(unique_explan_types):
    #   corrs_pearson = []
    #   corrs_spearman = []
    #   for i, range_accuracy in enumerate(unique_range_accuracies):
    #     filtered = treatment_effect_tracker.where(
    #         (treatment_effect_tracker['range_accuracy'] == range_accuracy) &
    #         (treatment_effect_tracker['explan_type'] == explan_type)
    #     ).dropna()
    #     x = filtered[x_key]
    #     y = filtered[y_key]
    #     corrs_pearson.append(scipy.stats.pearsonr(x, y)[0])
    #     corrs_spearman.append(scipy.stats.spearmanr(x, y)[0])
    #   ax1.plot(unique_range_accuracies, corrs_pearson, label=explan_type)
    #   ax2.plot(unique_range_accuracies, corrs_spearman, label=explan_type)
    #   if append_corr:
    #     # potential bug: what if missing elems are from the start or middle?
    #     if len(corrs_pearson) < 8:
    #       corrs_pearson.extend([np.nan] * (8 - len(corrs_pearson)))
    #     if len(corrs_spearman) < 8:
    #       corrs_spearman.extend([np.nan] * (8 - len(corrs_spearman)))
    #     corrs_pearson_all[explan_type].append(corrs_pearson)
    #     corrs_spearman_all[explan_type].append(corrs_spearman)

    #     with utils.file_handler(
    #         config.cfg.EXP_DIR_PATH,
    #         'corr_pearson_bands_%s.npy' % config.MEDIATION_TYPE,
    #         'wb',
    #     ) as f:
    #       pickle.dump(corrs_pearson_all, f, protocol=4)

    #     with utils.file_handler(
    #         config.cfg.EXP_DIR_PATH,
    #         'corr_spearman_bands_%s.npy' % config.MEDIATION_TYPE,
    #         'wb',
    #     ) as f:
    #       pickle.dump(corrs_spearman_all, f, protocol=4)
    # ax2.legend()
    # ax1.set_title('Pearson Corr.')
    # ax2.set_title('Spearman Rank Corr.')
    # # fig.tight_layout()
    # plt.subplots_adjust(hspace=0.1, wspace=0.1)
    # save_paper_figures('fig_8_%s_corr_%s.png' % (config.cfg.DATASET, suffix))

  plot_and_save_fig_7(
      treatment_effect_tracker,
      hue='h1_h2_str',
      hue_order=None,
      style='hparam_type',
      style_order=None,
      x_key='y_pred_ite',
      y_key='explan_ite',
      x_label=r'$ITE_{Y}(x)$',
      y_label=r'$ITE_{E}(x)$',
      suffix='%s_all' % kernel_type,
      plot_averages=False,
      append_corr=False,
  )

  # # similar to Fig 7 but at for each hparam
  # unique_hparam_types = treatment_effect_tracker['hparam_type'].unique()
  # for hparam_type in tqdm(unique_hparam_types):
  #   filtered = treatment_effect_tracker.where(
  #       (treatment_effect_tracker['hparam_type'] == hparam_type)
  #   ).dropna()
  #   hue = 'h1_h2_str'
  #   style = 'h1_h2_str'
  #   hue_order=config.MARKER_COLORS[:len(filtered[hue].unique())]
  #   style_order=config.MARKER_SHAPES[:len(filtered[style].unique())]
  #   plot_and_save_fig_7(
  #       filtered,
  #       hue=hue,
  #       hue_order=hue_order,
  #       style=style,
  #       style_order=style_order,
  #       x_key='y_pred_ite',
  #       y_key='explan_ite',
  #       x_label=r'$ITE_{Y}(x)$',
  #       y_label=r'$ITE_{E}(x)$',
  #       suffix='%s_%s' % (kernel_type, hparam_type),
  #       plot_averages=True,
  #       append_corr=True,
  #   )

  # # similar to Fig 7 but at for each hparam x each individual level
  # num_samples = 10
  # unique_hparam_types = treatment_effect_tracker_all['hparam_type'].unique()
  # for x_offset_idx in [f'x_{row_idx}' for row_idx in range(num_samples)]:
  #   filtered = treatment_effect_tracker_all.where(
  #       (treatment_effect_tracker_all['x_offset_idx'] == x_offset_idx)
  #   ).dropna()
  #   for hparam_type in tqdm(unique_hparam_types):
  #     filtered2 = filtered.where(filtered['hparam_type'] == hparam_type).dropna()
  #     hue = 'h1_h2_str'
  #     style = 'h1_h2_str'
  #     hue_order=config.MARKER_COLORS[:len(filtered2[hue].unique())]
  #     style_order=config.MARKER_SHAPES[:len(filtered2[style].unique())]
  #     plot_and_save_fig_7(
  #         filtered2,
  #         hue=hue,
  #         hue_order=hue_order,
  #         style=style,
  #         style_order=style_order,
  #         x_key='d_y_preds',
  #         y_key='d_explans',
  #         x_label='d_y_preds', # r'$d(Y_h(x), Y_{h\'}(x)$', # r'$||\phi(Y_h(x)) - \phi(Y_{h\'}(x))||^2_{\mathcal{G}}$', TODO(amirhkarimi): fix label printing
  #         y_label='d_explans', # r'$d(E_h(x), E_{h\'}(x)$', # r'$||\phi(E_h(x)) - \phi(E_{h\'}(x))||^2_{\mathcal{G}}$', TODO(amirhkarimi): fix label printing
  #         suffix='d_yy\'_ee\'_%s_%s_%s' % (kernel_type, hparam_type, x_offset_idx),
  #         plot_averages=True,
  #         append_corr=False,
  #     )

  ##############################################################################
  # Fig * new * : scatter ITE of Y vs ITE of E per explanation method.
  ##############################################################################
  kernel_type = 'rbf'
  unique_explan_types = treatment_effect_tracker['explan_type'].unique()
  fig, axes = plt.subplots(1,4)
  for idx, explan_type in enumerate(unique_explan_types):
    ax = axes[idx]
    tmp = treatment_effect_tracker.where(
        (treatment_effect_tracker['kernel_type'] == kernel_type) &
        (treatment_effect_tracker['explan_type'] == explan_type)
    ).dropna().sample(1000)
    g = sns.scatterplot(
        data=tmp,
        x='y_pred_ite',
        y='explan_ite',
        hue='h1_h2_str',
        style='hparam_type',
        ax=ax,
        alpha=0.35,
        s=15,
    )
    ax.legend().set_visible(False)
    ax.set_xlabel(r'$ITE_{Y}(x)$', fontsize=14)
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)
    if idx == 0:
      ax.set_ylabel(r'$ITE_{E}(x)$', fontsize=14)
    else:
      ax.set_ylabel('')
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_title(config.EXPLAN_NAME_CONVERTER[explan_type])
  # handles, labels = ax.get_legend_handles_labels()
  # fig.legend(  # TODO(amirhkarimi): why are some legend elems missing?
  #     handles,
  #     labels,
  #     bbox_to_anchor=(1.05, 0.5),
  #     loc="center left",
  #     # borderaxespad=0,
  #     # ncol=1 if len(handles) <= 5 else 2,
  #     ncol=2,
  #     markerscale=8,
  # )
  fig.set_size_inches(16, 4)
  # fig.tight_layout()
  # plt.subplots_adjust(hspace = 0.8)
  save_paper_figures('y_vs_e_ite_scattering_%s_all.png' % (config.cfg.DATASET))


def plot_y_vs_e_ite_corr_bands(treatment_effect_tracker):
  """Show correlation plots for ITE of Y vs E OR for d_yy' dee'."""
  with utils.file_handler(
      os.path.join(
          config.PAPER_BU_PATH,
          f'2022.05.18_final_{config.cfg.DATASET}'
      ),
      f'corr_pearson_bands_mediated.npy',
      'rb',
  ) as f:
    corrs_pearson_all_mediated = pickle.load(f)

  with utils.file_handler(
      os.path.join(
          config.PAPER_BU_PATH,
          f'2022.05.18_final_{config.cfg.DATASET}'
      ),
      f'corr_spearman_bands_mediated.npy',
      'rb',
  ) as f:
    corrs_spearman_all_mediated = pickle.load(f)

  try:
    with utils.file_handler(
        os.path.join(
            config.PAPER_BU_PATH,
            f'2022.05.18_final_{config.cfg.DATASET}'
        ),
        f'corr_pearson_bands_unmediated.npy',
        'rb',
    ) as f:
      corrs_pearson_all_unmediated = pickle.load(f)

    with utils.file_handler(
        os.path.join(
            config.PAPER_BU_PATH,
            f'2022.05.18_final_{config.cfg.DATASET}'
        ),
        f'corr_spearman_bands_unmediated.npy',
        'rb',
    ) as f:
      corrs_spearman_all_unmediated = pickle.load(f)

    assert \
      corrs_pearson_all_mediated.keys() == \
      corrs_spearman_all_mediated.keys() == \
      corrs_pearson_all_unmediated.keys() == \
      corrs_spearman_all_unmediated.keys()

  except:
    pass

  # ##############################################################################
  # # Fig 8.1
  # ##############################################################################

  # fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
  # unique_range_accuracies = treatment_effect_tracker['range_accuracy'].unique()
  # unique_range_accuracies = [e for e in unique_range_accuracies if e != '0.0_1.0']
  # x = unique_range_accuracies

  # # Plot Pearson correlations
  # for idx, key in enumerate(corrs_pearson_all_mediated.keys()):

  #   # General stuff.
  #   color = config.MARKER_COLORS[idx]
  #   explan_type = key

  #   # Plot mediated results.
  #   y_med = np.nanmean(np.array(corrs_pearson_all_mediated[key]), axis=0)
  #   ci = np.nanstd(np.array(corrs_pearson_all_mediated[key]), axis=0) / np.sqrt(len(x))
  #   axes[0].plot(x, y_med, color=color, alpha=.25, label=f'{key}-med.')
  #   axes[0].fill_between(x, (y_med - ci), (y_med + ci), alpha=.05, color=color)

  # # Plot Spearman correlations
  # for idx, key in enumerate(corrs_spearman_all_mediated.keys()):

  #   # General stuff.
  #   color = config.MARKER_COLORS[idx]
  #   explan_type = key

  #   # Plot mediated results.
  #   y_med = np.nanmean(np.array(corrs_spearman_all_mediated[key]), axis=0)
  #   ci = np.nanstd(np.array(corrs_spearman_all_mediated[key]), axis=0) / np.sqrt(len(x))
  #   axes[1].plot(x, y_med, color=color, alpha=.25, label=f'{key}-med.')
  #   axes[1].fill_between(x, (y_med - ci), (y_med + ci), alpha=.05, color=color)

  # # https://discuss.dizzycoding.com/getting-empty-tick-labels-before-showing-a-plot-in-matplotlib/
  # plt.draw()
  # fig.canvas.draw()

  # for ax in axes:
  #   labels = ax.get_xticklabels()
  #   for label in labels:
  #     range_accuracy = label.get_text()
  #     range_percentile = config.RANGE_ACCURACY_CONVERTER[range_accuracy]
  #     label.set_text(range_percentile)
  #   ax.set_xticklabels(labels, rotation=30)
  #   # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

  # axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
  # axes[0].set_title('Pearson Corr.')
  # axes[1].set_title('Spearman Rank Corr.')
  # # fig.tight_layout()
  # plt.subplots_adjust(hspace=0.1, wspace=0.1)
  # save_paper_figures('fig_8.1_%s_corr_%s.png' % (config.cfg.DATASET, 'bands'))

  ##############################################################################
  # Fig 8.2
  ##############################################################################

  fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
  unique_range_accuracies = treatment_effect_tracker['range_accuracy'].unique()
  unique_range_accuracies = [e for e in unique_range_accuracies if e != '0.0_1.0']
  x = unique_range_accuracies

  # Plot Pearson correlations
  for idx, key in enumerate(corrs_pearson_all_mediated.keys()):

    # General stuff.
    color = config.MARKER_COLORS[idx]
    explan_type = key

    # Plot mediated results.
    y_med = np.nanmean(np.array(corrs_pearson_all_mediated[key]), axis=0)
    ci = np.nanstd(np.array(corrs_pearson_all_mediated[key]), axis=0) / np.sqrt(len(x))
    axes[0].plot(x, y_med, color=color, linestyle='solid', alpha=.25)
    axes[0].fill_between(x, (y_med - ci), (y_med + ci), alpha=.05, color=color)

    # Plot unmediated results.
    y_unmed = np.nanmean(np.array(corrs_pearson_all_unmediated[key]), axis=0)
    ci = np.nanstd(np.array(corrs_pearson_all_unmediated[key]), axis=0) / np.sqrt(len(x))
    axes[0].plot(x, y_unmed, color=color, linestyle='dotted', alpha=.25)
    axes[0].fill_between(x, (y_unmed - ci), (y_unmed + ci), alpha=.05, color=color)

    # Plot mediated - unmediated results.
    axes[1].plot(x, y_med - y_unmed, color=color, linestyle='dashed', alpha=.25)

  # Plot Spearman correlations
  for idx, key in enumerate(corrs_spearman_all_mediated.keys()):

    # General stuff.
    color = config.MARKER_COLORS[idx]
    explan_type = key

    # Plot mediated results.
    y_med = np.nanmean(np.array(corrs_spearman_all_mediated[key]), axis=0)
    ci = np.nanstd(np.array(corrs_spearman_all_mediated[key]), axis=0) / np.sqrt(len(x))
    axes[2].plot(x, y_med, color=color, linestyle='solid', alpha=.25)
    axes[2].fill_between(x, (y_med - ci), (y_med + ci), alpha=.05, color=color)

    # Plot unmediated results.
    y_unmed = np.nanmean(np.array(corrs_spearman_all_unmediated[key]), axis=0)
    ci = np.nanstd(np.array(corrs_spearman_all_unmediated[key]), axis=0) / np.sqrt(len(x))
    axes[2].plot(x, y_unmed, color=color, linestyle='dotted', alpha=.25)
    axes[2].fill_between(x, (y_unmed - ci), (y_unmed + ci), alpha=.05, color=color)

    # Plot mediated - unmediated results.
    axes[3].plot(x, y_med - y_unmed, color=color, linestyle='dashed', alpha=.25)

  # Plot reference lines
  axes[1].plot(x, [0] * len(x), color='k', linestyle='dashdot', alpha=.25)
  axes[3].plot(x, [0] * len(x), color='k', linestyle='dashdot', alpha=.25)

  # https://discuss.dizzycoding.com/
  # getting-empty-tick-labels-before-showing-a-plot-in-matplotlib/
  plt.draw()
  fig.canvas.draw()

  for ax in axes:
    labels = ax.get_xticklabels()
    for label in labels:
      range_accuracy = label.get_text()
      range_percentile = config.RANGE_ACCURACY_CONVERTER[range_accuracy]
      label.set_text(range_percentile)
    ax.set_xticklabels(labels, rotation=30)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

  axes[0].set_title('Pearson Corr.')
  axes[1].set_title('Delta Pearson Corr.')
  axes[2].set_title('Spearman Rank Corr.')
  axes[3].set_title('Delta Spearman Rank Corr.')

  legend_handles = []
  legend_labels = []
  for idx, key in enumerate(corrs_pearson_all_mediated.keys()):
    color = config.MARKER_COLORS[idx]
    explan_type = key
    legend_handles.append(
        matplotlib.lines.Line2D([0], [0], color=color, lw=2, linestyle='solid')
    )
    legend_labels.append(config.EXPLAN_NAME_CONVERTER[explan_type])
  legend_handles.extend([
      matplotlib.lines.Line2D([0], [0], color='k', lw=2, linestyle='solid'),
      matplotlib.lines.Line2D([0], [0], color='k', lw=2, linestyle='dotted'),
      matplotlib.lines.Line2D([0], [0], color='k', lw=2, linestyle='dashed'),
      matplotlib.lines.Line2D([0], [0], color='k', lw=2, linestyle='dashdot'),
  ])
  legend_labels.extend([r'Med. $Y$', r'Unmed. $Y$', r'Med. - Unmed.', r'Ref.'])

  plt.legend(
      legend_handles,
      legend_labels,
      loc='center left',
      bbox_to_anchor=(1, 0.5),
      ncol=1,
      title='',
  )

  # Add line separator between subplots
  plt.plot([0.515, 0.515], [0, 1], color='lightgray', lw=1.5,
    transform=plt.gcf().transFigure, clip_on=False)

  plt.subplots_adjust(hspace=0.1, wspace=0.1)
  save_paper_figures('y_vs_e_ite_corr_bands_%s.png' % (config.cfg.DATASET))


def plot_paper_figures():

  ##############################################################################
  # Merge dataframes for all explan_types and all kernel_types.
  ##############################################################################
  treatment_effect_tracker = pd.DataFrame({
      'range_accuracy': [],
      'explan_type': [],
      'kernel_type': [],
      'x_offset_idx': [],
      'col_type': [],
      'h1_h2_str': [],
      'y_pred_ite': [],
      'explan_ite': [],
      'y_pred_ite_var': [],
      'explan_ite_var': [],
      'pred_correctness': [],
  })

  treatment_effect_tracker_all = pd.DataFrame({
      'range_accuracy': [],
      'explan_type': [],
      'kernel_type': [],
      'x_offset_idx': [],
      'col_type': [],
      'h1_h2_str': [],
      'd_y_preds': [],
      'd_explans': [],
  })

  if config.cfg.DATASET == 'cifar10':
    accuracy_ranges = [
      [0.0, 1.0],     #  0 - 100%
      [0.056, 0.154], #  0 - 20%
      [0.155, 0.253], #  20 - 40%
      [0.253, 0.330], #  40 - 60%
      [0.330, 0.385], #  60 - 80%
      [0.385, 0.461], #  80 - 90%
      [0.461, 0.501], #  90 - 95%
      [0.501, 0.521], #  95 - 99%
      [0.521, 0.575], #  99 - 100%
    ]
    if config.MEDIATION_TYPE == 'mediated':
      saved_data_path = '2022.05.12_00.50.41__'  # 100 base models; mediated Y
      # saved_data_path = '2022.08.01_22.42.01__'  # 100 base models; mediated Y
    elif config.MEDIATION_TYPE == 'unmediated':
      saved_data_path = '2022.05.18_01.48.03__'  # 100 base models; unmediated Y
  elif config.cfg.DATASET == 'svhn_cropped':
    accuracy_ranges = [
      [0.0, 1.0],     #  0 - 100%
      [0.070, 0.179], #  0 - 20%
      [0.179, 0.195], #  20 - 40%
      [0.195, 0.196], #  40 - 60%
      [0.196, 0.333], #  60 - 80%
      [0.333, 0.516], #  80 - 90%
      [0.516, 0.595], #  90 - 95%
      [0.595, 0.653], #  95 - 99%
      [0.653, 0.781], #  99 - 100%
    ]
    if config.MEDIATION_TYPE == 'mediated':
      saved_data_path = '2022.05.17_00.59.58__'  # 100 base models; mediated Y
    elif config.MEDIATION_TYPE == 'unmediated':
      saved_data_path = '2022.05.25_04.45.37__'  # 100 base models; unmediated Y
  elif config.cfg.DATASET == 'mnist':
    accuracy_ranges = [
      [0.0, 1.0],     #  0 - 100%
      [0.047, 0.113], #  0 - 20%
      [0.113, 0.359], #  20 - 40%
      [0.359, 0.739], #  40 - 60%
      [0.739, 0.898], #  60 - 80%
      [0.898, 0.955], #  80 - 90%
      [0.955, 0.969], #  90 - 95%
      [0.969, 0.974], #  95 - 99%
      [0.974, 0.986], #  99 - 100%
    ]
    if config.MEDIATION_TYPE == 'mediated':
      saved_data_path = '2022.06.05_12.37.25__'  # 100 base models; mediated Y
    elif config.MEDIATION_TYPE == 'unmediated':
      saved_data_path = '2022.06.05_04.09.00__'  # 100 base models; unmediated Y
  elif config.cfg.DATASET == 'fashion_mnist':
    accuracy_ranges = [
      [0.0, 1.0],     #  0 - 100%
      [0.016, 0.118], #  0 - 20%
      [0.118, 0.474], #  20 - 40%
      [0.474, 0.686], #  40 - 60%
      [0.686, 0.762], #  60 - 80%
      [0.762, 0.826], #  80 - 90%
      [0.826, 0.846], #  90 - 95%
      [0.846, 0.857], #  95 - 99%
      [0.857, 0.887], #  99 - 100%
    ]
    if config.MEDIATION_TYPE == 'mediated':
      saved_data_path = '2022.06.03_19.06.13__'  # 100 base models; mediated Y
    elif config.MEDIATION_TYPE == 'unmediated':
      saved_data_path = '2022.06.04_18.25.25__'  # 100 base models; unmediated Y

  for (min_accuracy, max_accuracy) in accuracy_ranges:
    for explan_type in config.ALLOWABLE_EXPLANATION_METHODS:
      for kernel_type in config.ALLOWABLE_TREATMENT_KERNELS:
        # Overwrite
        dir_path = (
            '/Users/amirhk/dev/invariant_explanations/_saved/'
            f'{saved_data_path}'
            f'dataset_{config.cfg.DATASET}_'
            'explanation_type_ig_'
            f'explan_norm_type_{config.cfg.EXPLAN_NORM_TYPE}_'
            'num_base_models_30000_'
            f'min_test_accuracy_{min_accuracy}_'
            f'max_test_accuracy_{max_accuracy}_'
            f'num_image_samples_{config.cfg.NUM_SAMPLES_PER_BASE_MODEL}_'
            'identical_samples_True'
            'batch_0_of_1'
        )

        file_name = (
            'treatment_effect_tracker_'
            f'{kernel_type}_{explan_type}.npy'
        )
        file_path = os.path.join(dir_path, file_name)
        try:
          file_data = pickle.load(open(file_path, 'rb'))
        except:
          logging.warning('Was not able to located file `%s`', file_name)
          continue
        file_data = file_data.dropna() # TODO(amirhkarimi): why do NaNs occur???
        count = file_data.shape[0]
        range_accuracy = f'{min_accuracy}_{max_accuracy}'
        file_data.insert(0, 'range_accuracy', [range_accuracy] * count, True)
        file_data.insert(1, 'explan_type', [explan_type] * count, True)
        file_data.insert(2, 'kernel_type', [kernel_type] * count, True)
        treatment_effect_tracker = pd.concat([treatment_effect_tracker, file_data], ignore_index=True)

        # if kernel_type == 'rbf':
        #   file_name = (
        #       'treatment_effect_tracker_all_'
        #       f'{kernel_type}_{explan_type}.npy'
        #   )
        #   file_path = os.path.join(dir_path, file_name)
        #   try:
        #     file_data = pickle.load(open(file_path, 'rb'))
        #   except:
        #     logging.warning('Was not able to located file `%s`', file_name)
        #     continue
        #   file_data = file_data.dropna() # TODO(amirhkarimi): why do NaNs occur???
        #   count = file_data.shape[0]
        #   range_accuracy = f'{min_accuracy}_{max_accuracy}'
        #   file_data.insert(0, 'range_accuracy', [range_accuracy] * count, True)
        #   file_data.insert(1, 'explan_type', [explan_type] * count, True)
        #   file_data.insert(2, 'kernel_type', [kernel_type] * count, True)
        #   treatment_effect_tracker_all = treatment_effect_tracker_all.append(
        #       file_data,
        #       ignore_index=True,
        #   )

  # Backwards compatibility; maybe remove later when experiments are re-run.
  rename_map = {'col_type': 'hparam_type'}
  treatment_effect_tracker = treatment_effect_tracker.rename(columns=rename_map)
  treatment_effect_tracker_all = treatment_effect_tracker_all.rename(columns=rename_map)

  treatment_effect_tracker['pred_correctness'] *= 0  # Disable for now (prevent duplicate samples???).

  sns.set_style("darkgrid")
  # sns.set_style("Whitegrid")
  latexify(40, 16, font_scale=1.2, large_fonts=True)

  # plot_vanilla_ite_values()
  plot_kernel_ite_comparison(treatment_effect_tracker)
  plot_hparam_ite_comparison(treatment_effect_tracker)
  plot_bucket_ite_comparison(treatment_effect_tracker)
  # plot_baseline_ite_comparison(treatment_effect_tracker)
  plot_y_vs_e_ite_scattering(treatment_effect_tracker, treatment_effect_tracker_all)
  plot_y_vs_e_ite_corr_bands(treatment_effect_tracker)

