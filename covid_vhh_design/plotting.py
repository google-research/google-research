# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Plotting helpers."""

import re
from typing import Any, Mapping, Optional, Sequence, Union

import altair as alt
import immutabledict
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from covid_vhh_design import helper
from covid_vhh_design import utils

Axes = mpl.axes.Axes
Figure = mpl.figure.Figure

CATEGORY_10 = (
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
)

PALETTE = {
    # Misc
    'Parent': '#000000',
    'Stratified': CATEGORY_10[9],
    'BLI': CATEGORY_10[2],
    # 'Top100': '#ffd92f',
    'Top100': CATEGORY_10[5],
    # Design methods
    utils.BASELINE: CATEGORY_10[0],
    utils.ML: CATEGORY_10[1],
    utils.SHUFFLED: CATEGORY_10[4],
    utils.RANDOM: CATEGORY_10[4],
    # ML models
    utils.VAE: CATEGORY_10[8],
    utils.LGB: CATEGORY_10[3],
    utils.VAE_RANDOM: CATEGORY_10[9],
    utils.CNN: '#ffd92f',
    'Linear': CATEGORY_10[0],
    'SARS-CoV-1': CATEGORY_10[0],
    'SARS-CoV-2': CATEGORY_10[1],
}


# Mapping of metric to corresponding axes label.
_AXIS_LABEL_MAPPING = immutabledict.immutabledict({
    'source_num_mutations': 'Number of mutations from VHH-72',
    'source_model': 'Model',
    'round': '',  # "Round" is added to ticks.
    'target_name': '',  # Target added to ticks.
    'target_short_name': '',  # Target added to ticks.
})


def update_rcparams():
  """Set default rcparams for plotting."""
  mpl.rcParams['font.size'] = 15
  mpl.rcParams['axes.grid'] = True
  mpl.rcParams['axes.axisbelow'] = True
  mpl.rcParams['grid.linestyle'] = '-.'
  mpl.rcParams['grid.linewidth'] = 0.65
  mpl.rcParams['grid.alpha'] = 1
  mpl.rcParams['axes.linewidth'] = 1.5
  mpl.rcParams['axes.edgecolor'] = 'gray'

  mpl.rcParams['xtick.labelsize'] = 13.0
  mpl.rcParams['legend.fontsize'] = 15.0
  mpl.rcParams['legend.title_fontsize'] = 18.0
  mpl.rcParams['axes.labelsize'] = 15.0


def replace_ml_design_by_model(
    df,
    design_col = 'source_design',
    model_col = 'source_model',
):
  """Replaces ML annotations by fine-grained model annotations."""
  df = df.copy()
  idx = df[design_col] == utils.ML
  df.loc[idx, design_col] = df.loc[idx, model_col]
  return df


################################################################################
# Axis labels and ticks.
################################################################################


def _format_xaxis(ax, metric):
  ax.set_xlabel(_AXIS_LABEL_MAPPING[metric])

  if metric == 'round':
    ax.set_xticklabels(
        [f'Round {int(item.get_text()) + 1}' for item in ax.get_xticklabels()]
    )

  # Increase font size of ticks if there is no xlabel.
  if not _AXIS_LABEL_MAPPING[metric]:
    ax.tick_params(axis='x', labelsize=16)
  return ax


def rotate_xlabels(ax, rotation = 30, ha = 'right'):
  """Rotates the x-tick labels of an axis."""
  # Force a draw to ensure tick labels are populated at correct positions.
  # See discussion: https://github.com/mwaskom/seaborn/issues/2717 and
  # https://github.com/matplotlib/matplotlib/issues/6103/
  ax.figure.canvas.draw()
  ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha=ha)


def _color_scale_from_dict(palette, **kwargs):
  """Converts a `dict` color palette to a `alt.Scale` object."""
  return alt.Scale(
      domain=list(palette.keys()), range=list(palette.values()), **kwargs
  )


def get_alt_color(
    variable,
    palette = None,
    ascending = None,
    value_type = None,
    scale_kwargs = None,
    **kwargs,
):
  """Helper function to create a `alt.Color` object.

  Args:
    variable: The name of the variable that is used for coloring.
    palette: The name of the color palette or a dict(Any, str). If the former, a
      special '_a' or '_d' suffix can  be used to indicate if colors should be
      used in ascending or descending order, respectively. E.g. 'spectral_d'
      means 'spectral' in descending order. The suffix is ignored if `ascending`
      is specified. Alternatively, in the case where `variable` is categorical
      this can be a dict mapping individual values to colors (hex or named).
    ascending: If set to `True` or `False`, colors are sorted in ascending or
      descending order respectively. If `None`, the default order is used unless
      the order is encoded in 'palette' (see above).
    value_type: The type of `variable`, e.g. 'quantitative' or 'nominal'. If
      `None`, the type will be inferred.
    scale_kwargs: An optional dict with additional keywords passed to
      `alt.Scale`.
    **kwargs: Named arguments passed to `alt.Color`.

  Returns:
    A `alt.Color` object.
  """
  if value_type:
    kwargs['type'] = value_type
  if scale_kwargs is None:
    scale_kwargs = dict()
  if palette:
    if isinstance(palette, dict):
      kwargs['scale'] = _color_scale_from_dict(palette, **scale_kwargs)
    else:
      match = re.fullmatch(r'(.+)_([ad])', palette)
      if match:
        palette = match.group(1)
        if ascending is None:
          ascending = match.group(2) == 'a'
      kwargs['scale'] = alt.Scale(scheme=palette, **scale_kwargs)
  if ascending is not None:
    kwargs['sort'] = 'ascending' if ascending else 'descending'
  return alt.Color(variable, **kwargs)


################################################################################
# Save legends as external figures to avoid repeating them in subplots.
################################################################################


def make_design_legend(ax, with_shuffled):
  """Scatterplot legend for design method."""
  ml = ax.scatter([], [], color=PALETTE['ML'], marker='.', s=200, label='ML')
  base = ax.scatter(
      [], [], color=PALETTE['Baseline'], marker='.', s=200, label='Baseline'
  )
  shuffled = ax.scatter(
      [], [], color=PALETTE['Shuffled'], marker='.', s=200, label='Shuffled'
  )
  handles = [ml, base] + ([shuffled] if with_shuffled else [])
  ax.legend(
      handles=handles,
      loc='lower center',
      ncol=5,
      labelspacing=0.0,
      frameon=False,
  )
  ax.axis('off')
  return ax


def make_target_legend(ax, with_parent):
  """Barplot legend for selected targets."""
  handles = []
  if with_parent:
    handles.append(
        mpl.lines.Line2D(
            [0],
            [0],
            color='k',
            label='Parent',
            linestyle='--',
        )
    )

  handles += [
      mpl.patches.Patch(facecolor=PALETTE[target], edgecolor='k', label=target)
      for target in ['SARS-CoV-1', 'SARS-CoV-2']
  ]

  ax.legend(
      handles=handles,
      loc='lower center',
      ncol=3,
      labelspacing=0.0,
      frameon=False,
  )
  ax.axis('off')
  return ax


def make_tsne_legend(ax, with_initial_bli):
  """Scatterplot legend for t-SNE plots."""

  def _scatter(ax, color, label, marker='*', linewidth=2, s=300, **kwargs):
    return ax.scatter(
        [],
        [],
        color=color,
        label=label,
        marker=marker,
        linewidth=linewidth,
        s=s,
        **kwargs,
    )

  handles = [
      _scatter(
          ax,
          color=PALETTE[utils.PARENT],
          label='VHH-72',
      ),
      _scatter(
          ax,
          color=PALETTE['BLI'],
          label='BLI',
          edgecolor=PALETTE['BLI'],
      ),
  ]

  if with_initial_bli:
    handles.append(
        _scatter(
            ax,
            color='w',
            edgecolor=PALETTE['BLI'],
            label=r'BLI ($\downarrow$ exp)',
        )
    )

  ax.legend(
      handles=handles,
      loc='lower center',
      ncol=9,
      labelspacing=0.0,
      frameon=False,
      handletextpad=0.01,
  )
  ax.axis('off')
  return ax


################################################################################
# Correlation plots
################################################################################


def plot_correlations(
    ax,
    df,
    x_col,
    y_col,
    hue_col = None,
):
  """Correlation plot annotated with Spearman's r."""
  df = df.dropna(subset=[x_col, y_col])
  sns.scatterplot(
      ax=ax, data=df, x=x_col, y=y_col, hue=hue_col, palette=PALETTE
  )
  xmin, xmax = ax.get_xlim()
  ymin, ymax = ax.get_ylim()
  axmin, axmax = min(xmin, ymin), max(xmax, ymax)
  sns.lineplot(
      ax=ax, x=[axmin, axmax], y=[axmin, axmax], color='k', linestyle='--'
  )

  ax.set_ylim(axmin, axmax)
  ax.set_xlim(axmin, axmax)

  ax.set_aspect('equal', adjustable='box')
  corr, _ = scipy.stats.spearmanr(df[x_col], df[y_col])
  ax.annotate(rf'$\rho$ = {corr:.2f}', xy=(0.04, 0.9), xycoords='axes fraction')
  ax.legend(
      bbox_to_anchor=(0.5, 1.1),
      ncol=4,
      loc='center',
      frameon=False,
  )
  return ax


################################################################################
# Log KD plots
################################################################################


def _plot_log_kd(
    ax,
    df,
    x_col,
    hue_col,
    parent_binding,
    show_iqr,
    parent_label='Parent',
    palette = None,
    **kwargs,
):
  """log KD plotting helper.

  Args:
    ax: Axes on which to draw content.
    df: pd.DataFrame of aggregated log KD measurements. Must contain at least
      columns ["value", `x_col`, `hue_col`].
    x_col: Column of `df` to use as x axis.
    hue_col: Column of `df` to stratify bars by.
    parent_binding: Binding of the parent sequence.
    show_iqr: If True, indicate the parent + 1 IQR and parent - 1 IQR by shaded
      region on the plot.
    parent_label: The label of the parent sequences in the color legend.
    palette: A color palette. PALETTE is the default.
    **kwargs: Additional keyword arguments passed to `sns.boxplot`

  Returns:
    Axes upon which content has been drawn.

  Returns:
    Axes, upon which content has been drawn.
  """
  ax.axhline(
      y=parent_binding,
      color=PALETTE['Parent'],
      linestyle='--',
      label=parent_label,
      zorder=1,
  )

  sns.boxplot(
      data=df,
      x=x_col,
      hue=hue_col,
      y='value',
      palette=PALETTE if palette is None else palette,
      showfliers=True,
      ax=ax,
      zorder=10,
      boxprops={'zorder': 10},
      whiskerprops={'zorder': 10},
      linewidth=2,
      **kwargs,
  )

  if show_iqr:
    xlim = ax.get_xlim()
    ax.fill_between(
        xlim,
        y1=parent_binding - 1,
        y2=parent_binding + 1,
        color=PALETTE['Parent'],
        # label=r'Parent $\pm$ 1 IQR',
        alpha=0.2,
        zorder=1,
    )
    ax.set_xlim(xlim)

  return ax


def plot_log_kd(
    ax,
    agg_df,
    x_col,
    hue_col,
    max_impute_inf,
    num_top_seqs = None,
    order = None,
    **kwargs,
):
  """Plots log KD stratified by `x_col` and `hue_col`."""
  df = agg_df.copy()
  parent_binding = helper.get_unique_value(
      utils.extract_parent_df(agg_df)['value']
  )

  if num_top_seqs is not None:
    seqs = utils.extract_best_sequences_by_category(
        df,
        num_top_seqs=num_top_seqs,
        value_col='value',
        category_col=[x_col, hue_col],
    )
    df = df[df['source_seq'].isin(seqs)].copy()

  if max_impute_inf:
    df['value'] = helper.max_impute_inf(df['value'])
  else:
    df = helper.drop_inf_df(df, 'value')

  df = df[df['source_num_mutations'] != 0]

  boxes = _plot_log_kd(
      ax=ax,
      df=df,
      x_col=x_col,
      hue_col=hue_col,
      parent_binding=parent_binding,
      order=sorted(df[x_col].unique()) if order is None else order,
      hue_order=sorted(df[hue_col].unique()),
      **kwargs,
  )

  # Add hatches to boxes with less than n observations, and update legend if
  # necessary.
  handles, labels = ax.get_legend_handles_labels()
  if num_top_seqs is not None:
    is_missing = (
        df.groupby([x_col, hue_col])['value']
        .size()
        .reset_index()
        .sort_values(by=[x_col, hue_col])
    )
    augment_legend = (is_missing['value'] < num_top_seqs).any()

    patches = [p for p in boxes.patches if isinstance(p, mpl.patches.PathPatch)]
    for patch, (_, row) in zip(patches, is_missing.iterrows()):
      if row['value'] < num_top_seqs:
        patch.set_hatch('//')

    if augment_legend:
      handles.append(
          mpl.patches.Patch(facecolor='w', edgecolor='k', hatch='//')
      )
      labels.append(f'< {num_top_seqs} measurements')

  ax.legend(
      handles,
      labels,
      bbox_to_anchor=(0.5, 1.1),
      ncol=4,
      loc='center',
      frameon=False,
  )
  ax.set_ylabel('Normalized log KD')
  ax = _format_xaxis(ax, x_col)
  return ax


################################################################################
# Hit rate plots.
################################################################################


def _format_count(num):
  return f'{num / 1000: .1f}k' if num >= 1000 else f'{int(num)}'


def _plot_hit_rate(
    ax,
    df,
    x_col,
    hue_col,
    show_counts,
    **kwargs,
):
  """Hit rate plotting helper.

  Args:
    ax: Axes on which to draw content.
    df: pd.DataFrame of computed hit rates. It must contain at least columns
      ['sum', 'mean', 'count', `x_col`, `hue_col`].
    x_col: Column of `df` to use as x axis.
    hue_col: Column of `df` to stratify bars by.
    show_counts: If true, add text over each bar indicating the number of hits
      and number of attempts in the corresponding category.
    **kwargs: Additional keyword arguments passed to `sns.barplot`.

  Returns:
    Axes, upon which content has been drawn.
  """
  df = df.assign(pct=df['mean'] * 100)
  bars = sns.barplot(
      data=df,
      x=x_col,
      y='pct',
      hue=hue_col,
      palette=PALETTE,
      ax=ax,
      zorder=2,
      **kwargs,
  )
  ymin, ymax = bars.get_ylim()

  if show_counts:
    if 'order' in kwargs:

      def sort_fn(series):
        return series.map(kwargs['order'].index)

    else:
      sort_fn = None

    num_hits = df.pivot(values='sum', index=x_col, columns=hue_col).sort_index(
        key=sort_fn
    )
    num_attempts = df.pivot(
        values='count', index=x_col, columns=hue_col
    ).sort_index(key=sort_fn)

    for bar_container in bars.containers:  # Iterating over hues
      for bar, hits, attempts in zip(
          bar_container,
          num_hits[bar_container.get_label()],
          num_attempts[bar_container.get_label()],
      ):
        if hits and not np.isnan(hits):
          bars.annotate(
              f'{_format_count(hits)}/{_format_count(attempts)}',
              (
                  bar.get_x() + bar.get_width() / 1.8,
                  bar.get_height() + (ymax - ymin) / 50,
              ),
              ha='center',
              va='bottom',
              size=11,
              rotation=90,
          )

    bars.set_ylim(ymin, ymax * 1.22)
  return ax


def plot_hit_rate(
    ax,
    agg_df,
    x_col,
    hue_col,
    how,
    show_counts,
    **kwargs,
):
  """Plots hit rates, optionally showing hit counts as labels."""
  if how == 'pvalue':
    measurement_col = 'pvalue_corrected'
    threshold = 0.05
    ylabel = r'Percentage of significant hits (p $\leq$ 0.05)'

  elif how == 'iqr':
    measurement_col = 'value'
    parent_value = helper.get_unique_value(
        utils.extract_parent_df(agg_df)['value']
    )
    threshold = parent_value - 1
    ylabel = 'Percentage of sequences with\n1 IQR improvement over VHH-72'
  else:
    raise ValueError(f'"how" must be one of ["iqr", "pvalue"], but was {how}.')

  hits_df = utils.compute_hit_rate(
      agg_df[agg_df['source_num_mutations'] != 0],
      groupby=[x_col, hue_col],
      thresholds=threshold,
      measurement_col=measurement_col,
      lesser_than=True,
  )

  _plot_hit_rate(
      ax=ax,
      df=hits_df,
      x_col=x_col,
      hue_col=hue_col,
      show_counts=show_counts,
      **kwargs,
  )
  _format_xaxis(ax, x_col)
  ax.set_ylabel(ylabel)
  ax.legend(bbox_to_anchor=(0.5, 1.1), ncol=3, loc='center', frameon=False)

  return ax


################################################################################
# Diversity-related plots
################################################################################


def plot_log_kd_by_distance(
    ax, df, target_name, n
):
  """Scatterplot of best sequences by distance to parent and log KD."""
  df = df[df['target_name'] == target_name].copy()
  df = replace_ml_design_by_model(df)

  # Plot parent
  sns.scatterplot(
      ax=ax,
      data=utils.extract_parent_df(df),
      x='source_num_mutations',
      y='value',
      marker='*',
      color=PALETTE['Parent'],
      s=800,
      edgecolor='w',
      label='Parent',
  )

  sns.scatterplot(
      ax=ax,
      data=df.sort_values(by='value', ascending=True)
      .head(n)
      .sort_values(by='source_model'),
      y='value',
      x='source_num_mutations',
      hue='source_design',
      hue_order=['Baseline', 'CNN', 'LGB'],
      s=60,
      palette=PALETTE,
      edgecolor='k',
      linewidth=0.5,
  )

  ax.legend(
      bbox_to_anchor=(0.48, 1.1),
      ncol=4,
      loc='center',
      frameon=False,
      # handlelength=1.3,
      columnspacing=0.8,
      handletextpad=0.3,
  )

  _format_xaxis(ax, 'source_num_mutations')
  ax.set_ylabel('Normalized log KD')
  ax.set_yticks([0, -1, -2, -3, -4])

  return ax


def plot_log_kd_by_mean_pairwise_distance(
    ax, df, target_name, n
):
  """Scatterplot of best sequences by mean pairwise distance."""
  df = df[df['target_name'] == target_name].copy()
  df = replace_ml_design_by_model(df)
  df = df.sort_values(by='value', ascending=True).head(n)
  all_seqs = df['source_seq'].values

  rows = []
  for i in range(n):
    previous_seqs = all_seqs[:i]
    cur_seq = all_seqs[i]
    rows.append(
        dict(
            source_design=df['source_design'].iloc[i],
            distance=utils.avg_distance_to_set(cur_seq, previous_seqs),
            value=df['value'].iloc[i],
        )
    )
  metrics_df = pd.DataFrame(rows)

  sns.scatterplot(
      ax=ax,
      data=metrics_df.sort_values(by='source_design'),
      y='value',
      x='distance',
      hue='source_design',
      palette=PALETTE,
      s=60,
      edgecolor='k',
      linewidth=0.5,
  )

  ax.legend(
      bbox_to_anchor=(0.48, 1.1),
      ncol=4,
      loc='center',
      frameon=False,
  )
  ax.set_xlabel('Average distance to better sequences')
  ax.set_ylabel('Normalized log KD')


def plot_tsne_by_round(ax, agg_df, round_idx):
  """Generates CoVID t-SNE plots."""
  # Replaced detailed annotation of non ML sequences by high-level annotation.

  sns.scatterplot(
      ax=ax,
      data=agg_df[agg_df['round'] != round_idx],
      x='latents_x',
      y='latents_y',
      s=20,
      edgecolor='k',
      color='w',
      lw=0.1,
      alpha=0.1,
  )

  hue_order = [
      'Baseline',
      'Shuffled',
      'VAE (random)',
      'VAE',
      'LGB',
      'CNN',
  ]

  cur_round = agg_df[
      (agg_df['round'] == round_idx) & (agg_df['source_num_mutations'] != 0)
  ]
  cur_round = replace_ml_design_by_model(cur_round)

  foreground = sns.scatterplot(
      ax=ax,
      data=cur_round,
      x='latents_x',
      y='latents_y',
      hue='source_design',
      s=40,
      palette=PALETTE,
      linewidth=0.8,
      hue_order=[
          m for m in hue_order if m in cur_round['source_design'].unique()
      ],
  )

  handles, labels = foreground.get_legend_handles_labels()

  sns.scatterplot(
      ax=ax,
      data=cur_round[cur_round['bli_v1']],
      x='latents_x',
      y='latents_y',
      marker='*',
      color='w',
      s=400,
      linewidth=2,
      edgecolor=PALETTE['BLI'],
      legend=False,
  )

  sns.scatterplot(
      ax=ax,
      data=cur_round[cur_round['bli_v2']],
      x='latents_x',
      y='latents_y',
      marker='*',
      color=PALETTE['BLI'],
      s=600,
      edgecolor='w',
      legend=False,
  )

  sns.scatterplot(
      ax=ax,
      data=utils.extract_parent_df(agg_df),
      x='latents_x',
      y='latents_y',
      marker='*',
      color=PALETTE['Parent'],
      s=800,
      edgecolor='w',
      legend=False,
  )

  ax.set_xlabel('t-SNE dimension 1')
  ax.set_ylabel('t-SNE dimension 2')

  ax.legend(
      handles=handles,
      labels=labels,
      bbox_to_anchor=(0.5, 1.07),
      ncol=3,
      loc='center',
      frameon=False,
      # handlelength=1.3,
      columnspacing=0.7,
      handletextpad=0.3,
  )

  return ax


def plot_hit_heatmap(
    ax,
    agg_df,
    design_by_sequence,
    min_hits = 3,
):
  """Heatmap of VHH-target binding events."""
  # To reduce the size of the plot, only keep sequences that bind to at least
  # `min_hits`` sequences.
  df = agg_df[agg_df.sum(axis=1) >= min_hits].copy()
  # Sort dataframe so that top rows and right-most columns have the most hits.
  hits_by_target = df.sum().to_dict()
  sorted_cols = sorted(
      df.columns, key=lambda x: hits_by_target[x], reverse=True
  )
  df = df.sort_values(by=sorted_cols, ascending=False)[reversed(sorted_cols)]

  # Replace hits (indicated by `1`) by an integer representing the method used
  # to design the sequence.
  design_to_int = {
      utils.BASELINE: 1,
      utils.ML: 2,
      utils.RANDOM: 3,
      utils.SHUFFLED: 4,
  }

  designs = df.index.map(design_by_sequence)
  df = df.multiply(designs.map(design_to_int), axis=0)

  # Rescale numbers to [0, 1], as required by custom colormaps.
  max_value = df.max().max()
  df = df / max_value

  # Create corresponding colormap.
  colors = [(0, 'white')] + [
      (design_to_int[design] / max_value, PALETTE[design])
      for design in designs.unique()
  ]
  colors = sorted(colors, key=lambda x: x[0])
  cmap = mpl.colors.LinearSegmentedColormap.from_list('', colors)

  ax = sns.heatmap(
      df.reset_index(drop=True).transpose(), cbar=False, cmap=cmap, ax=ax
  )
  xticks = np.arange(0, len(df), 10)

  ax.set_xticks(ticks=xticks + 0.5)
  ax.set_xticklabels(xticks + 1, fontsize=16, rotation=0)
  ax.set_ylabel('')
  ax.set_xlabel('Candidate VHH', fontsize=18)

  for i in range(len(df) + 1):
    ax.axvline(i, color='white', lw=0.2)
  for i in range(len(df.columns) + 1):
    ax.axhline(i, color='white', lw=0.2)

  return ax


def plot_dendogram(ax, df):
  """Dendogram of BLI sequences."""
  df = df.dropna(subset=['source_seq'])
  df = df[df['target_name'] == 'SARS-CoV2_RBD']

  vocab = utils.ProteinVocab()
  structures = np.vstack([vocab.encode(s) for s in df['source_seq'].values])

  labels = df['label'].values
  clustering = scipy.cluster.hierarchy.linkage(
      structures, method='single', metric=utils.hamming_distance
  )
  scipy.cluster.hierarchy.dendrogram(
      clustering, labels=labels, orientation='right', ax=ax
  )
  ax.tick_params(labelsize=16)
  ax.set_xlabel('# pairwise mutations')

  return ax
