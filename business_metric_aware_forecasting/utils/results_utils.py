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

"""Utility functions for displaying results."""

import locale

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

locale.setlocale(locale.LC_ALL, '')


def get_cost_cts(df):
  cost_cts = df[
      ['unit_holding_cost', 'unit_stockout_cost', 'unit_var_o_cost']
  ].value_counts()
  return cost_cts


def get_cost_d(cost_df):
  cost_df = cost_df.drop_duplicates(
      subset=['optimization objective'], keep='last', inplace=False
  )
  cost_d = cost_df.set_index('optimization objective').to_dict(orient='index')
  return cost_d, cost_df


def plot_circles(df, c_o_order, ax=None, var_title=True):
  """Plot relative performance across various tradeoffs.

  Radius of circle = 10 x proportional improvement of using total cost obj.
  versus MSE.

  Args:
    df: dataframe containing performance under various experiment configs.
    c_o_order: list of unit_var_o costs
    ax: axis to plot on
    var_title: whether to put unit_var_o cost as title
  """

  cost_cts = get_cost_cts(df)
  costs_to_winners = {}
  for _, row in cost_cts.reset_index().iterrows():
    c_h = row['unit_holding_cost']
    c_s = row['unit_stockout_cost']
    c_o = row['unit_var_o_cost']
    cost_df = df[
        (df['unit_holding_cost'] == c_h)
        & (df['unit_stockout_cost'] == c_s)
        & (df['unit_var_o_cost'] == c_o)
    ]
    cost_d, cost_df = get_cost_d(cost_df)

    mse_cost = ('MSE', cost_d['mse']['test_te_total_cost'])
    tc_cost = ('Total Cost', cost_d['total_cost']['test_te_total_cost'])

    winner = min([mse_cost, tc_cost], key=lambda x: x[1])
    loser = max([mse_cost, tc_cost], key=lambda x: x[1])

    winner_amt = -(winner[1] - loser[1]) / loser[1]

    costs_to_winners[(c_h, c_s, c_o)] = (winner[0], winner_amt)

  obj_to_color = {
      'MSE': 'tab:red',
      'Total Cost': 'tab:blue',
  }
  if ax is None:
    _, ax = plt.subplots(
        1, len(c_o_order), figsize=(len(c_o_order) * 3.5, 3), sharey=True
    )
  first = {k: True for k in list(obj_to_color.values()) + ['center']}

  for costs, winner in costs_to_winners.items():
    c_h, c_s, c_o = costs
    winner_name, winner_amt = winner

    o_idx = c_o_order.index(c_o)
    label = None
    if first['center'] and o_idx == len(c_o_order) - 1:
      label = 'unit costs'
      first['center'] = False
    ax[o_idx].plot([c_h], [c_s], 'o', markersize=1, color='black', label=label)

    color = obj_to_color[winner_name]
    label = None
    if first[color] and o_idx == len(c_o_order) - 1:
      first[color] = False
      label = {
          'tab:red': 'MSE better',
          'tab:blue': 'TC better',
      }[color]
    circle = plt.Circle(
        (c_h, c_s), winner_amt * 10, color=color, alpha=0.5, label=label
    )

    ax[o_idx].add_patch(circle)
    if var_title:
      ax[o_idx].set_title(r'$c_v = $' + f'{c_o:.0e}', fontsize=11)
  #         ax[o_idx].set_ylabel(r'$c_v = $' + f'\n{c_o:.0e}', rotation=0)

  for axis in ax:
    axis.set_xlim(0, 11)
    axis.set_ylim(0, 11)
    start, end = axis.get_xlim()
    axis.xaxis.set_ticks(np.arange(start, end, 2))


def get_summary(df):
  """Creates a dataframe summarizing performance across various costs.

  Args:
    df: full dataframe containing performance across various costs and configs.

  Returns:
    concise summary dataframe useful for plotting
  """
  cost_cts = get_cost_cts(df)
  # table
  summary = []
  for _, row in cost_cts.reset_index().iterrows():
    c_h = row['unit_holding_cost']
    c_s = row['unit_stockout_cost']
    c_o = row['unit_var_o_cost']
    cost_df = df[
        (df['unit_holding_cost'] == c_h)
        & (df['unit_stockout_cost'] == c_s)
        & (df['unit_var_o_cost'] == c_o)
    ].sort_values('max_steps')
    cost_d, cost_df = get_cost_d(cost_df)
    d = {
        'unit_holding_cost': c_h,
        'unit_stockout_cost': c_s,
        'unit_var_o_cost': c_o,
        'cost_tuple': f'({c_h:1g}, {c_s:1g}, {c_o:1g})',
    }
    for obj in ['mse', 'total_cost', 'rel_rms_stockout_2']:
      if obj in cost_d:
        d[f'{obj}_perf'] = cost_d[obj]['test_te_total_cost']
    summary.append(d)
  summary = pd.DataFrame(summary)
  summary['h_to_s_ratio'] = (
      summary['unit_holding_cost'] / summary['unit_stockout_cost']
  )
  if 'mse_perf' in summary.columns:
    imprv = (
        -(summary['total_cost_perf'] - summary['mse_perf'])
        / summary['mse_perf']
    )
    summary['rel_improvement_tc_to_mse'] = imprv

  if 'rel_rms_stockout_2_perf' in summary.columns:
    imprv = (
        -(summary['total_cost_perf'] - summary['rel_rms_stockout_2_perf'])
        / summary['rel_rms_stockout_2_perf']
    )
    summary['rel_improvement_tc_to_rel_rms'] = imprv
  return summary


def remove_nas(series):
  return series[~series.isna()]


def graph_relative_improvements(summary, ax, legend=True, xfontsize=14):
  """Graph relative percentage improvements vs. C_h / C_s ratios.

  Args:
    summary: summary dataframe returned by get_summary() function
    ax: axis to plot on
    legend: whether to include legend
    xfontsize: font size of x axis
  """
  # line graph
  cols = [c for c in summary.columns if c.startswith('rel_improvement')]
  line_summary = summary.groupby('h_to_s_ratio')[cols].mean()
  line_stds = summary.groupby('h_to_s_ratio')[cols].std()
  line_cts = summary.groupby('h_to_s_ratio')[cols].count()
  if 'rel_improvement_tc_to_mse' in cols:
    c = 'rel_improvement_tc_to_mse'
    mean = line_summary[c]
    std = line_stds[c]
    ct = line_cts[c]
    mean, std, ct = remove_nas(mean), remove_nas(std), remove_nas(ct)
    mean.plot(marker='o', label='imp. over MSE obj.', ax=ax, color='tab:blue')
    ax.fill_between(
        line_stds.index,
        mean - 1.96 * (std / ct),
        mean + 1.96 * (std / ct),
        color='tab:blue',
        alpha=0.1,
    )
  if 'rel_improvement_tc_to_rel_rms' in cols:
    c = 'rel_improvement_tc_to_rel_rms'
    mean = line_summary[c]
    std = line_stds[c]
    ct = line_cts[c]
    mean, std, ct = remove_nas(mean), remove_nas(std), remove_nas(ct)
    mean.plot(
        marker='o', label='imp. over RRMS obj.', ax=ax, color='tab:orange'
    )
    ax.fill_between(
        line_stds.index,
        mean - 1.96 * (std / ct),
        mean + 1.96 * (std / ct),
        color='tab:orange',
        alpha=0.1,
    )
  vals = ax.get_yticks()
  ax.set_yticklabels([f'{x:.0%}' for x in vals])
  ax.set_xscale('log')
  ax.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
  ax.set_xticklabels([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
  ax.set_xlabel(r'$c_h / c_s$ ratio', fontsize=xfontsize)
  if legend:
    ax.legend()


def plot_bar(cols, plot_df, figsize):
  """Plot bar plots grouping each cost tradeoff together.

  As the number of cost tradeoffs grows, the width of this plot should increase.

  Args:
    cols: columns of interest
    plot_df: dataframe to plot
    figsize: size of figures
  """
  plot_df = plot_df.sort_values([
      'h_to_s_ratio',
      'unit_var_o_cost',
      'unit_holding_cost',
  ])[[c for c in cols if c in plot_df.columns]]
  plot_df = plot_df.rename(
      {
          'cost_tuple': r'Costs ($C_h$, $C_s$, $C_o$)',
          'mse_perf': 'MSE Objective',
          'total_cost_perf': 'Total Cost Objective',
          'rel_rms_stockout_2_perf': 'Relative RMS Objective',
      },
      axis=1,
  )
  plot_df.set_index(r'Costs ($C_h$, $C_s$, $C_o$)').plot(
      kind='bar', figsize=figsize
  )
  plt.show()


def get_model_objective(x):
  if x['optimization objective'] != 'None':
    return f'{x.model_name} ({x["optimization objective"]})'
  else:
    return x.model_name


def process_tradeoff_df(df):
  """Pre-process dataframe to remove duplicates and improve naming.

  Args:
    df: raw dataframe containing performance under various configurations

  Returns:
    pre-processed dataframe
  """
  def get_cost_tuple(x):
    c_h, c_s, c_v = x.unit_holding_cost, x.unit_stockout_cost, x.unit_var_o_cost
    return  f'({int(c_h)}, {int(c_s)}, {c_v:.0e})'

  df['cost_tuple'] = df.apply(get_cost_tuple, axis=1)
  df['model_objective'] = df.apply(get_model_objective, axis=1)
  dedup = []
  unique_cols = ['model_name', 'optimization objective', 'cost_tuple']
  for idx, grp in df.groupby(unique_cols):
    obj = idx[1]
    if len(grp) > 1:
      vl_name = f'test_vl_{obj}'
      grp = grp[
          grp[vl_name] == grp[vl_name].min()
      ]  # drop duplicates based on validation perf
    if len(grp) > 1:
      grp = grp.drop_duplicates(unique_cols)
    dedup.append(grp)
  df = pd.concat(dedup, axis=0)
  return df


def bold_extreme_values(data, format_string='%.2f', max_=False):
  """Wraps extreme values in latex textbf decorator.

  Args:
    data: dataframe
    format_string: format string for numbers
    max_: whether to take maximum or minimum

  Returns:

  """
  if max_:
    extrema = data != data.max()
  else:
    extrema = data != data.min()

  def format_val(x, bold):
    if isinstance(x, str):
      x = locale.atof(x)
    if bold:
      fv = '\\textbf{%s}' % format_string % float(x)
    else:
      fv = x
    return fv

  bolded = data.apply(lambda x: format_val(x, bold=True))
  formatted = data.apply(lambda x: format_val(x, bold=False))
  return formatted.where(extrema, bolded)
