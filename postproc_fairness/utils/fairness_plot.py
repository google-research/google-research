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

"""Utilities to analyze fairness debiasing techniques."""

import itertools
import os
import numpy as np
from plotly import colors as plotly_colors
from plotly import graph_objs as go


# Color scale when comparing multiple plots.
COLOR_SCALE = plotly_colors.DEFAULT_PLOTLY_COLORS


def plot_performance_per_weight(
    data_df,
    output_dir=None,
    erm_stats=None,
    perf_metric='val_accuracy',
    fairness_weight_name='mindiff_weight',
    only_line=False,
    label=None,
    color=None,
    use_sem=True,
    fig=None,
):
  """Plots the accuracy against the adversarial/mindiff weight."""

  groups = data_df.groupby(fairness_weight_name)
  groups = groups.agg(['mean', 'std', 'sem']).reset_index()
  sorted_group = groups.sort_values([fairness_weight_name])
  symlog_x = symlog(sorted_group[fairness_weight_name])

  if use_sem:
    err = 'sem'
  else:
    err = 'std'

  if color is None:
    color = COLOR_SCALE[0]

  if fig is None:
    fig = go.Figure()

  fig.add_trace(
      go.Scatter(
          # x=sorted_group[fairness_weight_name],
          x=symlog_x,
          y=sorted_group[perf_metric]['mean'],
          legendgroup=f'{label}',
          name=f'{label}',
          line_color=color,
      )
  )
  y_upper = list(
      sorted_group[perf_metric]['mean'] + sorted_group[perf_metric][err]
  )
  y_lower = list(
      sorted_group[perf_metric]['mean'] - sorted_group[perf_metric][err]
  )
  fig.add_trace(
      go.Scatter(
          x=list(symlog_x) + list(symlog_x)[::-1],
          # x=list(sorted_group[fairness_weight_name])
          # + list(sorted_group[fairness_weight_name])[::-1],
          y=y_upper + y_lower[::-1],  # upper, then lower reversed
          fill='toself',
          fillcolor=color,
          legendgroup=f'{label}',
          opacity=0.25,
          line=dict(color='rgba(255,255,255,0)'),
          hoverinfo='skip',
          showlegend=False,
      )
  )
  if erm_stats is not None:
    fig.add_hline(
        y=erm_stats[perf_metric]['mean'], line_color='darkgray', line_dash='.'
    )
  if only_line:
    return
  xlist = list(symlog_x)
  fig.update_xaxes(
      range=[xlist[0], xlist[-1]],
      # range=[np.log(xlist[0] + 1e-10), np.log(xlist[-1])],
      # type='log',
      title='Fairness head weight',
  )
  fig.update_yaxes(title='Overall {}'.format(perf_metric))

  if output_dir is None:
    return fig

  output_path = os.path.join(
      output_dir, '{}_per_weight.png'.format(perf_metric)
  )
  fig.write_image(output_path)


def plot_multiple_performance_per_weight(
    data_dfs,
    names,
    output_dir=None,
    perf_metric='val_accuracy',
    varied_parameter_names=None,
):
  """Plots the accuracy against the adversarial/mindiff weight for 1+ models."""

  color_scale_cycle = itertools.cycle(COLOR_SCALE)
  fig = go.Figure()
  i = 0
  for data_df, name in zip(data_dfs, names):
    if varied_parameter_names is not None:
      fairness_weight_name = varied_parameter_names[i]
    else:
      fairness_weight_name = 'mindiff_weight'
    plot_performance_per_weight(
        data_df=data_df,
        output_dir=output_dir,
        perf_metric=perf_metric,
        fairness_weight_name=fairness_weight_name,
        only_line=True,
        label=name,
        color=next(color_scale_cycle),
        fig=fig,
    )
    i += 1
  fig.update_xaxes(
      type='log',
      title='Fairness head weight',
  )
  fig.update_yaxes(title='Overall {}'.format(perf_metric))

  if output_dir is None:
    return fig

  output_path = os.path.join(
      output_dir, '{}_per_weight.png'.format(perf_metric)
  )
  fig.write_image(output_path)


def plot_fairness_per_weight(
    data_df,
    output_dir=None,
    erm_stats=None,
    fairness_metric='val_fpr_gap',
    fairness_weight_name='mindiff_weight',
    only_line=False,
    label=None,
    color=None,
    use_sem=True,
    fig=None,
):
  """Plots the fairness metric against the weight of the head."""
  groups = data_df.groupby(fairness_weight_name)
  groups = groups.agg(['mean', 'std', 'sem']).reset_index()
  sorted_group = groups.sort_values([fairness_weight_name])

  if color is None:
    color = COLOR_SCALE[0]

  if fig is None:
    fig = go.Figure()

  if use_sem:
    err = 'sem'
  else:
    err = 'std'

  fig.add_trace(
      go.Scatter(
          x=sorted_group[fairness_weight_name],
          y=sorted_group[fairness_metric]['mean'],
          name=f'{label}',
          legendgroup=f'{label}',
          line_color=color,
      )
  )
  y_upper = list(
      sorted_group[fairness_metric]['mean'] + sorted_group[fairness_metric][err]
  )
  y_lower = list(
      sorted_group[fairness_metric]['mean'] - sorted_group[fairness_metric][err]
  )
  fig.add_trace(
      go.Scatter(
          x=list(sorted_group[fairness_weight_name])
          + list(sorted_group[fairness_weight_name])[::-1],
          y=y_upper + y_lower[::-1],  # upper, then lower reversed
          legendgroup=f'{label}',
          fill='toself',
          fillcolor=color,
          opacity=0.25,
          line=dict(color='rgba(255,255,255,0)'),
          hoverinfo='skip',
          showlegend=False,
      )
  )

  if erm_stats is not None:
    fig.add_hline(
        y=erm_stats[fairness_metric]['mean'],
        line_color='darkgray',
        line_dash='.',
    )
  if only_line:
    return
  xlist = list(sorted_group[fairness_weight_name])
  fig.update_xaxes(
      range=[np.log(xlist[0] + 1e-10), np.log(xlist[-1])],
      type='log',
      title='Fairness head weight',
  )
  fig.update_yaxes(title=fairness_metric)

  if output_dir is None:
    return fig

  output_path = os.path.join(
      output_dir, '{}_per_weight.png'.format(fairness_metric.replace(' ', '_'))
  )
  fig.write_image(output_path)


def plot_multiple_fairness_per_weight(
    data_dfs,
    names,
    output_dir=None,
    fairness_metric='val_fpr_gap',
    varied_parameter_names=None,
):
  """Plots the fairness against the adversarial/mindiff weight for 1+ models."""
  color_scale_cycle = itertools.cycle(COLOR_SCALE)

  fig = go.Figure()
  i = 0
  for data_df, name in zip(data_dfs, names):
    if varied_parameter_names is not None:
      fairness_weight_name = varied_parameter_names[i]
    else:
      fairness_weight_name = 'mindiff_weight'
    plot_fairness_per_weight(
        data_df=data_df,
        output_dir=output_dir,
        fairness_metric=fairness_metric,
        fairness_weight_name=fairness_weight_name,
        only_line=True,
        label=name,
        color=next(color_scale_cycle),
        fig=fig,
    )
    i += 1
  fig.update_xaxes(
      type='log',
      title='Fairness head weight',
  )
  fig.update_yaxes(title=fairness_metric)

  if output_dir is None:
    return fig

  output_path = os.path.join(
      output_dir, '{}_per_weight.png'.format(fairness_metric.replace(' ', '_'))
  )
  fig.write_image(output_path)


def plot_fairness_per_performance(
    data_df,
    output_dir=None,
    fairness_metric='val_fpr_gap',
    fairness_weight_name='mindiff_weight',
    perf_metric='val_accuracy',
    only_line=False,
    label=None,
    color=None,
    use_sem=True,
    showlegend=True,
    show_median_and_percentiles=False,
    fig=None,
    row=None,
    col=None,
):
  """Plots the perf metric against the fairness one."""

  def percentile(n):
    def percentile_(x):
      return x.quantile(n)

    percentile_.__name__ = 'percentile_{:02.0f}'.format(n * 100)
    return percentile_

  groups = data_df.groupby(fairness_weight_name)
  groups = groups.agg(
      ['mean', 'std', 'sem', 'median', percentile(0.1), percentile(0.9)]
  ).reset_index()

  if color is None:
    color = COLOR_SCALE[0]

  if fig is None:
    fig = go.Figure()
    row, col = None, None

  center = 'median' if show_median_and_percentiles else 'mean'

  sorted_group = groups.sort_values([(fairness_metric, 'mean')])
  fig.add_trace(
      go.Scatter(
          x=sorted_group[fairness_metric][center],
          y=sorted_group[perf_metric][center],
          mode='markers',
          name=f'{label}',
          legendgroup=f'{label}',
          marker_color=color,
          showlegend=showlegend,
      ),
      row=row,
      col=col,
  )
  if show_median_and_percentiles:
    error_y_dict, error_x_dict = None, None
  else:
    if use_sem:
      err = 'sem'
    else:
      err = 'std'
    error_y_dict = dict(
        type='data',
        array=sorted_group[perf_metric][err],
        color=color,
    )
    error_x_dict = dict(
        type='data',
        array=sorted_group[fairness_metric][err],
        color=color,
    )
  fig.add_trace(
      go.Scatter(
          x=sorted_group[fairness_metric][center],
          y=sorted_group[perf_metric]['mean'],
          text=sorted_group[fairness_weight_name],
          mode='markers',
          name=f'{label}',
          legendgroup=f'{label}',
          showlegend=False,
          error_y=error_y_dict,
          error_x=error_x_dict,
          marker=dict(color=color, size=8),
      ),
      row=row,
      col=col,
  )
  if only_line:
    return
  xlist = list(sorted_group[fairness_weight_name])
  fig.update_xaxes(
      range=[xlist[0], xlist[-1]],
      title=fairness_metric,
  )
  fig.update_yaxes(title=perf_metric)

  if output_dir is None:
    return fig

  filename = '{}_per_{}.png'.format(
      perf_metric, fairness_metric.replace(' ', '_')
  )
  output_path = os.path.join(output_dir, filename)
  fig.write_image(output_path)


def plot_multiple_fairness_per_performance(
    data_dfs,
    names,
    varied_parameter_names=None,
    output_dir=None,
    fairness_metric='val_fpr_gap',
    perf_metric='val_accuracy',
    fig=None,
    row=None,
    col=None,
    color_scale_cycle=None,
    **kwargs,
):
  """Plots the fairness against the performance for 1+ models."""
  if color_scale_cycle is None:
    color_scale_cycle = itertools.cycle(COLOR_SCALE)

  if fig is None:
    fig = go.Figure()
    row, col = None, None

  i = 0
  for data_df, name in zip(data_dfs, names):
    if varied_parameter_names is not None:
      fairness_weight_name = varied_parameter_names[i]
    else:
      fairness_weight_name = 'mindiff_weight'
    plot_fairness_per_performance(
        data_df,
        output_dir=output_dir,
        fairness_metric=fairness_metric,
        perf_metric=perf_metric,
        fairness_weight_name=fairness_weight_name,
        only_line=True,
        label=name,
        color=next(color_scale_cycle),
        fig=fig,
        row=row,
        col=col,
        **kwargs,
    )
    i += 1
  fig.update_xaxes(title=fairness_metric)
  fig.update_yaxes(title=perf_metric)

  if output_dir is None:
    return fig

  filename = '{}_per_{}.png'.format(
      perf_metric, fairness_metric.replace(' ', '_')
  )
  output_path = os.path.join(output_dir, filename)
  fig.write_image(output_path)


def symlog(arr, base=10, linthresh=2, linscale=1):
  """Symmetric logarithmic transformation."""

  linscale_adj = linscale / (1.0 - base**-1)
  log_base = np.log(base)
  arr = np.array(arr)
  abs_arr = np.abs(arr)

  # Arrays with all elements within the linear region of this symlog
  # transformation.
  linear = np.max(abs_arr, axis=0) < linthresh

  with np.errstate(divide='ignore', invalid='ignore'):
    out = (
        np.sign(arr)
        * linthresh
        * (linscale_adj + np.log(abs_arr / linthresh) / log_base)
    )
    inside = abs_arr <= linthresh

  out[inside] = arr[inside] * linscale_adj

  out = (
      out
      / np.max(np.abs(out), axis=0)
      * np.log(np.max(abs_arr, axis=0))
      / log_base
  )

  out[np.array([linear] * out.shape[0])] = arr[
      np.array([linear] * out.shape[0])
  ]

  return out
