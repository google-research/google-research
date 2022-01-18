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

"""Utility functions for plotting raw log files.

The only atypical function is plot_aggregate_xaxis that aggregates runs over
multiple runs.
"""
import collections
import math
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_run_names(log_dir, pattern='.*'):
  """Returns a list of run names with existing joblib log file.

  Args:
    log_dir: A string for the directory name of the runs.
    pattern: A regexp string to optionally filter runs.

  Returns:
    A list of run names that is the directory names matching pattern.
  """
  run_names = []
  for root, _, _ in os.walk(log_dir):
    log_exists = os.path.exists(os.path.join(root, 'log.jb'))
    if re.match(pattern, root) and log_exists:
      run_names += [root]
  run_names.sort()
  return run_names


def read_joblib_log(run_name):
  path = os.path.join(run_name, 'log.jb')
  with open(path, 'rb') as file:
    logdata = joblib.load(file)
  return logdata


def read_joblib_logs(run_names, summary_tag):
  """Returns the data of a summary tag for multiple runs.

  Args:
    run_names: A list of strings with the names of runs.
    summary_tag: A string or a list of strings with the summary tag to be
      extracted from data logs.

  Returns:
    A tuple of list of log data and configurations.
  """
  data = []
  config = []
  # TODO(fartash): multi-threaded read using ThreadPoolExecutor
  logdata = []
  for run_name in run_names:
    logdata += [read_joblib_log(run_name)]
  for i in range(len(logdata)):
    if not isinstance(summary_tag, list):
      d = None
      if summary_tag in logdata[i]:
        d = np.array(logdata[i][summary_tag]).T
    else:
      d = []
      for tag in summary_tag:
        if tag in logdata[i]:
          d += [np.array(logdata[i][tag][-1][1])]  # drop time, keep data
      d = np.vstack(d)
    data += [d]
    config += [logdata[i]['config']]
  return data, config


def plot_one_summary_tag(data,
                         config,
                         plot_texts,
                         legend_flags,
                         log_yscale=False,
                         ylim=None,
                         ncolor=None,
                         lg_replace='',
                         fig_dir='./',
                         plot_f=plt.plot,
                         noline=False,
                         filter_f=None,
                         log_xscale=False,
                         alpha=1):
  """Plots and saves a single figure for a single tag.

  Args:
    data: A list of data for individual plots, each with tuple of x and y data.
    config: A list of configuration dictionaries.
    plot_texts: A dictionary of texts to be displayed on the plot with keys:
      xlabel, ylabel, filename, title.
    legend_flags: A list of strings as the flags to be read from config and
      displayed in the legend.
    log_yscale: A boolean for displaying y-scale in log-scale.
    ylim: A tuple of two floats for the range of y-axis.
    ncolor: An integer for the number of colors to use. If there are more plots,
      different line-styles will be used.
    lg_replace: A list of tuples of two strings for replacing flags and values
      in the legend with human-readable text. Regular expression can be used.
    fig_dir: The director address for saving the figure.
    plot_f: The function to use for plotting, e.g. plt.plot.
    noline: A boolean specifying whether to plot lines connecting points.
    filter_f: A function f(config) that returns True for excluding runs.
    log_xscale: A boolean for displaying x-scale in log-scale.
    alpha: A float in [0, 1] that controls the transparency.
  """

  plt.rcParams.update({'font.size': 16})
  plt.figure(figsize=(7, 4))
  plt.tight_layout(pad=1., w_pad=3., h_pad=3.0)
  try:
    os.makedirs(fig_dir)
  except os.error:
    pass

  # Color-blind friendly palette and line styles
  color = sns.color_palette('bright', 10)
  if isinstance(ncolor, list):
    color = [color[c] for c in ncolor]
    ncolor = len(ncolor)
  color = color[:ncolor]
  # Semi-transparent
  for i, c in enumerate(color):
    color[i] = tuple(list(c) + [alpha])
  style = ['-', '--', ':', '-.'] if not noline else ['None'] * 10
  marker = ['s', 'o', 'X', 'p', '*', 'D', 'P', 'v', '^', '<', '>']

  # Filter out data
  if filter_f:
    data, config = zip(
        *[(d, c) for d, c in zip(data, config) if not filter_f(c)])

  legends = [
      get_legend(legend_flags, config[i], lg_replace)
      for i in range(len(config))
  ]
  srt_ord = np.argsort(legends)
  data, config, legends = zip(*[(data[i], config[i], legends[i])
                                for i in srt_ord])

  # Plot one curve per data entry
  plt.grid(linewidth=1)
  for i in range(len(data)):
    if data[i] is None:
      continue
    plot_kwargs = {
        'linestyle': style[i // len(color)],
        'color': color[i % len(color)],
        'linewidth': 2,
        'marker': marker[i % len(color)],
        'markersize': 15,
    }
    plot_f(data[i][0], data[i][1], **plot_kwargs)

  # Y-scale is set to log-scale for the preset list of plots in yscale_log
  ax = plt.gca()
  if log_yscale:
    ax.set_yscale('log')
  else:
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
  if log_xscale:
    ax.set_xscale('log')
  else:
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))

  if ylim is not None:
    plt.ylim(ylim)

  plt.xlabel(plot_texts['xlabel'])
  plt.ylabel(plot_texts['ylabel'])

  os.makedirs(fig_dir, exist_ok=True)

  # Save without legends and title
  for ext in ['pdf', 'png']:
    path = '{}/{}_nolegend_notitle.{}'.format(fig_dir, plot_texts['filename'],
                                              ext)
    with open(path, 'wb') as file:
      plt.savefig(file, format=ext, dpi=100, bbox_inches='tight')

  # Save with legends and title
  plt.title(plot_texts['title'])
  plt.legend(legends, bbox_to_anchor=(1, 1), loc='upper left')
  for ext in ['pdf', 'png']:
    path = '{}/{}.{}'.format(fig_dir, plot_texts['filename'], ext)
    with open(path, 'wb') as file:
      plt.savefig(file, format=ext, dpi=100, bbox_inches='tight')


def get_dotted_flag_value(config, flag):
  """Returns the value of a dotted tag name in a dictionary."""
  value = config
  for flag_i in flag.split('.'):
    if flag_i not in value:
      return None
    value = value[flag_i]
  return value


def copy_dotted_flags(config, keep_flags):
  """Returns a new dictionary with only the flags specified in a dotted format."""
  new_config = {}
  for flag in keep_flags:
    value = get_dotted_flag_value(config, flag)
    if value is None:
      continue
    new_value = new_config
    flag_i = flag
    for flag_i in flag.split('.'):
      if flag_i not in new_value:
        new_value[flag_i] = {}
      new_parent = new_value
      new_value = new_value[flag_i]
    new_parent[flag_i] = value
  return new_config


def ignore_dotted_flags(config, ignore_flags):
  """Returns a new dictionary without ignored flags given in dotted format."""
  new_config = dict(config)
  for flag in ignore_flags:
    value = get_dotted_flag_value(new_config, flag)
    if value is None:
      continue
    new_value = new_config
    flag_i = flag
    for flag_i in flag.split('.'):
      new_parent = new_value
      new_value = new_value[flag_i]
    new_parent[flag.split('.')[-1]] = None
  return new_config


def get_legend(legend_flags, config, lg_replace=''):
  """Returns human-readable legends for all runs from selected config flags.

  Args:
    legend_flags: List of flags to appear in the legend.
    config: Dictionary of flags.
    lg_replace: A list of string pairs to improve readability of legends.

  Returns:
    A string as the legend.
  """
  lg = ''
  for flag in legend_flags:
    value = get_dotted_flag_value(config, flag)
    if value is not None:
      lg += '{}={},'.format(flag, value)
  lg = lg.strip(',')
  for a, b in lg_replace:
    lg = re.sub(a, b, lg)
  return lg


def summarize_data_of_tag(data_of_tag, sum_type):
  """Summarize logged values during training into a single value."""

  if sum_type.startswith('max_eps_at'):
    data_of_tag = np.array(data_of_tag[1][-1])
    min_risk = float(sum_type[sum_type.rfind('_') + 1:])
    return max(data_of_tag[0][data_of_tag[1] <= min_risk], default=0)

  if '_' in sum_type:
    # For example process mean_70 as mean of last 70% of data
    start = int(
        math.ceil(data_of_tag.shape[1] *
                  (1 - float(sum_type[sum_type.rfind('_') + 1:])) / 100))
    data_of_tag = data_of_tag[:, start:]
    sum_type = sum_type[:sum_type.rfind('_')]

  if sum_type == 'min':
    return data_of_tag[1].min()
  if sum_type == 'max':
    return data_of_tag[1].max()
  if sum_type == 'end':
    return data_of_tag[1, -1]
  if sum_type == 'mean':
    return data_of_tag.mean(1)
  if sum_type == 'var':
    return data_of_tag[1].var()
  raise Exception('Invalid summary type: {}'.format(sum_type))


def get_one_value_from_multiple_flags(config, select_flags):
  """Returns a single float to be used as the x-axis."""
  if len(select_flags) == 1:
    # Return only the value of the selected flag
    return get_dotted_flag_value(config, select_flags[0])
  if len(select_flags) == 2:
    # If two flags are to be used, return their ratio, e.g. dim/num_train
    return float(get_dotted_flag_value(config, select_flags[0])) \
        / float(get_dotted_flag_value(config, select_flags[1]))
  raise Exception('Invalid list of multiple flags')


def aggregate_data(data, config, x_flags, legend_flags, sum_type):
  """Aggregates the log of multiple runs over a set of flags.

  Args:
    data: List of dictionaries containing log data.
    config: List of configurations matching each run in data.
    x_flags: A list of configuration flags (at most two) on the x-axis. Given
      two flags, the x-axis is the ratio of x_flags[0]/x_flags[1].
    legend_flags: A list of configuration flags that show up in the legend.
    sum_type: A string that specifies how logged values during a single training
      should be summarized into one value.

  Returns:
    Tuple of aggegated data and config to be passed to plot_one_summary_tag.
  """
  # Aggregate runs over any flags not given as legend_flags and x_flags
  agg_data = collections.OrderedDict()
  agg_config = collections.OrderedDict()
  for i in range(len(data)):
    legend_config = copy_dotted_flags(config[i], legend_flags)
    legend = get_legend(legend_flags, legend_config, '')
    if sum_type == 'full_last':  # used to plot risk/adv vs eps_tot
      run_data = data[i][1][-1]
    elif sum_type == 'last':  # used for margin vs risk
      run_data = data[i][:, 0]
    else:
      run_data = (get_one_value_from_multiple_flags(config[i], x_flags),
                  summarize_data_of_tag(data[i], sum_type))
    agg_data[legend] = agg_data.get(legend, []) + [run_data]
    agg_config[legend] = legend_config

  if sum_type == 'full_last':
    agg_data = [np.array(v).mean(0) for v in agg_data.values()]
  else:
    agg_data = [np.array(v).T for v in agg_data.values()]
  agg_config = list(agg_config.values())

  return agg_data, agg_config


def find_optimal_parameters(run_names,
                            metric_tag,
                            tune_flags,
                            sum_type,
                            max_is_optimal=False):
  """Returns the optimal value of a parameter that achieves lowest metric_tag."""
  ignore_flags = ['log_dir']
  assert isinstance(tune_flags, list), 'Tune flags should be a list.'
  data, config = read_joblib_logs(run_names, metric_tag)
  metric_values = collections.OrderedDict()
  for i in range(len(data)):
    if data[i] is None:
      continue
    config_new = dict(config[i])
    config_new = ignore_dotted_flags(config_new, tune_flags + ignore_flags)
    key = str(config_new)
    metric_values[key] = metric_values.get(
        key, []) + [(i, summarize_data_of_tag(data[i], sum_type))]

  optimal_run_names = []
  for key, value_pairs in metric_values.items():
    ids, values = zip(*value_pairs)
    if max_is_optimal:
      best_id = ids[np.nanargmax(values)]
    else:
      best_id = ids[np.nanargmin(values)]
    optimal_run_names += [run_names[best_id]]
  return optimal_run_names
