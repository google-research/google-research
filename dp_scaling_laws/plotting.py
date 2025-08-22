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

"""Setup plotting environment and helper plotting functions."""

from collections.abc import Sequence

import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def average_color(hex1, hex2):
  r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
  r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)
  r, g, b = (r1 + r2) // 2, (g1 + g2) // 2, (b1 + b2) // 2
  return '#' + ''.join('{:02X}'.format(a) for a in [r, g, b])


def light_color(hex):
  r, g, b = int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)
  r, g, b = (255 + r) // 2, (255 + g) // 2, (255 + b) // 2
  return '#' + ''.join('{:02X}'.format(a) for a in [r, g, b])


GOLDEN_RATIO = 1.618033988749


# google colors
BLUE, RED, GREEN, YELLOW = '#4285F4', '#EA4335', '#34A853', '#FBBC04'
PURPLE = average_color(BLUE, RED)
ORANGE = average_color(RED, YELLOW)
PINK = '#FF69B4'
BROWN = '#B87333'
GRAY = '#9E9E9E'
BLACK = '#000000'

COLORS = (BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE, PINK, BROWN, GRAY)

LIGHT_GRAY = '#EDEDED'

SMALLER_SIZE = 17
SMALL_SIZE = 19
MEDIUM_SIZE = 22
BIGGER_SIZE = 26

MARKER_SIZE = 8
LINEWIDTH = 1.5
MAJOR_TICK_SIZE = 6
MINOR_TICK_SIZE = 3
MAJOR_TICK_PAD = 2

RC_PARAMS = {
    'axes.linewidth': LINEWIDTH,
    'xtick.major.size': MAJOR_TICK_SIZE,
    'xtick.minor.size': MINOR_TICK_SIZE,
    'xtick.major.width': LINEWIDTH,
    'xtick.minor.width': LINEWIDTH,
    'xtick.major.pad': MAJOR_TICK_PAD,
    'xtick.direction': 'inout',
    'xtick.labelsize': SMALL_SIZE,  # font size of the tick labels
    'ytick.major.size': MAJOR_TICK_SIZE,
    'ytick.minor.size': MINOR_TICK_SIZE,
    'ytick.major.width': LINEWIDTH,
    'ytick.minor.width': LINEWIDTH,
    'ytick.major.pad': MAJOR_TICK_PAD,
    'ytick.direction': 'inout',
    'ytick.labelsize': SMALL_SIZE,  # font size of the tick labels
    'errorbar.capsize': 0,
    'axes.labelpad': 1.5,
    'axes.titlesize': MEDIUM_SIZE,  # font size of the axes title
    'axes.labelsize': MEDIUM_SIZE,  # font size of the x and y labels
    # 'font.family': 'sans-serif',
    # 'font.sans-serif': 'Roboto',
    'font.weight': 'normal',
    'font.size': SMALL_SIZE,  # controls default text sizes
    'grid.color': LIGHT_GRAY,
    'grid.linewidth': LINEWIDTH,
    'hatch.linewidth': LINEWIDTH - 0.3,
    'legend.frameon': False,  # don't show a box around the legend
    'legend.fontsize': SMALLER_SIZE,  # legend font size
    'legend.labelspacing': 0.35,
    'legend.handlelength': 0,
    'legend.handletextpad': 1,
    'figure.titlesize': BIGGER_SIZE,  # font size of the figure title
}

matplotlib.rcParams.update(RC_PARAMS)

# Define nice symbols

# Remove the line connecting symbols. If set to a finite number,
# you get a dashed line in between symbols.
SYMBOL_LINEWIDTH = 0.0

STANDARD_PARAMS = {
    'lw': SYMBOL_LINEWIDTH,
    'linestyle': '--',
    'dashes': (None, None),
    'dash_capstyle': 'round',
    'markersize': MARKER_SIZE,
    'markeredgewidth': LINEWIDTH,
}

SYMBOL_STYLES = {
    'round': {'marker': 'o', **STANDARD_PARAMS},
    'square': {'marker': 's', **STANDARD_PARAMS},
    'triangle': {
        'marker': '^',
        **STANDARD_PARAMS,
        **{'markersize': MARKER_SIZE + 2},
    },
    'diamond': {'marker': 'D', **STANDARD_PARAMS},
    'star': {
        'marker': '*',
        **STANDARD_PARAMS,
        **{'markersize': MARKER_SIZE * 1.8},
    },
    'plus': {'marker': 'P', **STANDARD_PARAMS},
    'cross': {'marker': 'X', **STANDARD_PARAMS},
    'hex': {'marker': 'H', **STANDARD_PARAMS},
    'spade': {'marker': 'd', **STANDARD_PARAMS},
    'pentagon': {'marker': 'p', **STANDARD_PARAMS},
    None: {'marker': None, **STANDARD_PARAMS},
}

SYMBOLS = tuple(SYMBOL_STYLES.keys())

STYLES = tuple(zip(COLORS, SYMBOL_STYLES.keys()))
LINESTYLES = ('-', '--', ':', '-.', '.')


def set_minor_ticks(ax, num_ticks_x=0, num_ticks_y=0):
  """Sets minor ticks for an axis frame.

  Caution: This can malfunction on logarithmic scales.

  Args:
    ax: The axis frame.
    num_ticks_x: An integer setting the number of minor ticks per major tick
      division on the x-axis.
    num_ticks_y: An integer setting the number of minor ticks per major tick
      division on the y-axis.
  """
  if num_ticks_x > 0:
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(num_ticks_x + 1))
  if num_ticks_y > 0:
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(num_ticks_y + 1))


def make_round_ticks(ax):
  """Changes capstyle of all tick marks to round.

  Args:
    ax: The axis frame.
  """
  for axis in [ax.xaxis, ax.yaxis]:
    for i in axis.get_majorticklines():
      i._marker._capstyle = 'round'
    for i in axis.get_minorticklines():
      i._marker._capstyle = 'round'


def lineplot(
    df,
    std = None,
    colors = COLORS,
    symbols = SYMBOLS,
    linestyles = ('-',) * 25,
    logx=False,
    logy=False,
    ax = None,
    fill_between = False,
    ncol = 1,
    legend = True,
):
  """Creates a production-quality lineplot from a dataframe.

  Args:
    df: A dataframe with columns to be plotted. The index will be used as the
      x-axis, and each column will be plotted as a separate line.
    std: Optional, a dataframe with the same index and columns as df, where each
      entry is the standard error of the corresponding entry in df. With careful
      indexing, can also pass in a dataframe with 2-level column index to give
      lower and upper bounds for the error bars.
    colors: A list of colors to use for each column.
    symbols: A list of symbols to use for each column.
    linestyles: A list of linestyles to use for each column.
    logx: Whether to use a log scale on the x-axis.
    logy: Whether to use a log scale on the y-axis.
    ax: The axis to use for plotting. If None, a new figure and axis will be
      created.
    fill_between: Whether to fill the between the error bars.
    ncol: The number of columns to use for the legend.
    legend: Whether to show the legend.

  Returns:
    An axis object for a production-quality line-plot.
  """
  if ax is None:
    fig = plt.figure(figsize=(10, 10 / GOLDEN_RATIO))
    ax = fig.add_subplot(111)

  if symbols is None:
    symbols = [None] * len(df.columns)

  xs = df.index
  for col, color, symbol, linestyle in zip(
      df.columns, colors, symbols, linestyles
  ):
    ys = df[col].values
    lw = LINEWIDTH if symbol is not None else 2 * LINEWIDTH
    label = col if symbol is None else None
    ax.plot(xs, ys, linestyle, color=color, lw=lw, label=label)
    if symbol is not None:
      ax.errorbar(
          xs,
          ys,
          yerr=std[col].values
          if std is not None and not fill_between
          else None,
          markeredgecolor=color,
          markerfacecolor=light_color(color),
          label=col,
          elinewidth=LINEWIDTH,
          ecolor=color,
          **SYMBOL_STYLES[symbol],
      )
    if fill_between and std is not None:
      errors = std[col].values
      ax.fill_between(
          xs,
          errors[:, 0],
          errors[:, 1],
          color=color,
          alpha=0.1,
      )

  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')

  # Tick labels shown as floating point numbers often have trailing zeros.
  # Get rid of them.
  ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

  set_minor_ticks(ax, num_ticks_x=1, num_ticks_y=1)
  for axis in [ax.xaxis, ax.yaxis]:
    axis.grid(True, which='major', lw=LINEWIDTH - 0.2, color=LIGHT_GRAY)
    axis.grid(True, which='minor', lw=LINEWIDTH - 0.6, color=LIGHT_GRAY)

  if legend:
    ax.legend(
        title=df.columns.name,
        frameon=True,
        handlelength=1,
        handletextpad=0.5,
        borderpad=0.5,
        fontsize='large',
        ncol=ncol,
    )

  return ax


def plot_vector_field(table):
  """Creates a vector field plot from a 2-dimensional table.

  Given a 2D table, this plot visualizes the relative change in the table
  entries for adjacent rows and columns.  The columns and rows of the table
  should have semantic menaing, and be indexed by values that are evenly spaced
  on a log scale.  This plot is designed to show multiplicative changes in the
  values of the table.

  Args:
    table: A 2-dimensional pandas DataFrame.
  """
  increase_x = (-table.apply(np.log).diff(axis=1)).apply(np.exp) - 1
  increase_y = (-table.apply(np.log).diff(axis=0)).apply(np.exp) - 1

  increase_x = increase_x.shift(-1, axis=0).shift(-1, axis=1).iloc[:-1, :-1]
  increase_y = increase_y.shift(-1, axis=0).shift(-1, axis=1).iloc[:-1, :-1]

  x, y = np.meshgrid(increase_x.columns, increase_x.index)

  color = np.log((increase_x + 1) / (increase_y + 1))
  color = np.clip(color, -1, 1)

  plt.figure(figsize=(6, 6))
  plt.quiver(
      x,
      y,
      increase_x,
      increase_y,
      color,
      units='dots',
      headwidth=2.5,
      headlength=3.5,
      headaxislength=3,
      cmap='viridis',
  )
  plt.loglog()
  plt.ylabel(table.index.name, fontsize='x-large')
  plt.xlabel(table.columns.name, fontsize='x-large')
