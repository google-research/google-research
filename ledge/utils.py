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

import tensorflow.compat.v1 as tf

from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 13})
import numpy as np


def graphBatch(lines, filename, labels=None, ylim=None, title=None):
  """Plots several lines or Tensors on several figures.

  Each figure is written to its own png file. Every figure must have
  the same number of lines. It's common to have just one figure.

  Args:
    lines: (list of points) A list of lines to be drawn on the figure.
      Each "points" is itself a list of y-values for that line.
    filename: (str) Output file name.
    labels: (list of str) labels[i] is the legend label for lines[i].
      If omitted, there is no legend.

  Returns: None
  """
  color_list  = ['blue', 'orange', 'purple', 'pink', 'cyan', 'red',
                 'yellow', 'magenta', 'teal']

  x_values = range(len(lines[0]))

  fig, ax = plt.subplots()
  axes = plt.gca()
  axes.set_ylim(ylim)

  for l in range(len(lines)):
    assert len(lines[l]) == len(lines[0])
    ax.plot(x_values, lines[l], label = None if not labels else labels[l],
            color=color_list[l % len(color_list)], marker='', linestyle='-')
    ax.grid()

  plt.title(title)

  if labels:
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=False,
              ncol=3, numpoints=1, fontsize=10)

  ax.axes.xaxis.set_ticklabels([])
  ax.axes.yaxis.set_ticklabels([])
  plt.grid(True)

  with tf.io.gfile.GFile(filename, 'wb') as f:
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
  print('wrote figure', filename)

  plt.close()
