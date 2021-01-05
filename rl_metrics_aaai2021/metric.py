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

"""Base Metric class."""

from __future__ import print_function
import abc
import collections
import os
import pickle

from absl import logging
import gin
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf


Statistics = collections.namedtuple(
    'statistics',
    ['tolerance',
     'time',
     'num_iterations',
     'metric_differences'])


@gin.configurable
class Metric(object):
  """Interface specification for metrics."""

  def __init__(self, name, label, env, base_dir, gamma=0.9, normalize=False):
    self.name = name
    self.label = label  # LaTeX label used for plotting.
    self.metric = None
    self.statistics = None
    self.env = env
    self.num_states = env.num_states
    self.num_actions = env.num_actions
    self.base_dir = os.path.join(base_dir, name)
    if not tf.io.gfile.exists(self.base_dir):
      tf.io.gfile.makedirs(self.base_dir)
    self.gamma = gamma
    self.normalize = normalize
    self.cmap = cm.get_cmap('plasma', 256)

  @abc.abstractmethod
  def _compute(self, tolerance, verbose=False):
    pass

  def compute(self, tolerance=1e-5, save_and_reload=True, verbose=False):
    """Compute the metric.

    Will call self._compute, which should be implemented by each subclass.

    Args:
      tolerance: float, maximum difference in metric estimate between successive
        iterations. Once this threshold is past, computation stops.
      save_and_reload: bool, whether to save and reload the metric.
      verbose: bool, whether to print verbose messages.
    """
    # First try reloading from file.
    if save_and_reload and self._try_reloading_metric():
      return
    self._compute(tolerance, verbose=verbose)
    if self.normalize:
      self.metric /= np.max(self.metric)
    if not save_and_reload:
      return
    # Save the metric to file so we save this step next time.
    path = self._build_metric_path()
    with tf.io.gfile.GFile(path, 'w') as f:
      pickle.dump(self.metric, f)

  def _build_metric_path(self):
    return os.path.join(self.base_dir, '{}_metric.pkl'.format(self.name))

  def _try_reloading_metric(self):
    path = self._build_metric_path()
    if tf.io.gfile.exists(path):
      with tf.io.gfile.GFile(path, 'rb') as f:
        self.metric = pickle.load(f)
      logging.info('Successfully reloaded %s metric from file', self.name)
      return True
    return False

  def maybe_pretty_print_and_compute_gaps(self):
    """Maybe print out a nice grid version of metric and compute the gaps.

    Returns:
      Average, maximum, and minimum gaps between d(s, t) and |V*(s) - V*(t)|.
    """
    if self.metric is None:
      logging.warning('%s metrics have not been computed yet.', self.name)
      return
    base_pdf_file = os.path.join(self.base_dir, '{}_metric'.format(self.name))
    min_nonzero_distance = np.min(
        np.where(self.metric == 0., np.inf, self.metric))
    max_distance = np.max(self.metric)
    min_gap = 0.
    avg_gap = 0.
    max_gap = 0.
    for s in range(self.num_states):
      # Plot the raw distances.
      filename = base_pdf_file + '_' + str(s) + '.pdf'
      s_distances = self.metric[s, :]
      norm = colors.Normalize(vmin=min_nonzero_distance,
                              vmax=max_distance)
      obs_image = self.env.render_custom_observation(
          self.env.reset(), s_distances, self.cmap,
          boundary_values=[min_nonzero_distance, max_distance])
      if obs_image is not None:
        m = cm.ScalarMappable(cmap=self.cmap, norm=norm)
        m.set_array(obs_image)
        plt.imshow(obs_image)
        plt.colorbar(m)
        plt.savefig(filename, format='pdf')
        plt.clf()
      # Plot the gap between metric distance and optimal value difference.
      s_values = np.ones(self.env.num_states) * self.env.values[s]
      diff = self.metric[s, :] - abs(s_values - self.env.values)
      min_gap = min(min_gap, np.min(diff))
      max_gap = max(max_gap, np.max(diff))
      avg_gap += np.sum(diff)
      min_val = np.min(np.where(diff == 0., np.inf, diff))
      max_val = np.max(diff)
      filename = base_pdf_file + '_gap_' + str(s) + '.pdf'
      norm = colors.Normalize(vmin=min_val, vmax=max_val)
      if min_val == max_val:
        # To avoid numerical issues.
        max_val *= 2
      obs_image = self.env.render_custom_observation(
          self.env.reset(), s_distances, self.cmap,
          boundary_values=[min_val, max_val])
      if obs_image is not None:
        m = cm.ScalarMappable(cmap=self.cmap, norm=norm)
        m.set_array(obs_image)
        plt.imshow(obs_image)
        plt.colorbar(m)
        plt.savefig(filename, format='pdf')
        plt.clf()
    avg_gap /= self.num_states
    return min_gap, avg_gap, max_gap

  def print_statistics(self):
    if self.statistics is None:
      return
    logging.info('**** %s statistics ***', self.name)
    logging.info('Number of states: %d', self.num_states)
    logging.info('Total number of iterations: %d',
                 self.statistics.num_iterations)
    logging.info('Total time: %f', self.statistics.time)
    logging.info('*************************')
