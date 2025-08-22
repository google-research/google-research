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

"""Base Metric class."""

from __future__ import print_function
import abc
import os.path as osp
import pickle

from absl import logging
import gin
import numpy as np
import tensorflow as tf


PARAMS_TO_SAVE = ['metric', 'k', 'tolerance', 'time', 'num_iterations',
                  'metric_differences']


@gin.configurable
class Metric(object):
  """Interface specification for metrics."""

  def __init__(self, name, label, env, base_dir, normalize=False,
               run_number=0):
    self.name = name
    self.label = label  # LaTeX label used for plotting.
    self.metric = None
    self.k = None
    self.env = env
    self.tolerance = None
    self.time = None
    self.num_iterations = None
    self.metric_differences = None
    self.num_states = env.num_states
    self.num_actions = env.num_actions
    self.base_dir = osp.join(base_dir, name)
    if not tf.io.gfile.exists(self.base_dir):
      tf.io.gfile.makedirs(self.base_dir)
    self.normalize = normalize
    # This is used to differentiate different counterexamples.
    self.run_number = run_number

  @abc.abstractmethod
  def _compute(self, tolerance, num_samples=None, max_iterations=None,
               verbose=False):
    pass

  def compute(self, tolerance=0.001, save_and_reload=True, verbose=False):
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
    return osp.join(self.base_dir, '{}_metric.pkl'.format(self.name))

  def _try_reloading_metric(self):
    path = self._build_metric_path()
    if tf.io.gfile.exists(path):
      with tf.io.gfile.GFile(path, 'rb') as f:
        self.metric = pickle.load(f)
      logging.info('Successfully reloaded %s metric from file', self.name)
      return True
    return False

  def compute_gap(self):
    r"""Compute the metric gap.

    Returns:
      The average, minimum, and maximum gap between d(x, y) and
      |V^{\pi}(x) - V^{\pi}{(y)|.
    """
    if self.metric is None:
      logging.warning('%s metrics have not been computed yet.', self.name)
      return
    avg_gap = 0.
    min_gap = np.inf
    max_gap = 0.
    for x in range(self.num_states):
      x_values = np.ones(self.env.num_states) * self.env.values[x]
      value_diff = abs(x_values - self.env.values)
      diff = self.metric[x, :] - value_diff
      min_gap = min(min_gap, np.min(diff))
      max_gap = max(max_gap, np.max(diff))
      avg_gap += np.sum(diff)
      min_val = np.min(np.where(diff == 0., np.inf, diff))
      max_val = np.max(diff)
      if min_val == max_val:
        # To avoid numerical issues.
        max_val *= 2
    avg_gap /= self.num_states**2
    if min_gap < -1e-5:
      logging.info('Counterexample found!')
      counterexample_dict = {
          'num_states': self.num_states,
          'num_actions': self.num_actions,
          'P': self.env.transition_probs,
          'R': self.env.rewards,
          'V': self.env.values,
          'd': self.metric,
      }
      if self.k is not None:
        counterexample_dict['k'] = self.k
      path = osp.join(self.base_dir,
                      f'{self.name}_counterexample_{self.run_number}.pkl')
      with tf.io.gfile.GFile(path, 'w') as f:
        pickle.dump(counterexample_dict, f)
    return avg_gap, min_gap, max_gap

  def print_statistics(self):
    logging.info('**** %s statistics ***', self.name)
    logging.info('Number of states: %d', self.num_states)
    logging.info('Total number of iterations: %d',
                 self.num_iterations)
    logging.info('Total time: %f', self.time)
    logging.info('*************************')

  def gram_matrix(self):
    return np.exp(-1 * self.metric)

  def bundle(self):
    metric_params = {}
    for key in PARAMS_TO_SAVE:
      metric_params[key] = self.__dict__[key]
    return metric_params

  def unbundle(self, policy_params):
    for key in PARAMS_TO_SAVE:
      assert key in policy_params
      self.__dict__[key] = policy_params[key]
