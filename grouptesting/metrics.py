# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Implements some metrics."""

import glob
import os
import os.path

from absl import logging
import jax
import jax.numpy as np
import jax.scipy.special
import numpy as onp
import sklearn.metrics


def auc(y_true, y_pred):
  """Area under curve."""
  sorted_y_true = y_true[np.argsort(y_pred)]
  false_count = np.cumsum(1 - sorted_y_true)
  area = np.sum(sorted_y_true * false_count)
  return area / (false_count[-1] * (len(sorted_y_true) - false_count[-1]))


def kl(x, y):
  """Kullback–Leibler divergence."""
  eps = 1e-15
  return np.sum(x * (np.log(x + eps) - np.log(y + eps) - 1.0)) + np.sum(y)


def binary_entropy(u):
  return jax.scipy.special.entr(u) + jax.scipy.special.entr(1.0 - u)


def entropy(table, axis=0):
  return np.sum(jax.scipy.special.entr(table), axis=axis)


class Metrics:
  """Some metrics to be kept while simulating."""

  def __init__(self,
               workdir=None,
               num_simulations=0,
               num_cycles=0,
               num_patients=0,
               num_tests_per_cycle=0):
    self.workdir = workdir
    self.num_simulations = num_simulations
    self.num_cycles = num_cycles
    self.num_patients = num_patients
    self.num_tests_per_cycle = num_tests_per_cycle
    self.ground_truth = onp.empty((num_simulations, num_patients))
    self.marginals = onp.empty((num_simulations, num_cycles, num_patients))
    self.marginals[:] = onp.NaN
    self.groups = onp.empty(
        (num_simulations, num_cycles, num_tests_per_cycle, num_patients))
    self.groups[:] = onp.NaN
    self.test_results = onp.empty(
        (num_simulations, num_cycles, num_tests_per_cycle))
    self.groups[:] = onp.NaN

  def update(self,
             simulation,
             cycle,
             marginal,
             ground_truth,
             groups=None,
             test_results=None):
    """Updates the values of the metrics for a given simulation and cycle."""
    logging.info('Exporting simulatiom %i at Cycle %i.', simulation, cycle)
    self.ground_truth[simulation] = ground_truth
    self.marginals[simulation, cycle] = marginal
    if groups is not None:
      self.groups[simulation, cycle, :groups.shape[0], :] = groups
      self.test_results[simulation, cycle, :groups.shape[0]] = test_results

  def load(self):
    """Loads metrics from files."""
    filenames = glob.glob(os.path.join(self.workdir, '*.npy'))
    for filename in filenames:
      name = os.path.basename(filename)[:-4]
      with open(filename, 'rb') as fp:
        setattr(self, name, onp.load(fp))

  def export(self):
    """Exports the data to files."""
    if self.workdir is None:
      return

    if not os.path.exists(self.workdir):
      os.makedirs(self.workdir)

    names = ['marginals', 'ground_truth', 'groups', 'test_results']
    arrays = [self.marginals, self.ground_truth, self.groups, self.test_results]
    for arr, name in zip(arrays, names):
      output_file = os.path.join(self.workdir, f'{name}.npy')
      with open(output_file, 'wb') as fp:
        onp.save(fp, arr)

  def _merge_one(self, other, name):
    """Merges one a single metrics from another object with the current one."""
    other_member = getattr(other, name, None)
    if other_member is None or not onp.any(other_member.shape):
      return

    curr = getattr(self, name, None)
    if curr is None or not onp.any(curr.shape):
      setattr(self, name, other_member)
    else:
      try:
        setattr(self, name, onp.concatenate([curr, other_member], axis=0))
      except ValueError as e:
        logging.warning(f'{name}: {e}, {curr.shape}, {other_member.shape}')

  def merge(self, other):
    """Merges the metrics of another object with the current ones."""
    attributes = [k for (k, v) in other.__dict__.items()
                  if isinstance(v, onp.ndarray)]
    for name in attributes:
      self._merge_one(other, name)

  def _extract_data(self, simulation=None, cycle = -1):
    """Returns the ground truth and marginals across all simulations."""
    if simulation is not None:
      gt = self.ground_truth[simulation]
      marginals = self.marginals[simulation, cycle]
    else:
      gt = onp.reshape(self.ground_truth, (-1,))
      cycle_first_marginals = onp.transpose(self.marginals, (1, 0, 2))[cycle]
      marginals = onp.reshape(cycle_first_marginals, (-1,))
    return gt[~onp.isnan(marginals)], marginals[~onp.isnan(marginals)]

  def roc_curve(self, simulation=None, cycle=-1):
    """Returns the ROC curves and its area under the curve.

    Args:
     simulation: the index of the simulation. If None consider them all.
     cycle: the index of the simulation.

    Returns:
     A 3-tuple false positive rate, true positive rate and auc.
    """
    gt_marginals = self._extract_data(simulation, cycle)
    if gt_marginals is None:
      return

    fpr, tpr, _ = sklearn.metrics.roc_curve(*gt_marginals)
    return fpr, tpr, sklearn.metrics.auc(fpr, tpr)

  def precision_recall_curve(self, simulation = None, cycle=-1):
    """Returns the precision recall curves and its area under the curve.

    Args:
     simulation: the index of the simulation. If None consider them all.
     cycle: the index of the simulation.

    Returns:
     A 3-tuple precision, recall and auc.
    """
    gt_marginals = self._extract_data(simulation, cycle)
    if gt_marginals is None:
      return

    precision, recall, _ = sklearn.metrics.precision_recall_curve(*gt_marginals)
    return precision, recall, sklearn.metrics.auc(recall, precision)
