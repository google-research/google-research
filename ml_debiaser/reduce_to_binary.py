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

"""Preprocessing method for debiasing multiclass datasets."""

import copy

import numpy as np

from ml_debiaser import randomized_threshold


class Reduce2Binary:
  """Debiase multiclass datasets via preprocessing (R2B Algorithm)."""

  def __init__(self, gamma=1.0, eps=0, eta=0.5, num_classes=2):
    """Instantiate object.

    Args:
      gamma: The regularization parameter gamma (for randomization). Set this to
        1 if the goal is to minimize changes to the original probability scores.
      eps: Tolerance parameter for bias >= 0.
      eta: The step size parameters in ADMM.
      num_classes: Number of classes (must be >= 2).
    """
    if num_classes < 2:
      raise ValueError('Number of classes (must be >= 2).')

    self.num_groups = 1
    self.gamma = gamma
    self.eps = eps
    self.eta = eta
    self.num_classes = num_classes

    # binary debiasers for each label
    self.debiasers = {}
    for k in range(num_classes):
      self.debiasers[k] = randomized_threshold.RandomizedThreshold(
          gamma=gamma+eta, eps=eps)

  def _compute_z(self, h_mat, u_mat):
    # Compute the Z matrix in the R2B algorithm.
    mult_by_ones = np.matmul(h_mat + u_mat, np.ones(self.num_classes,))
    over_k = 1.0 / self.num_classes*(mult_by_ones - np.ones(mult_by_ones.shape))
    j_mat = np.outer(over_k, np.ones(self.num_classes,))
    return h_mat + u_mat - j_mat

  def fit(self, y_orig, group_feature, sgd_steps, full_gradient_epochs=1_000,
          verbose=True, batch_size=256, max_admm_iter=100):
    """Debias scores w.r.t. the sensitive class in each demographic group.

    In the multiclass setting, we use ADMM to decompose the problem into
    separate debiasing tasks of binary labels before they are aggregated.

    Args:
      y_orig: Original probability scores.
      group_feature: An array containing the group id of each instance starting
        from group 0 to group K-1.
      sgd_steps: Number of minibatch steps in SGD.
      full_gradient_epochs: Number of full gradient descent steps.
      verbose: Set to True to display progress.
      batch_size: Size of minibatches in SGD.
      max_admm_iter: Maximum number of iteration of the ADMM procedure.

    Returns:
      None.
    """
    if len(y_orig.shape) != 2:
      raise ValueError('Original prob scores must be a 2-dimensional array.'
                       'Use RandomizedThreshold for binary classification.')

    y_orig_scores = copy.deepcopy(y_orig)

    # Initialize ADMM.
    f_mat = copy.deepcopy(y_orig_scores)
    h_mat = np.zeros_like(f_mat)
    u_mat = np.zeros_like(f_mat)
    z_mat = np.zeros_like(f_mat)

    for iterate in range(max_admm_iter):
      if verbose:
        print(f'ADMM Iteration {iterate}: ', end='\t   ')

      # Step 1: debias each label separately.
      for k in range(self.num_classes):
        if verbose:
          print('\b\b\b%02d%%'%int(100 * k / self.num_classes), end='')
        self.debiasers[k].fit(f_mat[:, k], group_feature, sgd_steps,
                              full_gradient_epochs,
                              verbose=False, batch_size=batch_size,
                              ignore_warnings=True)
        h_mat[:, k] = self.debiasers[k].predict(f_mat[:, k], group_feature,
                                                ignore_warnings=True)
      if verbose:
        print('\b\b\b100%')

      # Step 2: update ADMM variables.
      old_z = copy.deepcopy(z_mat)
      z_mat = self._compute_z(h_mat, u_mat)
      u_mat = u_mat + h_mat - z_mat
      f_mat = y_orig_scores + self.eta * (z_mat - u_mat)

      # Compute primal and dual residuals.
      s = np.linalg.norm(z_mat - old_z)
      r = np.linalg.norm(z_mat - h_mat)

      if verbose:
        print('primal residual: ', r)
        print('dual residual: ', s, '\n')

    # z_mat may contain small negative vals; e.g. if ADMM runs for a few rounds.
    # Remove negative values and normalize.
    z_mat = np.maximum(z_mat, 0)
    for i in range(z_mat.shape[0]):
      # No need to worry about division by zero here; it won't happen.
      z_mat[i, :] = z_mat[i, :] / sum(z_mat[i, :])
    return z_mat
