# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
"""Evaluation functions for SCAE and constellation models.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from monty.collections import AttrDict
import numpy as np
from scipy.optimize import linear_sum_assignment
import sklearn.cluster


def bipartite_match(pred, gt, n_classes=None, presence=None):
  """Does maximum biprartite matching between `pred` and `gt`."""

  if n_classes is not None:
    n_gt_labels, n_pred_labels = n_classes, n_classes
  else:
    n_gt_labels = np.unique(gt).shape[0]
    n_pred_labels = np.unique(pred).shape[0]

  cost_matrix = np.zeros([n_gt_labels, n_pred_labels], dtype=np.int32)
  for label in range(n_gt_labels):
    label_idx = (gt == label)
    for new_label in range(n_pred_labels):
      errors = np.equal(pred[label_idx], new_label).astype(np.float32)
      if presence is not None:
        errors *= presence[label_idx]

      num_errors = errors.sum()
      cost_matrix[label, new_label] = -num_errors

  row_idx, col_idx = linear_sum_assignment(cost_matrix)
  num_correct = -cost_matrix[row_idx, col_idx].sum()
  acc = float(num_correct) / gt.shape[0]
  return AttrDict(assingment=(row_idx, col_idx), acc=acc,
                  num_correct=num_correct)


def cluster_classify(features, gt_label, n_classes, kmeans=None, max_iter=100):
  """Performs clustering and evaluates it with bipartitate graph matching."""
  if kmeans is None:
    kmeans = sklearn.cluster.KMeans(
        n_clusters=n_classes,
        precompute_distances=True,
        n_jobs=-1,
        max_iter=max_iter,
    )

  kmeans = kmeans.fit(features)
  pred_label = kmeans.predict(features)
  return np.float32(bipartite_match(pred_label, gt_label, n_classes).acc)


def collect_results(sess, tensors, n_batches):
  """Collects `n_batches` of tensors and aggregates the results."""

  res = AttrDict({k: [] for k in tensors})

  print('Collecting: 0/{}'.format(n_batches), end='')
  for i in range(n_batches):
    print('\rCollecting: {}/{}'.format(i + 1, n_batches), end='')

    vals = sess.run(tensors)
    for k, v in vals.items():
      res[k].append(v)

  print('')
  for k, v in res.items():
    if v[0].shape:
      res[k] = np.concatenate(v, 0)
    else:
      res[k] = np.stack(v)

  return res
