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

"""Simple generator for synthetic multi-task data.

This implements the generation procedure described in "Modeling Task
Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts". Here,
the procedure is slightly generalized to generate multiple pairs of tasks,
where the relatedness between pairs is typically low, and the relatedness
within pairs can be controlled with a hyperparameter.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import scipy.stats


def generate_label(input_features, task_weights, alphas, betas, noise_std):
  """Generate a single label.

  The label is generated according to the definition in "Modeling Task
  Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts".

  Args:
    input_features: (numpy array of shape [`dim`]) input features.
    task_weights: (numpy array of shape [`dim`]) weights that define the task.
    alphas: (sequence of `m` floats) multipliers for the non-linear part
      of the label.
    betas: (sequence of `m` floats) biases for the non-linear part of the label.
    noise_std: (float) standard deviation for the noise added to the label.

  Returns:
    Generated label as a numpy scalar.
  """
  dot = np.dot(task_weights, input_features)
  non_linear_parts = [
      math.sin(alpha * dot + beta) for alpha, beta in zip(alphas, betas)]
  noise = np.random.normal(scale=noise_std)

  return dot + sum(non_linear_parts) + noise


def generate_task_pairs(
    num_task_pairs,
    num_samples,
    dim,
    relatedness,
    c=1.0,
    m=10,
    noise_std=0.01,
    shuffle_task_data=False):
  """Generate data for `num_task_pairs` pairs of tasks.

  Relatedness inside each pair of tasks can be configured, while the relatedness
  of tasks from different pairs will be low.

  Args:
    num_task_pairs: (int) number of pairs of tasks.
    num_samples: (int) number of samples to generate for each of the tasks.
    dim: (int) dimension for the synthetically generated data.
    relatedness: (float) between 0 and 1; the higher the more related the
      pairs of tasks will be.
    c: (float) scale for the linear part of the label.
    m: (int) number of summands in the non-linear part of the label.
    noise_std: (float) standard deviation for the noise added to the label.
    shuffle_task_data: (bool) if set to True, the features and labels for each
      task are randomly and independently reshuffled.

  Returns:
    Features and labels for all pairs of tasks as numpy arrays.

    If `shuffle_task_data` is set to False, then the returned feature
    array is the same for all tasks, so it has shape [`num_samples`, `dim`].
    The shape of the labels array is [2 * `num_task_pairs`, `num_samples`],
    where the first two tasks constitute the first pair, the next two
    the second pair, and so on.

    If `shuffle_task_data` is set to True, then a list of length
    2 * `num_task_pairs` is returned. Each element of the list corresponds to
    a single task, and contains features of shape [`num_samples`, `dim`], and
    labels of shape [`num_samples`].
  """
  assert 0.0 <= relatedness <= 1.0

  num_tasks = 2 * num_task_pairs

  # We need to generate `num_tasks` orthogonal basis vectors
  assert dim >= num_tasks

  # The original paper does not provide the values for these sequences
  alphas = range(1, m + 1)
  betas = [i ** 2 for i in range(m)]

  # Generate a random [`dim`, `dim`] orthogonal matrix, and select `num_tasks`
  # of first rows
  basis_vectors = scipy.stats.ortho_group.rvs(dim)[:num_tasks]
  basis_vectors = np.reshape(basis_vectors, (num_task_pairs, 2, dim))

  xs = np.random.normal(size=(num_samples, dim))
  task_ys = []

  for u1, u2 in basis_vectors:
    w1 = c * u1
    w2 = c * (relatedness * u1 + math.sqrt(1 - relatedness ** 2) * u2)

    for w in [w1, w2]:
      labels = [generate_label(x, w, alphas, betas, noise_std) for x in xs]
      labels = [np.expand_dims(label, axis=0) for label in labels]

      task_ys.append(np.concatenate(labels))

  if shuffle_task_data:
    data = []

    for y in task_ys:
      perm = np.random.permutation(len(xs))
      data.append((xs[perm], y[perm]))

    return data
  else:
    return xs, np.stack(task_ys)
