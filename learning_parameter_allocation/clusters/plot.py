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

"""Code that plots the allocation pattern."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

flags.DEFINE_string('dir', None, 'Path to the summaries file.')


def load_summaries(path):
  """Loads summaries under a given directory.

  Args:
    path: (string) path to summaries.

  Returns:
    List of summary tuples (`step`, `name`, `value`).
  """
  summaries = []

  for obj in tf.train.summary_iterator(path):
    if hasattr(obj.__class__, 'summary'):
      values = obj.summary.value

      if values:
        [value] = values
        summaries.append((obj.step, value.tag, value.simple_value))

  return summaries


def load_pattern(path, n_tasks, n_components_per_layer, layer_ids):
  """Loads the allocation pattern.

  Args:
    path: (string) path to summaries.
    n_tasks: (int) number of tasks.
    n_components_per_layer: (int) number of components per layer.
    layer_ids: (list of ints) ids of layers with non-trivial learned allocation.

  Returns:
    A numpy array of shape [`n_tasks`, `n_components`], where `n_components`
    is the total number of components across all layers in `layer_ids`.
  """
  n_layers = len(layer_ids)
  layer_names = ['layer%d' % i for i in layer_ids]

  step_to_task = {}
  step_to_path_prob = {}

  for (step, tag, value) in load_summaries(path):
    tag = tag.split('/')

    if tag[-1] == 'active' and tag[0] == 'layer0':
      task = int(tag[1][9:])

      assert step not in step_to_task
      step_to_task[step] = task
    elif tag[-1].startswith('prob_path_'):
      component_id = int(tag[-1][10:])

      if step not in step_to_path_prob:
        step_to_path_prob[step] = {}

      offset = layer_names.index(tag[0])
      step_to_path_prob[step][
          n_components_per_layer * offset + component_id] = value

  n_components_total = n_layers * n_components_per_layer

  last_step = {}
  pattern = np.zeros((n_tasks, n_components_total))

  for step in step_to_path_prob:
    task = step_to_task[step]
    probs = sorted(list(step_to_path_prob[step].items()))

    assert len(probs) == n_components_total

    probs = [y for (_, y) in probs]
    probs = [int(y > 0.5) for y in probs]

    if task not in last_step or last_step[task] < step:
      last_step[task] = step
      pattern[task] = probs

  return pattern


def compute_pairwise_similarity(n_tasks, allocation_pattern):
  """Compute cosine similarity of the allocation pattern for all pairs of tasks.

  Args:
    n_tasks: (int) number of tasks.
    allocation_pattern: (numpy array) allocation pattern as returned from
      `load_pattern`.

  Returns:
    A numpy array of shape [`n_tasks`, `n_tasks`]. The `(i, j)` entry equals
    `(1 - cosine similarity of allocation patterns for tasks i and j)`.
  """
  similarity = np.zeros((n_tasks, n_tasks))

  for i in range(n_tasks):
    for j in range(n_tasks):
      similarity[i][j] = 1.0 - spatial.distance.cosine(
          allocation_pattern[i], allocation_pattern[j])

  return similarity


def display_pattern(pattern, save_dir=None):
  """Displays the allocation pattern from the three task clusters experiment.

  Args:
    pattern: (numpy array) allocation pattern as returned from
      `load_pattern`.
    save_dir: (string) if not None, the resulting figure will be saved under
      this path.
  """
  fig = plt.figure()
  ax = fig.add_subplot(111)

  ax.set_yticks([-0.5, 19.5, 29.5, 39.5])
  ax.set_yticklabels(['', '', ''])

  ax.set_yticks([9.5, 24.1, 34.5], minor=True)
  ax.set_yticklabels(['CIFAR', 'MNIST', 'Fashion'], minor=True, va='center')

  ax.set_xticks([-0.5, 15.5, 31.5, 47.5])
  ax.set_xticklabels(['', '', ''])

  ax.set_xticks([7.5, 23.5, 39.5], minor=True)
  ax.set_xticklabels(['Layer #1', 'Layer #2', 'Layer #3'], minor=True)

  ax.tick_params(axis='both', which='major', length=10, top=True, bottom=False)
  ax.tick_params(axis='both', which='minor', length=0, top=True, bottom=False)
  ax.tick_params(axis='y', which='minor', rotation=90, top=True, bottom=False)

  plt.grid(which='major', color='w', linewidth=1.2)

  plt.imshow(pattern)

  if save_dir:
    plt.savefig(save_dir, bbox_inches='tight')


def display_similarity(similarity, save_dir=None):
  """Displays the pattern similarities from the three task clusters experiment.

  Args:
    similarity: (numpy array) pairwise allocation pattern similarities, as
      returned from `compute_pairwise_similarity`.
    save_dir: (string) if not None, the resulting figure will be saved under
      this path.
  """
  fig = plt.figure()
  ax = fig.add_subplot(111)

  ax.set_yticks([-0.5, 19.5, 29.5, 39.5])
  ax.set_yticklabels(['', '', ''])

  ax.set_yticks([9.5, 24.1, 34.5], minor=True)
  ax.set_yticklabels(['CIFAR', 'MNIST', 'Fashion'], minor=True, va='center')

  ax.set_xticks([-0.5, 19.5, 29.5, 39.5])
  ax.set_xticklabels(['', '', ''])

  ax.set_xticks([9.5, 24.1, 34.5], minor=True)
  ax.set_xticklabels(['CIFAR', 'MNIST', 'Fashion'], minor=True, ha='center')

  ax.tick_params(axis='both', which='major', length=10, top=True, bottom=False)
  ax.tick_params(axis='both', which='minor', length=0, top=True, bottom=False)
  ax.tick_params(axis='y', which='minor', rotation=90, top=True, bottom=False)

  plt.imshow(similarity)

  if save_dir:
    plt.savefig(save_dir, bbox_inches='tight')


def main(_):
  plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
  plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

  pattern = load_pattern(
      path=FLAGS.dir,
      n_tasks=40,
      n_components_per_layer=16,
      layer_ids=[10, 19, 28])

  similarity = compute_pairwise_similarity(
      n_tasks=40, allocation_pattern=pattern)

  display_pattern(pattern, save_dir='./pattern.pdf')
  display_similarity(similarity, save_dir='./similarity.pdf')


if __name__ == '__main__':
  app.run(main)
