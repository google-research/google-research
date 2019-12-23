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

"""A script for dumping experiment results to csv files.

This aims to make the data easy to consume via straight up python.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import re
import time

from absl import app
from absl import flags

import tensorflow.compat.v1 as tf

BATCH_SIZE = 128
MODEL_ROOT = ''
OUTPUT_DIR = '/tmp/igt_neurips19_imagenet_bs={}'.format(BATCH_SIZE)

flags.DEFINE_string(
    'root_dir',
    default=MODEL_ROOT,
    help='Root directory containing the experiment directories.')
flags.DEFINE_string('output_dir', default=OUTPUT_DIR, help='Output directory.')

FLAGS = flags.FLAGS


def get_experiment_dirs(root_dir):
  """Returns the list of directories in a root directory."""
  experiment_dirs = []
  for name in tf.io.gfile.listdir(root_dir):
    path = os.path.join(root_dir, name)
    if tf.io.gfile.isdir(path):
      experiment_dirs.append(path)
  return experiment_dirs


def get_metrics_from_file(file_path):
  """Returns metrics from a specific summary file."""
  steps = []
  losses = []
  accuracies = []
  for event in tf.compat.v1.train.summary_iterator(file_path):
    step = event.step
    got_loss = False
    got_accuracy = False
    for value in event.summary.value:
      if value.tag == 'loss':
        losses.append(value.simple_value)
        got_loss = True
      elif value.tag == 'top_1_accuracy':
        accuracies.append(value.simple_value)
        got_accuracy = True
    assert got_loss == got_accuracy
    if got_loss:
      steps.append(step)

  return steps, losses, accuracies


def get_metrics(metrics_dir):
  """Return metrics from a specific metrics directory."""
  files = []
  for name in tf.gfile.ListDirectory(metrics_dir):
    files.append(os.path.join(metrics_dir, name))

  data = []
  for file_path in files:
    steps, losses, accuracies = get_metrics_from_file(file_path)
    data.append((steps, losses, accuracies))
  data.sort()  # Sort by ascending step.

  all_steps = []
  all_losses = []
  all_accuracies = []
  for steps, losses, accuracies in data:
    all_steps.extend(steps)
    all_losses.extend(losses)
    all_accuracies.extend(accuracies)

  assert all_steps == sorted(all_steps)

  return all_steps, all_losses, all_accuracies


def filter_duplicates(steps, losses, accuracies):
  """Returns copies of the data with duplicates filtered out."""
  assert steps
  assert len(steps) == len(losses)
  assert len(steps) == len(accuracies)

  out_steps = [steps[0]]
  out_losses = [losses[0]]
  out_accuracies = [accuracies[0]]
  for cur in range(1, len(steps)):
    # Consider step for inclusion.
    prev = cur - 1
    if steps[cur] != steps[prev]:
      out_steps.append(steps[cur])
      out_losses.append(losses[cur])
      out_accuracies.append(accuracies[cur])
  return out_steps, out_losses, out_accuracies


def dump_metrics(experiment_dir, parameters='shift'):
  """Dump metrics from an experiment directory to a csv file.

  Args:
    experiment_dir: A string, the experiment directory.
    parameters: A string, the parameters for which to dump metrics (shift or
      true).
  """
  train_metrics_dir = 'eval_train_' + parameters
  train_metrics_dir = os.path.join(experiment_dir, train_metrics_dir)
  train_steps, train_losses, train_accuracies = get_metrics(train_metrics_dir)

  test_metrics_dir = 'eval_eval_' + parameters
  test_metrics_dir = os.path.join(experiment_dir, test_metrics_dir)
  test_steps, test_losses, test_accuracies = get_metrics(test_metrics_dir)

  # Observed some duplicates train / test steps.
  train_steps, train_losses, train_accuracies = filter_duplicates(
      train_steps, train_losses, train_accuracies)
  test_steps, test_losses, test_accuracies = filter_duplicates(
      test_steps, test_losses, test_accuracies)

  if train_steps != test_steps:
    print(train_steps)
    print(test_steps)
  assert train_steps == test_steps

  data = zip(train_steps, train_losses, train_accuracies, test_losses,
             test_accuracies)

  out_file = os.path.basename(experiment_dir)
  if parameters == 'true':
    out_file += '_true'
  out_file += '.csv'
  out_file = os.path.join(FLAGS.output_dir, out_file)
  tf.logging.info('Dumping results to %s', out_file)
  with tf.gfile.Open(out_file, 'w') as fd:
    spamwriter = csv.writer(fd)
    spamwriter.writerows(data)


def process_experiment(experiment_dir):
  dump_metrics(experiment_dir)
  if re.search(r'opt=eigt', os.path.basename(experiment_dir)):
    # Look for "true parameter" metrics.
    dump_metrics(experiment_dir, 'true')


def main(_):
  tf.logging.info('Using output directory: %s', FLAGS.output_dir)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  experiment_dirs = get_experiment_dirs(FLAGS.root_dir)
  tf.logging.info('Found %d experiments.', len(experiment_dirs))

  for i, experiment_dir in enumerate(experiment_dirs):
    start = time.time()
    tf.logging.info('Processing %d: %s', i, experiment_dir)
    process_experiment(experiment_dir)
    tf.logging.info('Processing took %d seconds.', time.time() - start)


if __name__ == '__main__':
  app.run(main)
