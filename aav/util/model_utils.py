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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Model utilities for extracting information from training checkpoints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pandas
import tensorflow as tf


def get_best_checkpoint_path(
    model_dir, metric='loss', eval_subdir='eval_one_pass'):
  """Gets the path of the best checkpoint by given metric.

  Args:
    model_dir: (str) Path to tf.Estimator model.
    metric: (str) Model evaluation metric over which to optimize.
    eval_subdir: (str) Subdir path within model_dir to search for evaluation
      events.
  Returns:
    (str) The path to the model best checkpoint.
  Raises:
    ValueError: If the given metric is not supported.
  """
  events = tf.event_accumulator.EventAccumulator(
      os.path.join(model_dir, eval_subdir))
  events.Reload()  # Actually read the event files into memory.

  step = None
  if metric == 'precision':
    step = _get_best_checkpoint_step(events, metric, higher_is_better=True)
  elif metric == 'loss':
    step = _get_best_checkpoint_step(events, metric, higher_is_better=False)
  elif metric == 'accuracy':
    step = _get_best_checkpoint_step(events, metric, higher_is_better=True)
  elif metric == 'recall':
    step = _get_best_checkpoint_step(events, metric, higher_is_better=True)
  else:
    raise ValueError('Unknown metric "%s" is not supported' % metric)

  return os.path.join(model_dir, 'model.ckpt-%d' % step)


def _get_best_checkpoint_step(
    events, metric_key='precision', higher_is_better=True):
  """Gets the global step number of the best checkpoint by given metric.

  Args:
    events: (tf.Events) The summary events for a model evaluation.
    metric_key: (str) The model evaluation metric key to optimize over.
    higher_is_better: (bool) Is a higher value of the metric better?
  Returns:
    (int) The global step number of the best checkpoint.
  """
  summary_df = pandas.DataFrame([
      {'step': entry.step, metric_key: entry.value}
      for entry in events.Scalars(metric_key)
  ])

  metric = summary_df[metric_key]
  best_index = None
  if higher_is_better:
    best_index = metric.idxmax()
  else:
    best_index = metric.idxmin()

  best_checkpoint = summary_df.iloc[best_index]
  return best_checkpoint.step
